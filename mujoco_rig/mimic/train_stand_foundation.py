"""Fresh stationary PPO; retain native standing checkpoints across uninterrupted training."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .baseline import sha256
from .checkpoints import BestCheckpoint, checkpoint_rank, validate_contract
from .continuous_stand import GATES, evaluate_retention, passing_cases
from .native_engine import NativeDummyEngine
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate, MIMICKIT_REVISION


def foundation_rank(metrics):
    if any(not np.isfinite(metrics[key]) or metrics[key] < 0 for key in
           ("worst_tail_foot_motion_m", "worst_mean_foot_error_m")):
        raise ValueError("Foot convergence ranking requires finite nonnegative measurements")
    base = checkpoint_rank(metrics)
    return (*base[:-1], -metrics["worst_tail_foot_motion_m"],
            -metrics["worst_mean_foot_error_m"], base[-1])


def add_foot_metrics(task, metrics, traces):
    q = torch.from_numpy(traces["qpos_after"])
    positions, _ = task.mapper.from_native_qpos(q.reshape(-1, q.shape[-1]))
    feet = positions[:, task.feet].reshape(len(q), 8, 2, 3)
    target, _ = task.mapper.from_native_qpos(task.sample_reference(task.offset)[0])
    error = torch.linalg.vector_norm(feet - target[:, task.feet], dim=-1).amax(-1)
    worst_motion, worst_mean = 0., 0.
    for case in metrics["cases"]:
        i = case["case"]
        end = int(traces["active"][:, i].sum())
        window = min(end, int(5. / task.dt))
        motion = float(torch.linalg.vector_norm(feet[end-window:end, i] - feet[end-window, i], dim=-1).max())
        mean = float(error[min(int(2./task.dt), end-1):end, i].mean())
        case.update(tail_foot_motion_m=motion, mean_foot_error_m=mean)
        worst_motion, worst_mean = max(worst_motion, motion), max(worst_mean, mean)
    metrics.update(worst_tail_foot_motion_m=worst_motion, worst_mean_foot_error_m=worst_mean)


def train(args):
    if args.envs < 32 or args.envs % 8 or args.seconds <= 0 or args.evaluation_interval < 1:
        raise ValueError("Require at least 32 worlds divisible by eight and positive budgets")
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .continuous_stand_task import ContinuousStandTask

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mp_util.init(0, 1, "cuda:0", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    native = ContinuousStandTask(rig, args.reference, engine_factory=NativeDummyEngine)
    contract = native.observation_contract()
    config = yaml.safe_load(args.config.read_text())
    config.update(test_episodes=0, iters_per_output=args.evaluation_interval)
    if args.actor_learning_rate is not None:
        if not 0 < args.actor_learning_rate < float("inf"):
            raise ValueError("Actor learning rate must be finite and positive")
        config["actor_optimizer"]["learning_rate"] = args.actor_learning_rate
    previous_anchor = float(config.get("standing_foot_anchor_sigma_m", 0.))
    if args.foot_anchor_sigma is not None:
        if not 0 < args.foot_anchor_sigma < float("inf"):
            raise ValueError("Foot anchor scale must be finite and positive")
        config["standing_foot_anchor_sigma_m"] = args.foot_anchor_sigma
    anchor_sigma = float(config.get("standing_foot_anchor_sigma_m", 0.))
    if args.late_states and not anchor_sigma:
        raise ValueError("Late-state curriculum requires the quiet foot-anchor objective")
    if args.resume and previous_anchor != anchor_sigma and args.calibration_steps < 1:
        raise ValueError("A changed standing reward requires frozen-actor critic calibration")
    if args.stable_normalization:
        config["observation_normalizer"] = "stable_moments_v1"
    if args.resume:
        validate_contract(json.loads((args.resume.parent / "contract.json").read_text()), contract)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "agent.yaml").write_text(yaml.safe_dump(config))
    (args.out / "contract.json").write_text(json.dumps(contract, indent=2))
    viewers = Path("logs/mimickit-viewer")
    viewer_hashes = {str(p): sha256(p) for p in (viewers / "stand.json", viewers / "ball.json") if p.exists()}
    protocol = dict(schema="mimic_fresh_stationary_training_v1", seed=args.seed,
        reference=str(args.reference), reference_sha256=sha256(args.reference),
        model_sha256=sha256(rig.path), config=config, seconds=args.seconds, envs=args.envs,
        initialization="fresh_actor_critic_and_normalization" if not args.resume else str(args.resume),
        initial_checkpoint_sha256=sha256(args.resume) if args.resume else None,
        normalization=config.get("observation_normalizer", "upstream running moments"),
        lifecycle="reset only at failure or 30-second timeout; independent native evaluation",
        selection="retain every incumbent native passing case; prioritize settled foot motion before root error; rejected evaluations do not overwrite best.pt",
        training_continuation="bounded fresh learning may continue after a rejected evaluation",
        foot_anchor_sigma_m=anchor_sigma, calibration_steps=args.calibration_steps,
        late_states=str(args.late_states) if args.late_states else None,
        late_states_sha256=sha256(args.late_states) if args.late_states else None,
        reset_distribution="50% normal starts / 50% posture-matched real late states" if args.late_states else "normal starts",
        gates=GATES, mimickit_revision=MIMICKIT_REVISION, viewer_hashes=viewer_hashes,
        source_hashes={name: sha256(Path(__file__).with_name(name)) for name in
            ("train_stand_foundation.py", "continuous_stand.py", "continuous_stand_task.py",
             "stand_task.py", "native_engine.py", "target_control.py", "newton_adapter.py",
             "stable_normalizer.py", "stand_foundation_task.py", "checkpoints.py")})
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (args.out / "runner.py").write_text(Path(__file__).read_text())
    if anchor_sigma:
        from .stand_foundation_task import FootAnchoredStandTask
        task = FootAnchoredStandTask(rig, args.reference, args.envs, "cuda:0", seed=args.seed,
                                    foot_anchor_sigma_m=anchor_sigma)
    else:
        task = ContinuousStandTask(rig, args.reference, args.envs, "cuda:0", seed=args.seed)
    if args.late_states:
        task.load_late_states(args.late_states)
    class FoundationAgent(PPOAgent):
        def _build_normalizers(self):
            super()._build_normalizers()
            if config.get("observation_normalizer") == "stable_moments_v1":
                from .stable_normalizer import install
                install(self)

    agent = FoundationAgent(config, task, "cuda:0")
    if args.resume:
        agent.load(str(args.resume))
    best = BestCheckpoint(args.out, contract, rank_function=foundation_rank,
        selection_label="Retain all passing stationary cases; rank standing gate and survival, then worst final-five-second foot motion, worst mean foot error, root error")
    curve = []
    started = time.monotonic()

    def assess(iteration):
        actor = StandPolicy(agent).cpu().eval()
        metrics, traces = evaluate_retention(native, actor)
        add_foot_metrics(native, metrics, traces)
        accepted = best.consider(agent.save, metrics, iteration=iteration, samples=agent._sample_count)
        if accepted:
            np.savez_compressed(args.out / "selected-native.npz", **traces)
        row = dict(iteration=iteration, elapsed_seconds=time.monotonic() - started,
                   metrics=metrics, selection=best.decisions[-1])
        curve.append(row)
        (args.out / "learning_curve.json").write_text(json.dumps(curve, indent=2))
        print(json.dumps(dict(iteration=iteration, native_cases=sorted(passing_cases(metrics)),
            survival=[round(c["survival_seconds"], 3) for c in metrics["cases"]],
            accepted=accepted, full_gate=metrics["passed"])), flush=True)
        return metrics

    agent.save(str(args.out / "initial.pt"))
    initial = assess(0)
    if args.calibration_steps:
        from .critic_calibration import calibrate
        from .train_continuous_stand import calibration_protocol
        calibration_spec = calibration_protocol(task)
        calibration_spec.reward.update(schema="mimic_stationary_foot_anchor_v1", foot_anchor_sigma_m=anchor_sigma)
        calibration = calibrate(agent, task, args.calibration_steps, args.out, protocol=calibration_spec)
        if not calibration["passed"]:
            (args.out / "report.json").write_text(json.dumps(dict(
                stop_reason="critic_calibration_rejected", iterations=0, full_gate_passed=False,
                selected_checkpoint=best.metadata, initial_native=initial), indent=2))
            return
        agent.save(str(args.out / "calibrated.pt"))
        print(json.dumps(dict(critic_calibration_passed=True, actor_unchanged=True)), flush=True)
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    agent._init_train()
    deadline = time.monotonic() + args.seconds
    stop_reason = "time_budget"
    while time.monotonic() < deadline and agent._iter < args.iterations:
        info = agent._train_iter()
        agent._sample_count = agent._update_sample_count()
        agent._iter += 1
        row = dict(iteration=agent._iter, samples=agent._sample_count,
                   elapsed_seconds=time.monotonic() - started,
                   max_episode_seconds=float(task.get_env_time().max()),
                   **{k: float(v) for k, v in info.items()})
        if not all(np.isfinite(v) for v in row.values()):
            raise FloatingPointError("Nonfinite stationary training metrics")
        with (args.out / "updates.jsonl").open("a") as file:
            file.write(json.dumps(row) + "\n")
        if agent._iter % 16 == 0:
            print(json.dumps({k: row[k] for k in ("iteration", "mean_ep_len", "mean_return", "critic_loss", "max_episode_seconds")}), flush=True)
        if agent._iter % args.evaluation_interval == 0:
            agent.save(str(args.out / f"evaluation_{agent._iter:06d}.pt"))
            metrics = assess(agent._iter)
            if metrics["passed"] and args.stop_on_pass:
                stop_reason = "native_gate_passed"
                break
    if agent._iter >= args.iterations and stop_reason == "time_budget":
        stop_reason = "iteration_budget"
    agent.save(str(args.out / "model.pt"))
    final = assess(agent._iter)
    agent.load(str(best.path))
    gpu = ContinuousStandTask(rig, args.reference, 8, "cuda:0", seed=args.seed)
    gpu_metrics, traces = evaluate_retention(gpu, StandPolicy(agent).eval())
    np.savez_compressed(args.out / "selected-gpu.npz", **traces)
    assert all(sha256(Path(p)) == digest for p, digest in viewer_hashes.items())
    report = dict(schema="mimic_fresh_stationary_report_v1", stop_reason=stop_reason,
        iterations=agent._iter, samples=agent._sample_count, elapsed_seconds=time.monotonic() - started,
        initial_native=initial, final_native=final, selected_checkpoint=best.metadata,
        selected_native=best.metadata["metrics"], selected_gpu=gpu_metrics,
        full_gate_passed=best.metadata["metrics"]["passed"] and gpu_metrics["passed"],
        viewers_unchanged=True, exported=False,
        scope="Training-posture validation; long holds and unseen initial disturbances still required")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(complete=True, full_gate_passed=report["full_gate_passed"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--evaluation-interval", type=int, default=64)
    parser.add_argument("--seed", type=int, default=270927)
    parser.add_argument("--stop-on-pass", action="store_true")
    parser.add_argument("--stable-normalization", action="store_true")
    parser.add_argument("--foot-anchor-sigma", type=float)
    parser.add_argument("--calibration-steps", type=int, default=0)
    parser.add_argument("--late-states", type=Path)
    parser.add_argument("--actor-learning-rate", type=float)
    train(parser.parse_args())
