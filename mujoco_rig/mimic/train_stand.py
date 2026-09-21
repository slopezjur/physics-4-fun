"""Bounded PPO with explicit warm starts, guarded updates and best-policy export."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .runtime import activate, MIMICKIT_REVISION
from .rig import Rig
from .policy import StandPolicy
from .evaluate_stand import evaluate
from .baseline import sha256
from .checkpoints import BestCheckpoint, standing_passed, validate_contract
from .export_policy import export_policy


def train(args):
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .stand_task import StandTask
    from .native_engine import NativeDummyEngine
    from .guarded_ppo import GuardedPPO
    from .ball_task import BallTask

    if args.envs < 8 or args.iterations < 1 or not np.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("Require at least eight worlds, one iteration and a positive time budget")
    if args.evaluation_interval < 8 or args.evaluation_interval % 8:
        raise ValueError("Evaluation interval must be a positive multiple of eight")
    if bool(args.initialize_from) != bool(args.source_contract):
        raise ValueError("Warm starts require both --initialize-from and --source-contract")
    if args.stability == "guarded" and not args.initialize_from:
        raise ValueError("Guarded fine-tuning requires a trained checkpoint and its contract")
    if not np.isfinite(args.actor_max_kl) or args.actor_max_kl <= 0:
        raise ValueError("Actor KL bound must be finite and positive")
    args.out.mkdir(parents=True, exist_ok=False)
    mp_util.init(0, 1, "cuda:0", None)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    is_ball = args.task == "ball"
    if is_ball and (args.control != "target_pd" or args.stability != "guarded"):
        raise ValueError("Ball experiments require guarded target-PD fine-tuning")
    options = dict(speed_min=args.ball_speed_min, speed_max=args.ball_speed_max, quiet_fraction=args.quiet_fraction)
    native = BallTask(rig, args.reference, 8, "cpu", **options) if is_ball else StandTask(
        rig, args.reference, 8, "cpu", engine_factory=NativeDummyEngine, control_mode=args.control)
    quiet = StandTask(rig, args.reference, 8, "cpu", engine_factory=NativeDummyEngine, control_mode=args.control) if is_ball else None
    def evaluate_native(actor):
        metrics, traces = evaluate(native, actor)
        if quiet is not None:
            quiet_metrics, _ = evaluate(quiet, actor)
            metrics.update(quiet_standing_passed=standing_passed(quiet_metrics), quiet_native=quiet_metrics)
        return metrics, traces
    contract = native.observation_contract()
    provenance = None
    if args.initialize_from:
        validate_contract(json.loads(args.source_contract.read_text()), contract)
        provenance = {"path": str(args.initialize_from.resolve()), "sha256": sha256(args.initialize_from),
                      "source_contract_sha256": sha256(args.source_contract),
                      "mode": "weights_and_normalization_only; new optimizer, rollout state and local counters"}
    task = BallTask(rig, args.reference, args.envs, seed=args.seed, **options) if is_ball else StandTask(
        rig, args.reference, args.envs, seed=args.seed, control_mode=args.control)
    experiment = task.experiment_contract() if is_ball else {"task": "stand"}
    config_path = args.mimickit / "data/agents/deepmimic_humanoid_ppo_agent.yaml"
    config = yaml.safe_load(config_path.read_text())
    config.update(iters_per_output=8, test_episodes=8)
    if args.stability == "guarded":
        config.update(actor_max_kl=args.actor_max_kl, observation_normalization="frozen_loaded_statistics",
                      actor_gradient_clip=1.0)
    (args.out / "agent.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    curve = []
    updates = []
    best = BestCheckpoint(args.out, contract)
    gates = {"required_successes": 8, "max_mean_root_tracking_error_m": 0.05}
    if is_ball:
        gates.update(required_confirmed_survived_hits=8, required_quiet_standing_pass=True,
                     ball_root_error="diagnostic/tie-break only; no fixed-position recovery gate")
    (args.out / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    (args.out / "protocol.json").write_text(json.dumps({"contract": contract, "stability": args.stability,
        "initialization": provenance, "experiment": experiment, "seconds": args.seconds, "iterations": args.iterations,
        "seed": args.seed, "envs": args.envs, "evaluation_interval": args.evaluation_interval,
        "mimickit_revision": MIMICKIT_REVISION, "agent_config": config}, indent=2), encoding="utf-8")

    class BudgetComplete(Exception):
        pass

    base_agent = GuardedPPO if args.stability == "guarded" else PPOAgent

    class BoundedPPO(base_agent):
        def _output_train_model(self, iteration, out_model_file, int_out_dir):
            super()._output_train_model(iteration, out_model_file, int_out_dir)
            if iteration % args.evaluation_interval == 0:
                metrics, _ = evaluate_native(StandPolicy(self).eval().cpu())
                curve.append({"iteration": iteration + 1, "samples": self._sample_count,
                              "elapsed_seconds": time.monotonic() - start, **metrics})
                self.save(str(args.out / f"evaluation_{iteration + 1:06d}.pt"))
                best.consider(self.save, metrics, iteration=iteration + 1, samples=self._sample_count)
                (args.out / "learning_curve.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")

        def _train_iter(self):
            if time.monotonic() >= self.deadline:
                raise BudgetComplete
            result = super()._train_iter()
            if not all(torch.isfinite(p).all() for p in self.parameters()):
                raise FloatingPointError("Nonfinite PPO parameters")
            for key, value in result.items():
                if not np.isfinite(float(value)):
                    raise FloatingPointError(f"Nonfinite training metric: {key}")
            if args.stability == "guarded":
                row = {"iteration": self._iter + 1, **{key: float(result[key]) for key in
                       ("actor_kl", "actor_attempted_kl", "actor_steps_accepted", "actor_steps_rejected")}}
                updates.append(row)
                with (args.out / "updates.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row) + "\n")
            return result

    agent = BoundedPPO(config, task, "cuda:0")
    if args.initialize_from:
        agent.load(str(args.initialize_from))
        if agent._obs_norm.get_count().item() <= 0:
            raise ValueError("Warm-start checkpoint has no trained observation normalization")
    agent.save(str(args.out / "initial.pt"))
    initial = StandPolicy(agent).eval().cpu()
    initial_cpu, _ = evaluate_native(initial)
    best.consider(agent.save, initial_cpu, iteration=0, samples=0)
    start = time.monotonic()
    agent.deadline = start + args.seconds
    stop_reason = "sample_budget"
    try:
        agent.train_model(args.iterations * args.envs * config["steps_per_iter"],
                          str(args.out), False, "txt")
    except BudgetComplete:
        stop_reason = "time_budget"
    elapsed = time.monotonic() - start
    agent.save(str(args.out / "model.pt"))
    if agent._sample_count == 0:
        raise RuntimeError("No training samples completed")
    trained = StandPolicy(agent).eval()
    trained_gpu, _ = evaluate(task, trained)
    trained_cpu, traces = evaluate_native(copy.deepcopy(trained).cpu())
    np.savez_compressed(args.out / "native_rollout.npz", **traces)
    best.consider(agent.save, trained_cpu, iteration=agent._iter, samples=agent._sample_count)
    if sha256(best.path) != best.metadata["checkpoint_sha256"]:
        raise ValueError("Best checkpoint hash changed before export")
    agent.load(str(best.path))
    selected = StandPolicy(agent).eval()
    selected_gpu, _ = evaluate(task, selected)
    selected_cpu, selected_traces = evaluate_native(copy.deepcopy(selected).cpu())
    error = export_policy(copy.deepcopy(selected).cpu(), selected_traces["observations"],
                          args.reference, contract, args.out / "export")
    native.episode_seconds = 5.0
    extended_cpu, _ = evaluate(native, copy.deepcopy(selected).cpu())
    quiet_extended = None
    if quiet is not None:
        quiet.episode_seconds = 5.
        quiet_extended, _ = evaluate(quiet, copy.deepcopy(selected).cpu())
    (args.out / "export" / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    report = {"seed": args.seed, "control_mode": args.control, "mimickit_revision": MIMICKIT_REVISION,
              "reference_sha256": task.reference.sha256, "envs": args.envs,
              "iterations": agent._iter, "samples": agent._sample_count,
              "training_seconds": elapsed, "stop_reason": stop_reason,
              "stability": args.stability, "initialization": provenance,
              "experiment": experiment, "selected_quiet_five_second_native": quiet_extended,
              "ball_gate_passed": (selected_cpu["quiet_standing_passed"] and selected_cpu["survived_hits"] == 8) if is_ball else None,
              "initial_native": initial_cpu, "trained_native": trained_cpu,
              "trained_newton": trained_gpu, "onnx_max_action_error": error,
              "selected_native": selected_cpu, "selected_newton": selected_gpu,
              "selected_five_second_native": extended_cpu, "selected_checkpoint": best.metadata,
              "export_source": "best.pt", "final_checkpoint": "model.pt",
              "standing_gate_passed": standing_passed(selected_cpu),
              "final_standing_gate_passed": standing_passed(trained_cpu),
              "guard_summary": {"max_accepted_rollout_kl": max((r["actor_kl"] for r in updates), default=0),
                                "accepted_actor_steps": sum(r["actor_steps_accepted"] for r in updates),
                                "rejected_actor_steps": sum(r["actor_steps_rejected"] for r in updates)},
              "scope": "Experimental motion-guided policy; fixed evaluation cases, not held-out motions. Viewer policy unchanged."}
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path(__file__).parent / "assets/stand_reference.npz")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--seed", type=int, default=210921)
    parser.add_argument("--control", choices=("torque", "target_pd"), default="torque")
    parser.add_argument("--task", choices=("stand", "ball"), default="stand")
    parser.add_argument("--ball-speed-min", type=float, default=1.)
    parser.add_argument("--ball-speed-max", type=float, default=2.)
    parser.add_argument("--quiet-fraction", type=float, default=.25)
    parser.add_argument("--stability", choices=("upstream", "guarded"), default="upstream")
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--source-contract", type=Path)
    parser.add_argument("--actor-max-kl", type=float, default=0.02)
    parser.add_argument("--evaluation-interval", type=int, default=64)
    train(parser.parse_args())
