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
from .export_policy import export_policy, validate_export_reference
from .ball_protocol import VALIDATION_EPISODES, VALIDATION_SCHEMA, validation_cases
from .evaluation_cache import EvaluationCache, actor_fingerprint, evaluation_key, evaluation_due
from .viewer_selection import select_run
from .motion_evaluation import tracking_metrics
from .motion_protocol import GATES as MOTION_GATES, validate_motion_reference
from .contact_contract import mode_from_contract


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
    contact_mode = getattr(args, "contact_mode", "legacy")
    if args.initialize_from:
        source = json.loads(args.source_contract.read_text())
        if source.get("diagnostic_only"):
            raise ValueError("A zero-shot diagnostic bundle is not a trained checkpoint contract")
        if mode_from_contract(source) != contact_mode:
            raise ValueError("Contact semantics changed: use a matching checkpoint contract; legacy weights are not implicitly migrated")
    if not np.isfinite(args.actor_max_kl) or args.actor_max_kl <= 0:
        raise ValueError("Actor KL bound must be finite and positive")
    validate_export_reference(args.reference)
    if args.task == "motion":
        validate_motion_reference(json.loads(args.reference.with_suffix(".json").read_text()))
    args.out.mkdir(parents=True, exist_ok=False)
    mp_util.init(0, 1, "cuda:0", None)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    is_ball = args.task == "ball"
    is_motion = args.task == "motion"
    if is_motion and (args.control != "target_pd" or not args.no_select_viewer):
        raise ValueError("Motion pilots require target PD and --no-select-viewer")
    if is_ball and args.envs < VALIDATION_EPISODES:
        raise ValueError(f"Ball validation requires at least {VALIDATION_EPISODES} environments")
    if is_ball and (args.control != "target_pd" or args.stability != "guarded"):
        raise ValueError("Ball experiments require guarded target-PD fine-tuning")
    options = dict(speed_min=args.ball_speed_min, speed_max=args.ball_speed_max,
                   quiet_fraction=args.quiet_fraction, reward_mode=args.ball_reward, contact_mode=contact_mode)
    native = BallTask(rig, args.reference, VALIDATION_EPISODES, "cpu", **options) if is_ball else StandTask(
        rig, args.reference, 8, "cpu", engine_factory=NativeDummyEngine, control_mode=args.control, contact_mode=contact_mode)
    quiet = StandTask(rig, args.reference, 8, "cpu", engine_factory=NativeDummyEngine, control_mode=args.control, contact_mode=contact_mode) if is_ball else None
    timing = {"rollout_seconds": 0., "optimization_seconds": 0., "initial_native_seconds": 0.,
              "training_native_seconds": 0., "final_native_seconds": 0.}
    timing_phase = "initial"
    native_cache = EvaluationCache()
    last_native_key = None
    def evaluate_native(actor):
        nonlocal last_native_key
        began = time.monotonic()
        context = {"contract": native.observation_contract(), "backend": "native",
                   "episode_seconds": native.episode_seconds,
                   "cases": native.validation_cases if is_ball else "stand_eight_phases",
                   "experiment": native.experiment_contract() if is_ball else None,
                   "quiet_seconds": quiet.episode_seconds if quiet is not None else None,
                   "evaluation_schema": VALIDATION_SCHEMA if is_ball else "mimic_motion_tracking_v1" if is_motion else "stand_eight_phases"}
        last_native_key = evaluation_key(actor, context)
        def compute():
            metrics, traces = evaluate(native, actor)
            if is_motion:
                metrics.update(evaluation_schema="mimic_motion_tracking_v1",
                               motion_tracking=tracking_metrics(native, metrics, traces))
            if quiet is not None:
                quiet_metrics, _ = evaluate(quiet, actor)
                metrics.update(quiet_standing_passed=standing_passed(quiet_metrics), quiet_native=quiet_metrics)
            return metrics, traces
        metrics, traces = native_cache.get_or_compute(last_native_key, compute)
        timing[f"{timing_phase}_native_seconds"] += time.monotonic() - began
        return metrics, traces
    contract = native.observation_contract()
    provenance = None
    if args.initialize_from:
        validate_contract(json.loads(args.source_contract.read_text()), contract)
        provenance = {"path": str(args.initialize_from.resolve()), "sha256": sha256(args.initialize_from),
                      "source_contract_sha256": sha256(args.source_contract),
                      "mode": "weights_and_normalization_only; new optimizer, rollout state and local counters"}
    task = BallTask(rig, args.reference, args.envs, seed=args.seed, **options) if is_ball else StandTask(
        rig, args.reference, args.envs, seed=args.seed, control_mode=args.control, contact_mode=contact_mode)
    experiment = task.experiment_contract() if is_ball else {"task": args.task}
    config_path = args.mimickit / "data/agents/deepmimic_humanoid_ppo_agent.yaml"
    config = yaml.safe_load(config_path.read_text())
    # Native validation selects checkpoints. Upstream's test_episodes=8 actually
    # runs at least one episode in EVERY training world; disable that duplicate work.
    config.update(iters_per_output=args.evaluation_interval, test_episodes=0)
    if args.stability == "guarded":
        config.update(actor_max_kl=args.actor_max_kl, observation_normalization="frozen_loaded_statistics",
                      actor_gradient_clip=1.0)
    (args.out / "agent.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    curve = []
    updates = []
    best = BestCheckpoint(args.out, contract)
    gates = {"required_successes": 8, "max_mean_root_tracking_error_m": 0.05}
    if is_motion:
        gates = dict(MOTION_GATES)
    if is_ball:
        gates.update(required_successes=VALIDATION_EPISODES, required_confirmed_survived_hits=VALIDATION_EPISODES,
                     required_recovered_hits=VALIDATION_EPISODES, required_quiet_standing_pass=True,
                     ball_root_error="diagnostic/tie-break only; no fixed-position recovery gate")
    (args.out / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    (args.out / "protocol.json").write_text(json.dumps({"contract": contract, "stability": args.stability,
        "initialization": provenance, "experiment": experiment, "seconds": args.seconds, "iterations": args.iterations,
        "seed": args.seed, "envs": args.envs, "evaluation_interval": args.evaluation_interval,
        "mimickit_revision": MIMICKIT_REVISION, "agent_config": config,
        "validation": {"schema": VALIDATION_SCHEMA, "cases": validation_cases(args.ball_speed_min, args.ball_speed_max)}
                      if is_ball else {"schema": "mimic_motion_tracking_v1" if is_motion else "stand_eight_phases"}}, indent=2), encoding="utf-8")

    class BudgetComplete(Exception):
        pass

    base_agent = GuardedPPO if args.stability == "guarded" else PPOAgent

    class BoundedPPO(base_agent):
        def test_model(self, num_episodes):
            if num_episodes == 0:
                # Upstream's zero-episode path still switches the training task to
                # TEST and resets it. Keep training resets stochastic instead.
                return {"mean_return": 0., "mean_ep_len": 0., "num_eps": 0}
            return super().test_model(num_episodes)

        def _rollout_train(self, num_steps):
            torch.cuda.synchronize()
            began = time.monotonic()
            result = super()._rollout_train(num_steps)
            torch.cuda.synchronize()
            timing["rollout_seconds"] += time.monotonic() - began
            return result

        def _update_model(self):
            torch.cuda.synchronize()
            began = time.monotonic()
            result = super()._update_model()
            torch.cuda.synchronize()
            timing["optimization_seconds"] += time.monotonic() - began
            return result

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
            completed = self._iter + 1
            if evaluation_due(completed, args.evaluation_interval):
                # Upstream output uses zero-based iteration indices. Schedule here
                # to evaluate after exactly N completed updates, never after update 1.
                samples = self._update_sample_count()
                metrics, _ = evaluate_native(StandPolicy(self).eval().cpu())
                curve.append({"iteration": completed, "samples": samples,
                              "elapsed_seconds": time.monotonic() - start, **metrics})
                self.save(str(args.out / f"evaluation_{completed:06d}.pt"))
                if best.consider(self.save, metrics, iteration=completed, samples=samples):
                    native_cache.pin(last_native_key)
                curve[-1]["checkpoint_selection"] = best.decisions[-1]
                (args.out / "learning_curve.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")
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
    native_cache.pin(last_native_key)
    start = time.monotonic()
    timing_phase = "training"
    agent.deadline = start + args.seconds
    stop_reason = "sample_budget"
    try:
        agent.train_model(args.iterations * args.envs * config["steps_per_iter"],
                          str(args.out), False, "txt")
    except BudgetComplete:
        stop_reason = "time_budget"
    elapsed = time.monotonic() - start
    timing_phase = "final"
    agent.save(str(args.out / "model.pt"))
    if agent._sample_count == 0:
        raise RuntimeError("No training samples completed")
    trained = StandPolicy(agent).eval()
    trained_fingerprint = actor_fingerprint(trained)
    trained_gpu, _ = evaluate(task, trained)
    trained_cpu, traces = evaluate_native(copy.deepcopy(trained).cpu())
    np.savez_compressed(args.out / "native_rollout.npz", **traces)
    if best.consider(agent.save, trained_cpu, iteration=agent._iter, samples=agent._sample_count):
        native_cache.pin(last_native_key)
    if sha256(best.path) != best.metadata["checkpoint_sha256"]:
        raise ValueError("Best checkpoint hash changed before export")
    agent.load(str(best.path))
    selected = StandPolicy(agent).eval()
    selected_matches_final = actor_fingerprint(selected) == trained_fingerprint
    selected_gpu = trained_gpu if selected_matches_final else evaluate(task, selected)[0]
    selected_cpu, selected_traces = evaluate_native(copy.deepcopy(selected).cpu())
    error = export_policy(copy.deepcopy(selected).cpu(), selected_traces["observations"],
                          args.reference, contract, args.out / "export")
    native.episode_seconds = 3.0 if is_motion else 5.0
    extended_cpu = None if is_motion else selected_cpu if is_ball else evaluate(native, copy.deepcopy(selected).cpu())[0]
    quiet_extended = None
    if quiet is not None:
        quiet.episode_seconds = 5.
        quiet_extended, _ = evaluate(quiet, copy.deepcopy(selected).cpu())
    (args.out / "export" / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    report = {"seed": args.seed, "control_mode": args.control, "mimickit_revision": MIMICKIT_REVISION,
              "reference_sha256": task.reference.sha256, "envs": args.envs,
              "iterations": agent._iter, "samples": agent._sample_count,
              "training_seconds": elapsed, "stop_reason": stop_reason,
              "timing": timing,
              "evaluation_cache": native_cache.statistics(),
              "selected_newton_reused_final": selected_matches_final,
              "stability": args.stability, "initialization": provenance,
              "experiment": experiment, "selected_quiet_five_second_native": quiet_extended,
              "ball_gate_passed": (selected_cpu["quiet_standing_passed"] and
                                   selected_cpu["recovered_hits"] == VALIDATION_EPISODES) if is_ball else None,
              "initial_native": initial_cpu, "trained_native": trained_cpu,
              "trained_newton": trained_gpu, "onnx_max_action_error": error,
              "selected_native": selected_cpu, "selected_newton": selected_gpu,
              "selected_five_second_native": extended_cpu, "selected_checkpoint": best.metadata,
              "checkpoint_selection": best.decisions,
              "export_source": "best.pt", "final_checkpoint": "model.pt",
              "standing_gate_passed": None if is_motion else standing_passed(selected_cpu),
              "final_standing_gate_passed": None if is_motion else standing_passed(trained_cpu),
              "motion_tracking_gate_passed": selected_cpu["motion_tracking"]["passed"] if is_motion else None,
              "guard_summary": {"max_accepted_rollout_kl": max((r["actor_kl"] for r in updates), default=0),
                                "accepted_actor_steps": sum(r["actor_steps_accepted"] for r in updates),
                                "rejected_actor_steps": sum(r["actor_steps_rejected"] for r in updates)},
              "scope": "Experimental motion-guided policy; fixed evaluation cases, not held-out motions. Viewer selection is for review, not recovery acceptance."}
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.no_select_viewer:
        report["viewer_selection"] = select_run(args.out)
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
    parser.add_argument("--contact-mode", choices=("legacy", "support_v2"), default="legacy")
    parser.add_argument("--task", choices=("stand", "ball", "motion"), default="stand")
    parser.add_argument("--ball-speed-min", type=float, default=1.)
    parser.add_argument("--ball-speed-max", type=float, default=2.)
    parser.add_argument("--quiet-fraction", type=float, default=.25)
    parser.add_argument("--ball-reward", choices=("reference", "recovery_v1"), default="reference")
    parser.add_argument("--stability", choices=("upstream", "guarded"), default="upstream")
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--source-contract", type=Path)
    parser.add_argument("--actor-max-kl", type=float, default=0.02)
    parser.add_argument("--evaluation-interval", type=int, default=64)
    parser.add_argument("--no-select-viewer", action="store_true", help="Keep the current F5/F6 export selected")
    train(parser.parse_args())
