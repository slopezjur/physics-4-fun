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
from . import weight_shift
from .probe_steps import validate_reference_substitution


def train(args):
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .stand_task import StandTask
    from .native_engine import NativeDummyEngine
    from .guarded_ppo import GuardedPPO
    from .ball_task import BallTask
    from .rehearsal_task import RehearsalTask

    if args.envs < 8 or args.iterations < 1 or not np.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("Require at least eight worlds, one iteration and a positive time budget")
    if args.evaluation_interval < 8 or args.evaluation_interval % 8:
        raise ValueError("Evaluation interval must be a positive multiple of eight")
    viewer_policy = getattr(args, "viewer_policy", "best")
    ball_curriculum = getattr(args, "ball_curriculum", "all_directions")
    if args.task != "ball" and (viewer_policy != "best" or ball_curriculum != "all_directions"):
        raise ValueError("Focused ball curricula and final-candidate review require the ball task")
    shift_seconds = getattr(args, "weight_shift_seconds", 3.)
    shift_quiet_fraction = getattr(args, "weight_shift_quiet_fraction", 0.)
    rehearsal = shift_seconds != 3. or shift_quiet_fraction != 0.
    shift_reward_mode = getattr(args, "weight_shift_reward", "reference")
    anchor_limit = getattr(args, "quiet_anchor_max_kl", 0.)
    calibration_steps = getattr(args, "critic_calibration_steps", 0)
    calibration_only = getattr(args, "calibrate_critic_only", False)
    actor_backtracks = getattr(args, "actor_backtracks", 0)
    actor_project_anchor = getattr(args, "actor_project_anchor", False)
    if type(actor_backtracks) is not int or not 0 <= actor_backtracks <= 12:
        raise ValueError("Actor backtracks must be between zero and twelve")
    if actor_backtracks and args.stability != "guarded":
        raise ValueError("Actor backtracking requires guarded PPO")
    if actor_project_anchor and (not actor_backtracks or anchor_limit <= 0):
        raise ValueError("Anchor projection requires backtracking and a positive fixed quiet-anchor limit")
    if (not np.isfinite(anchor_limit) or anchor_limit < 0 or calibration_steps < 0
            or shift_reward_mode not in ("reference", "precision_v1")):
        raise ValueError("Invalid weight-shift objective, anchor limit or calibration budget")
    if (shift_reward_mode != "reference" or anchor_limit or calibration_steps or calibration_only) and (
            not rehearsal or args.task != "weight_shift" or shift_quiet_fraction <= 0):
        raise ValueError("Precision reward, anchor and calibration require mixed weight-shift rehearsal")
    if (shift_reward_mode != "reference" or calibration_only) and not calibration_steps:
        raise ValueError("A changed reward requires explicit critic calibration before actor updates")
    if rehearsal and (args.task != "weight_shift" or not np.isfinite(shift_seconds)
                      or shift_seconds <= 3 or not np.isfinite(shift_quiet_fraction)
                      or not 0 <= shift_quiet_fraction < 1):
        raise ValueError("Rehearsal requires weight_shift, a horizon above three seconds and quiet fraction in [0,1)")
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
        if source.get("normalization_preparation") and args.task != "weight_shift":
            raise ValueError("Prepared motion normalization is limited to its weight-shift curriculum")
    if not np.isfinite(args.actor_max_kl) or args.actor_max_kl <= 0:
        raise ValueError("Actor KL bound must be finite and positive")
    critic_grad_clip = getattr(args, "critic_grad_clip", 0.)
    if not np.isfinite(critic_grad_clip) or critic_grad_clip < 0:
        raise ValueError("Critic gradient clip must be finite and nonnegative")
    validate_export_reference(args.reference)
    if args.task == "motion":
        validate_motion_reference(json.loads(args.reference.with_suffix(".json").read_text()))
    if args.task == "weight_shift":
        weight_shift.validate_reference(json.loads(args.reference.with_suffix(".json").read_text()))
        if args.stability != "guarded" or not args.initialize_from:
            raise ValueError("Weight-shift pilots require a guarded trained Stand warm start")
    args.out.mkdir(parents=True, exist_ok=False)
    mp_util.init(0, 1, "cuda:0", None)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    is_ball = args.task == "ball"
    is_weight_shift = args.task == "weight_shift"
    is_motion = args.task in ("motion", "weight_shift")
    if is_motion and (args.control != "target_pd" or not args.no_select_viewer):
        raise ValueError("Motion pilots require target PD and --no-select-viewer")
    if is_ball and args.envs < VALIDATION_EPISODES:
        raise ValueError(f"Ball validation requires at least {VALIDATION_EPISODES} environments")
    if is_ball and (args.control != "target_pd" or args.stability != "guarded"):
        raise ValueError("Ball experiments require guarded target-PD fine-tuning")
    options = dict(speed_min=args.ball_speed_min, speed_max=args.ball_speed_max,
                   quiet_fraction=args.quiet_fraction, reward_mode=args.ball_reward, contact_mode=contact_mode,
                   curriculum=ball_curriculum)
    native = BallTask(rig, args.reference, VALIDATION_EPISODES, "cpu", **options) if is_ball else StandTask(
        rig, args.reference, 8, "cpu", engine_factory=NativeDummyEngine, control_mode=args.control, contact_mode=contact_mode)
    quiet_reference = Path(__file__).parent / "assets/stand_reference.npz" if is_weight_shift else args.reference
    quiet = StandTask(rig, quiet_reference, 8, "cpu", engine_factory=NativeDummyEngine, control_mode=args.control, contact_mode=contact_mode) if is_ball or is_weight_shift else None
    extended = None
    if rehearsal:
        if shift_seconds > native.reference.duration:
            raise ValueError("Extended motion episodes must fit inside the reference")
        extended = StandTask(rig, args.reference, 8, "cpu", engine_factory=NativeDummyEngine,
                             control_mode=args.control, contact_mode=contact_mode)
        extended.episode_seconds = shift_seconds
        if shift_reward_mode != "reference":
            # Native evaluation must use exactly the same motion objective as CUDA.
            def precision_native():
                return RehearsalTask(rig, args.reference, quiet_reference, 8, "cpu",
                    engine_factory=NativeDummyEngine, quiet_fraction=0., motion_seconds=shift_seconds,
                    control_mode=args.control, contact_mode=contact_mode, reward_mode=shift_reward_mode)
            native, extended = precision_native(), precision_native()
            extended.episode_seconds = shift_seconds
    evaluation_schema = (weight_shift.REHEARSAL_SCHEMA if rehearsal else weight_shift.EVALUATION_SCHEMA if is_weight_shift else VALIDATION_SCHEMA if is_ball
                         else "mimic_motion_tracking_v1" if is_motion else "stand_eight_phases")
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
                   "quiet_contract": quiet.observation_contract() if quiet is not None else None,
                   "quiet_extended_seconds": 5. if is_weight_shift else None,
                   "extended_motion_seconds": shift_seconds if rehearsal else None,
                   "evaluation_schema": evaluation_schema}
        if rehearsal:
            context["reward_schema"] = shift_reward_mode
        last_native_key = evaluation_key(actor, context)
        def compute():
            metrics, traces = evaluate(native, actor)
            if is_weight_shift:
                metrics.update(evaluation_schema=evaluation_schema,
                               weight_shift_tracking=weight_shift.tracking_metrics(native, metrics, traces))
                if rehearsal:
                    from .shift_reward import contract as reward_contract
                    metrics["reward_schema"] = reward_contract(shift_reward_mode)["schema"]
                if extended is not None:
                    extended_metrics, extended_traces = evaluate(extended, actor)
                    extended_metrics.update(episode_seconds=shift_seconds,
                        weight_shift_tracking=weight_shift.tracking_metrics(extended, extended_metrics, extended_traces))
                    metrics.update(extended_motion_native=extended_metrics, extended_motion_seconds=shift_seconds)
            elif is_motion:
                metrics.update(evaluation_schema="mimic_motion_tracking_v1",
                               motion_tracking=tracking_metrics(native, metrics, traces))
            if quiet is not None:
                quiet_metrics, _ = evaluate(quiet, actor)
                metrics.update(quiet_standing_passed=standing_passed(quiet_metrics), quiet_native=quiet_metrics)
                if is_weight_shift:
                    quiet.episode_seconds = 5.
                    try:
                        quiet_five, _ = evaluate(quiet, actor)
                    finally:
                        quiet.episode_seconds = 3.
                    metrics.update(quiet_native_five_seconds=quiet_five,
                                   quiet_standing_passed=standing_passed(quiet_metrics) and standing_passed(quiet_five))
            return metrics, traces
        metrics, traces = native_cache.get_or_compute(last_native_key, compute)
        timing[f"{timing_phase}_native_seconds"] += time.monotonic() - began
        return metrics, traces
    contract = native.observation_contract()
    provenance = None
    if args.initialize_from:
        if is_weight_shift:
            # Continue the same admitted motion or explicitly substitute from Stand.
            validate_contract(source, contract if source.get("reference_sha256") == contract["reference_sha256"]
                              else quiet.observation_contract())
            validate_reference_substitution(source, contract)
        else:
            validate_contract(source, contract)
        provenance = {"path": str(args.initialize_from.resolve()), "sha256": sha256(args.initialize_from),
                      "source_contract_sha256": sha256(args.source_contract),
                      "mode": "weights_and_normalization_only; new optimizer, rollout state and local counters"}
        if is_weight_shift:
            provenance["reference_substitution"] = dict(source_sha256=source["reference_sha256"],
                target_sha256=contract["reference_sha256"], source_duration=source["reference_duration"],
                target_duration=contract["reference_duration"], scope="Explicit weight-shift pilot only; all other contract fields validated")
        if source.get("normalization_preparation"):
            from .prepare_motion_normalization import validate_preparation
            migration = validate_preparation(args.source_contract, args.initialize_from, contract["reference_sha256"])
            provenance["normalization_preparation"] = migration
            contract["normalization_lineage"] = migration
        elif source.get("normalization_lineage"):
            from .motion_features import validate_lineage
            validate_lineage(source["normalization_lineage"], contract)
            contract["normalization_lineage"] = copy.deepcopy(source["normalization_lineage"])
            provenance["normalization_lineage"] = contract["normalization_lineage"]
    if rehearsal:
        task = RehearsalTask(rig, args.reference, quiet_reference, args.envs, seed=args.seed,
                             quiet_fraction=shift_quiet_fraction, motion_seconds=shift_seconds,
                             control_mode=args.control, contact_mode=contact_mode, reward_mode=shift_reward_mode)
    else:
        task = BallTask(rig, args.reference, args.envs, seed=args.seed, **options) if is_ball else StandTask(
            rig, args.reference, args.envs, seed=args.seed, control_mode=args.control, contact_mode=contact_mode)
    experiment = task.experiment_contract() if is_ball or rehearsal else {"task": args.task}
    if is_weight_shift:
        experiment.update(quiet_reference_sha256=quiet.reference.sha256, tracking_gates=weight_shift.GATES,
                          scope="Double-support reference tracking pilot; no impact training or viewer promotion")
    if anchor_limit:
        experiment["quiet_anchor"] = dict(schema="mimic_quiet_anchor_v1", max_state_kl=anchor_limit,
            teacher_checkpoint_sha256=provenance["sha256"], horizon_seconds=5., state_stride=4,
            scope="Fixed starting-policy states; does not replace native retention checks")
    if calibration_steps:
        experiment["critic_calibration"] = dict(steps=calibration_steps, only=calibration_only)
    if actor_backtracks:
        experiment["actor_search"] = dict(schema="mimic_actor_backtracking_v1", max_backtracks=actor_backtracks,
            shrink_factor=.5, acceptance="Existing KL constraints and same-batch loss reduction above 1e-7 * max(1, abs(loss))",
            optimizer_state="Each retry restores the pre-step weights and optimizer state; base learning rate is unchanged")
        if actor_project_anchor:
            experiment["actor_search"].update(schema="mimic_actor_anchor_projection_v1",
                direction="Fallback tangent-plane projection at the worst protected state; 0.1% inward margin; all constraints rechecked")
    config_path = args.mimickit / "data/agents/deepmimic_humanoid_ppo_agent.yaml"
    config = yaml.safe_load(config_path.read_text())
    if contract.get("normalization_lineage", {}).get("migration_mode") == "additive_features":
        from .motion_features import validate_lineage
        validate_lineage(contract["normalization_lineage"], contract)
        config["motion_feature_indices"] = contract["normalization_lineage"]["motion_feature_indices"]
    if critic_grad_clip > 0:
        config["critic_optimizer"] = {**config["critic_optimizer"], "grad_clip": critic_grad_clip}
    # Native validation selects checkpoints. Upstream's test_episodes=8 actually
    # runs at least one episode in EVERY training world; disable that duplicate work.
    config.update(iters_per_output=args.evaluation_interval, test_episodes=0)
    if args.stability == "guarded":
        config.update(actor_max_kl=args.actor_max_kl, observation_normalization="frozen_loaded_statistics",
                      actor_gradient_clip=1.0)
        if actor_backtracks:
            config["actor_backtracks"] = actor_backtracks
            config["actor_project_anchor"] = actor_project_anchor
    (args.out / "agent.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    curve = []
    updates = []
    best = BestCheckpoint(args.out, contract)
    gates = {"required_successes": 8, "max_mean_root_tracking_error_m": 0.05}
    if is_motion:
        gates = dict(weight_shift.GATES if is_weight_shift else MOTION_GATES)
        if is_weight_shift:
            gates["quiet_standing_retention"] = dict(successes=8, max_mean_root_error_m=.05)
        if rehearsal:
            gates["extended_motion_seconds"] = shift_seconds
            gates["extended_motion_required"] = True
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
                      if is_ball else {"schema": evaluation_schema}}, indent=2), encoding="utf-8")

    class BudgetComplete(Exception):
        pass

    base_agent = GuardedPPO if args.stability == "guarded" else PPOAgent

    class BoundedPPO(base_agent):
        def _build_model(self, config):
            super()._build_model(config)
            if "motion_feature_indices" in config:
                from .motion_features import install
                install(self, config["motion_feature_indices"])

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
            invalid = [name for name, p in self.named_parameters() if not torch.isfinite(p).all()]
            if invalid:
                (args.out / "numerical_failure.json").write_text(json.dumps(dict(
                    iteration=self._iter + 1, nonfinite_parameters=invalid,
                    metrics={key: float(value) if np.isfinite(float(value)) else str(float(value)) for key, value in result.items()}),
                    indent=2), encoding="utf-8")
                raise FloatingPointError("Nonfinite PPO parameters: " + ", ".join(invalid))
            for key, value in result.items():
                if not np.isfinite(float(value)):
                    raise FloatingPointError(f"Nonfinite training metric: {key}")
            if args.stability == "guarded":
                row = {"iteration": self._iter + 1, **{key: float(result[key]) for key in
                       ("actor_kl", "actor_attempted_kl", "actor_steps_accepted", "actor_steps_rejected")}}
                if "quiet_anchor_max_kl" in result:
                    row["quiet_anchor_max_kl"] = float(result["quiet_anchor_max_kl"])
                if actor_backtracks:
                    row.update({key: float(result[key]) for key in (
                        "actor_search_trials", "actor_backtracked_steps", "actor_step_scale_sum",
                        "actor_search_loss_decrease", "actor_constraint_rejections", "actor_loss_rejections",
                        "actor_projected_steps", "actor_projected_trials")})
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
    if anchor_limit:
        from .quiet_anchor import QuietAnchor, valid_observations
        quiet.episode_seconds = 5.
        try:
            anchor_metrics, anchor_traces = evaluate(quiet, StandPolicy(agent).eval().cpu())
        finally:
            quiet.episode_seconds = 3.
        if not standing_passed(anchor_metrics):
            raise ValueError("Quiet anchor requires a starting policy that passes five-second standing")
        anchor_obs = valid_observations(anchor_traces, anchor_metrics, quiet.dt)
        np.savez_compressed(args.out / "quiet_anchor.npz", observations=anchor_obs.numpy())
        agent.quiet_anchor = QuietAnchor(agent, anchor_obs, anchor_limit)
        experiment["quiet_anchor"]["observations_sha256"] = sha256(args.out / "quiet_anchor.npz")
    if calibration_steps:
        from .critic_calibration import calibrate
        calibration = calibrate(agent, task, calibration_steps, args.out)
        if not calibration["passed"]:
            raise RuntimeError("Critic calibration failed validation; no actor updates were started. See critic_calibration.json")
        agent.save(str(args.out / "calibrated.pt"))
        experiment["critic_calibration"]["report_sha256"] = sha256(args.out / "critic_calibration.json")
    if anchor_limit or calibration_steps:
        protocol_path = args.out / "protocol.json"
        protocol = json.loads(protocol_path.read_text())
        protocol["experiment"] = experiment
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    if calibration_only:
        (args.out / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
        print(json.dumps(calibration, indent=2))
        return
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
    final_export = None
    if viewer_policy == "final":
        final_error = export_policy(copy.deepcopy(trained).cpu(), traces["observations"],
                                    args.reference, contract, args.out / "final-export")
        (args.out / "final-export" / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
        final_export = dict(checkpoint="model.pt", checkpoint_sha256=sha256(args.out / "model.pt"),
                            bundle="final-export", onnx_max_action_error=final_error,
                            purpose="Final trained candidate for visual review; not accepted by checkpoint-retention gates")
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
              "final_review_export": final_export,
              "standing_gate_passed": None if is_motion else standing_passed(selected_cpu),
              "final_standing_gate_passed": None if is_motion else standing_passed(trained_cpu),
              "motion_tracking_gate_passed": selected_cpu["motion_tracking"]["passed"] if is_motion and not is_weight_shift else None,
              "weight_shift_gate_passed": (selected_cpu["weight_shift_tracking"]["passed"] and selected_cpu["quiet_standing_passed"]
                    and (not rehearsal or selected_cpu["extended_motion_native"]["weight_shift_tracking"]["passed"])) if is_weight_shift else None,
              "guard_summary": {"max_accepted_rollout_kl": max((r["actor_kl"] for r in updates), default=0),
                                "accepted_actor_steps": sum(r["actor_steps_accepted"] for r in updates),
                                "rejected_actor_steps": sum(r["actor_steps_rejected"] for r in updates)},
              "scope": "Experimental motion-guided policy; fixed evaluation cases, not held-out motions. Viewer selection is for review, not recovery acceptance."}
    if anchor_limit:
        report["guard_summary"]["max_accepted_quiet_state_kl"] = max(
            (r["quiet_anchor_max_kl"] for r in updates), default=0.)
    if actor_backtracks:
        report["actor_search_summary"] = {key: sum(r[key] for r in updates) for key in (
            "actor_search_trials", "actor_backtracked_steps", "actor_step_scale_sum",
            "actor_search_loss_decrease", "actor_constraint_rejections", "actor_loss_rejections",
            "actor_projected_steps", "actor_projected_trials")}
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.no_select_viewer:
        report["viewer_selection"] = select_run(args.out, policy=viewer_policy)
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
    parser.add_argument("--task", choices=("stand", "ball", "motion", "weight_shift"), default="stand")
    parser.add_argument("--weight-shift-seconds", type=float, default=3., help="Motion training/extended-validation horizon; above three enables rehearsal mode")
    parser.add_argument("--weight-shift-quiet-fraction", type=float, default=0., help="Fraction of fixed training worlds for five-second original standing")
    parser.add_argument("--weight-shift-reward", choices=("reference", "precision_v1"), default="reference")
    parser.add_argument("--quiet-anchor-max-kl", type=float, default=0., help="Optional maximum per-state KL against fixed initial standing distributions")
    parser.add_argument("--critic-calibration-steps", type=int, default=0, help="Frozen-actor value fitting before PPO; required when changing the shift reward")
    parser.add_argument("--calibrate-critic-only", action="store_true", help="Stop after critic validation; no actor training or policy export")
    parser.add_argument("--ball-speed-min", type=float, default=1.)
    parser.add_argument("--ball-speed-max", type=float, default=2.)
    parser.add_argument("--quiet-fraction", type=float, default=.25)
    parser.add_argument("--ball-reward", choices=("reference", "recovery_v1"), default="reference")
    parser.add_argument("--ball-curriculum", choices=("all_directions", "front_back_torso_v1", "mixed_recovery_v1"), default="all_directions")
    parser.add_argument("--viewer-policy", choices=("best", "final"), default="best",
                        help="Ball only: export/select the final candidate for review while retaining best.pt separately")
    parser.add_argument("--stability", choices=("upstream", "guarded"), default="upstream")
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--source-contract", type=Path)
    parser.add_argument("--actor-max-kl", type=float, default=0.02)
    parser.add_argument("--actor-backtracks", type=int, default=0, help="Optional halving retries with same-batch loss improvement; zero preserves single-proposal updates")
    parser.add_argument("--actor-project-anchor", action="store_true", help="Fallback to a projected direction when ordinary backtracking cannot satisfy the fixed standing constraint")
    parser.add_argument("--critic-grad-clip", type=float, default=0., help="Optional critic gradient norm limit; zero preserves upstream behavior")
    parser.add_argument("--evaluation-interval", type=int, default=64)
    parser.add_argument("--no-select-viewer", action="store_true", help="Keep the current F5/F6 export selected")
    train(parser.parse_args())
