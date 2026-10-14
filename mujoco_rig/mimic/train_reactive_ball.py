"""Fine-tune stationary Stand on physical hits with quiet-standing retention."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .baseline import sha256
from .checkpoints import BestCheckpoint, validate_contract, recovery_step_rank
from .continuous_stand import evaluate_retention
from .native_engine import NativeDummyEngine
from .policy import StandPolicy
from .reactive_ball_eval import evaluate as evaluate_ball, repeated_cases, selection_cases, unseen_cases
from .rig import Rig
from .runtime import activate, MIMICKIT_REVISION


def train(args):
    amp_demonstrations = getattr(args, "amp_demonstration", None)
    if (args.amp_calibration_steps < 0 or args.amp_disc_learning_rate <= 0
            or not np.isfinite(args.amp_disc_learning_rate)
            or (args.amp_calibration_steps and not amp_demonstrations)):
        raise ValueError("Require valid AMP calibration settings")
    if args.envs < 32 or args.envs % 8 or args.iterations < 1 or args.seconds <= 0 \
            or args.evaluation_interval < 1 or args.actor_warmup < 0 \
            or args.quiet_anchor_max_kl < 0 or args.quiet_anchor_samples < 1 \
            or args.quiet_replay_weight < 0 or args.actor_stall_iterations < 1 \
            or not np.isfinite(args.impact_replay_weight) or args.impact_replay_weight < 0 \
            or args.impact_replay_samples < 2 \
            or bool(args.impact_replay_trace) != bool(args.impact_replay_weight) \
            or args.actor_max_kl < 0 \
            or (args.action_bound_weight is not None and
                (not np.isfinite(args.action_bound_weight) or args.action_bound_weight < 0)) \
            or (args.risk_leg_std is not None and not .05 <= args.risk_leg_std <= .4) \
            or (args.risk_exploration_gate != "speed" and args.risk_leg_std is None) \
            or (args.quiet_anchor_max_kl and args.quiet_replay_weight) \
            or args.checkpoint_interval < 0 \
            or (args.steps_per_iter is not None and args.steps_per_iter < 1) \
            or (args.discount is not None and not 0 < args.discount < 1) \
            or (args.td_lambda is not None and not 0 <= args.td_lambda <= 1) \
            or (args.recovery_reset_fraction and args.recovery_reset_bank is None):
        raise ValueError("Require positive budgets and complete posture replicas")
    activate(args.mimickit)
    from envs.base_env import EnvMode
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .continuous_stand_task import ContinuousStandTask
    from .reactive_ball_task import ReactiveBallTask
    from .stable_normalizer import install as install_normalizer
    from .guarded_ppo import GuardedPPO
    from .quiet_anchor import QuietAnchor
    AgentBase = GuardedPPO if args.quiet_anchor_max_kl or args.actor_max_kl else PPOAgent
    if amp_demonstrations:
        from learning.amp_agent import AMPAgent
        from .amp_recovery import AMPRecoveryMixin, actor_state_dict, amp_sidecar, validate_sidecar
        from .train_amp_step import load_stand_into_amp

        class RecoveryAMP(GuardedPPO, AMPAgent):
            def _record_disc_demo_data(self):
                count = self._exp_buffer.get_data_flat("disc_obs").shape[0]
                self._exp_buffer.set_data_flat("disc_obs_demo", self._env.fetch_disc_obs_demo(count))

            def save(self, out_file):
                torch.save(self.state_dict(), amp_sidecar(out_file))
                torch.save(actor_state_dict(self.state_dict()), out_file)

            def load(self, in_file):
                sidecar = amp_sidecar(in_file)
                if sidecar.exists():
                    validate_sidecar(torch.load(in_file, map_location="cpu", weights_only=True),
                                     torch.load(sidecar, map_location="cpu", weights_only=True))
                    super().load(str(sidecar))
                else:
                    load_stand_into_amp(self, Path(in_file))

        AgentBase = RecoveryAMP

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    validation_seed = args.validation_seed if args.validation_seed is not None else int(
        np.random.SeedSequence().generate_state(1)[0])
    mp_util.init(0, 1, "cuda:0", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    source = args.initialize_from
    metadata = json.loads((source.parent / "best.json").read_text())
    seed_manifest = source.parent / "seed_manifest.json"
    if args.initialize_from_quiet_report:
        from .training_quiet_admission import validate_quiet_initialization
        admission_path = args.initialize_from_quiet_report
        admission = json.loads(admission_path.read_text())
        validate_quiet_initialization(admission, source, args.reference, rig.path)
        if args.quiet_fraction < .25:
            raise ValueError("Experimental mass curriculum requires at least 25% quiet episodes")
        admitted = True
        source_kind = "quiet_validated_experimental_projectile_curriculum"
    elif args.resume_experimental_final:
        admission_path = source.parent / "arena-preview/validation.json"
        admission = json.loads(admission_path.read_text())
        source_report = json.loads((source.parent / "report.json").read_text())
        admitted = (source.name == "model.pt"
                    and admission.get("experimental_preview") is True
                    and admission["checkpoint_sha256"] == sha256(source)
                    and admission["training_report_sha256"] == sha256(source.parent / "report.json")
                    and admission["ball_model_sha256"] == sha256(args.ball_model)
                    and admission["reference_sha256"] == sha256(args.reference)
                    and source_report["final"]["quiet_standing_passed"] is True)
        source_kind = "experimental_arena_final_training_continuation"
    elif args.resume_selected_same_mass:
        admission_path = source.parent / "best.json"
        admission = metadata
        source_protocol = json.loads((source.parent / "protocol.json").read_text())
        admitted = (metadata["checkpoint_sha256"] == sha256(source)
                    and metadata["metrics"]["quiet_standing_passed"] is True
                    and metadata["metrics"]["confirmed_hits"] == 32
                    and source_protocol["experiment"]["ball_model_sha256"] == sha256(args.ball_model)
                    and source_protocol["experiment"]["reference_sha256"] == sha256(args.reference))
        source_kind = "selected_same_mass_training_continuation"
    elif seed_manifest.exists():
        admission_path = seed_manifest
        admission = json.loads(seed_manifest.read_text())
        admitted = (admission.get("selected_checkpoint_sha256") == sha256(source)
                    and admission.get("ready_for_next_mass_training") is True)
        source_kind = "validated_stationary_reactive_ball_seed"
    else:
        admission_path = source.parent / "export/validation.json"
        admission = json.loads(admission_path.read_text())
        admitted = (admission.get("checkpoint_sha256") == sha256(source)
                    and admission.get("stationary_stand_gate_passed") is True)
        source_kind = "validated_stationary_stand"
    if (not (args.resume_experimental_final or args.initialize_from_quiet_report)
            and metadata["checkpoint_sha256"] != sha256(source)) or not admitted:
        raise ValueError("Require a selected, admitted stationary checkpoint")
    contract = json.loads((source.parent / "contract.json").read_text())
    from .balance_observation import task_classes
    ContinuousStandTask, ReactiveBallTask = task_classes(contract)
    reference = args.reference
    quiet = ContinuousStandTask(rig, reference, engine_factory=NativeDummyEngine)
    validate_contract(contract, quiet.observation_contract())
    NativeTask = ReactiveBallTask
    amp_options = {}
    if amp_demonstrations:
        class NativeTask(AMPRecoveryMixin, ReactiveBallTask):
            pass
        amp_options = dict(demonstrations=amp_demonstrations, demo_weights=args.amp_demo_weights,
                           mirror_demos=args.amp_mirror_demos, foot_motion_weight=args.amp_foot_motion_weight)
    native = NativeTask(rig, reference, 32, "cpu", ball_path=args.ball_model,
                              speed_min=args.speed_min, speed_max=args.speed_max,
                              quiet_fraction=args.quiet_fraction, episode_seconds=args.episode_seconds,
                              recovery_mode=args.recovery_mode, target_set=args.target_set, **amp_options)
    validate_contract(contract, native.observation_contract())
    if seed_manifest.exists() and not (args.resume_selected_same_mass or args.resume_experimental_final or args.initialize_from_quiet_report) and not float(native.engine.rig.model.body_mass[
            native.engine.rig.model.body("ball").id]) > admission["ball_mass_kg"]:
        raise ValueError("Next mass stage must exceed the admitted seed mass")
    config = yaml.safe_load((source.parent / "agent.yaml").read_text())
    config["test_episodes"] = 0
    config["iters_per_output"] = args.evaluation_interval
    config["normalizer_samples"] = 0  # Preserve the admitted actor's input transform.
    config["actor_optimizer"]["learning_rate"] = args.actor_learning_rate
    if args.action_bound_weight is not None:
        config["action_bound_weight"] = args.action_bound_weight
    for name in ("steps_per_iter", "discount", "td_lambda"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    config["critic_eval_batch_size"] = 16384
    if args.quiet_anchor_max_kl:
        config.update(actor_max_kl=.005, actor_backtracks=4, actor_project_anchor=True)
    elif args.actor_max_kl:
        config.update(actor_max_kl=args.actor_max_kl, actor_backtracks=2, actor_project_anchor=False)
    if args.risk_leg_std is not None:
        config["risk_leg_std"] = args.risk_leg_std
        config["risk_exploration_gate"] = args.risk_exploration_gate
    deployment_config = dict(config)
    if amp_demonstrations:
        amp_config = yaml.safe_load((args.mimickit / "data/agents/amp_task_humanoid_agent.yaml").read_text())
        for key in ("disc_optimizer", "disc_epochs", "disc_batch_size", "disc_buffer_size",
                    "disc_replay_samples", "disc_logit_reg", "disc_grad_penalty", "disc_reward_scale"):
            config[key] = amp_config[key]
        config["model"] = {**config["model"], "disc_net": amp_config["model"]["disc_net"]}
        config.update(agent_name="AMP", task_reward_weight=.8, disc_reward_weight=.2,
                      disc_epochs=1, disc_batch_size=8, disc_eval_batch_size=16384)
        config["disc_optimizer"]["learning_rate"] = args.amp_disc_learning_rate
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "agent.yaml").write_text(yaml.safe_dump(deployment_config if amp_demonstrations else config))
    if amp_demonstrations:
        (args.out / "amp_agent.yaml").write_text(yaml.safe_dump(config))
    (args.out / "contract.json").write_text(json.dumps(contract, indent=2))
    cases = selection_cases(args.speed_max, args.episode_seconds, native.dt)
    protocol = dict(schema="mimic_reactive_ball_seed_training_v1", seed=args.seed,
                    validation_seed=validation_seed,
                    source_checkpoint=str(source), source_checkpoint_sha256=sha256(source),
                    source_kind=source_kind, source_validation_sha256=sha256(admission_path),
                    resume_selected_same_mass=args.resume_selected_same_mass,
                    resume_experimental_final=args.resume_experimental_final,
                    initialize_from_quiet_report=str(args.initialize_from_quiet_report) if args.initialize_from_quiet_report else None,
                    contract=contract, experiment=native.experiment_contract(),
                    selection_cases=cases, actor_warmup_iterations=args.actor_warmup,
                    training_second_shot_fraction=args.second_shot_fraction,
                    training_forward_fraction=args.forward_fraction,
                    training_target_set=args.target_set,
                    recovery_reset_fraction=args.recovery_reset_fraction,
                    recovery_reset_bank=str(args.recovery_reset_bank) if args.recovery_reset_bank else None,
                    recovery_reset_bank_sha256=sha256(args.recovery_reset_bank) if args.recovery_reset_bank else None,
                    quiet_anchor_max_kl=args.quiet_anchor_max_kl,
                    quiet_anchor_samples=args.quiet_anchor_samples,
                    quiet_replay_weight=args.quiet_replay_weight,
                    impact_replay_trace=str(args.impact_replay_trace) if args.impact_replay_trace else None,
                    impact_replay_weight=args.impact_replay_weight,
                    impact_replay_samples=args.impact_replay_samples,
                    risk_leg_std=args.risk_leg_std,
                    risk_exploration_gate=args.risk_exploration_gate,
                    prefer_recovered_steps=args.prefer_recovered_steps,
                    actor_max_kl=args.actor_max_kl,
                    actor_stall_iterations=args.actor_stall_iterations,
                    normalizer="frozen admitted stationary statistics", config=config,
                    iterations=args.iterations, seconds=args.seconds,
                    seconds_semantics="synchronized_training_iteration_wall_time_excludes_native_evaluations",
                    envs=args.envs,
                    mimickit_revision=MIMICKIT_REVISION,
                    source_hashes={name: sha256(Path(__file__).with_name(name)) for name in
                        ("train_reactive_ball.py", "reactive_ball_task.py", "reactive_ball_eval.py", "ball_protocol.py",
                         "ball.py", "ball_native.py", "ball_newton.py", "recovery_reward.py",
                         "target_control.py", "newton_adapter.py", "checkpoints.py",
                         "guarded_ppo.py", "quiet_anchor.py", "impact_replay.py", "capture_reward.py", "step_reward.py", "recovery_steps.py", "support_motion.py", "recovery_signals.py",
                         "risk_exploration.py", "runtime.py", "gated_recovery_net.py",
                         "balance_observation.py", "native_engine.py", "training_quiet_admission.py",
                         "validate_stand_foundation.py")})
    if amp_demonstrations:
        for name in ("amp_recovery.py", "amp_step.py", "train_amp_step.py", "recovery_quality.py",
                     "foot_motion.py", "calibrate_amp_recovery.py"):
            protocol["source_hashes"][name] = sha256(Path(__file__).with_name(name))
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (args.out / "runner.py").write_text(Path(__file__).read_text())
    (args.out / "source").mkdir()
    for name in protocol["source_hashes"]:
        (args.out / "source" / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    TrainingTask = ReactiveBallTask
    training_options = {}
    if amp_demonstrations:
        class TrainingTask(AMPRecoveryMixin, ReactiveBallTask):
            pass
        training_options.update(amp_options)
    task = TrainingTask(rig, reference, args.envs, "cuda:0", ball_path=args.ball_model,
                            speed_min=args.speed_min, speed_max=args.speed_max,
                            quiet_fraction=args.quiet_fraction, seed=args.seed,
                            episode_seconds=args.episode_seconds,
                            second_shot_fraction=args.second_shot_fraction,
                            recovery_mode=args.recovery_mode, forward_fraction=args.forward_fraction,
                            recovery_reset_fraction=args.recovery_reset_fraction, target_set=args.target_set,
                            **training_options)
    if amp_demonstrations:
        protocol["motion_prior"] = task.discriminator_contract()
        protocol["checkpoint_format"] = "PPO-compatible actor/critic .pt plus full training .amp.pt sidecar"
        (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    if args.recovery_reset_bank:
        task.load_recovery_resets(args.recovery_reset_bank)

    class SeedAgent(AgentBase):
        quiet_teacher = None
        impact_teacher = None

        def _compute_actor_loss(self, batch):
            info = super()._compute_actor_loss(batch)
            if self.quiet_teacher is not None:
                teacher = self.quiet_teacher
                divergence = teacher.teacher.kl(self._model.eval_actor(teacher.observations)).clamp_min(0).mean()
                info["actor_loss"] = info["actor_loss"] + args.quiet_replay_weight * divergence
                info["quiet_replay_kl"] = divergence
            if self.impact_teacher is not None:
                teacher = self.impact_teacher
                divergence = teacher.teacher.kl(self._model.eval_actor(teacher.observations)).clamp_min(0).mean()
                info["actor_loss"] = info["actor_loss"] + args.impact_replay_weight * divergence
                info["impact_replay_kl"] = divergence
            return info

        def _build_normalizers(self):
            super()._build_normalizers()
            install_normalizer(self)

        def _update_actor(self, batch_size, num_steps):
            if self._iter < args.actor_warmup:
                return dict(actor_loss=torch.zeros((), device=self._device),
                            clip_frac=torch.zeros((), device=self._device),
                            imp_ratio=torch.ones((), device=self._device))
            return super()._update_actor(batch_size, num_steps)

    agent = SeedAgent(config, task, "cuda:0")
    agent.load(str(source))
    if args.risk_leg_std is not None:
        from .risk_exploration import install as install_exploration
        install_exploration(agent, args.risk_leg_std, args.risk_exploration_gate)
    best = BestCheckpoint(args.out, contract,
        rank_function=recovery_step_rank if args.prefer_recovered_steps else None,
        selection_label=("Retain directional quiet/survival/recovery gates; prefer recovered steps "
                         "only when quiet, survival and recovery counts tie.") if args.prefer_recovered_steps else None)
    curve = []
    started = time.monotonic()
    initial_quiet_traces = None
    native_disc_samples = dict(train=[], heldout=[])

    def assess(iteration):
        nonlocal initial_quiet_traces
        actor = StandPolicy(agent).cpu().eval()
        quiet_metrics, quiet_traces = evaluate_retention(quiet, actor)
        if iteration == 0:
            initial_quiet_traces = quiet_traces
        quality = None
        if amp_demonstrations:
            from .recovery_quality import RecoveryQuality
            quality = RecoveryQuality(native, actor)
        def record_native(task, step, active, done):
            if quality is not None:
                quality.record(task, step, active, done)
            if iteration == 0 and args.amp_calibration_steps and step % 6 == 0:
                for split, worlds in (("train", task.ids < 24), ("heldout", task.ids >= 24)):
                    ids = active & worlds
                    if ids.any():
                        native_disc_samples[split].append(task._history.values[ids].flatten(1).clone())
        hit_metrics, traces = evaluate_ball(native, quality or actor, cases, record_native)
        if quality is not None:
            hit_metrics["motion_quality"] = quality.report()
        hit_metrics["quiet_standing_passed"] = bool(quiet_metrics["passed"])
        hit_metrics["quiet_native"] = quiet_metrics
        accepted = best.consider(agent.save, hit_metrics, iteration=iteration,
                                 samples=agent._sample_count)
        if accepted:
            np.savez_compressed(args.out / "selected-native.npz", **traces)
        row = dict(iteration=iteration, elapsed_seconds=time.monotonic()-started,
                   selection=best.decisions[-1], metrics=hit_metrics)
        curve.append(row)
        (args.out / "learning_curve.json").write_text(json.dumps(curve, indent=2))
        print(json.dumps(dict(iteration=iteration, quiet=quiet_metrics["passed"],
            hits=hit_metrics["confirmed_hits"], survived=hit_metrics["survived_hits"],
            recovered=hit_metrics["recovered_hits"], accepted=accepted)), flush=True)
        return hit_metrics

    agent.save(str(args.out / "initial.pt"))
    initial = assess(0)
    if args.amp_calibration_steps:
        from .calibrate_amp_recovery import calibrate
        calibration = calibrate(agent, task, native_disc_samples, args.out, args.amp_calibration_steps, args.seed + 97)
        (args.out / "discriminator-calibration.json").write_text(json.dumps(calibration, indent=2))
        print(json.dumps(dict(discriminator_calibrated=calibration["passed"],
                              before=calibration["source_mapping"], after=calibration["calibrated"])), flush=True)
        if not calibration["passed"]:
            agent.save(str(args.out / "calibration_failed.pt"))
            raise RuntimeError("Discriminator held-out calibration gate failed; policy training not started")
        agent.save(str(args.out / "calibrated.pt"))
        protocol["discriminator_calibration"] = dict(steps=args.amp_calibration_steps,
            report_sha256=sha256(args.out / "discriminator-calibration.json"), actor_unchanged=True)
        (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    if args.quiet_anchor_max_kl or args.quiet_replay_weight:
        observations = torch.as_tensor(initial_quiet_traces["observations"][initial_quiet_traces["active"]])
        from .gated_recovery_net import GatedRecoveryNet, quiet_rehearsal_indices
        if isinstance(agent._model._actor_layers, GatedRecoveryNet):
            picks = quiet_rehearsal_indices(observations, args.quiet_anchor_samples,
                support_activation=agent._model._actor_layers.support_activation,
                motion_activation=agent._model._actor_layers.motion_activation,
                damping_activation=agent._model._actor_layers.damping_activation)
        else:
            picks = torch.linspace(0, len(observations) - 1, min(args.quiet_anchor_samples,
                                                                 len(observations))).long()
        teacher = QuietAnchor(agent, observations[picks], args.quiet_anchor_max_kl or 1.)
        if args.quiet_anchor_max_kl:
            agent.quiet_anchor = teacher
        else:
            agent.quiet_teacher = teacher
        np.savez_compressed(args.out / "quiet_anchor_observations.npz",
                            observation=observations[picks].numpy())
    if args.impact_replay_trace:
        from .impact_replay import load_replay
        observations, indices, replay_protocol = load_replay(args.impact_replay_trace, source,
            reference, args.ball_model, rig.path, args.impact_replay_samples, native.dt)
        agent.impact_teacher = QuietAnchor(agent, observations, 1.)
        np.savez_compressed(args.out / "impact_replay_observations.npz",
                            observation=observations.numpy(), trace_indices=indices)
        replay_protocol["dataset_sha256"] = sha256(args.out / "impact_replay_observations.npz")
        (args.out / "impact_replay.json").write_text(json.dumps(replay_protocol, indent=2))
    task.set_mode(EnvMode.TRAIN)
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    agent._init_train()
    training_iteration_wall_seconds = 0.
    stalled_iterations = 0
    actor_parameters = list(agent._model.get_actor_params())
    stop_reason = None
    while training_iteration_wall_seconds < args.seconds and agent._iter < args.iterations:
        torch.cuda.synchronize()
        iteration_started = time.monotonic()
        before_actor = [p.detach().clone() for p in actor_parameters]
        info = agent._train_iter()
        actor_change = float(torch.stack([(p.detach() - old).square().sum()
                                          for p, old in zip(actor_parameters, before_actor)]).sum().sqrt())
        torch.cuda.synchronize()
        training_iteration_wall_seconds += time.monotonic() - iteration_started
        agent._sample_count = agent._update_sample_count()
        agent._iter += 1
        row = dict(iteration=agent._iter, samples=agent._sample_count,
                   elapsed_seconds=time.monotonic()-started,
                   training_iteration_wall_seconds=training_iteration_wall_seconds,
                   actor_parameter_change_l2=actor_change,
                   **{k: float(v) for k, v in info.items()})
        for key in ("capture_error_m", "swing_quality", "foot_motion_cost", "foot_acceleration_m_s2",
                    "rapid_contact_reversals", "useful_placement_events", "loaded_sole_slip_proxy_m_s"):
            if key in task._diagnostics:
                row["last_batch_" + key] = float(task._diagnostics[key])
        if task.recovery_signals is not None:
            signals = task.recovery_signals.drain()
            row.update({'rollout_'+key:value for key,value in signals.items() if key != 'scope'})
            with (args.out/'recovery_signals.jsonl').open('a') as file:
                file.write(json.dumps(dict(iteration=agent._iter,**signals))+'\n')
        if getattr(agent._model._actor_layers, "damping_activation", False):
            from .gated_recovery_net import damping_recovery_gate
            with torch.no_grad():
                damping_gate = damping_recovery_gate(agent._exp_buffer.get_data_flat("obs"),
                    agent._model._actor_layers.damping_gate_specification)
                row["damping_gate_active_fraction"] = float((damping_gate > 0).float().mean())
                row["damping_gate_mean"] = float(damping_gate.mean())
        if not all(np.isfinite(x) for x in row.values()):
            raise FloatingPointError("Nonfinite reactive training metrics")
        with (args.out / "updates.jsonl").open("a") as file:
            file.write(json.dumps(row) + "\n")
        if agent._iter % args.evaluation_interval == 0:
            agent.save(str(args.out / f"evaluation_{agent._iter:06d}.pt"))
            assess(agent._iter)
        elif args.checkpoint_interval and agent._iter % args.checkpoint_interval == 0:
            agent.save(str(args.out / f"candidate_{agent._iter:06d}.pt"))
        stalled_iterations = stalled_iterations + 1 if agent._iter > args.actor_warmup and actor_change == 0 else 0
        if stalled_iterations >= args.actor_stall_iterations:
            stop_reason = "actor_update_stalled"
            break
    agent.save(str(args.out / "model.pt"))
    final = curve[-1]["metrics"] if curve[-1]["iteration"] == agent._iter else assess(agent._iter)
    agent.load(str(best.path))
    selected = best.metadata["metrics"]
    gpu_quiet = ContinuousStandTask(rig, reference, 8, "cuda:0", seed=args.seed)
    gpu_metrics, _ = evaluate_retention(gpu_quiet, StandPolicy(agent).eval())
    heldout = ReactiveBallTask(rig, reference, 64, "cpu", ball_path=args.ball_model,
                               speed_min=args.speed_min, speed_max=args.speed_max,
                               episode_seconds=12.)
    hidden_cases = unseen_cases(validation_seed, 64, args.speed_min, args.speed_max, 12., heldout.dt)
    unseen, _ = evaluate_ball(heldout, StandPolicy(agent).cpu().eval(), hidden_cases)
    (args.out / "unseen_report.json").write_text(json.dumps(unseen, indent=2))
    repeated = None
    if args.second_shot_fraction:
        repeated_task = ReactiveBallTask(rig, reference, 32, "cpu", ball_path=args.ball_model,
                                         speed_min=args.speed_min, speed_max=args.speed_max,
                                         episode_seconds=max(12., args.episode_seconds))
        repeated, _ = evaluate_ball(repeated_task, StandPolicy(agent).cpu().eval(),
                                    repeated_cases(args.speed_max, max(12., args.episode_seconds),
                                                   repeated_task.dt))
        (args.out / "repeated_report.json").write_text(json.dumps(repeated, indent=2))
    report = dict(schema="mimic_reactive_ball_seed_report_v1",
        stop_reason=stop_reason or ("iteration_budget" if agent._iter >= args.iterations else "training_iteration_wall_time_budget"),
        iterations=agent._iter, samples=agent._sample_count,
        elapsed_seconds=time.monotonic()-started,
        training_iteration_wall_seconds=training_iteration_wall_seconds,
        initial=initial, final=final,
        selected_checkpoint=best.metadata, selected_gpu_quiet=gpu_metrics,
        selected_unseen=unseen,
        validation_seed=validation_seed,
        selected_repeated=repeated,
        quiet_standing_retained=selected["quiet_standing_passed"] and gpu_metrics["passed"],
        selection_hits_complete=selected["recovered_hits"] == len(cases),
        unseen_hits_complete=unseen["recovered_hits"] == len(hidden_cases),
        repeated_hits_complete=(repeated is not None and repeated["repeated_recoveries"] == 32),
        godot_selected=False,
        scope="First physical-ball fine-tuning seed. Further mass/angle and long-hold admission precedes Godot")
    if amp_demonstrations:
        report["motion_prior"] = protocol["motion_prior"]
        report["natural_stepping_gate"] = False
        report["scope"] = "Motion-prior pilot; paired slip/oscillation diagnostics are not proof of natural stepping; no Arena promotion"
        report["amp_selected_checkpoint_sha256"] = sha256(amp_sidecar(best.path))
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(complete=True, quiet_retained=report["quiet_standing_retained"],
                          selected_recovered=selected["recovered_hits"],
                          unseen_recovered=unseen["recovered_hits"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--initialize-from", "--stand-checkpoint", dest="initialize_from", type=Path, required=True)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume-selected-same-mass", action="store_true")
    resume.add_argument("--initialize-from-quiet-report", type=Path,
                        help="Experimental curriculum from an actor with exact 8x600s quiet evidence; not hit admission")
    resume.add_argument("--resume-experimental-final", action="store_true",
                        help="Continue the explicitly previewed final Arena actor with matching provenance")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--ball-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=600.)
    parser.add_argument("--evaluation-interval", type=int, default=16)
    parser.add_argument("--actor-warmup", type=int, default=16)
    parser.add_argument("--actor-learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--action-bound-weight", type=float,
                        help="Optional optimizer penalty ablation; physical action/target clipping is unchanged")
    parser.add_argument("--speed-min", type=float, default=1.)
    parser.add_argument("--speed-max", type=float, default=2.)
    parser.add_argument("--quiet-fraction", type=float, default=.25)
    parser.add_argument("--second-shot-fraction", type=float, default=0.)
    parser.add_argument("--quiet-anchor-max-kl", type=float, default=0.)
    parser.add_argument("--quiet-anchor-samples", type=int, default=128)
    parser.add_argument("--quiet-replay-weight", type=float, default=0.)
    parser.add_argument("--impact-replay-trace", type=Path)
    parser.add_argument("--impact-replay-weight", type=float, default=0.)
    parser.add_argument("--impact-replay-samples", type=int, default=512)
    parser.add_argument("--risk-leg-std", type=float,
                        help="Increase leg exploration only with observed horizontal root motion; deterministic actor unchanged")
    parser.add_argument("--risk-exploration-gate", choices=("speed", "support"), default="speed",
                        help="Optionally include observable capture error and foot unloading in leg exploration")
    parser.add_argument("--actor-max-kl", type=float, default=0.,
                        help="Per-iteration rollout KL guard, separate from cumulative quiet-state retention")
    parser.add_argument("--actor-stall-iterations", type=int, default=8)
    parser.add_argument("--recovery-mode", choices=("settling_v2", "capture_v3", "step_v4", "step_v5", "step_v6", "step_v7"), default="settling_v2")
    parser.add_argument("--steps-per-iter", type=int)
    parser.add_argument("--discount", type=float)
    parser.add_argument("--td-lambda", type=float)
    parser.add_argument("--forward-fraction", type=float, default=0.)
    parser.add_argument("--target-set", choices=("torso", "torso_head", "all"), default="torso")
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--prefer-recovered-steps", action="store_true")
    parser.add_argument("--recovery-reset-bank", type=Path)
    parser.add_argument("--recovery-reset-fraction", type=float, default=0.)
    parser.add_argument("--episode-seconds", type=float, default=8.)
    parser.add_argument("--seed", type=int, default=270933)
    parser.add_argument("--validation-seed", type=int,
                        help="Fresh held-out seed; omitted draws and records an independent seed")
    parser.add_argument("--amp-demonstration", type=Path, action="append",
                        help="Optional licensed idle/forward/backward style histories; preserves the stationary recovery objective")
    parser.add_argument("--amp-demo-weights", type=float, nargs="+")
    parser.add_argument("--amp-mirror-demos", action="store_true")
    parser.add_argument("--amp-foot-motion-weight", type=float, default=0.)
    parser.add_argument("--amp-disc-learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--amp-calibration-steps", type=int, default=0)
    train(parser.parse_args())
