"""Bounded Stand-policy pilot for the first failing pre-lift support pose."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .baseline import sha256
from .checkpoints import standing_passed
from .evaluate_stand import evaluate
from .policy import StandPolicy
from .probe_steps import validate_reference_substitution
from .rig import Rig
from .runtime import MIMICKIT_REVISION, activate
from .support_transition import (SupportTransitionReference, TRACKING_REWARD_SCHEMA,
                                 TRACKING_REWARD_WEIGHTS, TRACKING_POSITION_SCALE_M,
                                 TRACKING_VELOCITY_SCALE_M_S, support_tracking_reward)


def gate(metrics, traces, target, dt):
    if metrics["successes"] != 1:
        return dict(passed=False, reason="fall_before_episode_end")
    q = traces["qpos"][:, 0]
    tail = q[-round(2. / dt):]
    speed = np.linalg.norm(np.diff(tail[:, :2], axis=0) / dt, axis=1)
    target_root = target.qpos[-1, :3].cpu().numpy()
    result = dict(final_two_second_min_pelvis_height_m=float(tail[:, 2].min()),
                  final_two_second_max_horizontal_speed_m_s=float(speed.max()),
                  final_root_error_m=float(np.linalg.norm(tail[-1, :3] - target_root)))
    result["passed"] = bool(result["final_two_second_min_pelvis_height_m"] >= .75
                            and result["final_two_second_max_horizontal_speed_m_s"] <= .2
                            and result["final_root_error_m"] <= .05)
    return result


def run(args):
    activate(args.mimickit)
    from util import mp_util
    from .guarded_ppo import GuardedPPO
    from .native_engine import NativeDummyEngine
    from .newton_adapter import DummyNewtonEngine
    from .stand_task import StandTask
    from .support_rehearsal_task import SupportRehearsalTask

    if (args.out.exists() or args.seconds <= 0 or args.envs < 32 or args.envs % 32
            or args.evaluation_interval < 1 or args.retention_seconds < 6
            or args.quiet_fraction < 0 or args.quiet_fraction >= 1
            or args.quiet_fraction > 0 and args.training_mode != "transition"):
        raise ValueError("Require a new output directory and positive pilot settings")
    source_contract = json.loads(args.source_contract.read_text())
    if source_contract.get("observation_contract") != "mimic_stand_target_pd_v2":
        raise ValueError("Require a support-v2 Stand checkpoint")
    best_contract = None
    if args.initialize_from.name == "best.pt":
        best = json.loads((args.initialize_from.parent / "best.json").read_text())
        if best["checkpoint_sha256"] != sha256(args.initialize_from):
            raise ValueError("Source best.pt does not match best.json")
        best_contract = best["contract"]
        validate_reference_substitution(source_contract, best_contract)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mp_util.init(0, 1, "cuda:0", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")

    class SupportTask(StandTask):
        def reward(self):
            base = self.reference_reward()
            target, velocity = self.sample_reference(self.offset + self.steps * self.dt)
            return support_tracking_reward(base, self.engine.get_root_pos(0)[:, :2],
                                           target[:, :2], self.engine.get_root_vel(0)[:, :2],
                                           velocity[:, :2])

    def task(mode, worlds, device, engine_factory, reward_mode="reference"):
        task_type = SupportTask if reward_mode == TRACKING_REWARD_SCHEMA else StandTask
        result = task_type(rig, args.reference, worlds, device, seed=args.seed,
                           engine_factory=engine_factory, control_mode="target_pd",
                           contact_mode="support_v2",
                           reference_factory=lambda p, r, d: SupportTransitionReference(p, r, d, mode=mode))
        result.episode_seconds = result.reference.duration
        return result

    if args.quiet_fraction:
        train_task = SupportRehearsalTask(
            rig, args.reference, Path(__file__).parent / "assets/stand_reference.npz",
            args.envs, "cuda:0", quiet_fraction=args.quiet_fraction,
            support_seconds=args.support_seconds, quiet_seconds=3.,
            reward_mode=args.reward_mode, seed=args.seed, engine_factory=DummyNewtonEngine)
    else:
        train_task = task(args.training_mode, args.envs, "cuda:0", DummyNewtonEngine,
                          reward_mode=args.reward_mode)
    static = task("static", 1, "cpu", NativeDummyEngine)
    entry = task("transition", 1, "cpu", NativeDummyEngine)
    retention = task("transition", 1, "cpu", NativeDummyEngine) if args.retention_seconds > 6 else None
    if retention is not None:
        retention.episode_seconds = args.retention_seconds
        retention.reference_phase = lambda time: (time / retention.reference.duration).clamp(max=1.)
    quiet = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz", 8, "cpu",
                      engine_factory=NativeDummyEngine, control_mode="target_pd", contact_mode="support_v2")
    validate_reference_substitution(source_contract, train_task.observation_contract())
    validate_reference_substitution(source_contract, entry.observation_contract())
    if best_contract is not None:
        validate_reference_substitution(best_contract, train_task.observation_contract())
    config = yaml.safe_load((args.mimickit / "data/agents/deepmimic_humanoid_ppo_agent.yaml").read_text())
    config.update(actor_max_kl=args.actor_max_kl, test_episodes=0,
                  observation_normalization="frozen_loaded_statistics",
                  actor_gradient_clip=1.)
    config["critic_optimizer"] = {**config["critic_optimizer"], "grad_clip": 1.}
    args.out.mkdir(parents=True, exist_ok=False)
    train_task.reference.save(args.out / "training_reference.npz")
    static.reference.save(args.out / "static_reference.npz")
    entry.reference.save(args.out / "transition_reference.npz")
    (args.out / "agent.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    protocol = dict(schema="mimic_support_transition_pilot_v2", model_sha256=sha256(rig.path),
                    source_checkpoint_sha256=sha256(args.initialize_from),
                    source_contract_sha256=sha256(args.source_contract),
                    actor_contract=train_task.observation_contract(),
                    training_mode=args.training_mode,
                    rehearsal=train_task.experiment_contract() if args.quiet_fraction else None,
                    retention_seconds=args.retention_seconds,
                    reward_mode=args.reward_mode,
                    support_tracking_reward=dict(weights=TRACKING_REWARD_WEIGHTS,
                                                 position_scale_m=TRACKING_POSITION_SCALE_M,
                                                 velocity_scale_m_s=TRACKING_VELOCITY_SCALE_M_S)
                    if args.reward_mode == TRACKING_REWARD_SCHEMA else None,
                    training_reference_file_sha256=sha256(args.out / "training_reference.npz"),
                    static_reference=static.reference.contract(),
                    transition_reference=entry.reference.contract(),
                    static_reference_file_sha256=sha256(args.out / "static_reference.npz"),
                    transition_reference_file_sha256=sha256(args.out / "transition_reference.npz"),
                    seed=args.seed, envs=args.envs, seconds=args.seconds,
                    evaluation_interval=args.evaluation_interval,
                    mimickit_revision=MIMICKIT_REVISION, agent_config=config,
                    rollout_lifecycle="Connected episodes persist across native evaluations; only done worlds reset",
                    scope="Experimental support skill; no export or Godot policy selection")
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    class Pilot(GuardedPPO):
        def _train_iter(self):
            result = super()._train_iter()
            if (any(not torch.isfinite(p).all() for p in self.parameters())
                    or any(not np.isfinite(float(v)) for v in result.values())):
                raise FloatingPointError("Nonfinite support-control update")
            return result

    agent = Pilot(config, train_task, "cuda:0")
    agent.load(str(args.initialize_from))
    if agent._obs_norm.get_count().item() <= 0:
        raise ValueError("Require trained source normalization")

    def review():
        actor = copy.deepcopy(StandPolicy(agent)).eval().cpu()
        quiet_metrics, _ = evaluate(quiet, actor)
        static_metrics, static_traces = evaluate(static, actor, episodes=1)
        entry_metrics, entry_traces = evaluate(entry, actor, episodes=1)
        if retention is None:
            retention_metrics, retention_gate = entry_metrics, gate(entry_metrics, entry_traces,
                                                                   entry.reference, entry.dt)
        else:
            retention_metrics, retention_traces = evaluate(retention, actor, episodes=1)
            retention_gate = gate(retention_metrics, retention_traces,
                                  retention.reference, retention.dt)
        return dict(quiet=quiet_metrics, quiet_passed=standing_passed(quiet_metrics),
                    static=static_metrics, static_gate=gate(static_metrics, static_traces,
                                                            static.reference, static.dt),
                    transition=entry_metrics, transition_gate=gate(entry_metrics, entry_traces,
                                                                   entry.reference, entry.dt),
                    retention=retention_metrics, retention_gate=retention_gate)

    curve = []
    def rank(measurement):
        return (measurement["quiet_passed"], measurement["static_gate"]["passed"],
                measurement["transition_gate"]["passed"],
                measurement["retention_gate"]["passed"],
                measurement["retention"]["mean_survival_seconds"],
                measurement["static"]["mean_survival_seconds"],
                measurement["transition"]["mean_survival_seconds"])

    def save_best(measurement):
        checkpoint = args.out / "best.pt"
        agent.save(str(checkpoint))
        (args.out / "best.json").write_text(json.dumps(dict(
            checkpoint_sha256=sha256(checkpoint), iteration=agent._iter,
            samples=agent._sample_count, metrics=measurement,
            contract=train_task.observation_contract()), indent=2), encoding="utf-8")

    initial = review()
    agent.save(str(args.out / "initial.pt"))
    best_rank = rank(initial)
    save_best(initial)
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    agent._init_train()
    started = time.monotonic()
    while time.monotonic() < started + args.seconds:
        info = agent._train_iter()
        agent._sample_count = agent._update_sample_count()
        agent._iter += 1
        if agent._iter % args.evaluation_interval == 0:
            measurement = review()
            row = dict(iteration=agent._iter, samples=agent._sample_count,
                       elapsed_seconds=time.monotonic() - started,
                       actor_kl=float(info["actor_kl"]), measurement=measurement)
            curve.append(row)
            (args.out / "learning_curve.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")
            if rank(measurement) > best_rank:
                best_rank = rank(measurement)
                save_best(measurement)
            print(json.dumps(dict(iteration=agent._iter, quiet=measurement["quiet_passed"],
                                  static_seconds=measurement["static"]["mean_survival_seconds"],
                                  transition_seconds=measurement["transition"]["mean_survival_seconds"])), flush=True)
            if (measurement["quiet_passed"] and measurement["static_gate"]["passed"]
                    and measurement["transition_gate"]["passed"]
                    and measurement["retention_gate"]["passed"]):
                break
    elapsed = time.monotonic() - started
    agent.save(str(args.out / "final.pt"))
    final = review()
    report = dict(schema="mimic_support_transition_pilot_v2", training_seconds=elapsed,
                  iterations=agent._iter, samples=agent._sample_count, initial=initial,
                  final=final, learning_curve=curve, best_checkpoint_sha256=sha256(args.out / "best.pt"),
                  passed=bool(final["quiet_passed"] and final["static_gate"]["passed"]
                              and final["transition_gate"]["passed"]
                              and final["retention_gate"]["passed"]),
                  scope="One-seed native static/entry/retention/quiet evaluation; no deployment export")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(iterations=agent._iter, samples=agent._sample_count,
                          static_seconds=final["static"]["mean_survival_seconds"],
                          transition_seconds=final["transition"]["mean_survival_seconds"],
                          quiet=final["quiet_passed"], passed=report["passed"])), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--seconds", type=float, default=300.)
    parser.add_argument("--seed", type=int, default=240924)
    parser.add_argument("--evaluation-interval", type=int, default=16)
    parser.add_argument("--training-mode", choices=("static", "transition"), default="static")
    parser.add_argument("--quiet-fraction", type=float, default=0.)
    parser.add_argument("--support-seconds", type=float, default=12.)
    parser.add_argument("--retention-seconds", type=float, default=6.)
    parser.add_argument("--reward-mode", choices=("reference", TRACKING_REWARD_SCHEMA),
                        default="reference")
    parser.add_argument("--actor-max-kl", type=float, default=.02)
    run(parser.parse_args())
