"""Bounded right-leg-support PPO pilot with native quiet/hold selection gates."""
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
from .native_engine import NativeDummyEngine
from .policy import StandPolicy
from .probe_single_support import (HOLD_SECONDS, PoseHoldReference,
                                   evaluate_hold, physical_pose)
from .probe_steps import validate_reference_substitution
from .rig import Rig
from .runtime import MIMICKIT_REVISION, activate
from .track_transfer import CalibrationReference


SCHEMA = "mimic_single_support_ppo_pilot_v1"
TARGET_POSE_SECONDS = 8.


def support_reward(base, left_load_n, right_load_n, root_velocity_xy):
    """Retain imitation while rewarding loaded stance and quiet root motion."""
    right = torch.sigmoid((right_load_n - 20.) / 10.)
    left = torch.sigmoid((10. - left_load_n) / 5.)
    quiet = torch.exp(-root_velocity_xy.square().sum(dim=-1) / .15 ** 2)
    return .6 * base + .2 * right * left + .2 * quiet


def run(args):
    activate(args.mimickit)
    from util import mp_util
    from .guarded_ppo import GuardedPPO
    from .newton_adapter import DummyNewtonEngine
    from .stand_task import StandTask

    if (args.out.exists() or args.seconds <= 0 or args.envs < 32
            or args.envs % 32 or args.evaluation_interval < 1):
        raise ValueError("Require a new output directory and positive pilot settings")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mp_util.init(0, 1, "cuda:0", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    source_contract = json.loads(args.source_contract.read_text())
    best = json.loads(args.initialize_from.with_name("best.json").read_text())
    if best["checkpoint_sha256"] != sha256(args.initialize_from):
        raise ValueError("Source checkpoint hash mismatch")
    validate_reference_substitution(source_contract, best["contract"])
    parent = CalibrationReference(args.reference, rig)
    if not parent.placed_step or parent.requested_step_m <= 0:
        raise ValueError("Require the physically checked forward step")
    with np.load(parent.path, allow_pickle=False) as arrays:
        frame = round(TARGET_POSE_SECONDS / parent.dt)
        physical = physical_pose(rig, arrays, frame)
        pose = arrays["qpos"][frame].copy()
    if not physical["passed"] or physical["contact"] != [False, True]:
        raise ValueError("Target must be physically feasible right-leg support")

    def reference_factory(path, model_rig, device):
        return PoseHoldReference(pose, model_rig, parent.sha256,
                                 TARGET_POSE_SECONDS, device)

    class SingleSupportTask(StandTask):
        def reward(self):
            base = self.reference_reward()
            loads = self.engine.get_ground_contact_forces(0)[:, self.allowed_ground, 2].abs()
            return support_reward(base, loads[:, 0] + loads[:, 2],
                                  loads[:, 1] + loads[:, 3],
                                  self.engine.get_root_vel(0)[:, :2])

    train_task = SingleSupportTask(
        rig, args.reference, args.envs, "cuda:0", seed=args.seed,
        engine_factory=DummyNewtonEngine, control_mode="target_pd",
        contact_mode="support_v2", reference_factory=reference_factory)
    train_task.episode_seconds = HOLD_SECONDS
    hold = StandTask(rig, args.reference, 1, "cpu", engine_factory=NativeDummyEngine,
                     control_mode="target_pd", contact_mode="support_v2",
                     reference_factory=reference_factory)
    hold.episode_seconds = HOLD_SECONDS
    quiet = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz",
                      8, "cpu", engine_factory=NativeDummyEngine,
                      control_mode="target_pd", contact_mode="support_v2")
    validate_reference_substitution(source_contract, train_task.observation_contract())
    validate_reference_substitution(source_contract, hold.observation_contract())
    config = yaml.safe_load(args.agent_config.read_text())
    config.update(actor_max_kl=.02, test_episodes=0,
                  observation_normalization="frozen_loaded_statistics",
                  actor_gradient_clip=1.)
    config["critic_optimizer"] = {**config["critic_optimizer"], "grad_clip": 1.}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "agent.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    protocol = dict(schema=SCHEMA, source_checkpoint_sha256=sha256(args.initialize_from),
                    source_contract_sha256=sha256(args.source_contract),
                    model_sha256=sha256(rig.path), source_reference_sha256=parent.sha256,
                    target_pose_seconds=TARGET_POSE_SECONDS, target_contact=physical["contact"],
                    target_physical_checks=physical["checks"],
                    actor_contract=train_task.observation_contract(),
                    reward=dict(base_weight=.6, right_support_weight=.2,
                                quiet_velocity_weight=.2, left_unloaded=True),
                    native_gate=dict(hold_seconds=HOLD_SECONDS,
                                     quiet_standing_phases=8,
                                     final_settled_seconds=.5),
                    envs=args.envs, seconds=args.seconds, seed=args.seed,
                    evaluation_interval=args.evaluation_interval,
                    mimickit_revision=MIMICKIT_REVISION,
                    scope="One isolated right-leg-support pilot; no export or Godot selection")
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    agent = GuardedPPO(config, train_task, "cuda:0")
    agent.load(str(args.initialize_from))
    if agent._obs_norm.get_count().item() <= 0:
        raise ValueError("Require trained source normalization")

    def review():
        actor = copy.deepcopy(StandPolicy(agent)).eval().cpu()
        quiet_metrics, _ = evaluate(quiet, actor)
        hold_metrics, hold_gate = evaluate_hold(hold, actor, physical["contact"])
        return dict(quiet=quiet_metrics, quiet_passed=standing_passed(quiet_metrics),
                    hold=hold_metrics, hold_gate=hold_gate)

    def rank(result):
        return (result["quiet_passed"], result["hold_gate"]["passed"],
                result["hold"]["mean_survival_seconds"],
                result["hold_gate"]["final_settled_seconds"])

    def save_best(result):
        path = args.out / "best.pt"
        agent.save(str(path))
        (args.out / "best.json").write_text(json.dumps(
            dict(checkpoint_sha256=sha256(path), iteration=agent._iter,
                 samples=agent._sample_count, metrics=result,
                 contract=train_task.observation_contract()), indent=2), encoding="utf-8")

    initial = review()
    agent.save(str(args.out / "initial.pt"))
    best = initial
    save_best(initial)
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    agent._init_train()
    curve = []
    actor_steps_accepted = 0
    actor_steps_rejected = 0
    peak_attempted_kl = 0.
    started = time.monotonic()
    while time.monotonic() - started < args.seconds:
        info = agent._train_iter()
        actor_steps_accepted += int(info.get("actor_steps_accepted", 0))
        actor_steps_rejected += int(info.get("actor_steps_rejected", 0))
        peak_attempted_kl = max(peak_attempted_kl,
                                float(info.get("actor_attempted_kl", 0.)))
        agent._sample_count = agent._update_sample_count()
        agent._iter += 1
        if agent._iter % args.evaluation_interval == 0:
            result = review()
            row = dict(iteration=agent._iter, samples=agent._sample_count,
                       elapsed_seconds=time.monotonic() - started,
                       actor_kl=float(info["actor_kl"]),
                       actor_steps_accepted=actor_steps_accepted,
                       actor_steps_rejected=actor_steps_rejected,
                       peak_attempted_kl=peak_attempted_kl,
                       measurement=result)
            curve.append(row)
            (args.out / "learning_curve.json").write_text(json.dumps(curve, indent=2),
                                                            encoding="utf-8")
            if rank(result) > rank(best):
                best = result
                save_best(result)
            print(json.dumps(dict(iteration=agent._iter,
                                  quiet=result["quiet_passed"],
                                  hold_seconds=result["hold"]["mean_survival_seconds"],
                                  hold_passed=result["hold_gate"]["passed"])), flush=True)
            if result["quiet_passed"] and result["hold_gate"]["passed"]:
                break
    elapsed = time.monotonic() - started
    agent.save(str(args.out / "final.pt"))
    final = review()
    report = dict(schema=SCHEMA, training_seconds=elapsed,
                  iterations=agent._iter, samples=agent._sample_count,
                  initial=initial, final=final, best=best, learning_curve=curve,
                  actor_steps_accepted=actor_steps_accepted,
                  actor_steps_rejected=actor_steps_rejected,
                  peak_attempted_kl=peak_attempted_kl,
                  best_checkpoint_sha256=sha256(args.out / "best.pt"),
                  passed=bool(best["quiet_passed"] and best["hold_gate"]["passed"]),
                  scope="Single-seed right-leg-support pilot; no deployment export")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(iterations=agent._iter, samples=agent._sample_count,
                          best_hold_seconds=best["hold"]["mean_survival_seconds"],
                          best_quiet=best["quiet_passed"], passed=report["passed"])),
          flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--seconds", type=float, default=300.)
    parser.add_argument("--seed", type=int, default=250925)
    parser.add_argument("--evaluation-interval", type=int, default=8)
    run(parser.parse_args())
