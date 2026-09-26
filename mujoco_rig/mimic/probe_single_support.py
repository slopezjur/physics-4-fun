"""Separate static support feasibility from the controller's ability to hold it."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .baseline import sha256
from .com_feedback_teacher import ComFeedbackTeacher
from .native_engine import NativeDummyEngine
from .observation_com import ObservationCom
from .policy import StandPolicy
from .probe_steps import validate_reference_substitution
from .retarget_steps import validate_motion
from .rig import Rig
from .runtime import activate
from .support_transition import SupportTransitionReference
from .track_transfer import CalibrationReference


SCHEMA = "mimic_single_support_hold_probe_v1"
POSE_SECONDS = (7., 8., 9.)
HOLD_SECONDS = 3.
FINAL_SETTLE_SECONDS = .5
PHYSICAL_CHECKS = ("finite", "joint_limits", "penetration", "stance_slide",
                   "ground_support", "stance_contact", "foot_tracking", "joint_speed",
                   "sole_contact", "force_consistency", "root_wrench_consistency",
                   "actuation_feasibility")


class PoseHoldReference:
    def __init__(self, pose, rig, source_hash, phase_seconds, device="cpu"):
        self.dt = .05
        self.duration = 6.
        self.qpos = torch.as_tensor(np.tile(pose, (121, 1)), dtype=torch.float32,
                                    device=device)
        self.qvel = torch.zeros((121, rig.model.nv), device=device)
        digest = hashlib.sha256()
        digest.update(SCHEMA.encode())
        digest.update(source_hash.encode())
        digest.update(str(phase_seconds).encode())
        digest.update(np.asarray(pose).tobytes())
        self.sha256 = digest.hexdigest()

    def sample(self, time):
        return (self.qpos[0].expand(*time.shape, self.qpos.shape[-1]),
                self.qvel[0].expand(*time.shape, self.qvel.shape[-1]))


def physical_pose(rig, arrays, frame):
    pose = arrays["qpos"][frame]
    qpos = np.tile(pose, (3, 1))
    qvel = np.zeros((3, rig.model.nv))
    targets = np.tile(arrays["foot_targets"][frame], (3, 1, 1))
    contact = np.tile(arrays["contact"][frame], (3, 1))
    soles = np.tile(arrays["sole_contact"][frame], (3, 1))
    checks, metrics = validate_motion(rig, qpos, qvel, targets, contact, .05, soles)
    return dict(passed=all(checks[name] for name in PHYSICAL_CHECKS),
                checks=checks, metrics=metrics, contact=contact[0].tolist())


def hold_metrics(task, survived, contact):
    q = task.engine.native_qpos()[0]
    loads = task.engine.get_ground_contact_forces(0)[0, task.allowed_ground, 2].abs()
    left, right = float(loads[0] + loads[2]), float(loads[1] + loads[3])
    upright = bool(q[2] >= .75 and 1 - 2 * (q[4].square() + q[5].square()) >= .9659)
    slow = bool(torch.linalg.vector_norm(task.engine.get_root_vel(0)[0, :2]) <= .2 and
                torch.linalg.vector_norm(task.engine.get_root_ang_vel(0)[0]) <= 1.)
    support = (bool(left >= 5. and right >= 5.) if all(contact)
               else bool(left < 5. and right >= 5.))
    return dict(final_left_load_n=left, final_right_load_n=right,
                final_upright=upright, final_slow=slow,
                passed=bool(survived and support and upright and slow))


def evaluate_hold(task, controller, contact):
    from envs.base_env import DoneFlags, EnvMode

    task.set_mode(EnvMode.TEST)
    task.reset()
    settled_steps = 0
    count = 0
    done = DoneFlags.NULL.value
    for _ in range(round(HOLD_SECONDS / task.dt)):
        observation = task.observations()
        with torch.no_grad():
            action = controller(observation)
        _, _, flags, _ = task.step(action)
        done = int(flags[0])
        count += 1
        state = hold_metrics(task, False, contact)
        correctly_loaded = (state["final_left_load_n"] >= 5.
                            if all(contact) else state["final_left_load_n"] < 5.)
        correctly_loaded &= state["final_right_load_n"] >= 5.
        settled = (state["final_upright"] and state["final_slow"]
                   and correctly_loaded and done != DoneFlags.FAIL.value)
        settled_steps = settled_steps + 1 if settled else 0
        if done != DoneFlags.NULL.value:
            break
    survived = done == DoneFlags.TIME.value
    hold = hold_metrics(task, survived, contact)
    hold["final_settled_seconds"] = float(settled_steps * task.dt)
    hold["passed"] &= hold["final_settled_seconds"] >= FINAL_SETTLE_SECONDS
    metrics = dict(successes=int(survived), mean_survival_seconds=float(count * task.dt))
    return metrics, hold


def run(args):
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .stand_task import StandTask

    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    source = json.loads(args.source_contract.read_text())
    best = json.loads(args.initialize_from.with_name("best.json").read_text())
    if best["checkpoint_sha256"] != sha256(args.initialize_from):
        raise ValueError("Source checkpoint hash mismatch")
    validate_reference_substitution(source, best["contract"])
    original = CalibrationReference(args.reference, rig)
    if not original.placed_step or original.requested_step_m <= 0:
        raise ValueError("Require the validated forward placed-step reference")
    support = StandTask(rig, args.reference, 1, "cpu", engine_factory=NativeDummyEngine,
                        control_mode="target_pd", contact_mode="support_v2",
                        reference_factory=lambda p, r, d: SupportTransitionReference(
                            p, r, d, mode="transition"))
    validate_reference_substitution(source, support.observation_contract())
    agent = PPOAgent(yaml.safe_load(args.agent_config.read_text()), support, "cpu")
    agent.load(str(args.initialize_from))
    actor = StandPolicy(agent).eval()
    observer = ObservationCom(rig, source)
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    with np.load(original.path, allow_pickle=False) as arrays:
        for phase in POSE_SECONDS:
            frame = round(phase / original.dt)
            physical = physical_pose(rig, arrays, frame)
            result = dict(pose_seconds=phase, physical=physical, controllers=[])
            if physical["passed"]:
                reference = PoseHoldReference(arrays["qpos"][frame], rig,
                                              original.sha256, phase)
                for label in ("reference_pd", "frozen_actor", "observation_com_teacher"):
                    task = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz",
                                     1, "cpu", engine_factory=NativeDummyEngine,
                                     control_mode="target_pd", contact_mode="support_v2")
                    task.reference = reference
                    validate_reference_substitution(source, task.observation_contract())
                    task.episode_seconds = HOLD_SECONDS
                    if label == "reference_pd":
                        controller = lambda obs: torch.zeros((len(obs), len(rig.actuators)))
                    elif label == "frozen_actor":
                        controller = actor
                    else:
                        controller = ComFeedbackTeacher(task, actor,
                                                        position_gain=1.5, velocity_gain=1.5,
                                                        observer=observer)
                    metrics, hold = evaluate_hold(task, controller, physical["contact"])
                    result["controllers"].append(dict(label=label, evaluation=metrics,
                                                      hold=hold))
                    print(json.dumps(dict(pose_seconds=phase, label=label,
                                          survival_seconds=metrics["mean_survival_seconds"],
                                          **hold)), flush=True)
            rows.append(result)
            (args.out / "progress.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report = dict(schema=SCHEMA, model_sha256=sha256(rig.path),
                  parent_reference_sha256=original.sha256,
                  source_checkpoint_sha256=sha256(args.initialize_from),
                  source_contract_sha256=sha256(args.source_contract),
                  hold_seconds=HOLD_SECONDS, final_settle_seconds=FINAL_SETTLE_SECONDS,
                  poses=rows,
                  scope="Static support and native controller probe; no training, export or Godot selection")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
