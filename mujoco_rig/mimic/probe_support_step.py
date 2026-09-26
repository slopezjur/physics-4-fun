"""Native-only 10 cm landing attempt inside a measured right-leg support window."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .baseline import sha256
from .capture_step_reflex import (SOURCE_ACTION_RADIANS, STEP_SECONDS,
                                  WIDE_ACTION_RADIANS, WideStepTargetControl,
                                  swing_target)
from .native_engine import NativeDummyEngine
from .probe_single_support_entry import (CONTROLLERS, ScheduledStabilizer,
                                         build_entry)
from .rig import Rig
from .runtime import activate
from .track_transfer import CalibrationReference
from .target_control import DelayedTargetControl


SCHEMA = "mimic_native_support_step_probe_v2"
STRIDE_M = .10
TRIGGER_SECONDS = {7.: 1.35, 3.: 4.75}


class WindowStepController:
    def __init__(self, task, balance, trigger_seconds, action_radians,
                 latch_landing_reference=False):
        self.task, self.balance = task, balance
        self.trigger_seconds = trigger_seconds
        self.action_radians = action_radians
        self.latch_landing_reference = latch_landing_reference
        self.latched_at_seconds = None
        model = task.rig.model
        names = [model.actuator(int(a)).name for a in task.rig.actuators]
        joints = [model.joint(name).id for name in
                  ("Thigh_L_rx", "Shin_L_rx", "Foot_L_rx")]
        self.dofs = model.jnt_dofadr[joints]
        self.qpos = model.jnt_qposadr[joints]
        self.actions = [names.index(name) for name in
                        ("Thigh_L_rx", "Shin_L_rx", "Foot_L_rx")]
        self.foot = model.body("Foot_L").id
        landing_joints = [j for j in range(1, model.njnt)
                          if model.joint(j).name.startswith(
                              ("Thigh_L", "Shin_L", "Foot_L", "Toe_L"))]
        self.landing_qpos = model.jnt_qposadr[landing_joints]
        self.landing_qvel = model.jnt_dofadr[landing_joints]
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.start = None
        self.peak_lift = 0.
        self.max_shift = 0.
        self.samples = 0
        self.saturated = 0

    def __call__(self, observation):
        action = self.balance(observation)
        action *= SOURCE_ACTION_RADIANS / self.action_radians
        time = float(self.task.steps[0] * self.task.dt)
        if time < self.trigger_seconds:
            return action
        data = self.task.engine.datas[0]
        foot = data.xpos[self.foot]
        if self.start is None:
            self.start = foot[[0, 2]].copy()
        self.peak_lift = max(self.peak_lift, float(foot[2] - self.start[1]))
        self.max_shift = max(self.max_shift, float(foot[0] - self.start[0]))
        desired = swing_target(self.start, STRIDE_M, time - self.trigger_seconds)
        mujoco.mj_jacBody(self.task.rig.model, data, self.jacp, self.jacr, self.foot)
        jacobian = self.jacp[[0, 2]][:, self.dofs]
        error = desired - foot[[0, 2]]
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T +
                                             .002 * np.eye(2), error)
        target = data.qpos[self.qpos] + np.clip(delta, -.2, .2)
        reference, _ = self.task.reference.sample(
            self.task.offset + self.task.steps * self.task.dt)
        residual = np.clip((target - reference[0, self.qpos].numpy()) /
                           self.action_radians,
                           -1., 1.)
        self.samples += 1
        self.saturated += int(np.any(np.abs(residual) >= .99))
        action[0, self.actions] = torch.as_tensor(residual, dtype=action.dtype)
        return action

    def latch_reference(self, time):
        if not self.latch_landing_reference or self.latched_at_seconds is not None:
            return
        data = self.task.engine.datas[0]
        frame = min(round(time / self.task.reference.dt),
                    len(self.task.reference.qpos) - 1)
        self.task.reference.qpos[frame:, self.landing_qpos] = torch.as_tensor(
            data.qpos[self.landing_qpos], dtype=torch.float32)
        self.task.reference.qvel[frame:, self.landing_qvel] = 0.
        self.latched_at_seconds = time


def evaluate(task, controller):
    from envs.base_env import DoneFlags, EnvMode

    task.set_mode(EnvMode.TEST)
    task.reset()
    first_contact = None
    continuous_contact = None
    longest_contact = 0.
    records = []
    for step in range(round(task.episode_seconds / task.dt)):
        with torch.no_grad():
            action = controller(task.observations())
        _, _, flags, _ = task.step(action)
        time = float(task.steps[0] * task.dt)
        q = task.engine.native_qpos()[0]
        data = task.engine.datas[0]
        foot = data.xpos[controller.foot]
        loads = task.engine.get_ground_contact_forces(0)[0, task.allowed_ground, 2].abs()
        left, right = float(loads[0] + loads[2]), float(loads[1] + loads[3])
        displacement = (0. if controller.start is None
                        else float(foot[0] - controller.start[0]))
        upright = bool(q[2] >= .75 and 1 - 2 * (q[4].square() + q[5].square()) >= .9659)
        placed_contact = (time >= controller.trigger_seconds + STEP_SECONDS
                          and controller.peak_lift >= .03 and displacement >= .05
                          and left >= 5. and right >= 5. and upright)
        if placed_contact:
            first_contact = time if first_contact is None else first_contact
            continuous_contact = time if continuous_contact is None else continuous_contact
            longest_contact = max(longest_contact,
                                  time - continuous_contact + task.dt)
            controller.latch_reference(time)
        else:
            continuous_contact = None
        if step == 0 or time - records[-1]["seconds"] >= .1 or flags[0] != DoneFlags.NULL.value:
            records.append(dict(seconds=time, pelvis_height_m=float(q[2]),
                                left_load_n=left, right_load_n=right,
                                foot_shift_m=displacement,
                                foot_lift_m=0. if controller.start is None else
                                float(foot[2] - controller.start[1]),
                                upright=upright))
        if flags[0] != DoneFlags.NULL.value:
            break
    survived = flags[0] == DoneFlags.TIME.value
    retained = (0. if continuous_contact is None else
                time - continuous_contact + task.dt)
    return dict(passed=bool(survived and first_contact is not None and retained >= 3.),
                survived=bool(survived), survival_seconds=time,
                first_placed_contact_seconds=first_contact,
                reference_latched_seconds=controller.latched_at_seconds,
                longest_placed_contact_seconds=longest_contact,
                final_placed_contact_seconds=retained,
                peak_foot_lift_m=controller.peak_lift,
                peak_foot_shift_m=controller.max_shift,
                ik_saturated_fraction=controller.saturated / max(1, controller.samples),
                trace=records)


def run(args):
    activate(args.mimickit)
    from util import mp_util
    from .stand_task import StandTask

    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    source = CalibrationReference(args.reference, rig)
    with np.load(source.path, allow_pickle=False) as arrays:
        reference, _, times, torques, com = build_entry(
            rig, arrays, args.start_seconds)
    control = (DelayedTargetControl if args.action_radians == SOURCE_ACTION_RADIANS
               else WideStepTargetControl)
    def engine_factory(model_rig, worlds, device, control_factory=None):
        return NativeDummyEngine(model_rig, worlds, device, control_factory=control)
    task = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz",
                     1, "cpu", engine_factory=engine_factory,
                     control_mode="target_pd", contact_mode="support_v2")
    task.reference = reference
    task.episode_seconds = reference.duration
    balance = ScheduledStabilizer(task, times, torques, com,
                                  CONTROLLERS["balanced_axes"])
    trigger = TRIGGER_SECONDS[args.start_seconds]
    controller = WindowStepController(task, balance, trigger,
                                      args.action_radians,
                                      latch_landing_reference=args.latch_landing_reference)
    result = evaluate(task, controller)
    report = dict(schema=SCHEMA, source_reference_sha256=source.sha256,
                  model_sha256=sha256(rig.path),
                  start_source_seconds=args.start_seconds,
                  trigger_seconds=trigger, requested_stride_m=STRIDE_M,
                  swing_seconds=STEP_SECONDS,
                  action_radians=args.action_radians,
                  latch_landing_reference=args.latch_landing_reference,
                  controller_parameters=CONTROLLERS["balanced_axes"],
                  evaluation=result,
                  scope="Native oracle-COM and native-foot IK diagnostic; no training, export or viewer selection")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--start-seconds", type=float, choices=tuple(TRIGGER_SECONDS), default=7.)
    parser.add_argument("--action-radians", type=float,
                        choices=(SOURCE_ACTION_RADIANS, WIDE_ACTION_RADIANS),
                        default=SOURCE_ACTION_RADIANS)
    parser.add_argument("--latch-landing-reference", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
