"""Native-only capture-point step probe around the observation-COM controller."""
import math

import mujoco
import numpy as np
import torch

from .com_feedback_teacher import ComFeedbackTeacher
from .target_control import DelayedTargetControl


STEP_SECONDS = .55
MAX_STRIDE_M = .15
MIN_STRIDE_M = .10
LIFT_M = .06
SHOE_HALF_LENGTH_M = .09
TRIGGER_EXCESS_M = .02
SOURCE_ACTION_RADIANS = .25
WIDE_ACTION_RADIANS = .5


class WideStepTargetControl(DelayedTargetControl):
    """Isolated native diagnostic with a wider target envelope."""
    residual_radians = WIDE_ACTION_RADIANS


def capture_excess(com_x, com_vx, pelvis_z, foot_x):
    """Signed distance of the linear capture point beyond the planted feet."""
    omega = math.sqrt(9.81 / max(float(pelvis_z), .5))
    capture = float(com_x + com_vx / omega)
    lower = min(foot_x) - SHOE_HALF_LENGTH_M
    upper = max(foot_x) + SHOE_HALF_LENGTH_M
    return min(0., capture - lower) + max(0., capture - upper)


def swing_target(start_xz, stride, elapsed):
    progress = min(1., max(0., elapsed / STEP_SECONDS))
    smooth = progress * progress * (3. - 2. * progress)
    lift = LIFT_M * math.sin(math.pi * progress)
    return np.asarray((start_xz[0] + stride * smooth, start_xz[1] + lift))


def planned_stride(excess):
    return math.copysign(min(MAX_STRIDE_M, max(MIN_STRIDE_M, abs(excess) + .05)), excess)


class CaptureStepReflex:
    """One bounded foot placement after a confirmed hit; never an export policy."""

    def __init__(self, task, actor, observer, *, action_radians=SOURCE_ACTION_RADIANS):
        if task.device != "cpu" or observer is None or not hasattr(task.engine, "datas"):
            raise ValueError("Capture-step probe requires native data and observation COM")
        if action_radians not in (SOURCE_ACTION_RADIANS, WIDE_ACTION_RADIANS):
            raise ValueError("Unknown diagnostic action envelope")
        self.task, self.observer = task, observer
        self.action_radians = action_radians
        self.balance = ComFeedbackTeacher(task, actor, position_gain=1.5,
                                          velocity_gain=1.5, observer=observer)
        model = task.engine.rig.model
        names = [task.rig.model.actuator(int(a)).name for a in task.rig.actuators]
        self.legs = []
        for side in ("L", "R"):
            joint_names = (f"Thigh_{side}_rx", f"Shin_{side}_rx", f"Foot_{side}_rx")
            joints = [model.joint(name).id for name in joint_names]
            self.legs.append(dict(body=model.body(f"Foot_{side}").id,
                                  dofs=np.asarray(model.jnt_dofadr[joints]),
                                  qpos=np.asarray(model.jnt_qposadr[joints]),
                                  action=np.asarray([names.index(name) for name in joint_names])))
        n = task.get_num_envs()
        self.triggered = np.zeros(n, dtype=bool)
        self.finished = np.zeros(n, dtype=bool)
        self.trigger_step = np.zeros(n, dtype=int)
        self.trigger_seconds = np.zeros(n)
        self.swing = np.zeros(n, dtype=int)
        self.stride = np.zeros(n)
        self.start_xz = np.zeros((n, 2))
        self.peak_lift = np.zeros(n)
        self.peak_shift = np.zeros(n)
        self.landing_shift = np.zeros(n)
        self.landed = np.zeros(n, dtype=bool)
        self.ik_samples = np.zeros(n, dtype=int)
        self.saturated_samples = np.zeros(n, dtype=int)
        self.peak_ik_error = np.zeros(n)
        self.trigger_excess = np.zeros(n)
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def __call__(self, observation):
        action = self.balance(observation)
        action *= SOURCE_ACTION_RADIANS / self.action_radians
        state = self.observer(observation)
        feet = self.task.engine.get_body_pos(0)[:, self.task.feet, :].cpu().numpy()
        ground = self.task.engine.get_ground_contact_forces(0).cpu().numpy()
        loads = observation[:, 93:95].cpu().numpy()
        reference, _ = self.task.reference.sample(self.task.offset + self.task.steps * self.task.dt)
        reference = reference.cpu().numpy()
        model = self.task.engine.rig.model
        for world, data in enumerate(self.task.engine.datas):
            if self.finished[world]:
                continue
            if self.task.engine.hit[world] and not self.triggered[world]:
                excess = capture_excess(state[world, 0], state[world, 1],
                                        data.qpos[2], feet[world, :, 0])
                if abs(excess) > TRIGGER_EXCESS_M:
                    self.triggered[world] = True
                    self.trigger_step[world] = int(self.task.steps[world])
                    self.trigger_seconds[world] = float(self.task.steps[world] * self.task.dt)
                    self.trigger_excess[world] = excess
                    self.swing[world] = int(np.argmin(loads[world]))
                    foot = feet[world, self.swing[world]]
                    self.start_xz[world] = foot[[0, 2]]
                    self.stride[world] = planned_stride(excess)
            if not self.triggered[world]:
                continue
            leg = self.legs[self.swing[world]]
            foot = data.xpos[leg["body"]]
            self.peak_lift[world] = max(self.peak_lift[world],
                                        float(foot[2] - self.start_xz[world, 1]))
            self.peak_shift[world] = max(self.peak_shift[world],
                                         abs(float(foot[0] - self.start_xz[world, 0])))
            elapsed = (int(self.task.steps[world]) - self.trigger_step[world]) * self.task.dt
            if elapsed >= STEP_SECONDS and not self.landed[world]:
                self.landing_shift[world] = float(foot[0] - self.start_xz[world, 0])
                foot_index = int(self.task.feet[self.swing[world]])
                toe_index = int(self.task.allowed_ground[2 + self.swing[world]])
                load = abs(ground[world, foot_index, 2]) + abs(ground[world, toe_index, 2])
                self.landed[world] = bool(self.peak_lift[world] >= .03
                                          and abs(self.landing_shift[world]) >= .05
                                          and foot[2] <= self.start_xz[world, 1] + .02
                                          and load >= 5.)
            desired = swing_target(self.start_xz[world], self.stride[world], elapsed)
            mujoco.mj_jacBody(model, data, self.jacp, self.jacr, leg["body"])
            jacobian = self.jacp[[0, 2]][:, leg["dofs"]]
            error = desired - foot[[0, 2]]
            self.peak_ik_error[world] = max(self.peak_ik_error[world],
                                            float(np.linalg.norm(error)))
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T +
                                                 .002 * np.eye(2), error)
            target = data.qpos[leg["qpos"]] + np.clip(delta, -.2, .2)
            residual = np.clip((target - reference[world, leg["qpos"]]) /
                               self.action_radians, -1., 1.)
            self.ik_samples[world] += 1
            self.saturated_samples[world] += int(np.any(np.abs(residual) >= .99))
            action[world, leg["action"]] = torch.as_tensor(residual, dtype=action.dtype)
        return action

    def deactivate(self, ids):
        self.finished[ids.cpu().numpy()] = True

    def case_metrics(self):
        return [dict(step_triggered=bool(self.triggered[i]),
                     trigger_seconds=float(self.trigger_seconds[i]),
                     trigger_capture_excess_m=float(self.trigger_excess[i]),
                     commanded_stride_m=float(self.stride[i]),
                     peak_swing_foot_lift_m=float(self.peak_lift[i]),
                     peak_swing_foot_shift_m=float(self.peak_shift[i]),
                     landing_shift_m=float(self.landing_shift[i]),
                     placed_at_least_5cm=bool(self.landed[i]),
                     peak_ik_position_error_m=float(self.peak_ik_error[i]),
                     ik_samples=int(self.ik_samples[i]),
                     ik_saturated_sample_fraction=float(self.saturated_samples[i] /
                                                        max(1, self.ik_samples[i])))
                for i in range(len(self.triggered))]

    def contract(self):
        return dict(schema="mimic_native_capture_step_probe_v1",
                    balance=self.balance.contract(), step_seconds=STEP_SECONDS,
                    min_stride_m=MIN_STRIDE_M, max_stride_m=MAX_STRIDE_M, lift_m=LIFT_M,
                    action_radians=self.action_radians,
                    shoe_half_length_m=SHOE_HALF_LENGTH_M,
                    trigger_excess_m=TRIGGER_EXCESS_M,
                    state_source="Observation-derived COM plus native foot pose/contact for diagnostic IK",
                    deployment="Native diagnostic only; no Godot parity or training admission")
