"""Native-only COM feedback teacher layered over the retained Stand actor."""
import math

import mujoco
import numpy as np
import torch


class ComFeedbackTeacher:
    def __init__(self, task, actor, *, position_gain, velocity_gain,
                 max_correction=.5, hip_fraction=0., observer=None):
        parameters = (position_gain, velocity_gain, max_correction, hip_fraction)
        if (task.device != "cpu" or task.get_num_envs() < 1
                or not hasattr(task.engine, "datas")
                or any(not math.isfinite(value) for value in parameters)
                or position_gain < 0 or velocity_gain < 0
                or not 0 < max_correction <= 1 or not 0 <= hip_fraction <= 1):
            raise ValueError("Require native worlds and finite, bounded feedback parameters")
        self.task, self.actor = task, actor
        self.observer = observer
        self.position_gain, self.velocity_gain = position_gain, velocity_gain
        self.max_correction, self.hip_fraction = max_correction, hip_fraction
        self.records = []
        names = [task.rig.model.actuator(int(a)).name for a in task.rig.actuators]
        self.ankles = [names.index(name) for name in ("Foot_L_rx", "Foot_R_rx")]
        self.hips = [names.index(name) for name in ("Thigh_L_rx", "Thigh_R_rx")]
        target = mujoco.MjData(task.rig.model)
        positions, velocities = [], []
        for q, v in zip(task.reference.qpos.cpu().numpy(), task.reference.qvel.cpu().numpy()):
            target.qpos[:], target.qvel[:] = q, v
            mujoco.mj_forward(task.rig.model, target)
            mujoco.mj_subtreeVel(task.rig.model, target)
            positions.append(target.subtree_com[1, 0])
            velocities.append(target.subtree_linvel[1, 0])
        self.target_position = np.asarray(positions)
        self.target_velocity = np.asarray(velocities)

    def __call__(self, observation):
        if self.observer is None:
            state = []
            for data in self.task.engine.datas:
                mujoco.mj_subtreeVel(self.task.engine.rig.model, data)
                state.append((data.subtree_com[1, 0], data.subtree_linvel[1, 0]))
            state = np.asarray(state)
        else:
            state = self.observer(observation)
        if state.shape != (self.task.get_num_envs(), 2):
            raise ValueError("COM observer returned the wrong shape")
        com, velocity = state[:, 0], state[:, 1]
        time = (self.task.offset + self.task.steps * self.task.dt).cpu().numpy()
        frame = np.clip(time / self.task.reference.dt, 0., len(self.target_position) - 1)
        target_position = np.interp(frame, np.arange(len(self.target_position)),
                                    self.target_position)
        target_velocity = np.interp(frame, np.arange(len(self.target_velocity)),
                                    self.target_velocity)
        position_error = np.asarray(com) - target_position
        velocity_error = np.asarray(velocity) - target_velocity
        correction = np.clip(-self.position_gain * position_error
                             - self.velocity_gain * velocity_error,
                             -self.max_correction, self.max_correction)
        with torch.no_grad():
            action = self.actor(observation).clone()
        correction_tensor = torch.as_tensor(correction, dtype=action.dtype, device=action.device)[:, None]
        action[:, self.ankles] += correction_tensor
        action[:, self.hips] += self.hip_fraction * correction_tensor
        action.clamp_(-1., 1.)
        if self.task.get_num_envs() == 1:
            self.records.append(dict(seconds=float(time[0]), com_x_m=float(com[0]),
                                     com_vx_m_s=float(velocity[0]),
                                     target_com_x_m=float(target_position[0]),
                                     position_error_m=float(position_error[0]),
                                     velocity_error_m_s=float(velocity_error[0]),
                                     correction=float(correction[0])))
        return action

    def contract(self):
        return dict(schema="mimic_native_com_teacher_v1",
                    feedback="Paired ankle-pitch target residual; optional paired hip pitch",
                    position_gain=self.position_gain, velocity_gain=self.velocity_gain,
                    max_correction=self.max_correction, hip_fraction=self.hip_fraction,
                    target="Reference whole-body COM X and velocity",
                    state_source=("Native MuJoCo subtree COM/velocity oracle" if self.observer is None
                                  else self.observer.contract()))
