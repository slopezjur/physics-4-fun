"""Diagnostic joint-space PD using only reference/state channels and policy torques."""
import torch
import mujoco
import numpy as np
from .rig import DelayedTorqueControl


class ImmediateTorqueDiagnostic(DelayedTorqueControl):
    """Counterfactual only: immediate torques, with zero padding in the task's queue channels."""
    def apply(self, action):
        super().apply(action)
        self.ctrl[:, self.indices] = action.clamp(-1, 1) * self.scale
        self.pending.zero_()
        return self.ctrl


class ReferencePD:
    def __init__(self, task, gain, damping_time, inertia_scaled=False, gravity_feedforward=False):
        self.task = task
        self.gain = gain
        self.damping_time = damping_time
        rig = task.rig
        self.dofs = torch.tensor(rig.model.jnt_dofadr[rig.model.actuator_trnid[rig.actuators, 0]],
                                 device=task.device)
        self.hold = False
        self.kp = torch.full((len(rig.actuators),), gain, device=task.device)
        self.kd = self.kp * damping_time
        self.feedforward = None
        if inertia_scaled or gravity_feedforward:
            data = mujoco.MjData(rig.model)
            data.qpos[:] = task.reference.qpos[0].cpu().numpy()
            mujoco.mj_forward(rig.model, data)
            if inertia_scaled:
                mass = np.zeros((rig.model.nv, rig.model.nv))
                mujoco.mj_fullM(rig.model, mass, data.qM)
                inertia = np.diag(mass)[self.dofs.cpu().numpy()]
                omega = 2 * np.pi * gain
                self.kp = torch.tensor(inertia * omega**2 / rig.action_scale, device=task.device, dtype=torch.float32)
                self.kd = torch.tensor(2 * damping_time * inertia * omega / rig.action_scale, device=task.device, dtype=torch.float32)
            if gravity_feedforward:
                feedforward = []
                for pose in task.reference.qpos.cpu().numpy():
                    data.qpos[:] = pose
                    mujoco.mj_forward(rig.model, data)
                    feedforward.append((data.qfrc_bias - data.qfrc_passive)[self.dofs.cpu().numpy()] / rig.action_scale)
                self.feedforward = torch.tensor(np.asarray(feedforward), device=task.device, dtype=torch.float32)

    @torch.no_grad()
    def __call__(self, observation):
        phase = self.task.offset if self.hold else self.task.offset + self.task.get_env_time()
        q, v = self.task.reference.sample(phase)
        velocity = observation[:, 54:93] * 10
        target_velocity = torch.zeros_like(v[:, self.dofs]) if self.hold else v[:, self.dofs]
        # kp = gain * torque_limit; kd = damping_time * kp. Returning normalized
        # torque preserves the shared clipping, authority and two-step command delay.
        action = (self.kp * (q[:, self.dofs + 1] - observation[:, 15:54][:, self.dofs - 6])
                  + self.kd * (target_velocity - velocity[:, self.dofs - 6]))
        if self.feedforward is not None:
            frame = (phase / self.task.reference.dt).clamp(0, len(self.feedforward) - 1)
            index = frame.long().clamp(max=len(self.feedforward) - 2)
            action += torch.lerp(self.feedforward[index], self.feedforward[index + 1], (frame - index)[:, None])
        return action
