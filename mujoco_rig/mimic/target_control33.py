"""Rig-sized delayed target commands with bounded local joint feedback."""
import numpy as np
import torch

from mujoco_rig.rl.env_config import ACTION_LATENCY_STEPS


class DelayedTargetControl33:
    residual_radians = 0.25
    gain = 4.0
    damping_time = 0.02

    def __init__(self, rig, num_envs, device):
        self.count = len(rig.actuators)
        if self.count != 33 or ACTION_LATENCY_STEPS != 2:
            raise ValueError('This experiment requires 33 motors and two delayed commands')
        joints = rig.model.actuator_trnid[rig.actuators, 0]
        if not np.all(rig.model.jnt_limited[joints]):
            raise ValueError('All controlled hinges must have limits')
        self.indices = torch.as_tensor(rig.actuators, device=device)
        self.q_indices = torch.as_tensor(rig.model.jnt_qposadr[joints], device=device)
        self.v_indices = torch.as_tensor(rig.model.jnt_dofadr[joints], device=device)
        self.scale = torch.as_tensor(rig.action_scale, dtype=torch.float32, device=device)
        self.lower = torch.as_tensor(rig.model.jnt_range[joints, 0], dtype=torch.float32, device=device)
        self.upper = torch.as_tensor(rig.model.jnt_range[joints, 1], dtype=torch.float32, device=device)
        self.ctrl = torch.zeros((num_envs, rig.model.nu), device=device)
        self.pending = torch.zeros((ACTION_LATENCY_STEPS, num_envs, 2*self.count+1), device=device)
        self.active = torch.zeros_like(self.pending[0])
        self.cursor = 0

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.ctrl[ids] = 0
        self.pending[:, ids] = 0
        self.active[ids] = 0
        if env_ids is None:
            self.cursor = 0

    def set_reference(self, q, v):
        self.reference_q, self.reference_v = q, v

    def target_positions(self, action):
        return (self.reference_q[:, self.q_indices] + self.residual_radians*action.clamp(-1, 1)).clamp(self.lower, self.upper)

    def apply(self, action):
        if action.shape != (len(self.ctrl), self.count) or not torch.isfinite(action).all():
            raise ValueError('Expected finite [worlds, 33] target residuals')
        self.active.copy_(self.pending[self.cursor])
        slot = self.pending[self.cursor]
        slot[:, :self.count] = self.target_positions(action)
        slot[:, self.count:2*self.count] = self.reference_v[:, self.v_indices]
        slot[:, 2*self.count] = 1
        self.cursor = (self.cursor+1) % len(self.pending)
        return self.ctrl

    def physics_control(self, q, v):
        error = self.active[:, :self.count]-q[:, self.q_indices]
        velocity_error = self.active[:, self.count:2*self.count]-v[:, self.v_indices]
        normalized = self.gain*error + self.gain*self.damping_time*velocity_error
        self.ctrl.zero_()
        self.ctrl[:, self.indices] = normalized.clamp(-1, 1)*self.scale*self.active[:, 2*self.count:]
        return self.ctrl

    def contract(self):
        return dict(action_mode='reference_relative_target_pd_33', motor_count=self.count,
            residual_radians=self.residual_radians, pd_gain_per_torque_limit=self.gain,
            pd_damping_time=self.damping_time, target_qpos_indices=self.q_indices.tolist(),
            target_qvel_indices=self.v_indices.tolist(), target_lower_radians=self.lower.tolist(),
            target_upper_radians=self.upper.tolist(), target_timing='sample_at_issue_hold_after_delay',
            initial_motor_control='zero_until_valid_target', pd_update='every_physics_step', feedforward='none')
