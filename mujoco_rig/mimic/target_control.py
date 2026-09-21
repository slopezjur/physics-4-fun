"""Versioned reference-relative targets; delay commands, not physics feedback."""
import numpy as np
import torch

from mujoco_rig.rl.env_config import ACTION_LATENCY_STEPS


class DelayedTargetControl:
    # Fixed before the bounded PPO comparison. No support-force oracle/feedforward.
    residual_radians = 0.25
    gain = 4.0
    damping_time = 0.02

    def __init__(self, rig, num_envs, device):
        joints = rig.model.actuator_trnid[rig.actuators, 0]
        self.indices = torch.as_tensor(rig.actuators, device=device)
        self.q_indices = torch.as_tensor(rig.model.jnt_qposadr[joints], device=device)
        self.v_indices = torch.as_tensor(rig.model.jnt_dofadr[joints], device=device)
        self.scale = torch.tensor(rig.action_scale, dtype=torch.float32, device=device)
        self.lower = torch.tensor(rig.model.jnt_range[joints, 0], dtype=torch.float32, device=device)
        self.upper = torch.tensor(rig.model.jnt_range[joints, 1], dtype=torch.float32, device=device)
        if not np.all(rig.model.jnt_limited[joints]):
            raise ValueError("Target control expects bounded hinge joints")
        self.ctrl = torch.zeros((num_envs, rig.model.nu), device=device)
        # Per slot: absolute target position, target velocity, initialized flag.
        self.pending = torch.zeros((ACTION_LATENCY_STEPS, num_envs, 61), device=device)
        self.active = torch.zeros((num_envs, 61), device=device)
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

    def apply(self, action):
        if action.shape != (len(self.ctrl), 30) or not torch.isfinite(action).all():
            raise ValueError("Expected finite [worlds, 30] residual actions")
        self.active.copy_(self.pending[self.cursor])
        slot = self.pending[self.cursor]
        slot[:, :30] = (self.reference_q[:, self.q_indices] +
                       action.clamp(-1, 1) * self.residual_radians).clamp(self.lower, self.upper)
        slot[:, 30:60] = self.reference_v[:, self.v_indices]
        slot[:, 60] = 1
        self.cursor = (self.cursor + 1) % len(self.pending)
        return self.ctrl

    def physics_control(self, q, v):
        error = self.active[:, :30] - q[:, self.q_indices]
        velocity_error = self.active[:, 30:60] - v[:, self.v_indices]
        normalized = self.gain * error + (self.gain * self.damping_time) * velocity_error
        self.ctrl.zero_()
        self.ctrl[:, self.indices] = normalized.clamp(-1, 1) * self.scale * self.active[:, 60:61]
        return self.ctrl

    def contract(self):
        return {"action_mode": "reference_relative_target_pd", "residual_radians": self.residual_radians,
                "pd_gain_per_torque_limit": self.gain, "pd_damping_time": self.damping_time,
                "target_qpos_indices": self.q_indices.tolist(), "target_qvel_indices": self.v_indices.tolist(),
                "target_lower_radians": self.lower.tolist(), "target_upper_radians": self.upper.tolist(),
                "target_timing": "sample_at_issue_hold_after_delay", "initial_motor_control": "zero_until_valid_target",
                "pd_update": "every_physics_step", "feedforward": "none"}
