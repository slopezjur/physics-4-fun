"""Newton projectile world with unchanged character observation/action views."""
import mujoco_warp
import numpy as np
import torch
import warp as wp

from .ball import load_ball_rig
from .newton_adapter import DummyNewtonEngine
from .target_control import DelayedTargetControl


class BallNewtonEngine(DummyNewtonEngine):
    def __init__(self, rig, num_envs, device="cuda:0", control_factory=DelayedTargetControl):
        self.character = rig
        super().__init__(load_ball_rig(rig), num_envs, device, control_factory)

    def _apply_start_xform(self):
        # This MJCF contains two articulations in one builder. Its authored initial
        # transforms are correct; the complete native reset follows initialization.
        self._sim_state.eval_fk()

    def reset_envs(self, ids, qpos, qvel):
        q = torch.tensor(self.rig.rest, device=self._device, dtype=torch.float32).repeat(len(ids), 1)
        v = torch.zeros((len(ids), self.rig.model.nv), device=self._device)
        q[:, :46], v[:, :45] = qpos, qvel
        super().reset_envs(ids, q, v)

    def launch_ball(self, ids, position, velocity):
        position = torch.as_tensor(position, device=self._device, dtype=torch.float32)
        velocity = torch.as_tensor(velocity, device=self._device, dtype=torch.float32)
        state = self._sim_state.raw_state
        for raw in (state, self._state_swap_buffer):
            q = wp.to_torch(raw.joint_q).view(self.get_num_envs(), -1)
            v = wp.to_torch(raw.joint_qd).view(self.get_num_envs(), -1)
            q[ids, 46:49] = position
            q[ids, 49:53] = torch.tensor([0., 0., 0., 1.], device=self._device)
            v[ids, 45:48] = velocity
            v[ids, 48:51] = 0
        self._sim_state.eval_fk()
        for field in ("body_q", "body_qd"):
            wp.copy(getattr(self._state_swap_buffer, field), getattr(state, field))
        q = wp.to_torch(self._solver.mjw_data.qpos)
        v = wp.to_torch(self._solver.mjw_data.qvel)
        q[ids, 46:49] = position
        q[ids, 49:53] = torch.tensor([1., 0., 0., 0.], device=self._device)
        v[ids, 45:48] = velocity
        v[ids, 48:51] = 0
        mujoco_warp.forward(self._solver.mjw_model, self._solver.mjw_data)

    def native_qpos(self): return super().native_qpos()[:, :46]
    def get_ground_contact_forces(self, obj_id): return super().get_ground_contact_forces(obj_id)[:, :18]
    def ball_position(self): return wp.to_torch(self._sim_state.raw_state.joint_q).view(self.get_num_envs(), -1)[:, 46:49]
