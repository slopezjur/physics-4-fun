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
        # Keep hit state on the same device as the simulation.  Newton exposes contact
        # shape ids after each control interval; the sticky flag gives BallTask and the
        # parity probe the same observable that BallNativeEngine already provides.
        self.hit = torch.zeros(num_envs, dtype=torch.bool, device=device)
        super().__init__(load_ball_rig(rig), num_envs, device, control_factory)
        labels = self._sim_model.shape_label
        total_shapes = self._sim_model.shape_count
        if total_shapes % self.get_num_envs():
            raise ValueError("Newton projectile shape buffer is not environment-strided")
        self._shape_stride = total_shapes // self.get_num_envs()
        ball_shapes = [i for i, label in enumerate(labels) if label.endswith("/g_ball")]
        floor_shapes = [i for i, label in enumerate(labels) if label.endswith("/floor")]
        if not ball_shapes or any(i % self._shape_stride != ball_shapes[0] % self._shape_stride
                                  for i in ball_shapes) or not floor_shapes:
            raise ValueError("Projectile world must contain one consistent g_ball/floor shape per world")
        self._ball_shape = ball_shapes[0] % self._shape_stride
        self._floor_shape = floor_shapes[0] % self._shape_stride
        if self._shape_stride <= self._ball_shape:
            raise ValueError("Invalid Newton projectile shape mapping")

    def _record_ball_contacts(self):
        """Mark worlds with a current ball/character contact in Newton's contact buffer."""
        count = int(wp.to_torch(self._contacts.rigid_contact_count).reshape(-1)[0].item())
        if count <= 0:
            return
        shape0 = wp.to_torch(self._contacts.rigid_contact_shape0)[:count]
        shape1 = wp.to_torch(self._contacts.rigid_contact_shape1)[:count]
        local0 = shape0.remainder(self._shape_stride)
        local1 = shape1.remainder(self._shape_stride)
        ball0 = local0 == self._ball_shape
        ball1 = local1 == self._ball_shape
        has_ball = ball0 | ball1
        if not bool(has_ball.any()):
            return
        other = torch.where(ball0, local1, local0)
        # The ball may also touch the floor.  Only a ball/character pair is a gameplay hit.
        character = (other != self._ball_shape) & (other != self._floor_shape)
        valid = has_ball & character
        if not bool(valid.any()):
            return
        world0 = torch.div(shape0, self._shape_stride, rounding_mode="floor")
        world1 = torch.div(shape1, self._shape_stride, rounding_mode="floor")
        worlds = torch.where(ball0, world0, world1)[valid].long()
        worlds = worlds[(worlds >= 0) & (worlds < self.hit.numel())]
        if worlds.numel():
            self.hit[worlds] = True

    def step(self):
        super().step()
        self._record_ball_contacts()

    def _apply_start_xform(self):
        # This MJCF contains two articulations in one builder. Its authored initial
        # transforms are correct; the complete native reset follows initialization.
        self._sim_state.eval_fk()

    def reset_envs(self, ids, qpos, qvel):
        q = torch.tensor(self.rig.rest, device=self._device, dtype=torch.float32).repeat(len(ids), 1)
        v = torch.zeros((len(ids), self.rig.model.nv), device=self._device)
        q[:, :46], v[:, :45] = qpos, qvel
        super().reset_envs(ids, q, v)
        self.hit[ids] = False

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
