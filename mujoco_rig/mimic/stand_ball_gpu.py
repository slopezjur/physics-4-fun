"""33-target projectile physics; import only after activating pinned MimicKit."""
import mujoco_warp
import numpy as np
import torch
import warp as wp

from .representation_engine import RepresentationGPU
from .rig import Rig
from .target_control33 import DelayedTargetControl33


@wp.kernel
def accumulate_impulses(count: wp.array[int], geometries: wp.array[wp.vec2i],
                        worlds: wp.array[int], frames: wp.array[wp.mat33],
                        bodies: wp.array[int], forces: wp.array[wp.spatial_vector],
                        ball_body: int, dt: float, normal: wp.array[float],
                        vector: wp.array[wp.vec3]):
    contact = wp.tid()
    if contact >= count[0]:
        return
    pair = geometries[contact]
    if pair[0] < 0 or pair[1] < 0:
        return
    a, b = bodies[pair[0]], bodies[pair[1]]
    if not ((a == ball_body and b > 0 and b < ball_body)
            or (b == ball_body and a > 0 and a < ball_body)):
        return
    wrench = forces[contact]
    if wrench[0] <= 0.0:
        return
    sign = float(1.0)
    if b == ball_body:
        sign = -1.0
    force = wp.transpose(frames[contact]) * wp.vec3(wrench[0], wrench[1], wrench[2])
    wp.atomic_add(normal, worlds[contact], wrench[0] * dt)
    wp.atomic_add(vector, worlds[contact], sign * dt * force)


class StandBallGPU(RepresentationGPU):
    def __init__(self, character, num_envs, device='cuda:0', *, ball_path, trace=False):
        rig = Rig.load(ball_path, policy_exclude=())
        if (character.model.nq, character.model.nv, character.model.nbody) != (46, 45, 19):
            raise ValueError('Unexpected character topology')
        joint = rig.model.joint('ball_free').id
        if (rig.model.nq, rig.model.nv, rig.model.nbody, len(rig.actuators)) != (53, 51, 20, 33) \
                or rig.model.jnt_qposadr[joint] != 46 or rig.model.jnt_dofadr[joint] != 45:
            raise ValueError('Projectile must be appended after the character')
        self.ball_body = rig.model.body('ball').id
        super().__init__(rig, num_envs, device, control_factory=DelayedTargetControl33, trace=trace)

    def _apply_start_xform(self):
        # Two authored articulations; the complete native reset follows initialization.
        self._sim_state.eval_fk()

    def _build_solver(self):
        super()._build_solver()
        data = self._solver.mjw_data
        self._contact_ids = wp.array(np.arange(data.naconmax, dtype=np.int32), device=self._device)
        self._contact_wrenches = wp.zeros(data.naconmax, dtype=wp.spatial_vector, device=self._device)
        self._geom_bodies = wp.array(self.rig.model.geom_bodyid.astype(np.int32), device=self._device)
        self._normal_impulse = wp.zeros(self.get_num_envs(), device=self._device)
        self._contact_impulse = wp.zeros(self.get_num_envs(), dtype=wp.vec3, device=self._device)
        self.normal_impulse = wp.to_torch(self._normal_impulse)
        self.contact_impulse = wp.to_torch(self._contact_impulse)

    def _simulate(self):
        # Pinned NewtonEngine order, with read-only solved contact sampling per substep.
        state0, state1 = self._sim_state.raw_state, self._state_swap_buffer
        control = self._controls.control
        self._normal_impulse.zero_()
        self._contact_impulse.zero_()
        self._sim_state.pre_step_update()
        for _ in range(self._sim_steps):
            self._pre_sim_step(state0, control)
            self._solver.step(state0, state1, control, self._contacts, self._sim_timestep)
            data = self._solver.mjw_data
            mujoco_warp.contact_force(self._solver.mjw_model, data, self._contact_ids,
                                      False, self._contact_wrenches)
            wp.launch(accumulate_impulses, dim=data.naconmax,
                inputs=[data.nacon, data.contact.geom, data.contact.worldid, data.contact.frame,
                        self._geom_bodies, self._contact_wrenches, self.ball_body,
                        self._sim_timestep, self._normal_impulse, self._contact_impulse], device=self._device)
            state0, state1 = state1, state0
        if self._sim_steps % 2:
            self._sim_state.copy(self._state_swap_buffer)
        self._sim_state.post_step_update()

        def clear_body_forces():
            self._sim_state.clear_forces()
            self._has_body_forces.zero_()

        wp.capture_if(self._has_body_forces, on_true=clear_body_forces)

    def reset_envs(self, ids, qpos, qvel):
        ids = torch.as_tensor(ids, device=self._device, dtype=torch.long)
        q = torch.as_tensor(self.rig.rest, device=self._device, dtype=torch.float32).repeat(len(ids), 1)
        v = torch.zeros((len(ids), self.rig.model.nv), device=self._device)
        q[:, :46], v[:, :45] = qpos, qvel
        super().reset_envs(ids, q, v)
        self.normal_impulse[ids] = 0.
        self.contact_impulse[ids] = 0.

    def launch_ball(self, ids, position, velocity):
        ids = torch.as_tensor(ids, device=self._device, dtype=torch.long)
        position = torch.as_tensor(position, device=self._device, dtype=torch.float32).expand(len(ids), 3)
        velocity = torch.as_tensor(velocity, device=self._device, dtype=torch.float32).expand(len(ids), 3)
        if not torch.isfinite(position).all() or not torch.isfinite(velocity).all():
            raise ValueError('Projectile launch state must be finite')
        state = self._sim_state.raw_state
        for raw in (state, self._state_swap_buffer):
            q = wp.to_torch(raw.joint_q).view(self.get_num_envs(), -1)
            v = wp.to_torch(raw.joint_qd).view(self.get_num_envs(), -1)
            q[ids, 46:49] = position
            q[ids, 49:53] = torch.tensor([0., 0., 0., 1.], device=self._device)
            v[ids, 45:48], v[ids, 48:51] = velocity, 0.
        self._sim_state.eval_fk()
        # Copy only selected worlds; peer state and delayed controls must survive launch.
        for field in ('body_q', 'body_qd'):
            source = wp.to_torch(getattr(state, field)).view(self.get_num_envs(), -1)
            wp.to_torch(getattr(self._state_swap_buffer, field)).view(self.get_num_envs(), -1)[ids] = source[ids]
        data = self._solver.mjw_data
        q, v = wp.to_torch(data.qpos), wp.to_torch(data.qvel)
        q[ids, 46:49] = position
        q[ids, 49:53] = torch.tensor([1., 0., 0., 0.], device=self._device)
        v[ids, 45:48], v[ids, 48:51] = velocity, 0.
        mujoco_warp.forward(self._solver.mjw_model, data)

    def ball_state(self):
        q = wp.to_torch(self._sim_state.raw_state.joint_q).view(self.get_num_envs(), -1)[:, 46:].clone()
        q[:, 3:7] = q[:, [6, 3, 4, 5]]
        v = wp.to_torch(self._sim_state.raw_state.joint_qd).view(self.get_num_envs(), -1)[:, 45:].clone()
        return q, v  # Ball angular velocity is in Newton's world coordinates.

    def native_qpos(self): return super().native_qpos()[:, :46]
    def get_dof_vel(self, obj_id): return super().get_dof_vel(obj_id)[:, :39]
    def get_body_pos(self, obj_id): return super().get_body_pos(obj_id)[:, :18]
    def get_body_rot(self, obj_id): return super().get_body_rot(obj_id)[:, :18]
    def get_ground_contact_forces(self, obj_id): return super().get_ground_contact_forces(obj_id)[:, :18]
