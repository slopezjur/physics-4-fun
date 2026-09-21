"""MimicKit Newton engine configured to preserve the game's MuJoCo plant.

Call runtime.activate before importing this module. StandTask supplies the matching
30-channel action contract; stock humanoid tasks use different action assumptions.
"""
from __future__ import annotations

import mujoco
import mujoco_warp
import newton
import numpy as np
import torch
import warp as wp

from engines.engine import ObjType
from engines.newton_engine import Controls, NewtonEngine

from .rig import DelayedTorqueControl, Rig
from .model_contract import model_checks
from .target_control import DelayedTargetControl


@wp.kernel
def target_pd(q: wp.array(dtype=float), v: wp.array(dtype=float),
              targets: wp.array2d(dtype=float), actuators: wp.array(dtype=wp.int32),
              qi: wp.array(dtype=wp.int32), vi: wp.array(dtype=wp.int32),
              limits: wp.array(dtype=float), ctrl: wp.array(dtype=float),
              nq: int, nv: int, nu: int, gain: float, damping: float):
    world, motor = wp.tid()
    error = targets[world, motor] - q[world * nq + qi[motor]]
    velocity_error = targets[world, motor + 30] - v[world * nv + vi[motor]]
    normalized = gain * error + (gain * damping) * velocity_error
    ctrl[world * nu + actuators[motor]] = wp.clamp(normalized, -1.0, 1.0) * limits[motor] * targets[world, 60]


class DummyNewtonEngine(NewtonEngine):
    def __init__(self, rig: Rig, num_envs: int, device="cuda:0", control_factory=DelayedTorqueControl):
        self.rig = rig
        super().__init__({"env_spacing": 5, "sim_freq": 240, "control_freq": 60,
                          "control_mode": "torque"}, num_envs, device, False)
        self._sim_timestep = rig.model.opt.timestep
        self._sim_steps = rig.decimation
        self._timestep = self._sim_timestep * self._sim_steps
        self.policy_control = control_factory(rig, num_envs, device)
        for _ in range(num_envs):
            env_id = self.create_env()
            self.create_obj(env_id, ObjType.articulated, str(rig.path), "dummy",
                            start_pos=rig.rest[:3], start_rot=rig.rest[[4, 5, 6, 3]])
        self.initialize_sim()
        self.reset()

    def _build_ground(self):
        # The original MJCF already owns the floor, friction and collision settings.
        pass

    def _create_obj_builder(self, asset_file, fix_root, is_visual, enable_self_collisions):
        if fix_root or is_visual or not enable_self_collisions:
            raise ValueError("This compatibility engine only accepts the unmodified dynamic dummy")
        if asset_file not in self._builder_cache:
            builder = self._create_model_builder()
            builder.add_mjcf(asset_file, ignore_inertial_definitions=False,
                             collapse_fixed_joints=False, enable_self_collisions=True,
                             convert_3d_hinge_to_ball_joints=False, ctrl_direct=True)
            self._builder_cache[asset_file] = builder
        return self._builder_cache[asset_file]

    def _build_controls(self):
        # ctrl_direct retains native motor/neck actuators and passive spring/damping.
        self._controls = Controls(self._sim_model, self.get_num_envs())
        self.native_ctrl = wp.to_torch(self._controls.control.mujoco.ctrl).view(self.get_num_envs(), -1)
        if isinstance(self.policy_control, DelayedTargetControl):
            c = self.policy_control
            self._pd_arrays = [wp.from_torch(c.active), wp.from_torch(c.indices.to(torch.int32)),
                               wp.from_torch(c.q_indices.to(torch.int32)), wp.from_torch(c.v_indices.to(torch.int32)),
                               wp.from_torch(c.scale), wp.from_torch(self.native_ctrl.flatten())]

    def _pre_sim_step(self, raw_state, control):
        super()._pre_sim_step(raw_state, control)
        if isinstance(self.policy_control, DelayedTargetControl):
            c = self.policy_control
            wp.launch(target_pd, dim=(self.get_num_envs(), 30),
                      inputs=[raw_state.joint_q, raw_state.joint_qd, *self._pd_arrays,
                              self.rig.model.nq, self.rig.model.nv, self.rig.model.nu, c.gain, c.damping_time])

    def _build_solver(self):
        # Import source MJCF solver/integrator options instead of MimicKit overrides.
        source = self.rig.model
        self._solver = newton.solvers.SolverMuJoCo(
            self._sim_model, njmax=450, nconmax=150,
            enable_multiccd=not bool(source.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)))
        imported = self._solver.mj_model
        if imported.njnt != source.njnt or not np.array_equal(imported.jnt_bodyid, source.jnt_bodyid):
            raise ValueError("Newton changed joint topology; native control mapping is invalid")
        # Newton translates limit time constants to direct stiffness/damping and uses
        # a default CPU timestep. Keep the source representation on both sides.
        imported.opt.timestep = source.opt.timestep
        imported.jnt_solref[:] = source.jnt_solref
        self._solver.mjw_model.opt.timestep.fill_(source.opt.timestep)
        solref = self._solver.mjw_model.jnt_solref
        solref.assign(np.broadcast_to(source.jnt_solref, solref.numpy().shape).astype(np.float32).copy())
        # Preserve explicitly authored direct contact stiffness (the projectile),
        # which Newton otherwise converts through its generic material parameters.
        imported.geom_solref[:] = source.geom_solref
        geom_solref = self._solver.mjw_model.geom_solref
        geom_solref.assign(np.broadcast_to(source.geom_solref, geom_solref.numpy().shape).astype(np.float32).copy())
        failed = [name for name, matches in model_checks(source, imported).items() if not matches]
        if failed:
            raise ValueError(f"Newton changed the source plant: {', '.join(failed)}")

    def _build_ground_contact_sensor(self):
        self._ground_contact_sensor = newton.sensors.SensorContact(
            self._sim_model, sensing_obj_bodies="*", counterpart_shapes="*floor", include_total=True)
        forces = wp.to_torch(self._ground_contact_sensor.net_force).view(self.get_num_envs(), -1, 2, 3)
        self._contact_forces = [forces[:, :, 0, :]]
        self._ground_contact_forces = [forces[:, :, 1, :]]

    def _build_dof_force_tensors(self):
        self.native_actuator_force = wp.to_torch(self._solver.mjw_data.actuator_force)

    def get_dof_forces(self, obj_id):
        self._check_obj(obj_id)
        forces = torch.zeros_like(self.get_dof_vel(obj_id))
        joints = self.rig.model.actuator_trnid[:, 0]
        indices = torch.as_tensor(self.rig.model.jnt_dofadr[joints] - 6, device=self._device)
        forces[:, indices] = self.native_actuator_force
        return forces

    def set_cmd(self, obj_id, cmd):
        self._check_obj(obj_id)
        self.native_ctrl.copy_(self.policy_control.apply(cmd))

    def set_external_force(self, body, forces):
        """Match native MuJoCo's world-frame COM force, cleared after each interval."""
        forces = torch.as_tensor(forces, dtype=torch.float32, device=self._device)
        if not 0 < body < self.rig.model.nbody or forces.shape != (self.get_num_envs(), 3) or not torch.isfinite(forces).all():
            raise ValueError("Expected a valid moving body and one finite force per world")
        self.set_body_forces(None, 0, body - 1, forces)

    @staticmethod
    def _check_obj(obj_id):
        if obj_id != 0:
            raise ValueError("Only the dummy object is available in this compatibility engine")

    def reset(self, qpos=None):
        q = self.rig.rest.copy() if qpos is None else np.asarray(qpos).copy()
        if q.shape != (self.rig.model.nq,) or not np.isfinite(q).all():
            raise ValueError("Expected one finite native MuJoCo pose")
        native_q = q.copy()
        q[3:7] = q[[4, 5, 6, 3]]
        for joint in np.flatnonzero(self.rig.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[1:]:
            start = self.rig.model.jnt_qposadr[joint] + 3
            q[start:start + 4] = q[[start + 1, start + 2, start + 3, start]]
        state = self._sim_state.raw_state
        wp.to_torch(state.joint_q).view(self.get_num_envs(), -1).copy_(
            torch.as_tensor(q, dtype=torch.float32, device=self._device))
        state.joint_qd.zero_()
        state.clear_forces()
        self._sim_state.eval_fk()
        for field in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f"):
            wp.copy(getattr(self._state_swap_buffer, field), getattr(state, field))
        self._sim_state.clear_forces()
        self._has_body_forces.zero_()
        self.policy_control.reset()
        self.native_ctrl.zero_()
        # Reset solver history as well as exposed Newton state. Cached MuJoCo body
        # transforms otherwise survive resets and can affect the first force update.
        mujoco_warp.reset_data(self._solver.mjw_model, self._solver.mjw_data)
        wp.to_torch(self._solver.mjw_data.qpos).copy_(
            torch.as_tensor(native_q, dtype=torch.float32, device=self._device))
        mujoco_warp.forward(self._solver.mjw_model, self._solver.mjw_data)
        self._solver.update_contacts(self._contacts, self._sim_state.raw_state)
        self._update_contact_sensors()
        self._sim_step_count = 0

    def native_qpos(self):
        q = wp.to_torch(self._sim_state.raw_state.joint_q).view(self.get_num_envs(), -1).clone()
        q[:, 3:7] = q[:, [6, 3, 4, 5]]
        return q

    def reset_envs(self, env_ids, qpos, qvel):
        """Reset selected worlds from native MuJoCo coordinates without clearing peers."""
        from util.torch_util import quat_rotate
        ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        if ids.numel() == 0:
            return
        q = torch.as_tensor(qpos, dtype=torch.float32, device=self._device).clone()
        v = torch.as_tensor(qvel, dtype=torch.float32, device=self._device).clone()
        if q.shape != (len(ids), self.rig.model.nq) or v.shape != (len(ids), self.rig.model.nv):
            raise ValueError("Invalid batched native reset state")
        if not torch.isfinite(q).all() or not torch.isfinite(v).all():
            raise ValueError("Non-finite reset state")
        newton_q = q.clone()
        newton_q[:, 3:7] = q[:, [4, 5, 6, 3]]
        for joint in np.flatnonzero(self.rig.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[1:]:
            start = self.rig.model.jnt_qposadr[joint] + 3
            newton_q[:, start:start + 4] = q[:, [start + 1, start + 2, start + 3, start]]
        newton_v = v.clone()
        newton_v[:, 3:6] = quat_rotate(newton_q[:, 3:7], v[:, 3:6])
        state = self._sim_state.raw_state
        for field, value in (("joint_q", newton_q), ("joint_qd", newton_v)):
            wp.to_torch(getattr(state, field)).view(self.get_num_envs(), -1)[ids] = value
        self._sim_state.eval_fk()
        for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
            source = wp.to_torch(getattr(state, field)).view(self.get_num_envs(), -1)
            wp.to_torch(getattr(self._state_swap_buffer, field)).view(self.get_num_envs(), -1)[ids] = source[ids]
        for raw in (state, self._state_swap_buffer):
            wp.to_torch(raw.body_f).view(self.get_num_envs(), -1)[ids] = 0
        wp.to_torch(self._sim_state._wp_body_force).view(self.get_num_envs(), -1)[ids] = 0
        self.policy_control.reset(ids)
        self.native_ctrl[ids] = 0
        mask = torch.zeros(self.get_num_envs(), device=self._device, dtype=torch.bool)
        mask[ids] = True
        mujoco_warp.reset_data(self._solver.mjw_model, self._solver.mjw_data, reset=wp.from_torch(mask))
        wp.to_torch(self._solver.mjw_data.qpos)[ids] = q
        wp.to_torch(self._solver.mjw_data.qvel)[ids] = v
        mujoco_warp.forward(self._solver.mjw_model, self._solver.mjw_data)
        self._solver.update_contacts(self._contacts, state)
        self._update_contact_sensors()
