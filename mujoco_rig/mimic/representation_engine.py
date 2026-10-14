"""Isolated native/GPU engines for the 33-motor representation comparison."""
import mujoco
import numpy as np
import torch
import warp as wp

from .diagnose_stand_foundations import DiagnosticNativeEngine
from .newton_adapter import DummyNewtonEngine
from .rig import DelayedTorqueControl
from .target_control33 import DelayedTargetControl33


@wp.kernel
def target_feedback(q: wp.array(dtype=float), v: wp.array(dtype=float),
                    targets: wp.array2d(dtype=float), indices: wp.array(dtype=wp.int32),
                    qi: wp.array(dtype=wp.int32), vi: wp.array(dtype=wp.int32),
                    scale: wp.array(dtype=float), ctrl: wp.array(dtype=float),
                    nq: int, nv: int, nu: int, count: int, gain: float, damping: float):
    world, motor = wp.tid()
    error = targets[world, motor]-q[world*nq+qi[motor]]
    speed_error = targets[world, motor+count]-v[world*nv+vi[motor]]
    value = gain*error + gain*damping*speed_error
    ctrl[world*nu+indices[motor]] = wp.clamp(value, -1.0, 1.0)*scale[motor]*targets[world, 2*count]


@wp.kernel
def record_motor_step(q: wp.array(dtype=float), v: wp.array(dtype=float), ctrl: wp.array(dtype=float),
                      indices: wp.array(dtype=wp.int32), qi: wp.array(dtype=wp.int32), vi: wp.array(dtype=wp.int32),
                      recorded_q: wp.array3d(dtype=float), recorded_v: wp.array3d(dtype=float),
                      recorded_torque: wp.array3d(dtype=float), slot: int, nq: int, nv: int, nu: int):
    world, motor = wp.tid()
    recorded_q[slot, world, motor] = q[world*nq+qi[motor]]
    recorded_v[slot, world, motor] = v[world*nv+vi[motor]]
    recorded_torque[slot, world, motor] = ctrl[world*nu+indices[motor]]


class RepresentationGPU(DummyNewtonEngine):
    def __init__(self, rig, num_envs, device='cuda:0', *, control_factory=DelayedTorqueControl, trace=False):
        self.record_substeps = trace
        self._record_slot = 0
        super().__init__(rig, num_envs, device, control_factory=control_factory)

    def _build_controls(self):
        super()._build_controls()
        c, model = self.policy_control, self.rig.model
        joints = model.actuator_trnid[self.rig.actuators, 0]
        self._motor_indices = torch.as_tensor(self.rig.actuators, dtype=torch.int32, device=self._device)
        self._motor_qi = torch.as_tensor(model.jnt_qposadr[joints], dtype=torch.int32, device=self._device)
        self._motor_vi = torch.as_tensor(model.jnt_dofadr[joints], dtype=torch.int32, device=self._device)
        self._motor_arrays = [wp.from_torch(t) for t in (self._motor_indices, self._motor_qi, self._motor_vi)]
        if isinstance(c, DelayedTargetControl33):
            self._target_arrays = [wp.from_torch(c.active), *self._motor_arrays,
                wp.from_torch(c.scale), wp.from_torch(self.native_ctrl.flatten())]
        if self.record_substeps:
            shape = (self.rig.decimation, self.get_num_envs(), len(self.rig.actuators))
            self.substep_q = torch.empty(shape, device=self._device)
            self.substep_v = torch.empty(shape, device=self._device)
            self.substep_torque = torch.empty(shape, device=self._device)
            self._record_arrays = [wp.from_torch(t) for t in (self.substep_q, self.substep_v, self.substep_torque)]

    def _pre_sim_step(self, raw_state, control):
        super()._pre_sim_step(raw_state, control)
        c, model = self.policy_control, self.rig.model
        if isinstance(c, DelayedTargetControl33):
            wp.launch(target_feedback, dim=(self.get_num_envs(), c.count),
                inputs=[raw_state.joint_q, raw_state.joint_qd, *self._target_arrays,
                    model.nq, model.nv, model.nu, c.count, c.gain, c.damping_time])
        if self.record_substeps:
            wp.launch(record_motor_step, dim=(self.get_num_envs(), len(self.rig.actuators)),
                inputs=[raw_state.joint_q, raw_state.joint_qd, wp.from_torch(self.native_ctrl.flatten()),
                    *self._motor_arrays, *self._record_arrays, self._record_slot,
                    model.nq, model.nv, model.nu])
            self._record_slot = (self._record_slot+1) % self.rig.decimation


class RepresentationNative(DiagnosticNativeEngine):
    def __init__(self, rig, num_envs, device='cpu', *, control_factory=DelayedTorqueControl, trace=False):
        super().__init__(rig, num_envs, device, control_factory=control_factory)
        self.record_substeps = trace
        joints = rig.model.actuator_trnid[rig.actuators, 0]
        self._qi, self._vi = rig.model.jnt_qposadr[joints], rig.model.jnt_dofadr[joints]
        if trace:
            shape = (rig.decimation, num_envs, len(rig.actuators))
            self.substep_q, self.substep_v, self.substep_torque = (torch.empty(shape) for _ in range(3))

    def step(self):
        bad = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC)
        for substep in range(self.rig.decimation):
            q, v = self._array('qpos'), self._array('qvel')
            if isinstance(self.policy_control, DelayedTargetControl33):
                self.policy_control.physics_control(q, v)
            if self.record_substeps:
                self.substep_q[substep] = q[:, self._qi]
                self.substep_v[substep] = v[:, self._vi]
                self.substep_torque[substep] = self.native_ctrl[:, self.rig.actuators]
            for case, data in enumerate(self.datas):
                if not self.active[case] or self.solver_warning[case]:
                    continue
                data.ctrl[:] = self.native_ctrl[case].numpy()
                mujoco.mj_step(self.rig.model, data)
                self.solver_warning[case] |= any(data.warning[item].number for item in bad)
        for case, data in enumerate(self.datas):
            if self.active[case]:
                self._ground_force_cache[case] = self._read_ground_forces(data)
                mujoco.mj_forward(self.rig.model, data)
                data.xfrc_applied[:] = 0
