"""Small native-MuJoCo evaluation backend for the same experimental task contract."""
import mujoco
import numpy as np
import torch

from .rig import DelayedTorqueControl


class NativeDummyEngine:
    def __init__(self, rig, num_envs, device="cpu", control_factory=DelayedTorqueControl):
        if device != "cpu":
            raise ValueError("Native evaluation runs on CPU")
        self.rig = rig
        self.datas = [mujoco.MjData(rig.model) for _ in range(num_envs)]
        self.policy_control = control_factory(rig, num_envs, device)
        self.native_ctrl = self.policy_control.ctrl

    def get_timestep(self):
        return self.rig.model.opt.timestep * self.rig.decimation

    def reset_envs(self, ids, qpos, qvel):
        for i, q, v in zip(ids.tolist(), np.asarray(qpos), np.asarray(qvel)):
            data = self.datas[i]
            mujoco.mj_resetData(self.rig.model, data)
            data.qpos[:], data.qvel[:] = q, v
            mujoco.mj_forward(self.rig.model, data)
        self.policy_control.reset(ids)

    def set_cmd(self, obj_id, command):
        self.policy_control.apply(command)

    def set_external_force(self, body, forces):
        """World-frame force at body COM, held for one complete control interval."""
        forces = torch.as_tensor(forces, dtype=torch.float32, device="cpu")
        if not 0 < body < self.rig.model.nbody or forces.shape != (len(self.datas), 3) or not torch.isfinite(forces).all():
            raise ValueError("Expected a valid moving body and one finite force per world")
        for data, force in zip(self.datas, forces.numpy()):
            data.xfrc_applied[body, :3] = force

    def step(self):
        if hasattr(self.policy_control, "physics_control"):
            for _ in range(self.rig.decimation):
                self.policy_control.physics_control(self._array("qpos"), self._array("qvel"))
                for i, data in enumerate(self.datas):
                    data.ctrl[:] = self.native_ctrl[i].numpy()
                    mujoco.mj_step(self.rig.model, data)
                    self._after_physics_step(i, data)
            for data in self.datas:
                mujoco.mj_forward(self.rig.model, data)
                data.xfrc_applied[:] = 0
            return
        for i, data in enumerate(self.datas):
            data.ctrl[:] = self.native_ctrl[i].numpy()
            for _ in range(self.rig.decimation):
                mujoco.mj_step(self.rig.model, data)
                self._after_physics_step(i, data)
            mujoco.mj_forward(self.rig.model, data)
            data.xfrc_applied[:] = 0

    def _after_physics_step(self, world, data):
        pass

    def _array(self, field):
        return torch.tensor(np.stack([getattr(d, field) for d in self.datas]), dtype=torch.float32)

    def native_qpos(self):
        return self._array("qpos")

    def get_root_pos(self, obj_id):
        return self._array("qpos")[:, :3]

    def get_root_rot(self, obj_id):
        return self._array("qpos")[:, [4, 5, 6, 3]]

    def get_root_vel(self, obj_id):
        return self._array("qvel")[:, :3]

    def get_root_ang_vel(self, obj_id):
        return torch.tensor(np.stack([d.xmat[1].reshape(3, 3) @ d.qvel[3:6] for d in self.datas]), dtype=torch.float32)

    def get_dof_vel(self, obj_id):
        return self._array("qvel")[:, 6:]

    def get_body_pos(self, obj_id):
        return self._array("xpos")[:, 1:]

    def get_body_rot(self, obj_id):
        return self._array("xquat")[:, 1:, [1, 2, 3, 0]]

    def get_ground_contact_forces(self, obj_id):
        model = self.rig.model
        forces = np.zeros((len(self.datas), model.nbody - 1, 3))
        wrench = np.zeros(6)
        for i, data in enumerate(self.datas):
            for k, contact in enumerate(data.contact[:data.ncon]):
                if 0 not in contact.geom:
                    continue
                mujoco.mj_contactForce(model, data, k, wrench)
                world_force = contact.frame.reshape(3, 3).T @ wrench[:3]
                for geom, sign in zip(contact.geom, (-1, 1)):
                    body = model.geom_bodyid[geom]
                    if body:
                        forces[i, body - 1] += sign * world_force
        return torch.tensor(forces, dtype=torch.float32)
