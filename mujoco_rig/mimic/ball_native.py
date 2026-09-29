"""Native projectile world exposing only the original dummy to the Stand policy."""
import mujoco
import numpy as np
import torch

from .ball import load_ball_rig
from .native_engine import NativeDummyEngine
from .target_control import DelayedTargetControl


class BallNativeEngine(NativeDummyEngine):
    def __init__(self, rig, num_envs, device="cpu", control_factory=DelayedTargetControl, ball_path=None):
        self.character = rig
        super().__init__(load_ball_rig(rig, ball_path), num_envs, device, control_factory)
        self.ball_body = self.rig.model.body("ball").id
        self.hit = torch.zeros(num_envs, dtype=torch.bool)
        self.hit_impulse = torch.zeros((num_envs, 3), dtype=torch.float64)
        self.hit_normal_impulse = torch.zeros(num_envs, dtype=torch.float64)
        self.hit_bodies = [set() for _ in range(num_envs)]

    def reset_envs(self, ids, qpos, qvel):
        q = np.tile(self.rig.rest, (len(ids), 1))
        v = np.zeros((len(ids), self.rig.model.nv))
        q[:, :46], v[:, :45] = np.asarray(qpos), np.asarray(qvel)
        super().reset_envs(ids, q, v)
        self.hit[ids] = False
        self.hit_impulse[ids] = 0.
        self.hit_normal_impulse[ids] = 0.
        for index in ids.tolist():
            self.hit_bodies[index].clear()

    def launch_ball(self, ids, position, velocity):
        for i, p, v in zip(ids.tolist(), np.asarray(position), np.asarray(velocity)):
            data = self.datas[i]
            data.qpos[46:49], data.qpos[49:53] = p, [1, 0, 0, 0]
            data.qvel[45:48], data.qvel[48:51] = v, 0
            mujoco.mj_forward(self.rig.model, data)

    def _after_physics_step(self, world, data):
        for index, contact in enumerate(data.contact[:data.ncon]):
            a, b = self.rig.model.geom_bodyid[contact.geom]
            if (a == self.ball_body and 0 < b < self.ball_body) or (b == self.ball_body and 0 < a < self.ball_body):
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.rig.model, data, index, wrench)
                if wrench[0] > 0:
                    self.hit[world] = True
                    sign = 1. if a == self.ball_body else -1.
                    force = sign * (contact.frame.reshape(3, 3).T @ wrench[:3])
                    self.hit_impulse[world] += torch.from_numpy(force * self.rig.model.opt.timestep)
                    self.hit_normal_impulse[world] += wrench[0] * self.rig.model.opt.timestep
                    self.hit_bodies[world].add(int(b if a == self.ball_body else a))

    def native_qpos(self): return super().native_qpos()[:, :46]
    def get_dof_vel(self, obj_id): return super().get_dof_vel(obj_id)[:, :39]
    def get_body_pos(self, obj_id): return super().get_body_pos(obj_id)[:, :18]
    def get_body_rot(self, obj_id): return super().get_body_rot(obj_id)[:, :18]
    def get_ground_contact_forces(self, obj_id): return super().get_ground_contact_forces(obj_id)[:, :18]
    def ball_position(self): return self._array("qpos")[:, 46:49]
