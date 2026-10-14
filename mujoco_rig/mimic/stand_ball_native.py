"""Isolated native ball assessment with the admitted 33-target controller."""
from functools import partial
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .representation_engine import RepresentationNative
from .representation_task import RepresentationMixin
from .rig import Rig
from .target_control33 import DelayedTargetControl33
from .torque_stand_task import TorqueStandTask


def write_world(character, template, destination, mass):
    """Copy the Arena projectile, retaining the character's authored physics."""
    if not np.isfinite(mass) or not .1 <= mass <= 10.:
        raise ValueError('Ball mass must be within 0.1–10 kg')
    plain, world = ET.parse(character).getroot(), ET.parse(template).getroot()
    ball = world.find("./worldbody/body[@name='ball']")
    if ball is None:
        raise ValueError('Missing appended projectile')
    world.find('worldbody').remove(ball)
    for root in (plain, world):
        keyframe = root.find('keyframe')
        if keyframe is not None:
            root.remove(keyframe)
        root.attrib.pop('model', None)
        for node in root.iter():
            node.text = node.tail = None
    if ET.tostring(plain) != ET.tostring(world):
        raise ValueError('Ball template changes character physics')
    # Start from the original character tree; append a single parked ball.
    tree = ET.parse(character)
    root = tree.getroot()
    ball.set('pos', '40 40 0.09')
    geom = ball.find("geom[@name='g_ball']")
    if geom is None or ball.find('inertial') is not None:
        raise ValueError('Expected one geom-derived projectile inertia')
    geom.set('mass', str(float(mass)))
    root.find('worldbody').append(ball)
    for key in root.findall('./keyframe/key'):
        key.set('qpos', key.attrib['qpos']+' 40 40 0.09 1 0 0 0')
        if 'qvel' in key.attrib:
            key.set('qvel', key.attrib['qvel']+' 0 0 0 0 0 0')
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    tree.write(destination, encoding='utf-8', xml_declaration=True)


class StandBallNative(RepresentationNative):
    def __init__(self, character, num_envs, device='cpu', *, ball_path):
        rig = Rig.load(ball_path, policy_exclude=())
        if (character.model.nq, character.model.nv, character.model.nbody) != (46, 45, 19):
            raise ValueError('Unexpected character topology')
        joint = rig.model.joint('ball_free').id
        if (rig.model.nq, rig.model.nv, rig.model.nbody, len(rig.actuators)) != (53, 51, 20, 33) \
                or rig.model.jnt_qposadr[joint] != 46 or rig.model.jnt_dofadr[joint] != 45:
            raise ValueError('Projectile must be appended after the character')
        super().__init__(rig, num_envs, device, control_factory=DelayedTargetControl33, trace=True)
        self.ball_body = rig.model.body('ball').id
        self.contact_impulse = np.zeros((num_envs, 3))
        self.normal_impulse = np.zeros(num_envs)

    def reset_envs(self, ids, qpos, qvel):
        q = np.tile(self.rig.rest, (len(ids), 1))
        v = np.zeros((len(ids), self.rig.model.nv))
        q[:, :46], v[:, :45] = np.asarray(qpos), np.asarray(qvel)
        super().reset_envs(ids, q, v)

    def launch_ball(self, ids, position, velocity):
        position = np.broadcast_to(np.asarray(position), (len(ids), 3))
        velocity = np.broadcast_to(np.asarray(velocity), (len(ids), 3))
        if not np.isfinite(position).all() or not np.isfinite(velocity).all():
            raise ValueError('Projectile launch state must be finite')
        for index, p, v in zip(ids.tolist(), position, velocity):
            data = self.datas[index]
            data.qpos[46:53] = [*p, 1., 0., 0., 0.]
            data.qvel[45:51] = [*v, 0., 0., 0.]
            mujoco.mj_forward(self.rig.model, data)

    def _record_contacts(self, case, data):
        for index, contact in enumerate(data.contact[:data.ncon]):
            a, b = self.rig.model.geom_bodyid[contact.geom]
            if not ((a == self.ball_body and 0 < b < self.ball_body)
                    or (b == self.ball_body and 0 < a < self.ball_body)):
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.rig.model, data, index, wrench)
            if wrench[0] > 0:
                sign = 1. if a == self.ball_body else -1.
                self.contact_impulse[case] += sign * (contact.frame.reshape(3, 3).T @ wrench[:3]) * self.rig.model.opt.timestep
                self.normal_impulse[case] += wrench[0] * self.rig.model.opt.timestep

    def step(self):
        # Same delayed PD/substep order as RepresentationNative, with contact recording.
        self.contact_impulse.fill(0.)
        self.normal_impulse.fill(0.)
        bad = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC)
        for substep in range(self.rig.decimation):
            q, v = self._array('qpos'), self._array('qvel')
            self.policy_control.physics_control(q, v)
            self.substep_q[substep] = q[:, self._qi]
            self.substep_v[substep] = v[:, self._vi]
            self.substep_torque[substep] = self.native_ctrl[:, self.rig.actuators]
            for case, data in enumerate(self.datas):
                if not self.active[case] or self.solver_warning[case]:
                    continue
                data.ctrl[:] = self.native_ctrl[case].numpy()
                mujoco.mj_step(self.rig.model, data)
                self._record_contacts(case, data)
                self.solver_warning[case] |= any(data.warning[item].number for item in bad)
        for case, data in enumerate(self.datas):
            if self.active[case]:
                self._ground_force_cache[case] = self._read_ground_forces(data)
                mujoco.mj_forward(self.rig.model, data)
                data.xfrc_applied[:] = 0

    def native_qpos(self): return super().native_qpos()[:, :46]
    def get_dof_vel(self, obj_id): return super().get_dof_vel(obj_id)[:, :39]
    def get_body_pos(self, obj_id): return super().get_body_pos(obj_id)[:, :18]
    def get_body_rot(self, obj_id): return super().get_body_rot(obj_id)[:, :18]
    def get_ground_contact_forces(self, obj_id): return super().get_ground_contact_forces(obj_id)[:, :18]


class StandBallTask(RepresentationMixin, TorqueStandTask):
    def __init__(self, rig, reference, *, ball_path, seconds):
        self.representation = 'target_pd'
        # RepresentationMixin normally selects a character-only engine.
        TorqueStandTask.__init__(self, rig, reference, 8, 'cpu', reward_mode='reference_v9',
            episode_seconds=seconds, engine_factory=partial(StandBallNative, ball_path=ball_path))
