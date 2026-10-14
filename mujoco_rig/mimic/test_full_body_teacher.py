"""Whole-body solver frame convention, bounds, and isolation from live physics."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch

from .full_body_teacher import FullBodyTeacher, rotation_error, upright_rotation
from .recovery_motor import RecoveryTargetControl
from .rig import Rig


class WholeBodyTeacherTests(unittest.TestCase):
    def setUp(self):
        self.rig = Rig.load(Path(__file__).resolve().parents[1] / 'dummy.xml')
        self.live = mujoco.MjData(self.rig.model)
        self.live.qpos[:] = self.rig.rest
        mujoco.mj_forward(self.rig.model, self.live)
        control = RecoveryTargetControl(self.rig, 1, 'cpu')
        self.task = SimpleNamespace(device='cpu', recovery_motor_radians=1.25, rig=self.rig,
            get_num_envs=lambda: 1, engine=SimpleNamespace(policy_control=control, datas=[self.live]),
            observation_contract=lambda: self.rig.contract())
        self.teacher = FullBodyTeacher(self.task, lambda obs: torch.zeros(1, 30))

    def test_orientation_error_has_world_frame_corrective_sign(self):
        angle = .1
        current = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                            [-np.sin(angle), 0., np.cos(angle)]])
        np.testing.assert_allclose(rotation_error(current, np.eye(3)), [0., -np.sin(angle), 0.])
        np.testing.assert_allclose(upright_rotation(current), np.eye(3))

    def test_floating_root_solver_preserves_live_plant_and_bounds(self):
        teacher = self.teacher
        original_q = self.live.qpos.copy()
        original_v = self.live.qvel.copy()
        goals = self.live.xpos[teacher.body].copy()
        goals[0, 2] += .035
        rotations = np.array([upright_rotation(self.live.xmat[b].reshape(3, 3)) for b in teacher.body])
        com_goal = self.live.subtree_com[teacher.pelvis].copy()
        com_goal[1] += .03
        root_goal = self.live.xpos[teacher.pelvis].copy()
        root_goal[1] += .03
        root_goal[2] -= .02
        radius = np.where(self.task.engine.policy_control.leg_mask[teacher.motors].numpy(), 1.25, .25)
        result = teacher.solve(original_q, original_q, original_q[teacher.indices], radius,
                               goals, rotations, 1, com_goal, root_goal)
        np.testing.assert_array_equal(self.live.qpos, original_q)
        np.testing.assert_array_equal(self.live.qvel, original_v)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.all(abs(result - original_q[teacher.indices]) <= radius + 1e-8))
        self.assertTrue(np.all(result >= self.rig.model.jnt_range[teacher.joints, 0]))
        self.assertTrue(np.all(result <= self.rig.model.jnt_range[teacher.joints, 1]))
        self.assertGreater(abs(result - original_q[teacher.indices]).max(), .05)
        self.assertFalse(teacher.contract()['plant_root_writes'])

    def quiet_task(self):
        task = self.task
        task.steps = torch.ones(1, dtype=torch.long)
        task.offset = torch.zeros(1)
        task.feet = torch.tensor(self.teacher.body) - 1
        task.engine.get_body_pos = lambda _: torch.tensor(self.live.xpos[None, 1:], dtype=torch.float32)
        task.engine.get_center_of_mass_velocity = lambda: torch.zeros(1, 3)
        task.engine.native_qpos = lambda: torch.tensor(self.live.qpos[None], dtype=torch.float32)
        task.center_of_mass = lambda: torch.tensor(self.live.subtree_com[None, self.teacher.pelvis], dtype=torch.float32)
        task.sample_reference = lambda _: (task.engine.native_qpos(), torch.zeros(1, self.rig.model.nv))
        return task

    def test_observed_two_foot_standing_releases_stale_lift_without_a_clock(self):
        task = self.quiet_task()
        teacher = FullBodyTeacher(task, self.teacher.actor, release_settled=True, composition='source_offset')
        teacher.phase[0] = 2
        obs = torch.zeros(1, 363); obs[:, 93:95] = .3
        result = teacher(obs)
        torch.testing.assert_close(result, torch.zeros(1, 30), rtol=0, atol=0)
        self.assertEqual(teacher.phase[0], 0)
        self.assertEqual(teacher.settled_releases[0], 1)

    def test_rearming_requires_quiet_joint_and_support_state(self):
        task = self.quiet_task()
        task.engine._array = lambda _: torch.zeros(1, self.rig.model.nv)
        teacher = FullBodyTeacher(task, self.teacher.actor, release_settled=True, rearm_quiet=True)
        teacher.armed[:] = False
        obs = torch.zeros(1, 363); obs[:, 93:95] = .1
        teacher(obs)
        self.assertFalse(teacher.armed[0])
        obs[:, 93:95] = .3
        moving = torch.zeros(1, self.rig.model.nv); moving[:, teacher.dofs[0]] = .2
        task.engine._array = lambda _: moving
        teacher(obs)
        self.assertFalse(teacher.armed[0])
        task.engine._array = lambda _: torch.zeros(1, self.rig.model.nv)
        teacher(obs)
        self.assertTrue(teacher.armed[0])


if __name__ == '__main__':
    unittest.main()
