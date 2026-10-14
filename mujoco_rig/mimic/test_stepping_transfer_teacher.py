"""Bounded inverse-command and short native checks for transfer examples."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .runtime import activate


class SteppingTransferTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
        from .rig import Rig
        from .run_learned_support import PARENT
        from .run_support_skill_study import FOUNDATION
        from .stand_experiment_evidence import load_actor
        from .stepping_skill_core import StepSkillEnvironment
        from .stepping_transfer_teacher import SteppingTransferTeacher, fit_transfer_goal
        from .interval_target_support import IntervalTargetSupport
        torch.set_num_threads(1)
        cls.environment_type = StepSkillEnvironment
        cls.teacher_type, cls.parent_type = SteppingTransferTeacher, IntervalTargetSupport
        cls.fit_goal = staticmethod(fit_transfer_goal)
        cls.rig = Rig.load(FOUNDATION / 'dummy.xml', policy_exclude=())
        cls.anchor = load_actor(cls.rig, FOUNDATION / 'stationary_reference.npz',
            json.loads((FOUNDATION / 'protocol.json').read_text()), PARENT)

    def setUp(self):
        self.env = self.environment_type(self.rig, self.anchor)
        self.teacher = self.teacher_type(self.env.task)

    def test_reachable_box_and_finite_inverse_with_asymmetric_and_clipped_targets(self):
        env, teacher = self.env, self.teacher
        control = env.engine.policy_control
        low, high = control.lower.numpy(), control.upper.numpy()
        radius = env.radius.numpy().astype(float)
        reference = env.engine.datas[0].qpos.copy()
        scenarios = [('interior', .5 * (low + high), np.zeros(33)),
                     ('asymmetric', .5 * (low + high), radius * .95),
                     ('clipped_upper', high.copy(), np.full(33, .06)),
                     ('clipped_lower', low.copy(), np.full(33, -.06))]
        for name, nominal, previous in scenarios:
            with self.subTest(name=name):
                residual_low = np.maximum(-radius * (1 - 1e-6), previous - .02)
                residual_high = np.minimum(radius * (1 - 1e-6), previous + .02)
                expected_low = np.clip(nominal + residual_low, low, high)
                expected_high = np.clip(nominal + residual_high, low, high)

                def proposal(instance, world, goal, source, box_radius):
                    self.assertIs(instance, teacher)
                    self.assertEqual(world, 0)
                    np.testing.assert_allclose(source - box_radius, expected_low, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(source + box_radius, expected_high, atol=1e-12, rtol=0)
                    return expected_low + .73 * (expected_high - expected_low), {'mocked': True}

                with patch.object(self.parent_type, 'target', autospec=True, side_effect=proposal) as spy:
                    latent, details = teacher.command(0, reference, nominal, previous, radius)
                self.assertEqual(spy.call_count, 1)
                self.assertEqual(latent.dtype, np.float32)
                self.assertTrue(np.isfinite(latent).all())
                applied = previous + np.clip(radius * np.tanh(latent.astype(float)) - previous, -.02, .02)
                issued = np.clip(nominal + applied, low, high)
                np.testing.assert_allclose(issued, details['planned_target'], atol=2e-6, rtol=0)
                self.assertTrue((np.abs(applied) <= radius + 1e-7).all())
                self.assertTrue((np.abs(applied - previous) <= .02 + 1e-7).all())
                if name.startswith('clipped'):
                    self.assertEqual(details['collapsed_target_dimensions'], 33)
                    self.assertGreater(float(np.abs(applied).min()), 0.)

    def test_private_solver_mutation_is_rejected(self):
        for kind in ('qpos', 'queue'):
            with self.subTest(kind=kind):
                env = self.environment_type(self.rig, self.anchor)
                teacher = self.teacher_type(env.task)
                control = env.engine.policy_control
                nominal = .5 * (control.lower.numpy() + control.upper.numpy())

                def mutate(instance, world, goal, source, radius):
                    if kind == 'qpos':
                        env.engine.datas[world].qpos[0] += .001
                    else:
                        control.pending[0, world, 0] += .001
                    return source, {}

                with patch.object(self.parent_type, 'target', autospec=True, side_effect=mutate):
                    with self.assertRaisesRegex(RuntimeError, 'modified the live plant'):
                        teacher.command(0, env.engine.datas[0].qpos.copy(), nominal, np.zeros(33), env.radius.numpy())

    def test_private_ik_exact_xy_fixed_feet_and_unchanged_nonlegs(self):
        env = self.env
        before_q = np.stack([data.qpos.copy() for data in env.engine.datas])
        before_v = np.stack([data.qvel.copy() for data in env.engine.datas])
        leg_joints = [joint for joint in range(1, self.rig.model.njnt)
                      if self.rig.model.joint(joint).name.startswith(('Thigh_', 'Shin_', 'Foot_', 'Toe_'))]
        untouched = np.setdiff1d(np.arange(7, self.rig.model.nq), self.rig.model.jnt_qposadr[leg_joints])
        for world in (0, 1, 8, 9):
            with self.subTest(world=world):
                reference = env.task.reference.qpos[world % 8].numpy().astype(float)
                target = env.curriculum.transfer_target[world].numpy()
                goal, report = self.fit_goal(self.rig.model, reference, target, lean_radians=np.deg2rad(4.))
                np.testing.assert_array_equal(goal[untouched], reference[untouched])
                self.assertLessEqual(report['foot_error_m'], .001)
                self.assertLessEqual(report['com_error_m'], .001)
                self.assertLessEqual(report['penetration_m'], .0002)
                np.testing.assert_allclose(report['fitted_com_m'][:2], target, atol=.001, rtol=0)
        np.testing.assert_array_equal(np.stack([data.qpos for data in env.engine.datas]), before_q)
        np.testing.assert_array_equal(np.stack([data.qvel for data in env.engine.datas]), before_v)

    def test_native_teacher_query_preserves_state_and_matches_issued_target(self):
        from .target_control33 import DelayedTargetControl33
        env, teacher = self.env, self.teacher
        env.reset(level=0, onset=1.)
        active = torch.zeros(16, dtype=torch.bool)
        active[0] = True
        for _ in range(int(np.ceil(1. / env.dt)) + 1):
            if bool(env.curriculum.observations()[0, 13]):
                break
            env.step(torch.zeros(16, 33), active)
        self.assertTrue(env.curriculum.observations()[0, 13])
        goal, _ = self.fit_goal(self.rig.model, env.task.reference.qpos[0].numpy(),
                                env.curriculum.transfer_target[0].numpy(), lean_radians=np.deg2rad(4.))
        control = env.engine.policy_control
        control.set_reference(*env.task.sample_reference(env.task.offset + env.task.steps * env.dt))
        with torch.no_grad():
            source = env.anchor(env.task.observations())
            nominal = DelayedTargetControl33.target_positions(control, source)
        before_q, before_v = env.engine.native_qpos().clone(), env.engine._array('qvel').clone()
        pending, cursor = control.pending.clone(), control.cursor
        latent, details = teacher.command(0, goal, nominal[0].numpy(), env.previous[0].numpy(), env.radius.numpy())
        self.assertTrue(torch.equal(env.engine.native_qpos(), before_q))
        self.assertTrue(torch.equal(env.engine._array('qvel'), before_v))
        self.assertTrue(torch.equal(control.pending, pending))
        self.assertEqual(control.cursor, cursor)
        command = torch.zeros(16, 33)
        command[0] = torch.from_numpy(latent)
        _, _, info = env.step(command, active)
        np.testing.assert_allclose(info['issued'][0].numpy(), details['planned_target'], atol=2e-6, rtol=0)
        self.assertFalse(info['warning'].any())
        self.assertFalse(info['force'].any())
        self.assertGreater(teacher.private_steps, 0)


if __name__ == '__main__':
    unittest.main()
