"""Physical curriculum and explicit controller-memory contract regressions."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from .ball import set_curriculum_ball_mass
from .ball_native import BallNativeEngine
from .rig import Rig
from .train_recovery_residual import controller_observation


class RecoveryResidualTests(unittest.TestCase):
    def test_mass_curriculum_changes_only_projectile_mass_and_inertia(self):
        rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        engine = BallNativeEngine(rig, 2)
        model = engine.rig.model
        masses, inertias = model.body_mass.copy(), model.body_inertia.copy()
        joints = model.jnt_range.copy()
        controls = model.actuator_forcerange.copy()
        set_curriculum_ball_mass(engine, .25)
        body = engine.ball_body
        np.testing.assert_array_equal(model.body_mass[:body], masses[:body])
        np.testing.assert_array_equal(model.body_inertia[:body], inertias[:body])
        np.testing.assert_allclose(model.body_inertia[body], inertias[body] * .25 / masses[body])
        self.assertEqual(model.body_mass[body], .25)
        np.testing.assert_array_equal(model.jnt_range, joints)
        np.testing.assert_array_equal(model.actuator_forcerange, controls)
        set_curriculum_ball_mass(engine, 8.)
        np.testing.assert_allclose(model.body_mass, masses)
        np.testing.assert_allclose(model.body_inertia, inertias)
        with self.assertRaises(ValueError):
            set_curriculum_ball_mass(engine, 0.)

    def test_observation_contains_action_and_remembered_controller_state(self):
        teacher = SimpleNamespace(
            landing=SimpleNamespace(landed=True, com_kp=15., com_kd=3.),
            controller=SimpleNamespace(offsets=[0., 0., 0., .2]),
            capture_adjustment=[-.03, .01],
            balance=SimpleNamespace(stabilizer=SimpleNamespace(k_hr=12., d_hr=2.)))
        raw, base = torch.randn(1, 362), torch.randn(1, 30)
        result = controller_observation(raw, teacher, base)
        self.assertEqual(result.shape, (1, 400))
        torch.testing.assert_close(result[:, :362], raw)
        torch.testing.assert_close(result[:, 362:392], base)
        torch.testing.assert_close(result[:, 392:], torch.tensor([[1., .2, -.03, .01, 12., 2., 15., 3.]]))


if __name__ == "__main__":
    unittest.main()
