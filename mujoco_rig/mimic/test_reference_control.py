"""Diagnostic controller boundaries: signs, passive joints, delay and support equations."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from .reference import StandReference
from .reference_controller import ReferencePD, ImmediateTorqueDiagnostic
from .rig import Rig, DelayedTorqueControl
from .static_support import solve_support


class ReferenceControlTests(unittest.TestCase):
    def setUp(self):
        self.rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        self.reference = StandReference(Path(__file__).parent / "assets/stand_reference.npz", self.rig, "cpu")

    def test_feedback_opposes_error_without_driving_passive_joints(self):
        task = SimpleNamespace(rig=self.rig, reference=self.reference, device="cpu",
                               offset=torch.zeros(1), get_env_time=lambda: torch.zeros(1))
        controller = ReferencePD(task, 2, 0.05)
        q, v = self.reference.sample(task.offset)
        obs = torch.zeros((1, 300)); obs[:, 15:54] = q[:, 7:]; obs[:, 54:93] = v[:, 6:] * 0.1
        torch.testing.assert_close(controller(obs), torch.zeros((1, 30)), atol=1e-6, rtol=0)
        dof = int(controller.dofs[0]) - 6
        obs[0, 15 + dof] += 0.1
        action = controller(obs)
        self.assertAlmostEqual(float(action[0, 0]), -0.2, places=6)
        self.assertEqual(int(torch.count_nonzero(action.abs() > 1e-6)), 1)
        passive = next(i for i in range(39) if i + 6 not in controller.dofs)
        obs[0, 15 + passive] += 0.5
        torch.testing.assert_close(controller(obs), action)

    def test_counterfactual_is_explicit_and_keeps_torque_limits(self):
        delayed = DelayedTorqueControl(self.rig, 1, "cpu")
        immediate = ImmediateTorqueDiagnostic(self.rig, 1, "cpu")
        action = torch.full((1, 30), 2.)
        self.assertEqual(int(torch.count_nonzero(delayed.apply(action))), 0)
        np.testing.assert_allclose(immediate.apply(action)[0, self.rig.actuators], self.rig.action_scale)
        self.assertEqual(int(torch.count_nonzero(immediate.pending)), 0)

    def test_support_solution_respects_torque_budget_and_balance_equations(self):
        result = solve_support(self.rig, self.reference.qpos[0].numpy(), passive_tolerance=None, minimum_norm=True)
        self.assertTrue(result["feasible"])
        self.assertLessEqual(result["max_torque_budget_fraction"], 1)
        self.assertLess(result["controlled_equilibrium_residual"], 1e-5)
        # The exact reference's one-foot contact set cannot support its COM statically.
        self.assertFalse(solve_support(self.rig, self.reference.qpos[0].numpy(), actual_contacts=True)["feasible"])
        airborne = self.reference.qpos[0].numpy().copy()
        airborne[2] += 1
        self.assertFalse(solve_support(self.rig, airborne, actual_contacts=True)["feasible"])


if __name__ == "__main__":
    unittest.main()
