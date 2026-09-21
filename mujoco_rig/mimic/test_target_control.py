"""Target transport and feedback invariants, independent of learning quality."""
from pathlib import Path
import unittest

import torch

from .rig import Rig
from .target_control import DelayedTargetControl


class TargetControlTests(unittest.TestCase):
    def setUp(self):
        self.rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        self.control = DelayedTargetControl(self.rig, 2, "cpu")
        self.q = torch.tensor(self.rig.rest, dtype=torch.float32).repeat(2, 1)
        self.v = torch.zeros(2, self.rig.model.nv)
        self.control.set_reference(self.q, self.v)

    def test_two_step_target_delay_and_physics_feedback(self):
        c = self.control
        action = torch.zeros(2, 30)
        action[:, 0] = 0.4
        for _ in range(2):
            c.apply(action)
            self.assertEqual(int(torch.count_nonzero(c.physics_control(self.q, self.v))), 0)
        c.apply(torch.zeros_like(action))
        torque = c.physics_control(self.q, self.v).clone()
        self.assertAlmostEqual(float(torque[0, c.indices[0]]), 0.4 * float(c.scale[0]), places=4)
        # Feedback changes immediately, without enqueueing another action.
        self.q[:, c.q_indices[0]] += 0.15
        self.assertLess(float(c.physics_control(self.q, self.v)[0, c.indices[0]]), 0)
        self.assertEqual(int(torch.count_nonzero(c.ctrl[:, [i for i in range(self.rig.model.nu)
                                                         if i not in self.rig.actuators]])), 0)

    def test_absolute_targets_are_frozen_at_issue_and_clamped(self):
        c = self.control
        c.apply(torch.full((2, 30), 100.0))
        target = c.pending[0].clone()
        self.assertTrue(bool((target[:, :30] <= c.upper).all()))
        self.assertTrue(bool((target[:, :30] >= c.lower).all()))
        self.q[:, 7:] += 2
        c.apply(torch.zeros(2, 30))
        c.apply(torch.zeros(2, 30))
        torch.testing.assert_close(c.active, target)
        c.physics_control(self.q, self.v)
        self.assertTrue(bool((c.ctrl[:, c.indices].abs() <= c.scale).all()))

    def test_partial_reset_clears_only_selected_targets_and_active_feedback(self):
        c = self.control
        for _ in range(3):
            c.apply(torch.ones(2, 30))
        peer = c.pending[:, 1].clone(), c.active[1].clone()
        c.reset([0])
        self.assertEqual(int(torch.count_nonzero(c.pending[:, 0])), 0)
        self.assertEqual(int(torch.count_nonzero(c.active[0])), 0)
        torch.testing.assert_close(c.pending[:, 1], peer[0])
        torch.testing.assert_close(c.active[1], peer[1])
        c.apply(torch.zeros(2, 30))
        self.assertEqual(int(torch.count_nonzero(c.active[0])), 0)

    def test_invalid_action_does_not_mutate_delay(self):
        for action in (torch.zeros(2, 39), torch.full((2, 30), float("nan"))):
            with self.assertRaises(ValueError):
                self.control.apply(action)
        self.assertEqual(self.control.cursor, 0)
        self.assertEqual(int(torch.count_nonzero(self.control.pending)), 0)


if __name__ == "__main__":
    unittest.main()
