"""Acceptance distinguishes one placed foot from the two-foot walking gate."""
import unittest

from .placed_step import single_step_checks


class SingleStepChecksTests(unittest.TestCase):
    def test_one_foot_lifts_and_places_without_moving_support_foot(self):
        checks = dict(finite=True, penetration=True, root_wrench_consistency=True,
                      actuation_feasibility=True, foot_lift=False)
        metrics = {"foot_lift_range_m": [.06, 0.]}
        self.assertTrue(all(single_step_checks(checks, metrics, .15, .15).values()))
        self.assertFalse(single_step_checks(checks, metrics, .03, .15)["placement"])
        self.assertFalse(single_step_checks(checks, {"foot_lift_range_m": [.02, 0.]}, .15, .15)["swing_foot_lift"])
        self.assertFalse(single_step_checks({**checks, "actuation_feasibility": False}, metrics, .15, .15)["physical"])
        self.assertTrue(all(single_step_checks(checks, metrics, -.15, -.15).values()))


if __name__ == "__main__":
    unittest.main()
