"""The first planned touchdown must stay loaded through recovery."""
import unittest

from .recovery_step_motion import loaded_landing_metrics
from .probe_recovery_robustness import disturbed_recovery_passed


class LoadedLandingGateTests(unittest.TestCase):
    def test_late_replant_does_not_replace_lost_first_touchdown(self):
        dt = .05
        samples = [(step * dt, 2.55 <= step * dt < 2.7 or 3.7 <= step * dt <= 8.15)
                   for step in range(1, 164)]

        result = loaded_landing_metrics(samples, 2.55, dt)

        self.assertLess(result["first_loaded_contact_seconds"], .2)
        self.assertGreater(result["longest_loaded_landing_seconds"], 4.)
        self.assertFalse(result["touchdown_retained"])
        self.assertAlmostEqual(result["final_loaded_landing_start_seconds"], 3.7)

    def test_first_touchdown_must_begin_near_schedule_and_last_two_seconds(self):
        dt = .05
        sustained = [(step * dt, 2.6 <= step * dt <= 5.)
                     for step in range(1, 101)]
        delayed = [(step * dt, 3.1 <= step * dt <= 5.)
                   for step in range(1, 101)]

        self.assertTrue(loaded_landing_metrics(sustained, 2.55, dt)["touchdown_retained"])
        self.assertFalse(loaded_landing_metrics(delayed, 2.55, dt)["touchdown_retained"])

    def test_hit_allows_replant_only_with_sustained_final_support(self):
        recovered = dict(survived=True, final_slow=True,
                         peak_swing_lift_m=.1, final_left_foot_shift_m=.2,
                         final_loaded_landing_start_seconds=3.05,
                         final_loaded_landing_seconds=5.1)
        self.assertTrue(disturbed_recovery_passed(recovered, 2.55))
        self.assertFalse(disturbed_recovery_passed(
            {**recovered, "final_loaded_landing_seconds": .5}, 2.55))
        self.assertFalse(disturbed_recovery_passed(
            {**recovered, "survived": False}, 2.55))


if __name__ == "__main__":
    unittest.main()
