"""Regression for a float32 timeout that previously marked survivors as failures."""
import unittest
import hashlib
import json

import torch

from .reactive_ball_eval import repeated_cases, timeout_for_steps, unseen_cases


class ReactiveBallEvalTests(unittest.TestCase):
    def test_original_torso_validation_fixture_is_unchanged(self):
        cases = unseen_cases(281425, 64, 2, 2, 33, .016668)
        digest = hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, "32ced2e88a09688d7af12a5986c1d6137010a4ed56bb46634ac9fa10efd55e16")

    def test_head_validation_covers_every_posture_and_direction(self):
        cases = unseen_cases(281610, 64, 2, 2, 33, .016668, target_bodies=("Head",))
        self.assertTrue(all(c["target_body"] == "Head" for c in cases))
        self.assertEqual({(c["posture"], c["direction"]) for c in cases},
                         {(p, d) for p in range(8) for d in range(4)})

    def test_unseen_targets_reject_empty_duplicate_or_unknown_bodies(self):
        for targets in ((), ("Head", "Head"), ("unknown",)):
            with self.assertRaisesRegex(ValueError, "unique supported"):
                unseen_cases(281610, 64, 2, 2, 33, .016668, target_bodies=targets)

    def test_repeated_suite_rotates_second_hit_and_leaves_settling_time(self):
        cases = repeated_cases(1.5, 12.)
        self.assertEqual(len(cases), 32)
        self.assertTrue(all(c["second_launch_step"] == 360 for c in cases))
        self.assertTrue(all(c["trial_steps"] - c["second_launch_step"] > 300 for c in cases))
        self.assertTrue(all(c["second_target_body"] != c["target_body"] for c in cases))

    def test_float32_clock_reaches_final_twelve_second_step(self):
        dt, steps = .016668, 720
        elapsed = torch.arange(steps + 1, dtype=torch.float32) * dt
        # The old Python-float endpoint exceeded the final float32 elapsed value.
        self.assertLess(float(elapsed[-1]), steps * dt)
        endpoint = timeout_for_steps(steps, dt)
        self.assertLess(float(elapsed[-2]), endpoint)
        self.assertGreaterEqual(float(elapsed[-1]), endpoint)


if __name__ == "__main__":
    unittest.main()
