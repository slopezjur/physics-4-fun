"""Reject apparent steps caused by sliding, rocking, hopping or late collapse."""
import unittest

import numpy as np

from .corrective_step_audit import measure_steps


class CorrectiveStepAuditTests(unittest.TestCase):
    def setUp(self):
        self.time = np.arange(1, 251)*.02
        self.xy = np.zeros((250, 2, 2))
        self.xy[:, 1, 1] = .3
        self.clear = np.zeros((250, 2))
        self.loads = np.full((250, 2), 300.)
        self.alive = np.ones(250, bool)
        self.valid = np.ones(250, bool)

    def result(self):
        return measure_steps(self.time, self.xy, self.clear, self.loads, self.alive,
                             self.valid, hit_time=.5)

    def step(self, foot=0):
        self.loads[40:50, foot] = 0
        self.clear[40:50, foot] = .03
        self.xy[40:50, foot, 0] = np.linspace(0, .08, 10)
        self.xy[50:, foot, 0] = .08
        self.valid[40:55] = False

    def test_either_foot_replants_and_settles(self):
        for foot in (0, 1):
            self.setUp()
            self.step(foot)
            events = self.result()
            self.assertEqual(len(events), 1)
            self.assertIsNotNone(events[0]['settled_window_start'])

    def test_grounded_slide_is_not_a_step(self):
        self.xy[50:, 0, 0] = .1
        self.assertEqual(self.result(), [])

    def test_rocking_or_brief_clearance_is_not_a_step(self):
        self.step()
        self.clear[40:50, 0] = 0
        self.clear[42, 0] = .03
        self.assertEqual(self.result(), [])

    def test_hop_and_sliding_support_are_rejected(self):
        self.step()
        self.loads[40:50, 1] = 0
        self.assertEqual(self.result(), [])
        self.loads[:, 1] = 300
        self.xy[45:, 1, 0] = .04
        self.assertEqual(self.result(), [])

    def test_landing_then_collapse_is_not_settled(self):
        self.step()
        self.alive[100:] = False
        self.assertIsNone(self.result()[0]['settled_window_start'])

    def test_pre_hit_swing_is_not_credited(self):
        self.step()
        result = measure_steps(self.time, self.xy, self.clear, self.loads, self.alive,
            self.valid, hit_time=.9)
        self.assertEqual(result, [])

    def test_next_hit_cannot_complete_previous_settling_window(self):
        self.step()
        result = measure_steps(self.time, self.xy, self.clear, self.loads, self.alive,
            self.valid, hit_time=.5, end_time=2.)
        self.assertIsNone(result[0]['settled_window_start'])


if __name__ == '__main__':
    unittest.main()
