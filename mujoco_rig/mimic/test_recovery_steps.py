import unittest

import torch

from .recovery_steps import RecoveryStepTracker


class RecoveryStepTests(unittest.TestCase):
    def setUp(self):
        self.points = torch.zeros(1, 2, 2, 3)
        self.tracker = RecoveryStepTracker(self.points, .02)
        self.active = torch.tensor([True])

    def tick(self, points, loads, hit=True, count=1):
        for _ in range(count):
            self.tracker.update(points, torch.tensor([loads]), self.active, torch.tensor([hit]))

    def test_loaded_sliding_and_brief_bounces_are_not_steps(self):
        moved = self.points.clone()
        moved[:, 0, :, 0] = .1
        self.tick(moved, [100., 100.], count=10)
        moved[:, 0, :, 2] = .02
        self.tick(moved, [0., 100.])
        self.tick(moved, [100., 100.], count=3)
        self.assertEqual(int(self.tracker.steps.sum()), 0)

    def test_either_foot_requires_clearance_translation_and_loaded_landing(self):
        for foot in (0, 1):
            self.tracker = RecoveryStepTracker(self.points, .02)
            moved = self.points.clone()
            moved[:, foot, :, 2] = .02
            loads = [100., 100.]
            loads[foot] = 0.
            self.tick(moved, loads, count=3)
            moved[:, foot, :, 0] = .05
            self.tick(moved, loads)
            self.tick(moved, [100., 100.], count=2)
            self.assertEqual(int(self.tracker.steps.sum()), 0)
            self.tick(moved, [100., 100.])
            self.assertEqual(int(self.tracker.steps[0, foot]), 1)
            self.tick(moved, [100., 100.], count=20)
            self.assertEqual(int(self.tracker.steps.sum()), 1)

    def test_no_pre_hit_or_finished_episode_credit(self):
        moved = self.points.clone()
        moved[:, 0, :, 2] = .1
        self.tick(moved, [0., 100.], hit=False, count=10)
        self.active[:] = False
        self.tick(moved, [0., 100.], count=10)
        self.assertFalse(self.tracker.lifted.any())


if __name__ == "__main__":
    unittest.main()
