"""Recovery credit must follow support and momentum, without choosing a stepping foot."""
import unittest

import torch

from .capture_reward import CaptureRecoveryReward, capture_distance


class CaptureRewardTests(unittest.TestCase):
    def setUp(self):
        self.com = torch.tensor([[0., 0., 1.]])
        self.feet = torch.tensor([[[0., -.1, 0.], [0., .1, 0.]]])
        self.loads = torch.tensor([[100., 100.]])

    def test_momentum_warns_before_height_or_uprightness_changes(self):
        stopped = capture_distance(self.com, torch.zeros(1, 3), self.feet, self.loads)
        moving = capture_distance(self.com, torch.tensor([[.8, 0., 0.]]), self.feet, self.loads)
        self.assertEqual(float(stopped), 0.)
        self.assertGreater(float(moving), .19)
        args = (torch.ones(1), torch.ones(1), torch.ones(1), torch.ones(1),
                torch.zeros(1, 3), torch.zeros(1, 3))
        reward = CaptureRecoveryReward()
        self.assertGreater(float(reward(*args, stopped) - reward(*args, moving)), .4)

    def test_either_loaded_foot_can_capture_motion_but_airborne_foot_cannot(self):
        velocity = torch.tensor([[.8, 0., 0.]])
        before = capture_distance(self.com, velocity, self.feet, self.loads)
        moved = self.feet.clone()
        moved[0, 0] = torch.tensor([.26, 0., 0.])
        after = capture_distance(self.com, velocity, moved, self.loads)
        airborne = capture_distance(self.com, velocity, moved, torch.tensor([[0., 100.]]))
        self.assertLess(float(after), float(before))
        self.assertGreater(float(airborne), float(after))
        torch.testing.assert_close(after, capture_distance(self.com, velocity, moved.flip(1), self.loads.flip(1)))

    def test_world_translation_and_yaw_do_not_change_balance_score(self):
        velocity = torch.tensor([[.6, .1, 0.]])
        original = capture_distance(self.com, velocity, self.feet, self.loads)
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        shift = torch.tensor([2., -3., 0.])
        transformed = capture_distance(self.com @ rotation.T + shift, velocity @ rotation.T,
                                       self.feet @ rotation.T + shift, self.loads)
        torch.testing.assert_close(original, transformed, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
