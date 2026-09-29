import unittest

import torch

from .step_reward import StepRecoveryReward, TransferRecoveryReward, swing_quality


class StepRewardTests(unittest.TestCase):
    def test_transfer_has_signal_before_liftoff_but_still_prefers_lift(self):
        points = torch.zeros(1, 2, 2, 3)
        com, velocity = torch.tensor([[0., 0., 1.]]), torch.zeros(1, 3)
        heights = torch.zeros(1, 2, 2)
        def score(loads):
            return swing_quality(com, velocity, points, heights, torch.tensor([loads]), preparation_weight=.3)
        equal, shifting = score([150., 150.]), score([75., 225.])
        self.assertEqual(float(equal), 0.)
        self.assertGreater(float(shifting), 0.)
        torch.testing.assert_close(shifting, score([225., 75.]))
        points[:, 0, :, 2] = .02
        self.assertGreater(float(score([0., 300.])), float(shifting))
        self.assertEqual(float(score([0., 0.])), 0.)

    def test_final_two_foot_stance_outscores_balanced_single_support(self):
        reward = TransferRecoveryReward()
        args = (torch.ones(1), torch.ones(1), torch.ones(1), torch.ones(1),
                torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1), torch.ones(1))
        self.assertGreater(float(reward(*args, torch.tensor([[100., 100.]]))),
                           float(reward(*args, torch.tensor([[0., 200.]]))))

    def test_settled_balance_remains_better_than_swing_and_airborne_foot_needs_support(self):
        reward = StepRecoveryReward()
        args = (torch.ones(1), torch.ones(1), torch.ones(1), torch.ones(1),
                torch.zeros(1, 3), torch.zeros(1, 3))
        stable = reward(*args, torch.zeros(1), torch.zeros(1))
        swinging = reward(*args, torch.tensor([.08]), torch.ones(1))
        self.assertGreater(float(stable), float(swinging))
        points = torch.zeros(1, 2, 2, 3)
        points[:, 0, :, 2] = .04
        com, velocity = torch.tensor([[0., 0., 1.]]), torch.zeros(1, 3)
        heights = torch.zeros(1, 2, 2)
        supported = swing_quality(com, velocity, points, heights, torch.tensor([[0., 300.]]))
        floating = swing_quality(com, velocity, points, heights, torch.zeros(1, 2))
        self.assertGreater(float(supported), .9)
        self.assertEqual(float(floating), 0.)

    def test_swing_credit_follows_capture_location_and_is_foot_symmetric(self):
        points = torch.zeros(1, 2, 2, 3)
        points[:, 0, :, 2] = .04
        com, velocity = torch.tensor([[.15, 0., 1.]]), torch.zeros(1, 3)
        heights, loads = torch.zeros(1, 2, 2), torch.tensor([[0., 300.]])
        before = swing_quality(com, velocity, points, heights, loads)
        points[:, 0, :, 0] = .15
        after = swing_quality(com, velocity, points, heights, loads)
        self.assertGreater(float(after), float(before))
        torch.testing.assert_close(after, swing_quality(com, velocity, points.flip(1), heights.flip(1), loads.flip(1)))


if __name__ == "__main__":
    unittest.main()
