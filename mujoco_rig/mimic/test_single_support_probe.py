"""A one-leg hold cannot pass by quietly putting the swing foot down."""
from types import SimpleNamespace
import unittest

import torch

from .probe_single_support import hold_metrics
from .train_single_support import support_reward


class SingleSupportHoldTests(unittest.TestCase):
    def task(self, left_load):
        q = torch.zeros((1, 46))
        q[0, 2], q[0, 3] = .9, 1.
        loads = torch.zeros((1, 4, 3))
        loads[0, 0, 2] = left_load
        loads[0, 1, 2] = 20.
        engine = SimpleNamespace(native_qpos=lambda: q,
                                 get_ground_contact_forces=lambda _: loads,
                                 get_root_vel=lambda _: torch.zeros((1, 3)),
                                 get_root_ang_vel=lambda _: torch.zeros((1, 3)))
        return SimpleNamespace(engine=engine, allowed_ground=torch.arange(4))

    def test_single_support_requires_unloaded_swing_foot(self):
        self.assertTrue(hold_metrics(self.task(0.), True, [False, True])["passed"])
        self.assertFalse(hold_metrics(self.task(10.), True, [False, True])["passed"])
        self.assertTrue(hold_metrics(self.task(10.), True, [True, True])["passed"])

    def test_reward_prefers_loaded_stance_and_quiet_root(self):
        base = torch.ones(3)
        velocity = torch.tensor([[0., 0.], [0., 0.], [1., 0.]])
        reward = support_reward(base, torch.tensor([0., 100., 0.]),
                                torch.tensor([100., 100., 100.]), velocity)
        self.assertGreater(reward[0], reward[1])
        self.assertGreater(reward[0], reward[2])


if __name__ == "__main__":
    unittest.main()
