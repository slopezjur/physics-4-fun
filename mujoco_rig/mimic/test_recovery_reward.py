"""Recovery incentives and unchanged quiet/observation behavior."""
import os
from pathlib import Path
import unittest

import torch

from .recovery_reward import RecoveryReward


class RecoveryRewardTests(unittest.TestCase):
    def score(self, *, height=.9, up=1., speed=0., angular_speed=0., feet=(100., 100.), imitation=.8):
        return RecoveryReward()(
            torch.tensor([imitation]), torch.tensor([height]), torch.tensor([.9]),
            torch.tensor([up]), torch.tensor([[speed, 0., 0.]]),
            torch.tensor([[0., angular_speed, 0.]]), torch.tensor([feet]))

    def test_settling_is_better_than_drifting_spinning_crouching_or_leaning(self):
        settled = self.score()
        for disturbance in (dict(speed=.6), dict(angular_speed=2.), dict(height=.7), dict(up=.8)):
            with self.subTest(disturbance=disturbance):
                self.assertGreater(float(settled), float(self.score(**disturbance)))

    def test_support_is_small_bonus_not_a_requirement_to_step(self):
        one_foot = self.score(feet=(100., 0.))
        self.assertGreater(float(one_foot), .8)
        self.assertAlmostEqual(float(self.score() - one_foot), .075, places=6)
        self.assertLess(float(self.score(speed=1.)), float(one_foot))

    def test_reward_bounded_and_motion_prior_retained(self):
        self.assertAlmostEqual(float(self.score(imitation=1.)), 1.)
        self.assertAlmostEqual(float(self.score(imitation=1.) - self.score(imitation=0.)), .25)
        self.assertGreaterEqual(float(self.score(height=0., up=-1., speed=100., imitation=0.)), 0.)

    def test_native_contact_gate_reset_and_translation_invariance(self):
        checkout = os.environ.get("MIMICKIT_PATH")
        if not checkout:
            self.skipTest("Set MIMICKIT_PATH for native task integration")
        from .runtime import activate
        activate(Path(checkout))
        from envs.base_env import EnvMode
        from .ball_task import BallTask
        from .rig import Rig
        root = Path(__file__).resolve().parent
        rig = Rig.load(root.parent / "dummy.xml")
        task = BallTask(rig, root / "assets/stand_reference.npz", 4, "cpu", reward_mode="recovery_v1")
        task.set_mode(EnvMode.TEST)
        task.reset()
        task.offset.zero_()
        q, v = task.reference.sample(task.offset)
        # Paired identical poses, displaced 0.5 m horizontally in the second pair.
        q[2:, 0] += .5
        task.engine.reset_envs(task.ids, q, v)
        before = task.observations().clone()
        standing = task.reference_reward()
        relaxed = task.reference_reward(track_root=False)
        self.assertLess(float(standing[2]), float(standing[0]))
        torch.testing.assert_close(relaxed[2:], relaxed[:2])
        # Reported contact only activates enabled shots; launch without contact does not.
        task.shot_enabled[:] = torch.tensor([False, True, True, True])
        task.engine.hit[:] = torch.tensor([True, False, True, True])
        reward = task.reward()
        torch.testing.assert_close(reward[:2], standing[:2], rtol=0, atol=0)
        torch.testing.assert_close(before, task.observations(), rtol=0, atol=0)
        task.engine.hit[:] = True
        task.shot_enabled[:] = True
        reward = task.reward()
        torch.testing.assert_close(reward[2:], reward[:2])
        task.reset(torch.tensor([0, 2]))
        torch.testing.assert_close(task.engine.hit, torch.tensor([False, True, False, True]))
        torch.testing.assert_close(task.reward()[[0, 2]], task.reference_reward()[[0, 2]], rtol=0, atol=0)
        task.reward_mode = "reference"
        torch.testing.assert_close(task.reward(), task.reference_reward(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
