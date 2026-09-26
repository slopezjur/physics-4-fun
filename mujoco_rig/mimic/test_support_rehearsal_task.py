"""Quiet rehearsal and long support holds must share one policy contract."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from .native_engine import NativeDummyEngine
from .rig import Rig
from .runtime import activate
from .support_transition import TRACKING_REWARD_SCHEMA


class SupportRehearsalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("MIMICKIT_PATH"):
            raise unittest.SkipTest("Set MIMICKIT_PATH for support rehearsal tests")
        activate(Path(os.environ["MIMICKIT_PATH"]))
        from envs.base_env import EnvMode
        from .reference import StandReference
        from .support_rehearsal_task import SupportRehearsalTask
        cls.Mode, cls.Reference, cls.Task = EnvMode, StandReference, SupportRehearsalTask
        cls.quiet = Path(__file__).parent / "assets/stand_reference.npz"
        cls.rig = Rig.load(cls.quiet.parents[2] / "dummy.xml")

    def task(self, reward="reference"):
        with patch("mujoco_rig.mimic.support_rehearsal_task.SupportTransitionReference",
                   side_effect=lambda p, r, d, mode: self.Reference(self.quiet, r, d)):
            task = self.Task(self.rig, self.quiet, self.quiet, 4, "cpu", seed=19,
                             quiet_fraction=.25, support_seconds=12., quiet_seconds=3.,
                             reward_mode=reward, engine_factory=NativeDummyEngine)
        task.set_mode(self.Mode.TRAIN)
        task.reset()
        return task

    def test_world_groups_and_partial_reset(self):
        task = self.task()
        self.assertEqual(task.quiet_mask().tolist(), [True, False, False, False])
        self.assertEqual(task.episode_duration().tolist(), [3., 12., 12., 12.])
        self.assertEqual(task.offset[1:].tolist(), [0., 0., 0.])
        self.assertLessEqual(float(task.offset[0]), task.quiet_reference.duration - 3.)
        before = task.engine.native_qpos().clone()
        task.reset(torch.tensor([0, 2]))
        torch.testing.assert_close(task.engine.native_qpos()[[1, 3]], before[[1, 3]])
        self.assertEqual(task.reference_phase(torch.tensor([1., 7., 12., 15.])).tolist()[1:],
                         [1., 1., 1.])

    def test_tracking_reward_does_not_change_quiet_worlds(self):
        task = self.task(TRACKING_REWARD_SCHEMA)
        q, v = task.sample_reference(task.offset)
        q[:, 0] += .14
        task.engine.reset_envs(task.ids, q, v)
        base = task.reference_reward()
        tracked = task.reward()
        torch.testing.assert_close(tracked[:1], base[:1])
        self.assertTrue(bool((tracked[1:] < base[1:] - .15).all()))


if __name__ == "__main__":
    unittest.main()
