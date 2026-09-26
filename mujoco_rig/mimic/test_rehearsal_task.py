"""Mixed references must agree in reset, observations, rewards and delayed targets."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .baseline import sha256
from .native_engine import NativeDummyEngine
from .rig import Rig
from .runtime import activate


class RehearsalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("MIMICKIT_PATH"):
            raise unittest.SkipTest("Set MIMICKIT_PATH for rehearsal task tests")
        activate(Path(os.environ["MIMICKIT_PATH"]))
        from .rehearsal_task import RehearsalTask
        from .stand_task import StandTask
        from envs.base_env import EnvMode, DoneFlags
        cls.Task, cls.StandTask, cls.Mode, cls.Done = RehearsalTask, StandTask, EnvMode, DoneFlags
        cls.quiet = Path(__file__).parent / "assets/stand_reference.npz"
        cls.rig = Rig.load(cls.quiet.parents[2] / "dummy.xml")

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.motion = Path(self.folder.name) / "motion.npz"
        with np.load(self.quiet) as source:
            data = {k: source[k].copy() for k in source.files}
        data["qpos"][:, 0] += .02
        data["dt"] *= 2
        data["qvel"] *= .5
        np.savez(self.motion, **data)
        meta = json.loads(self.quiet.with_suffix(".json").read_text())
        meta["reference_sha256"] = sha256(self.motion)
        self.motion.with_suffix(".json").write_text(json.dumps(meta))

    def task(self):
        task = self.Task(self.rig, self.motion, self.quiet, 4, "cpu", seed=19,
                         engine_factory=NativeDummyEngine, control_mode="target_pd", contact_mode="support_v2")
        task.set_mode(self.Mode.TRAIN)
        task.reset()
        return task

    def test_world_allocation_partial_resets_and_seeded_phases(self):
        a, b = self.task(), self.task()
        torch.testing.assert_close(a.offset, b.offset)
        self.assertEqual(a.quiet_mask().tolist(), [True, True, False, False])
        self.assertTrue(bool((a.offset <= a.reference_duration() - a.episode_duration()).all()))
        before = a.engine.native_qpos().clone()
        a.reset(torch.tensor([1, 3]))
        torch.testing.assert_close(a.engine.native_qpos()[[0, 2]], before[[0, 2]])
        q, _ = a.sample_reference(a.offset)
        torch.testing.assert_close(a.engine.native_qpos(), q)

    def test_mixed_observations_reward_and_targets_match_single_reference_tasks(self):
        mixed = self.task()
        for reference, indices in ((self.quiet, [0, 1]), (self.motion, [2, 3])):
            task = self.StandTask(self.rig, reference, 4, "cpu", engine_factory=NativeDummyEngine,
                                  control_mode="target_pd", contact_mode="support_v2")
            task.offset[:] = mixed.offset
            task.steps[:] = 0
            task.engine.reset_envs(task.ids, *task.reference.sample(task.offset))
            torch.testing.assert_close(mixed.observations()[indices], task.observations()[indices])
            torch.testing.assert_close(mixed.reward()[indices], task.reward()[indices])
        actions = torch.zeros((4, 30))
        q, v = mixed.sample_reference(mixed.offset)
        mixed.step(actions)
        control = mixed.engine.policy_control
        packet = control.pending[(control.cursor - 1) % len(control.pending)]
        torch.testing.assert_close(packet[:, :30], q[:, control.q_indices].clamp(control.lower, control.upper))
        torch.testing.assert_close(packet[:, 30:60], v[:, control.v_indices])

    def test_horizons_and_test_mode_do_not_mix_references(self):
        task = self.task()
        with patch.object(task.engine, "step"):
            task.steps[:] = round(3. / task.dt) - 1
            done = task.step(torch.zeros((4, 30)))[2]
            self.assertFalse(bool((done == self.Done.TIME.value).any()))
            task.steps[:] = round(5. / task.dt) - 1
            done = task.step(torch.zeros((4, 30)))[2]
            self.assertEqual((done == self.Done.TIME.value).tolist(), [True, True, False, False])
        task.set_mode(self.Mode.TEST)
        task.reset()
        self.assertFalse(bool(task.quiet_mask().any()))
        self.assertEqual(task.episode_duration(), 3.)
        torch.testing.assert_close(task.engine.native_qpos(), task.reference.sample(task.offset)[0])

    def test_invalid_horizons_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "fit inside"):
            self.Task(self.rig, self.motion, self.quiet, 4, "cpu", motion_seconds=20.,
                      engine_factory=NativeDummyEngine, control_mode="target_pd")

    def test_precision_reward_only_changes_motion_and_keeps_test_semantics(self):
        task = self.Task(self.rig, self.motion, self.quiet, 4, "cpu", reward_mode="precision_v1",
                         engine_factory=NativeDummyEngine, control_mode="target_pd", contact_mode="support_v2")
        task.set_mode(self.Mode.TRAIN)
        task.reset()
        q, v = task.sample_reference(task.offset)
        q[:, 0] += .01
        task.engine.reset_envs(task.ids, q, v)
        base = task.reference_reward()
        reward = task.reward()
        torch.testing.assert_close(reward[:2], base[:2])
        self.assertTrue(bool((reward[2:] < base[2:] - .3).all()))
        task.set_mode(self.Mode.TEST)
        task.reset()
        q, v = task.reference.sample(task.offset)
        q[:, 0] += .01
        task.engine.reset_envs(task.ids, q, v)
        self.assertTrue(bool((task.reward() < task.reference_reward() - .3).all()))


if __name__ == "__main__":
    unittest.main()
