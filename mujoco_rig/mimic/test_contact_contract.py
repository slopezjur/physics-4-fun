"""Versioned loads and contact snapshots must not change legacy export semantics."""
import copy
from argparse import Namespace
import json
import os
from pathlib import Path
import unittest
import tempfile

import mujoco
import numpy as np
import torch

from .contact_contract import LEGACY, SUPPORT_V2, mode_from_contract
from .checkpoints import validate_contract
from .foundation_audit import sensor_poses
from .native_engine import NativeDummyEngine
from .rig import Rig
from .runtime import activate


class ContactContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("MIMICKIT_PATH"):
            raise unittest.SkipTest("Set MIMICKIT_PATH for task contract tests")
        activate(Path(os.environ["MIMICKIT_PATH"]))
        from .stand_task import StandTask
        from .ball_task import BallTask
        cls.StandTask, cls.BallTask = StandTask, BallTask
        cls.root = Path(__file__).resolve().parent
        cls.rig = Rig.load(cls.root.parent / "dummy.xml")

    def task(self, mode, ball=False):
        options = dict(contact_mode=mode)
        if not ball:
            options.update(engine_factory=NativeDummyEngine, control_mode="target_pd")
        return (self.BallTask if ball else self.StandTask)(
            self.rig, self.root / "assets/stand_reference.npz", 3, "cpu", **options)

    def test_same_width_does_not_make_contracts_compatible(self):
        old, new = (self.task(mode).observation_contract() for mode in (LEGACY, SUPPORT_V2))
        self.assertEqual(old["num_obs"], new["num_obs"])
        self.assertEqual(mode_from_contract(old), LEGACY)
        self.assertEqual(mode_from_contract(new), SUPPORT_V2)
        for actual, expected in ((old, new), (new, old)):
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                validate_contract(actual, expected)
        malformed = copy.deepcopy(new)
        malformed["support_bodies"][0].reverse()
        with self.assertRaises(ValueError):
            mode_from_contract(malformed)
        with self.assertRaises(ValueError):
            mode_from_contract({**old, "ground_contact_sampling": new["ground_contact_sampling"]})

    def test_toe_only_support_changes_only_the_two_load_channels(self):
        old, new = self.task(LEGACY), self.task(SUPPORT_V2)
        poses = torch.tensor(sensor_poses(self.rig), dtype=torch.float32)
        for task in (old, new):
            task.offset.zero_()
            task.engine.reset_envs(task.ids, poses, torch.zeros(3, self.rig.model.nv))
        before, after = old.observations(), new.observations()
        group = new.observation_contract()["observation_layout"][6]
        start = group["offset"]
        self.assertTrue(torch.all(before[2, start:start+2] == 0))
        self.assertTrue(torch.all(after[2, start:start+2] > .01))
        before[:, start:start+2] = after[:, start:start+2]
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_snapshot_survives_forward_and_partial_reset_preserves_peers(self):
        task = self.task(SUPPORT_V2)
        task.engine.reset_envs(task.ids, sensor_poses(self.rig), np.zeros((3, self.rig.model.nv)))
        task.step(torch.zeros(3, 30))
        engine = task.engine
        saved = engine.get_ground_contact_forces(0).clone()
        for data in engine.datas:
            data.qpos[2] += .1
            mujoco.mj_forward(engine.rig.model, data)
        torch.testing.assert_close(saved, engine.get_ground_contact_forces(0), rtol=0, atol=0)
        engine.reset_envs(torch.tensor([1]), self.rig.rest[None], np.zeros((1, self.rig.model.nv)))
        updated = engine.get_ground_contact_forces(0)
        torch.testing.assert_close(saved[[0, 2]], updated[[0, 2]], rtol=0, atol=0)
        np.testing.assert_allclose(updated[1], engine._read_ground_forces(engine.datas[1]), atol=1e-4)

    def test_ball_launch_preserves_v2_snapshot(self):
        task = self.task(SUPPORT_V2, ball=True)
        task.step(torch.zeros(3, 30))
        saved = task.engine.get_ground_contact_forces(0).clone()
        task.engine.launch_ball(torch.tensor([0]), [[3., 0., 2.]], [[-1., 0., 0.]])
        torch.testing.assert_close(saved, task.engine.get_ground_contact_forces(0), rtol=0, atol=0)

    def test_training_rejects_implicit_migration_and_diagnostic_normalization(self):
        from .train_stand import train
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "contract.json"
            args = Namespace(mimickit=Path(os.environ["MIMICKIT_PATH"]), envs=8, iterations=1,
                             seconds=1., evaluation_interval=8, initialize_from=Path(directory) / "best.pt",
                             source_contract=source, stability="guarded", contact_mode=SUPPORT_V2)
            source.write_text(json.dumps(self.task(LEGACY).observation_contract()))
            with self.assertRaisesRegex(ValueError, "not implicitly migrated"):
                train(args)
            contract = self.task(SUPPORT_V2).observation_contract()
            contract["diagnostic_only"] = True
            source.write_text(json.dumps(contract))
            with self.assertRaisesRegex(ValueError, "zero-shot diagnostic"):
                train(args)


if __name__ == "__main__":
    unittest.main()
