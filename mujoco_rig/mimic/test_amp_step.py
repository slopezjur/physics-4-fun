"""AMP feature timing and demonstration-license contract tests."""
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import torch

from .rig import Rig
from .runtime import activate


class AMPStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("MIMICKIT_PATH"):
            raise unittest.SkipTest("Set MIMICKIT_PATH for AMP adapter tests")
        activate(Path(os.environ["MIMICKIT_PATH"]))
        from .amp_step import AMPStepTask, discriminator_state, load_demonstrations, reference_state
        from .native_engine import NativeDummyEngine
        from .stand_task import StandTask
        cls.discriminator_state = staticmethod(discriminator_state)
        cls.reference_state = staticmethod(reference_state)
        cls.load_demonstrations = staticmethod(load_demonstrations)
        cls.AMPStepTask = AMPStepTask
        cls.NativeDummyEngine, cls.StandTask = NativeDummyEngine, StandTask
        cls.asset = Path(__file__).parent / "assets/stand_reference.npz"
        cls.rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")

    def test_reference_and_live_features_share_velocity_frame(self):
        q = torch.zeros((1, self.rig.model.nq))
        q[:, 2] = 1.
        q[:, 3] = 1.
        v = torch.randn((1, self.rig.model.nv))
        live = self.discriminator_state(q, v[:, :3], v[:, 3:6], v[:, 6:])
        demo = self.reference_state(q, v)
        torch.testing.assert_close(live, demo)
        self.assertEqual(live.shape, (1, 91))

    def test_demonstration_rejects_noncommercial_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "motion.npz"
            shutil.copyfile(self.asset, path)
            manifest = json.loads(self.asset.with_suffix(".json").read_text())
            manifest["source"]["license"] = "CC-BY-NC-4.0"
            path.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "license"):
                self.load_demonstrations([path], self.rig, "cpu")

    def test_native_reset_matches_reference_motion_features(self):
        task = self.StandTask(self.rig, self.asset, 1, "cpu",
                              engine_factory=self.NativeDummyEngine,
                              control_mode="target_pd", contact_mode="support_v2")
        q, v = task.reference.sample(task.offset)
        engine = task.engine
        live = self.discriminator_state(engine.native_qpos(), engine.get_root_vel(0),
                                        engine.get_root_ang_vel(0), engine.get_dof_vel(0))
        torch.testing.assert_close(live, self.reference_state(q, v), rtol=1e-5, atol=1e-6)

    def test_reference_features_are_finite_with_expected_width(self):
        ref = self.load_demonstrations([self.asset], self.rig, "cpu")[0]
        q, v = ref.sample(torch.tensor([1., 1.05]))
        pair = torch.cat((self.reference_state(q, v), self.reference_state(q, v)), dim=-1)
        self.assertEqual(pair.shape, (2, 182))
        self.assertTrue(torch.isfinite(pair).all())

    def test_pre_lift_reward_prefers_support_weight_transfer(self):
        desired = torch.tensor([[[0., .10, .9], [0., -.15, .0], [0., .15, .0]]])
        unshifted = desired.clone()
        unshifted[:, 0, 1] = 0.
        root = torch.zeros((1, 46))
        root[:, 2:4] = torch.tensor([.9, 1.])
        current = [unshifted]
        fake = SimpleNamespace(
            sample_reference=lambda _: (root, None),
            mapper=SimpleNamespace(from_native_qpos=lambda _: (desired, None)),
            engine=SimpleNamespace(get_body_pos=lambda _: current[0], native_qpos=lambda: root,
                                   get_root_vel=lambda _: torch.zeros((1, 3))),
            feet=torch.tensor([1, 2]), _body_mass=torch.tensor([10., 1., 1.]),
            offset=torch.zeros(1), steps=torch.tensor([240]), dt=1 / 60,
            _diagnostics={})
        before = self.AMPStepTask.reward(fake)
        before_error = fake._diagnostics["support_com_error_m"].clone()
        current[0] = desired
        after = self.AMPStepTask.reward(fake)
        self.assertGreater(float(after), float(before))
        self.assertGreater(float(before_error), float(fake._diagnostics["support_com_error_m"]))


if __name__ == "__main__":
    unittest.main()
