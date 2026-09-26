"""Support-transition reference and native retention gate checks."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .rig import Rig
from .support_transition import SupportTransitionReference, support_tracking_reward
from .train_support_transition import gate


class SupportTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        with np.load(Path(__file__).parent / "assets/stand_reference.npz") as arrays:
            cls.pose = arrays["qpos"][0].copy()

    def reference(self, mode):
        def sample(times):
            pose = np.repeat(self.pose[None, :], len(times), axis=0)
            pose[:, 0] += (times.numpy() - 3.25) * .1
            return torch.tensor(pose, dtype=torch.float32), None

        source = SimpleNamespace(placed_step=True, requested_step_m=.15, path=Path("source"),
                                 sha256="source-hash", report_sha256="report-hash", dt=.05,
                                 sample=sample)
        with patch("mujoco_rig.mimic.support_transition.CalibrationReference", return_value=source):
            return SupportTransitionReference(Path("source"), self.rig, mode=mode)

    def test_static_and_transition_endpoints_are_distinct_and_versioned(self):
        static = self.reference("static")
        entry = self.reference("transition")
        self.assertEqual(static.contract()["schema"], "mimic_prelift_support_gate_v1")
        self.assertNotEqual(static.sha256, entry.sha256)
        self.assertEqual(static.qpos.shape[0], 121)
        self.assertAlmostEqual(float(static.qpos[:, 0].max() - static.qpos[:, 0].min()), 0.)
        self.assertAlmostEqual(float(entry.qpos[0, 0] - self.pose[0]), 0., places=5)
        self.assertAlmostEqual(float(entry.qpos[-1, 0] - self.pose[0]), .025, places=5)
        self.assertAlmostEqual(float(entry.qpos[10, 0] - self.pose[0]), 0., places=5)
        self.assertAlmostEqual(float(entry.qpos[30, 0] - self.pose[0]), .025, places=5)
        self.assertTrue(torch.isfinite(entry.qvel).all())

    def test_gate_requires_full_survival_and_settled_tail(self):
        reference = self.reference("static")
        poses = np.repeat(reference.qpos[-1].cpu().numpy()[None, None, :], 120, axis=0)
        self.assertTrue(gate({"successes": 1}, {"qpos": poses}, reference, .05)["passed"])
        self.assertFalse(gate({"successes": 0}, {"qpos": poses}, reference, .05)["passed"])
        poses[-1, 0, 0] += .1
        self.assertFalse(gate({"successes": 1}, {"qpos": poses}, reference, .05)["passed"])

    def test_tracking_reward_values_horizontal_position_and_velocity(self):
        base = torch.ones(3)
        root = torch.tensor([[0., 0.], [.14, 0.], [0., 0.]])
        velocity = torch.tensor([[0., 0.], [0., 0.], [.12, 0.]])
        reward = support_tracking_reward(base, root, torch.zeros_like(root),
                                         velocity, torch.zeros_like(velocity))
        self.assertAlmostEqual(reward[0].item(), 1., places=6)
        self.assertLess(reward[1].item(), .82)
        self.assertLess(reward[2].item(), .96)
        self.assertTrue(torch.all(reward >= 0))


if __name__ == "__main__":
    unittest.main()
