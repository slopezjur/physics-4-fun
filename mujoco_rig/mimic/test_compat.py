"""Regression tests for the physical/control boundary, not training outcomes."""
from pathlib import Path
import os
import unittest

import mujoco
import numpy as np
import torch

from .rig import DelayedTorqueControl, Rig
from .runtime import activate
from .model_contract import model_checks

ROOT = Path(__file__).resolve().parents[2]


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.rig = Rig.load(ROOT / "mujoco_rig/dummy.xml")
        self.control = DelayedTorqueControl(self.rig, 2, "cpu")

    def test_delay_clip_neck_and_partial_reset(self):
        actions = torch.full((2, len(self.rig.actuators)), 4.0)
        self.assertEqual(torch.count_nonzero(self.control.apply(actions)), 0)
        self.assertEqual(torch.count_nonzero(self.control.apply(actions)), 0)
        ctrl = self.control.apply(actions)
        np.testing.assert_allclose(ctrl[0, self.rig.actuators], self.rig.action_scale)
        neck = [i for i in range(self.rig.model.nu) if i not in self.rig.actuators]
        self.assertEqual(torch.count_nonzero(ctrl[:, neck]), 0)
        self.control.reset([0])
        ctrl = self.control.apply(torch.zeros_like(actions))
        self.assertEqual(torch.count_nonzero(ctrl[0]), 0)
        np.testing.assert_allclose(ctrl[1, self.rig.actuators], self.rig.action_scale)

    def test_reset_discards_pending_commands(self):
        actions = torch.ones((2, len(self.rig.actuators)))
        self.control.apply(actions)
        self.control.reset()
        for _ in range(4):
            self.assertEqual(torch.count_nonzero(self.control.apply(torch.zeros_like(actions))), 0)

    def test_invalid_actions_do_not_enter_queue(self):
        with self.assertRaises(ValueError):
            self.control.apply(torch.zeros(2, 39))
        with self.assertRaises(ValueError):
            self.control.apply(torch.full((2, 30), float("nan")))
        self.assertEqual(torch.count_nonzero(self.control.pending), 0)

    def test_model_gate_rejects_changed_mechanics(self):
        other = mujoco.MjModel.from_xml_path(str(self.rig.path))
        self.assertTrue(all(model_checks(self.rig.model, other).values()))
        other.dof_damping[6] *= 2
        self.assertFalse(model_checks(self.rig.model, other)["dof_damping"])
        other.actuator_gear[0, 0] = 2
        self.assertFalse(model_checks(self.rig.model, other)["actuator_gear"])


class MapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkout = os.environ.get("MIMICKIT_PATH")
        if not checkout:
            raise RuntimeError("Set MIMICKIT_PATH to the pinned MimicKit checkout")
        activate(Path(checkout))
        from .kinematics import DummyKinematics
        cls.rig = Rig.load(ROOT / "mujoco_rig/dummy.xml")
        cls.mapper = DummyKinematics(cls.rig, dtype=torch.float64)

    def test_offset_pivots_and_hinge_order_against_native(self):
        model = self.rig.model
        rng = np.random.default_rng(917)
        poses = np.tile(self.rig.rest, (16, 1))
        poses[:, 7:] = rng.uniform(model.jnt_range[1:, 0], model.jnt_range[1:, 1], (16, model.nq - 7))
        positions, rotations = self.mapper.from_native_qpos(torch.tensor(poses))
        for i, pose in enumerate(poses):
            data = mujoco.MjData(model)
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(positions[i], data.xpos[1:], atol=1e-12)
            dots = (rotations[i].numpy() * data.xquat[1:, [1, 2, 3, 0]]).sum(axis=-1)
            np.testing.assert_allclose(np.abs(dots), 1, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
