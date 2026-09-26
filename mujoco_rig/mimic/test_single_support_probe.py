"""A one-leg hold cannot pass by quietly putting the swing foot down."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .probe_single_support import PHYSICAL_CHECKS, hold_metrics
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


    def test_coronal_sagittal_stabilizer_output_and_feedback(self):
        from .probe_single_support import CoronalSagittalStabilizer
        from .rig import Rig
        rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        eq_torque = np.zeros(len(rig.actuators))
        target_com = [0.0, 0.137, 0.92]
        data = SimpleNamespace(subtree_com=np.array([[0., 0., 0.], [0.05, 0.10, 0.90]]))
        engine = SimpleNamespace(
            datas=[data],
            get_root_vel=lambda _: torch.zeros((1, 3)),
            get_root_ang_vel=lambda _: torch.zeros((1, 3)),
        )
        task = SimpleNamespace(rig=rig, engine=engine, dt=0.01667)
        stabilizer = CoronalSagittalStabilizer(task, eq_torque, target_com)
        obs = torch.zeros((1, 10))
        action = stabilizer(obs)
        self.assertEqual(action.shape, (1, len(rig.actuators)))
        self.assertTrue(bool((action >= -1.0).all() and (action <= 1.0).all()))
        # Verify non-zero feedback corrections were applied to hip and ankle actuators
        names = [rig.model.actuator(int(a)).name for a in rig.actuators]
        self.assertNotEqual(float(action[0, names.index("Foot_R_rx")]), 0.0)
        self.assertNotEqual(float(action[0, names.index("Thigh_R_rz")]), 0.0)
        self.assertNotEqual(float(action[0, names.index("Foot_R_rz")]), 0.0)
        stabilizer.set_target(np.ones(len(rig.actuators)) * .25, target_com)
        self.assertAlmostEqual(float(stabilizer.eq[names.index("Foot_R_rx")]), .25)
        np.testing.assert_array_equal(stabilizer.target_com, target_com)

    def test_physical_pose_uses_declared_support_feet(self):
        from .probe_single_support import physical_pose
        rig = SimpleNamespace(model=SimpleNamespace(nv=6))
        arrays = dict(qpos=np.zeros((1, 7)), foot_targets=np.zeros((1, 2, 3)),
                      sole_contact=np.zeros((1, 2)))
        support_cases = (([True, False], ("Foot_L", "Toe_L")),
                         ([False, True], ("Foot_R", "Toe_R")),
                         ([True, True], ("Foot_L", "Foot_R", "Toe_L", "Toe_R")))
        with patch("mujoco_rig.mimic.probe_single_support.settle_passive_joints",
                   return_value=(arrays["qpos"][0], {"passed": True})), \
             patch("mujoco_rig.mimic.probe_single_support.validate_motion",
                   return_value=({name: True for name in PHYSICAL_CHECKS}, {})), \
             patch("mujoco_rig.mimic.probe_single_support.solve_support",
                   return_value={"feasible": True}) as solve:
            for contacts, expected in support_cases:
                with self.subTest(contacts=contacts):
                    arrays["contact"] = np.asarray([contacts])
                    self.assertTrue(physical_pose(rig, arrays, 0)["passed"])
                    self.assertEqual(solve.call_args.kwargs["support_names"], expected)

    def test_physical_pose_passes_passive_and_static_support(self):
        from .probe_single_support import physical_pose
        from .rig import Rig
        rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        ref_path = Path(__file__).resolve().parents[2] / (
            "logs/mimickit-steps/placed-step-forward-09/diagnostic_poses.npz")
        if not ref_path.exists():
            self.skipTest("placed-step diagnostic artifact is unavailable")
        with np.load(ref_path) as arrays:
            frame = round(9.0 / float(arrays["dt"]))
            res = physical_pose(rig, arrays, frame)
            self.assertTrue(res["passed"])
            self.assertTrue(res["checks"]["passive_equilibrium"])
            self.assertTrue(res["checks"]["static_support"])


if __name__ == "__main__":
    unittest.main()
