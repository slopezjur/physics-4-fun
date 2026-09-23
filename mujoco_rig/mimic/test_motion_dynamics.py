"""Reject force-inconsistent references even when every frame touches the floor."""
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import torch
import mujoco

from .motion_dynamics import force_consistency, root_wrench_consistency, action_window_consistency, delayed_pd_bounds, frame_dynamics, audit
from .target_control import DelayedTargetControl
from .motion_protocol import REFERENCE_SCHEMA, validate_motion_reference
from .rig import Rig


class MotionDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")

    def test_constant_velocity_is_not_mistaken_for_acceleration(self):
        poses = np.tile(self.rig.rest, (10, 1))
        poses[:, 0] += np.arange(10) * .1
        self.assertTrue(force_consistency(self.rig, poses, .03)["passed"])

    def test_delayed_bounds_match_runtime_pd_at_action_extremes(self):
        poses = np.tile(self.rig.rest, (6, 1))
        joints = self.rig.model.actuator_trnid[self.rig.actuators, 0]
        qi = self.rig.model.jnt_qposadr[joints]
        # Include joint-limit clipping and a moving reference.
        poses[:, qi[0]] = self.rig.model.jnt_range[joints[0], 1] - .01 - np.arange(6) * .002
        reference_v = np.zeros((6, self.rig.model.nv))
        reference_v[:, self.rig.model.jnt_dofadr[joints]] = np.arange(6)[:, None] * .03
        dt, frame = .04, 3
        bounds = delayed_pd_bounds(self.rig, poses, reference_v, dt, frame)
        data = mujoco.MjData(self.rig.model)
        frame_dynamics(self.rig.model, data, poses, frame, dt)
        delay = 2 * self.rig.decimation * self.rig.model.opt.timestep
        times = np.arange(6) * dt
        issued = np.array([np.interp(frame * dt - delay, times, poses[:, i]) for i in range(self.rig.model.nq)])
        issued_v = np.array([np.interp(frame * dt - delay, times, reference_v[:, i])
                             for i in range(self.rig.model.nv)])
        for column, action in enumerate((-1., 1.)):
            control = DelayedTargetControl(self.rig, 1, "cpu")
            control.set_reference(torch.tensor(issued[None], dtype=torch.float32),
                                  torch.tensor(issued_v[None], dtype=torch.float32))
            for _ in range(len(control.pending) + 1):
                control.apply(torch.full((1, len(qi)), action))
            force = control.physics_control(torch.tensor(data.qpos[None], dtype=torch.float32),
                                            torch.tensor(data.qvel[None], dtype=torch.float32))
            np.testing.assert_allclose(force[0, self.rig.actuators].numpy() / self.rig.action_scale,
                                       bounds[:, column], atol=2e-6)

    def test_grounded_horizontal_impulse_exceeds_friction(self):
        poses = np.tile(self.rig.rest, (10, 1))
        poses[5:, 0] += .1
        report = force_consistency(self.rig, poses, .03)
        self.assertFalse(report["passed"])
        self.assertEqual(report["failing_frames"], [4, 5])

    def test_floor_cannot_pull_a_downward_accelerating_body(self):
        poses = np.tile(self.rig.rest, (10, 1))
        time = np.arange(10) * .03
        poses[:, 2] -= 10 * time**2
        report = force_consistency(self.rig, poses, .03)
        self.assertFalse(report["passed"])
        self.assertLess(report["min_normal_force_per_mass_m_s2"], 0)

    def test_old_contact_only_manifest_cannot_start_training(self):
        metadata = dict(schema=REFERENCE_SCHEMA, passed=True,
                        checks=dict(ground_support=True, stance_contact=True))
        with self.assertRaisesRegex(ValueError, "force-consistency"):
            validate_motion_reference(metadata)
        metadata["checks"]["force_consistency"] = True
        with self.assertRaisesRegex(ValueError, "root-wrench"):
            validate_motion_reference(metadata)
        metadata["checks"]["root_wrench_consistency"] = True
        with self.assertRaisesRegex(ValueError, "actuation-feasibility"):
            validate_motion_reference(metadata)
        metadata["checks"]["actuation_feasibility"] = True
        validate_motion_reference(metadata)
        metadata["schema"] = "mimic_step_reference_v4"
        with self.assertRaises(ValueError):
            validate_motion_reference(metadata)
        metadata["schema"] = "mimic_step_reference_v2"
        with self.assertRaises(ValueError):
            validate_motion_reference(metadata)

    def test_wrench_detects_missing_support_even_with_valid_aggregate_force(self):
        poses = np.tile(self.rig.rest, (3, 1))
        poses[:, 2] += .1
        self.assertTrue(force_consistency(self.rig, poses, .03)["passed"])
        self.assertEqual(root_wrench_consistency(self.rig, poses, .03)["infeasible_frames"], [1])

    def test_static_reference_support_is_wrench_feasible(self):
        path = Path(__file__).parent / "assets/stand_reference.npz"
        with np.load(path, allow_pickle=False) as reference:
            pose = reference["qpos"][0]
        report = root_wrench_consistency(self.rig, np.tile(pose, (3, 1)), .03)
        self.assertTrue(report["all_frames_feasible"], report)

    def test_file_audit_honors_explicit_soles_and_preserves_legacy_support(self):
        with np.load(Path(__file__).parent / "assets/stand_reference.npz") as source:
            poses = np.tile(source["qpos"][0], (5, 1))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy, explicit = root / "legacy.npz", root / "explicit.npz"
            arrays = dict(qpos=poses, qvel=np.zeros((5, self.rig.model.nv)), dt=.04)
            np.savez(legacy, **arrays)
            # A grounded pose cannot borrow forces from soles declared inactive.
            np.savez(explicit, **arrays, sole_contact=np.zeros((5, 4), dtype=bool))
            audit([legacy, explicit], root / "report.json")
            rows = json.loads((root / "report.json").read_text())["results"]
            self.assertTrue(rows[0]["root_wrench"]["all_frames_feasible"])
            self.assertEqual(rows[1]["root_wrench"]["infeasible_frames"], [1, 2, 3])
            self.assertEqual(rows[1]["action_window"]["root_supported_frames"], [])

    def test_action_window_cannot_substitute_for_missing_ground_support(self):
        poses = np.tile(self.rig.rest, (5, 1))
        poses[:, 2] += .1
        report = action_window_consistency(self.rig, poses, np.zeros((5, self.rig.model.nv)), .04)
        self.assertEqual(report["root_supported_frames"], [])
        self.assertEqual(report["failing_frames"], {"torque": [], "0.25": [], "0.5": []})

    def test_wider_window_cannot_remove_previously_feasible_commands(self):
        with np.load(Path(__file__).parent / "assets/steps/backward_reference.npz") as source:
            report = action_window_consistency(self.rig, source["qpos"], source["qvel"], float(source["dt"]))
        failures = report["failing_frames"]
        self.assertTrue(set(failures["torque"]) <= set(failures["0.5"]) <= set(failures["0.25"]))
        self.assertGreater(len(report["root_supported_frames"]), 0)


if __name__ == "__main__":
    unittest.main()
