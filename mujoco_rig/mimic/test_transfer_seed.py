"""Warm starts retain source provenance, physical bounds and fixed history."""
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from .baseline import sha256
from .force_retarget import refine_support, rotate_root
from .rig import Rig
from .transfer_seed import load_seed_probe, rotation_offsets
from .transfer_calibration import interpolate_poses, stopped_interpolation


class TransferSeedTests(unittest.TestCase):
    def test_world_rotation_offsets_reconstruct_root_and_ignore_quaternion_sign(self):
        original = np.array([[1., 0., 0., 0.], [np.sqrt(.5), 0., np.sqrt(.5), 0.]])
        offsets = np.array([[.03, -.02, .01], [-.04, .02, .03]])
        seed = rotate_root(original, offsets)
        np.testing.assert_allclose(rotation_offsets(original, seed), offsets, atol=1e-12)
        np.testing.assert_allclose(rotation_offsets(original, -seed), offsets, atol=1e-12)

    def test_smooth_pose_interpolation_preserves_unit_roots_and_static_coordinates(self):
        q = np.zeros((3, 10))
        q[:, 3] = 1.
        q[:, 3:7] = rotate_root(q[:, 3:7], np.array([[0., 0., 0.], [.1, -.1, 0.], [0., 0., 0.]]))
        q[:, 7:] = [.2, .3, .4]
        result = interpolate_poses(q, [0., 1., 2.], np.linspace(0, 2, 21))
        np.testing.assert_allclose(np.linalg.norm(result[:, 3:7], axis=1), 1., atol=1e-12)
        np.testing.assert_allclose(result[[0, 10, 20]], q, atol=1e-12)
        np.testing.assert_allclose(result[:, 7:], np.tile(q[0, 7:], (21, 1)))

    def test_stationary_fit_cannot_manufacture_support_through_acceleration(self):
        from .force_retarget import reference_velocities
        r = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        with np.load(Path(__file__).parent / "assets/stand_reference.npz") as arrays:
            q = np.tile(arrays["qpos"][0], (3, 1))
        d = mujoco.MjData(r.model)
        d.qpos[:] = q[0]
        mujoco.mj_kinematics(r.model, d)
        feet = [r.model.body(n).id for n in ("Foot_L", "Foot_R")]
        targets = np.tile(d.xpos[feet], (3, 1, 1))
        contact = np.ones((3, 2), dtype=bool)
        result, _, _, fit = refine_support(r, q, targets, contact, .05, stationary=True, max_evaluations=2)
        np.testing.assert_array_equal(result, np.tile(result[0], (3, 1)))
        np.testing.assert_allclose(reference_velocities(r.model, result, .05), 0., atol=1e-12)
        self.assertTrue(fit["stationary"])
        with self.assertRaisesRegex(ValueError, "post-delay"):
            refine_support(r, q, targets, contact, .01, stationary=True)
        headings = np.tile(np.eye(3), (3, 2, 1, 1))
        headings[1, 0] = np.diag([-1., -1., 1.])
        with self.assertRaisesRegex(ValueError, "identical poses"):
            refine_support(r, q, targets, contact, .05, stationary=True, stance_rotations=headings)
        q[1, 0] += .001
        with self.assertRaisesRegex(ValueError, "identical poses"):
            refine_support(r, q, targets, contact, .05, stationary=True)

    def test_stopped_segments_do_not_overshoot_and_have_smooth_joins(self):
        values = np.array([[0., 1.], [1., 0.], [0., 1.]])
        times = [0., 1., 3.]
        result = stopped_interpolation(values, times, np.linspace(0, 3, 301))
        self.assertGreaterEqual(result.min(), -1e-12)
        self.assertLessEqual(result.max(), 1. + 1e-12)
        np.testing.assert_allclose(result[[0, 100, 300]], values, atol=1e-12)
        h = 1e-4
        near = stopped_interpolation(values, times, [1-h, 1., 1+h])
        np.testing.assert_allclose((near[2] - near[0]) / (2*h), 0., atol=1e-6)
        np.testing.assert_allclose((near[0] - 2*near[1] + near[2]) / h**2, 0., atol=.002)
        q = np.zeros((2, 10)); q[:, 3] = 1.
        q[:, 3:7] = rotate_root(q[:, 3:7], np.array([[0., 0., 0.], [.2, -.1, .1]]))
        rotations = stopped_interpolation(q, [0., 1.], np.linspace(0, 1, 11), root_rotations=True)
        np.testing.assert_allclose(np.linalg.norm(rotations[:, 3:7], axis=1), 1., atol=1e-12)
        np.testing.assert_allclose(rotations[[0, -1]], q, atol=1e-12)
        for invalid in [[0., 0., 1.], [0., 2., 1.]]:
            with self.assertRaises(ValueError):
                stopped_interpolation(values, invalid, [0.])
        with self.assertRaises(ValueError):
            stopped_interpolation(values, times, [-.01])

    def test_probe_provenance_timestep_and_window_are_required(self):
        q = np.zeros((7, 9))
        q[:, 3] = 1.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            seed = q[2:5].copy()
            seed[:, 0] = .01
            np.savez(path / "diagnostic_poses.npz", qpos=seed, dt=.04)
            report = dict(schema="mimic_transition_probe_v1", source_reference_sha256="source",
                          model_sha256="rig", source_frame_start=2, source_frame_stop_exclusive=5)
            (path / "report.json").write_text(json.dumps(report))
            result, record = load_seed_probe(path, "source", "rig", q, .04, (1, 6))
            np.testing.assert_array_equal(result[[0, 1, 5, 6]], q[[0, 1, 5, 6]])
            np.testing.assert_array_equal(result[2:5], seed)
            self.assertFalse(record["historical_content_hash_verified"])
            for reference, model, dt, window in [("other", "rig", .04, (1, 6)),
                                                 ("source", "other", .04, (1, 6)),
                                                 ("source", "rig", .05, (1, 6)),
                                                 ("source", "rig", .04, (3, 6))]:
                with self.assertRaises(ValueError):
                    load_seed_probe(path, reference, model, q, dt, window)
            report["diagnostic_sha256"] = sha256(path / "diagnostic_poses.npz")
            (path / "report.json").write_text(json.dumps(report))
            self.assertTrue(load_seed_probe(path, "source", "rig", q, .04, (1, 6))[1]["historical_content_hash_verified"])
            np.savez(path / "diagnostic_poses.npz", qpos=seed, dt=.04, changed=True)
            with self.assertRaisesRegex(ValueError, "content hash"):
                load_seed_probe(path, "source", "rig", q, .04, (1, 6))

    def test_seed_initializes_rotation_without_replacing_targets_or_fixed_poses(self):
        r = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        with np.load(Path(__file__).parent / "assets/stand_reference.npz") as arrays:
            q = np.tile(arrays["qpos"][0], (5, 1))
        d = mujoco.MjData(r.model)
        d.qpos[:] = q[0]
        mujoco.mj_kinematics(r.model, d)
        feet = [r.model.body(n).id for n in ("Foot_L", "Foot_R")]
        targets = np.tile(d.xpos[feet], (5, 1, 1))
        for foot, name in enumerate(("g_Foot_L", "g_Foot_R")):
            geom = r.model.geom(name).id
            targets[:, foot, 2] = r.model.geom_size[geom, 2] - r.model.geom_pos[geom, 2]
        seed, fixed = q.copy(), np.array([True, True, False, True, True])
        seed[2:3, 3:7] = rotate_root(q[2:3, 3:7], np.array([[.005, -.004, .003]]))
        seed[2, 0] += .002
        result, returned, _, report = refine_support(r, q, targets, np.ones((5, 2), dtype=bool), .04,
                                                     max_evaluations=1, fixed_frames=fixed, initial_poses=seed,
                                                     root_seed_drop_m=0., linear_solver="exact")
        np.testing.assert_array_equal(result[fixed], q[fixed])
        np.testing.assert_allclose(result[2, :7], seed[2, :7], atol=1e-12)
        np.testing.assert_array_equal(returned, targets)
        self.assertEqual(report["clipped_initial_coordinates"], 0)
        broken = seed.copy()
        broken[0, 0] += .001
        with self.assertRaisesRegex(ValueError, "fixed context"):
            refine_support(r, q, targets, np.ones((5, 2), dtype=bool), .04,
                           fixed_frames=fixed, initial_poses=broken, root_seed_drop_m=0.)
        controlled = r.model.jnt_qposadr[r.model.actuator_trnid[r.actuators, 0]]
        passive = np.setdiff1d(np.arange(7, r.model.nq), controlled)
        broken = seed.copy()
        broken[2, passive[0]] += .001
        with self.assertRaisesRegex(ValueError, "uncontrolled"):
            refine_support(r, q, targets, np.ones((5, 2), dtype=bool), .04,
                           fixed_frames=fixed, initial_poses=broken, root_seed_drop_m=0.)


if __name__ == "__main__":
    unittest.main()
