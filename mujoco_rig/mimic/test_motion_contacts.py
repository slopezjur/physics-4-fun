"""Toe-off anchors, explicit support admission and backward-compatible defaults."""
from pathlib import Path
import json
import tempfile
import unittest

import mujoco
import numpy as np

from .contact_projection import StanceProjection, bounded_ik_step
from .motion_contacts import (infer_toe_off, sole_schedule, validate_sole_schedule, TOE_REFERENCE_SCHEMA,
                              stance_phase_ids, expand_landing_offsets)
from .motion_dynamics import support_columns
from .motion_protocol import validate_motion_reference
from .retarget_steps import validate_motion
from .rig import Rig
from .baseline import sha256
from .reference import StandReference


class MotionContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")

    def test_bound_active_ik_redistributes_correction_instead_of_clipping_it_away(self):
        jac = np.array([[1., 1.], [0., .1]])
        error = np.array([1., 0.])
        limits = np.array([[-1., 0.], [-2., 2.]])
        position = np.array([-1e-6, 0.])
        step = bounded_ik_step(jac, error, position, limits, max_step=2.)
        self.assertLessEqual(step[0], 1e-12)
        self.assertGreater(step[1], .98)
        self.assertLess(np.linalg.norm(jac @ step - error), .11)

    def test_landing_adjustments_are_constant_through_stance_and_zero_in_swing(self):
        contact = np.array([[True, False], [True, True], [False, True], [True, True]])
        phases, count = stance_phase_ids(contact)
        self.assertEqual(count, 3)
        offsets = np.array([[.01, -.02], [-.01, .02], [.005, 0.]])
        expanded = expand_landing_offsets(phases, offsets)
        np.testing.assert_equal(expanded[:2, 0], np.tile(offsets[0], (2, 1)))
        np.testing.assert_equal(expanded[3, 0], offsets[1])
        np.testing.assert_equal(expanded[1:, 1], np.tile(offsets[2], (3, 1)))
        np.testing.assert_equal(expanded[~contact], 0.)

    def test_adjusted_landing_keeps_toe_pivot_fixed_and_validation_targets_original(self):
        m = self.rig.model
        with np.load(Path(__file__).parent / "assets/stand_reference.npz") as source:
            poses = np.tile(source["qpos"][0], (3, 1))
        d = mujoco.MjData(m)
        d.qpos[:] = poses[0]
        mujoco.mj_kinematics(m, d)
        feet = [m.body("Foot_" + side).id for side in ("L", "R")]
        targets = np.tile(d.xpos[feet], (3, 1, 1))
        poses[:, 2] -= .02
        poses[:, m.jnt_qposadr[m.joint("Toe_L_rx").id]] = [0., .2, .35]
        contact = np.ones((3, 2), dtype=bool)
        toe_only = np.array([[False, False], [True, False], [True, False]])
        offsets = np.tile([[.005, -.005], [0., 0.]], (3, 1, 1))
        projection = StanceProjection(self.rig, poses, targets, contact, toe_only, offsets)
        fitted = projection.project(poses)
        toe_positions = []
        for pose in fitted:
            d.qpos[:] = pose
            mujoco.mj_kinematics(m, d)
            toe_positions.append(d.xpos[m.body("Toe_L").id].copy())
        expected = projection.anchors[1, 0] + np.r_[offsets[1, 0], 0.]
        np.testing.assert_allclose(toe_positions, np.tile(expected, (3, 1)), atol=1e-5)
        np.testing.assert_equal(projection.targets[:, :, :2], targets[:, :, :2])

    def test_toe_pivot_lifts_heel_without_sliding_or_inventing_foot_support(self):
        m = self.rig.model
        with np.load(Path(__file__).parent / "assets/stand_reference.npz") as source:
            poses = np.tile(source["qpos"][0], (3, 1))
        d = mujoco.MjData(m)
        d.qpos[:] = poses[0]
        mujoco.mj_kinematics(m, d)
        feet = [m.body("Foot_" + s).id for s in ("L", "R")]
        targets = np.tile(d.xpos[feet], (3, 1, 1))
        poses[:, 2] -= .02
        toe = m.jnt_qposadr[m.joint("Toe_L_rx").id]
        poses[:, toe] = [0, .2, .35]
        contact = np.ones((3, 2), dtype=bool)
        toe_only = np.array([[False, False], [True, False], [True, False]])
        projection = StanceProjection(self.rig, poses, targets, contact, toe_only)
        fitted = projection.project(poses)
        toe_positions, foot_heights = [], []
        for pose in fitted:
            d.qpos[:] = pose
            mujoco.mj_kinematics(m, d)
            toe_positions.append(d.xpos[m.body("Toe_L").id].copy())
            foot_heights.append(d.xpos[feet[0], 2])
        np.testing.assert_allclose(toe_positions, np.tile(toe_positions[0], (3, 1)), atol=1e-5)
        self.assertGreater(foot_heights[-1] - foot_heights[0], .015)
        soles = sole_schedule(contact, toe_only)
        checks, metrics = validate_motion(self.rig, fitted, np.zeros((3, m.nv)), targets, contact, .04, soles)
        self.assertTrue(checks["sole_contact"])
        self.assertTrue(checks["stance_slide"])
        self.assertLess(metrics["max_stance_slide_m_s"], 1e-3)
        columns = support_columns(m, d, np.array([False, False, True, False]))
        toe_dof = m.jnt_dofadr[m.joint("Toe_R_rx").id]
        np.testing.assert_equal(columns[toe_dof], 0)
        self.assertEqual(support_columns(m, d, np.zeros(4, dtype=bool)).shape, (m.nv, 0))
        # A shallow heel lift leaves only the front edge within the contact tolerance.
        shallow = poses.copy()
        shallow[1:, toe] = .1
        d.qpos[:] = projection.project(shallow)[1]
        mujoco.mj_forward(m, d)
        explicit = support_columns(m, d, np.array([True, False, False, False]))
        self.assertEqual(explicit.shape[1], 8, "Only two near-floor corners may supply four friction rays each")

    def test_only_heel_up_contact_gaps_get_toe_off(self):
        original = np.array([[True, False], [True, False], [False, False], [False, True]])
        completed = original.copy()
        completed[2, 0] = True
        rotations = np.tile(np.eye(3), (4, 2, 1, 1))
        rotations[2, 0, 2, 0] = -.2
        toe = infer_toe_off(original, completed, rotations)
        np.testing.assert_equal(toe[:, 0], [True, True, True, False])
        rotations[2, 0, 2, 0] = .2
        self.assertFalse(infer_toe_off(original, completed, rotations).any())

    def test_inconsistent_contact_phase_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_sole_schedule(np.ones((3, 2), dtype=bool), np.zeros((3, 4), dtype=bool))
        with self.assertRaises(ValueError):
            sole_schedule(np.zeros((3, 2), dtype=bool), np.ones((3, 2), dtype=bool))

    def test_v6_cannot_claim_v5_validation_without_sole_checks(self):
        metadata = dict(schema=TOE_REFERENCE_SCHEMA, passed=True, checks={k: True for k in
                        ("ground_support", "stance_contact", "force_consistency", "root_wrench_consistency", "actuation_feasibility")})
        with self.assertRaisesRegex(ValueError, "sole-contact"):
            validate_motion_reference(metadata)
        metadata["checks"]["sole_contact"] = True
        validate_motion_reference(metadata)
        metadata["force_fit"] = dict(self_clearance_m=.005)
        with self.assertRaisesRegex(ValueError, "self-clearance"):
            validate_motion_reference(metadata)
        metadata["checks"]["self_clearance"] = True
        validate_motion_reference(metadata)

    def test_clearance_gate_exposes_near_collisions_without_lifting_floor_support(self):
        path = Path(__file__).parent / "assets/steps/backward_reference.npz"
        with np.load(path) as a:
            checks, metrics = validate_motion(self.rig, a["qpos"], a["qvel"], a["foot_targets"],
                                              a["contact"], float(a["dt"]), self_clearance_m=.005)
        self.assertTrue(checks["ground_support"])
        self.assertFalse(checks["self_clearance"])
        self.assertLess(metrics["self_clearance"]["minimum_m"], .005)
        self.assertNotIn("floor", metrics["self_clearance"]["closest_pair"])

    def test_v6_loader_requires_one_explicit_phase_per_pose(self):
        asset = Path(__file__).parent / "assets/stand_reference.npz"
        with np.load(asset) as source:
            arrays = {key: source[key] for key in source.files}
        metadata = json.loads(asset.with_suffix(".json").read_text())
        metadata.update(schema=TOE_REFERENCE_SCHEMA, passed=True, checks={key: True for key in
                        ("ground_support", "stance_contact", "force_consistency", "root_wrench_consistency",
                         "actuation_feasibility", "sole_contact")})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.npz"

            def write():
                np.savez_compressed(path, **arrays)
                metadata["reference_sha256"] = sha256(path)
                path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")

            arrays["contact"] = np.ones((len(arrays["qpos"]) - 1, 2), dtype=bool)
            arrays["sole_contact"] = np.ones((len(arrays["contact"]), 4), dtype=bool)
            write()
            with self.assertRaisesRegex(ValueError, "every pose"):
                StandReference(path, self.rig, "cpu")
            arrays["contact"] = np.ones((len(arrays["qpos"]), 2), dtype=bool)
            arrays["sole_contact"] = np.ones((len(arrays["qpos"]), 4), dtype=bool)
            write()
            loaded = StandReference(path, self.rig, "cpu")
            np.testing.assert_allclose(loaded.qpos.numpy(), arrays["qpos"], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
