"""Physical constraints and derivative semantics of force-aware retargeting."""
from pathlib import Path
import unittest

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from .force_retarget import complete_stance_targets, friction_rays, reference_velocities, refine_support, project_actuated_force, rotate_root
from .contact_projection import StanceProjection
from .motion_dynamics import root_wrench_consistency
from .rig import Rig


class ForceRetargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")

    def test_nonnegative_forces_cannot_pull_or_exceed_circular_friction(self):
        friction = .7
        coefficients = np.random.default_rng(23).uniform(0, 3, (100, 4))
        forces = np.vstack((friction_rays(friction), coefficients @ friction_rays(friction)))
        self.assertTrue((forces[:, 2] >= 0).all())
        self.assertTrue((np.linalg.norm(forces[:, :2], axis=1) <= friction * forces[:, 2] + 1e-12).all())

    def test_root_rotation_preserves_unit_quaternions_and_zero_offset(self):
        q = np.array([[1., 0., 0., 0.], [np.sqrt(.5), 0., np.sqrt(.5), 0.]])
        np.testing.assert_allclose(rotate_root(q, np.zeros((2, 3))), q)
        result = rotate_root(q, np.array([[0., 0., .15], [.1, -.12, .05]]))
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1., atol=1e-12)
        np.testing.assert_allclose(result[0], [np.cos(.075), 0, 0, np.sin(.075)])

    def test_motor_overload_cannot_be_hidden_by_ground_force(self):
        contact = np.zeros((7, 1))
        contact[2] = 1
        required = np.array([0., 0., 100., 0., 0., 0., 1.5])
        error = project_actuated_force(contact, required, np.arange(7), np.ones(1), 100., np.array([[-1., 1.]]))
        np.testing.assert_allclose(error[:6], 0, atol=1e-12)
        self.assertAlmostEqual(error[6], .5)

    def test_fixed_saturated_motor_interval_is_eliminated_exactly(self):
        contact = np.zeros((7, 1))
        contact[2] = 1
        required = np.array([0., 0., 100., 0., 0., 0., -1.])
        error = project_actuated_force(contact, required, np.arange(7), np.ones(1), 100., np.array([[-1., -1.]]))
        np.testing.assert_allclose(error, 0, atol=1e-12)

    def test_eliminating_independent_motors_preserves_the_full_bounded_solve(self):
        contact = np.zeros((8, 1))
        contact[2], contact[6] = 1., .2
        required = np.array([0., 0., 100., 0., 0., 0., 32., 7.])
        scale = np.r_[np.full(6, 100.), 10., 5.]
        motors = np.zeros((8, 2))
        motors[6:] = np.eye(2)
        matrix = np.column_stack((contact * 100 / scale[:, None], motors))
        target = required / scale
        full = lsq_linear(matrix, target, bounds=([0, -1, -1], [np.inf, 1, 1]), method="bvls", tol=1e-9)
        reduced = project_actuated_force(contact, required, np.arange(8), scale[6:], 100., np.array([[-1., 1.], [-1., 1.]]))
        np.testing.assert_allclose(reduced, target - matrix @ full.x, atol=1e-9)

    def test_pelvis_motion_keeps_stance_feet_at_their_world_anchors(self):
        with np.load(Path(__file__).parent / "assets/stand_reference.npz") as source:
            poses = np.tile(source["qpos"][0], (3, 1))
        data = mujoco.MjData(self.rig.model)
        data.qpos[:] = poses[0]
        mujoco.mj_kinematics(self.rig.model, data)
        feet = [self.rig.model.body(name).id for name in ("Foot_L", "Foot_R")]
        targets = np.tile(data.xpos[feet], (3, 1, 1))
        poses[:, 2] -= .02
        projection = StanceProjection(self.rig, poses, targets, np.ones((3, 2), dtype=bool))
        perturbed = poses.copy()
        perturbed[:, :3] += [.005, .002, -.002]
        perturbed[:, 3:7] = rotate_root(perturbed[:, 3:7], np.tile([.02, -.015, .01], (3, 1)))
        result = projection.project(perturbed)
        for frame, pose in enumerate(result):
            data.qpos[:] = pose
            mujoco.mj_kinematics(self.rig.model, data)
            self.assertLess(np.abs(data.xpos[feet] - projection.targets[frame]).max(), 1e-5)
            self.assertLess(np.abs(data.xmat[feet].reshape(2, 3, 3) - projection.rotations[frame]).max(), 1e-5)

    def test_exported_velocities_use_central_derivatives(self):
        poses = np.tile(self.rig.rest, (5, 1))
        dt = .04
        poses[:, 0] += (np.arange(5) * dt)**2
        velocities = reference_velocities(self.rig.model, poses, dt)
        np.testing.assert_allclose(velocities[1:-1, 0], 2 * np.arange(1, 4) * dt)
        np.testing.assert_allclose(velocities[:, 1:], 0, atol=1e-12)

    def test_supported_standing_is_not_destroyed_by_refinement(self):
        with np.load(Path(__file__).parent / "assets/stand_reference.npz") as reference:
            poses = np.tile(reference["qpos"][0], (3, 1))
        data = mujoco.MjData(self.rig.model)
        data.qpos[:] = poses[0]
        mujoco.mj_forward(self.rig.model, data)
        feet = [self.rig.model.body(name).id for name in ("Foot_L", "Foot_R")]
        targets = np.tile(data.xpos[feet], (3, 1, 1))
        original = poses.copy()
        contact = np.ones((3, 2), dtype=bool)
        result, _, schedule, _ = refine_support(self.rig, poses, targets, contact, .04, max_evaluations=5)
        np.testing.assert_equal(poses, original)
        np.testing.assert_equal(schedule, contact)
        data.qpos[:] = result[1]
        mujoco.mj_forward(self.rig.model, data)
        self.assertLess(np.linalg.norm(data.xpos[feet] - targets[1], axis=1).max(), .001)
        self.assertTrue(root_wrench_consistency(self.rig, result, .04)["all_frames_feasible"])

    def test_invalid_time_step_is_rejected_before_optimization(self):
        with self.assertRaises(ValueError):
            refine_support(self.rig, np.tile(self.rig.rest, (3, 1)), np.zeros((3, 2, 3)),
                           np.ones((3, 2), dtype=bool), 0)

    def test_landing_fit_preserves_validation_targets_and_continues_absolute_offsets(self):
        with np.load(Path(__file__).parent / "assets/stand_reference.npz") as source:
            poses = np.tile(source["qpos"][0], (3, 1))
        data = mujoco.MjData(self.rig.model)
        data.qpos[:] = poses[0]
        mujoco.mj_kinematics(self.rig.model, data)
        feet = [self.rig.model.body(name).id for name in ("Foot_L", "Foot_R")]
        targets = np.tile(data.xpos[feet], (3, 1, 1))
        contact = np.ones((3, 2), dtype=bool)
        result, returned, _, report = refine_support(self.rig, poses, targets, contact, .04,
                                                     max_evaluations=2, adjust_landings=True,
                                                     landing_seed=np.array([[.003, -.002], [-.003, .002]]))
        offsets = np.asarray(report["stance_anchor_offsets_xy"])
        self.assertEqual(offsets.shape, (2, 2))
        self.assertLessEqual(np.abs(offsets).max(), .02)
        np.testing.assert_equal(returned[:, :, :2], targets[:, :, :2])
        continued, again, _, continued_report = refine_support(
            self.rig, result, returned, contact, .04, max_evaluations=1,
            root_seed_drop_m=0., landing_seed=offsets)
        np.testing.assert_equal(again, returned)
        np.testing.assert_allclose(continued_report["stance_anchor_offsets_xy"], offsets)
        np.testing.assert_allclose(continued, result, atol=1e-5)
        for invalid in (np.zeros((1, 2)), np.full((2, 2), np.nan), np.full((2, 2), .021)):
            with self.subTest(seed=invalid), self.assertRaisesRegex(ValueError, "Landing offsets"):
                refine_support(self.rig, poses, targets, contact, .04, landing_seed=invalid)

    def test_extended_contact_uses_existing_stance_anchor_and_preserves_swing(self):
        targets = np.arange(30, dtype=float).reshape(5, 2, 3)
        contact = np.array([[False, False], [False, False], [True, False], [True, False], [False, True]])
        schedule = contact.copy()
        schedule[1, 0] = True
        result = complete_stance_targets(targets, contact, schedule)
        np.testing.assert_equal(result[1:4, 0], np.tile(targets[2, 0], (3, 1)))
        np.testing.assert_equal(result[~schedule], targets[~schedule])
        np.testing.assert_equal(targets, np.arange(30).reshape(5, 2, 3))


if __name__ == "__main__":
    unittest.main()
