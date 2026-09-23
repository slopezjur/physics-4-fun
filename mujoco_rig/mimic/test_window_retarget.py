"""Fixed-context windows preserve delayed dynamics and cannot hide regressions."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from .force_retarget import refine_support, reference_velocities, rotate_root
from .contact_projection import StanceProjection
from .baseline import sha256
from .motion_dynamics import frame_dynamics, delayed_pd_bounds
from .rig import Rig
from .window_retarget import context_interval, regression_reasons, refine_windows


class WindowRetargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")

    def test_context_reproduces_full_clip_forces_and_delayed_commands(self):
        r = self.rig
        dt = .033334
        q = np.tile(r.rest, (24, 1))
        joints = r.model.jnt_qposadr[r.model.actuator_trnid[r.actuators, 0]]
        q[:, joints] += .003 * np.sin(np.arange(24)[:, None] * .7)
        q[8:12, joints] += .004
        left, right, fixed = context_interval(r, len(q), dt, 8, 12)
        self.assertEqual((left, right), (4, 16))
        with self.assertRaisesRegex(ValueError, "surrounding context"):
            context_interval(r, len(q), dt, 0, len(q))
        np.testing.assert_equal(np.flatnonzero(~fixed), np.arange(4, 8))
        whole_v = reference_velocities(r.model, q, dt)
        local = q[left:right]
        local_v = reference_velocities(r.model, local, dt)
        d = mujoco.MjData(r.model)
        for frame in range(7, 15):
            with self.subTest(frame=frame):
                np.testing.assert_allclose(frame_dynamics(r.model, d, q, frame, dt),
                                           frame_dynamics(r.model, d, local, frame - left, dt), atol=1e-9)
                np.testing.assert_allclose(delayed_pd_bounds(r, q, whole_v, dt, frame),
                                           delayed_pd_bounds(r, local, local_v, dt, frame - left), atol=1e-9)

    def test_window_keeps_the_original_stance_heading_when_cutting_mid_stance(self):
        r = self.rig
        with np.load(Path(__file__).parent / "assets/stand_reference.npz") as arrays:
            q = np.tile(arrays["qpos"][0], (3, 1))
        d = mujoco.MjData(r.model)
        d.qpos[:] = q[0]
        mujoco.mj_kinematics(r.model, d)
        feet = [r.model.body(name).id for name in ("Foot_L", "Foot_R")]
        targets = np.tile(d.xpos[feet], (3, 1, 1))
        contact = np.ones((3, 2), dtype=bool)
        toe = np.array([[False, False], [True, False], [True, False]])
        q[1:, 3:7] = rotate_root(q[1:, 3:7], np.tile([0., 0., .03], (2, 1)))
        whole = StanceProjection(r, q, targets, contact, toe)
        cropped = StanceProjection(r, q[1:], targets[1:], contact[1:], toe[1:], stance_rotations=whole.rotations[1:])
        inferred = StanceProjection(r, q[1:], targets[1:], contact[1:], toe[1:])
        np.testing.assert_array_equal(cropped.rotations, whole.rotations[1:])
        np.testing.assert_array_equal(cropped.anchors, whole.anchors[1:])
        self.assertGreater(np.abs(inferred.rotations - cropped.rotations).max(), .02)

    def test_optimizer_never_reprojects_or_moves_fixed_context(self):
        r = self.rig
        with np.load(Path(__file__).parent / "assets/stand_reference.npz") as arrays:
            q = np.tile(arrays["qpos"][0], (5, 1))
        d = mujoco.MjData(r.model)
        d.qpos[:] = q[0]
        mujoco.mj_kinematics(r.model, d)
        feet = [r.model.body(name).id for name in ("Foot_L", "Foot_R")]
        targets = np.tile(d.xpos[feet], (5, 1, 1))
        fixed = np.array([True, True, False, True, True])
        original = q.copy()
        result, _, _, report = refine_support(r, q, targets, np.ones((5, 2), dtype=bool), .04,
                                              max_evaluations=2, fixed_frames=fixed, linear_solver="exact")
        np.testing.assert_array_equal(result[fixed], original[fixed])
        np.testing.assert_array_equal(q, original)
        self.assertEqual(report["fixed_frame_indices"], [0, 1, 3, 4])
        with self.assertRaisesRegex(ValueError, "landing anchors"):
            refine_support(r, q, targets, np.ones((5, 2), dtype=bool), .04,
                           fixed_frames=fixed, adjust_landings=True)

    def test_regression_guard_uses_frame_identity_and_combined_pd_failures(self):
        def assessment(root, pd, penetration=.001):
            return ({"finite": True}, dict(root_wrench_consistency=dict(infeasible_frames=root),
                     action_window_consistency=dict(failing_frames={"0.25": pd}),
                     max_penetration_m=penetration, max_stance_slide_m_s=0., max_foot_target_error_m=.02))
        baseline = assessment([2], [])
        self.assertEqual(regression_reasons(baseline, assessment([], [2])), [])
        self.assertIn("new root-support failures", regression_reasons(baseline, assessment([3], [])))
        self.assertIn("new combined root/PD failures", regression_reasons(baseline, assessment([], [3])))
        self.assertIn("increased max_penetration_m", regression_reasons(baseline, assessment([], [], .002)))

    def test_direct_forces_preserve_native_standing_and_fixed_context(self):
        from .motion_dynamics import root_wrench_consistency, action_window_consistency
        from .constrained_fit import constrained_least_squares
        r = self.rig
        with np.load(Path(__file__).parent / "assets/stand_reference.npz") as arrays:
            q = np.tile(arrays["qpos"][0], (7, 1))
        d = mujoco.MjData(r.model)
        d.qpos[:] = q[0]
        mujoco.mj_kinematics(r.model, d)
        feet = [r.model.body(name).id for name in ("Foot_L", "Foot_R")]
        targets = np.tile(d.xpos[feet], (7, 1, 1))
        fixed = np.ones(7, dtype=bool)
        fixed[3] = False
        soles = np.ones((7, 4), dtype=bool)
        def check_assembled_jacobians(evaluate, initial, lower, upper, budget, **options):
            first = options["numerical_columns"]
            trial = initial.copy()
            trial[first:] += .03
            jr, jc = options["analytic_tail"](trial)
            direction = np.random.default_rng(51).normal(size=len(initial) - first)
            delta = np.r_[np.zeros(first), direction * 1e-6]
            rp, cp = evaluate(trial + delta)
            rm, cm = evaluate(trial - delta)
            np.testing.assert_allclose(jr @ direction, (rp - rm) / 2e-6, atol=1e-5, rtol=1e-6)
            np.testing.assert_allclose(jc @ direction, (cp - cm) / 2e-6, atol=1e-6, rtol=1e-6)
            return constrained_least_squares(evaluate, initial, lower, upper, budget, **options)

        with patch("mujoco_rig.mimic.force_retarget.constrained_least_squares", side_effect=check_assembled_jacobians):
            result, _, _, report = refine_support(r, q, targets, np.ones((7, 2), dtype=bool), .04,
                                                  max_evaluations=2, fixed_frames=fixed, constrained=True,
                                                  direct_forces=True, root_seed_drop_m=0., source_soles=soles)
        np.testing.assert_array_equal(result[fixed], q[fixed])
        self.assertGreater(report["force_variables"], 0)
        self.assertTrue(report["constraints_satisfied"])
        self.assertTrue(report["used_feasible_incumbent"])
        self.assertTrue(root_wrench_consistency(r, result, .04, soles)["all_frames_feasible"])
        pd = action_window_consistency(r, result, reference_velocities(r.model, result, .04), .04, sole_contact=soles)
        self.assertEqual(pd["failing_frames"]["0.25"], [])

    def test_constraints_exclude_immutable_geometry_but_include_affected_fixed_dynamics(self):
        r = self.rig
        with np.load(Path(__file__).parent / "assets/stand_reference.npz") as arrays:
            q = np.tile(arrays["qpos"][0], (9, 1))
        d = mujoco.MjData(r.model)
        d.qpos[:] = q[0]
        mujoco.mj_kinematics(r.model, d)
        feet = [r.model.body(name).id for name in ("Foot_L", "Foot_R")]
        targets = np.tile(d.xpos[feet], (9, 1, 1))
        q[0, 0] += 1.  # Impossible placement in distant immutable context.
        fixed = np.ones(9, dtype=bool)
        fixed[4] = False
        result, _, _, report = refine_support(r, q, targets, np.ones((9, 2), dtype=bool), .04,
                                              max_evaluations=1, fixed_frames=fixed, constrained=True,
                                              root_seed_drop_m=0., source_soles=np.ones((9, 4), dtype=bool))
        np.testing.assert_array_equal(result[fixed], q[fixed])
        self.assertLess(report["constraint_violations"]["placement"], 1e-8)
        self.assertEqual(report["constrained_dynamics_frames"], [3, 4, 5, 6])
        # Frame 6's delayed command depends on editable frame 4, but its own
        # central dynamics use only fixed frames 5, 6 and 7. A root failure here
        # cannot be repaired locally and must remain a full-clip failure.
        q[6, 0] += .1
        from .motion_dynamics import root_wrench_consistency
        self.assertIn(6, root_wrench_consistency(r, q, .04, np.ones((9, 4), dtype=bool))["infeasible_frames"])
        deferred, _, _, report = refine_support(r, q, targets, np.ones((9, 2), dtype=bool), .04,
                                                max_evaluations=1, fixed_frames=fixed, constrained=True,
                                                root_seed_drop_m=0., source_soles=np.ones((9, 4), dtype=bool))
        self.assertEqual(report["constrained_dynamics_frames"], [3, 4, 5])
        self.assertEqual([row["frame"] for row in report["deferred_immutable_dynamics"]], [6])
        np.testing.assert_array_equal(deferred[fixed], q[fixed])
        self.assertIn(6, root_wrench_consistency(r, deferred, .04, np.ones((9, 4), dtype=bool))["infeasible_frames"])

    def test_window_output_is_hash_bound_and_outside_frames_are_unchanged(self):
        r = self.rig
        with np.load(Path(__file__).parent / "assets/stand_reference.npz") as arrays:
            q = np.tile(arrays["qpos"][0], (7, 1))
        d = mujoco.MjData(r.model)
        d.qpos[:] = q[0]
        mujoco.mj_kinematics(r.model, d)
        feet = [r.model.body(name).id for name in ("Foot_L", "Foot_R")]
        targets = np.tile(d.xpos[feet], (7, 1, 1))
        metadata = json.loads((Path(__file__).parent / "assets/steps/forward_reference.json").read_text())
        metadata.update(schema="mimic_step_reference_v6", force_fit=dict(stance_method="fixture"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, out = root / "source.npz", root / "result"
            np.savez(reference, qpos=q, qvel=reference_velocities(r.model, q, .04), dt=.04,
                     foot_targets=targets, contact=np.ones((7, 2), dtype=bool), sole_contact=np.ones((7, 4), dtype=bool))
            metadata["reference_sha256"] = sha256(reference)
            reference.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
            report = refine_windows(reference, out, [(2, 4), (3, 5)], max_evaluations=1)
            self.assertEqual(report["parent_reference_sha256"], sha256(reference))
            self.assertEqual(report["reference_sha256"], sha256(out / "step_reference.npz"))
            self.assertFalse(report["passed"], "A static fixture must still fail the walking lift gate")
            with np.load(out / "step_reference.npz") as result:
                np.testing.assert_array_equal(result["qpos"][[0, 1, 5, 6]], q[[0, 1, 5, 6]])
                np.testing.assert_array_equal(result["foot_targets"], targets)
            with self.assertRaises(FileExistsError):
                refine_windows(reference, out, [(2, 4)], max_evaluations=1)


if __name__ == "__main__":
    unittest.main()
