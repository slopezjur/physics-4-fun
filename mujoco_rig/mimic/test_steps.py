"""Reference provenance, stance isolation and unchanged physical contracts for step probes."""
import unittest
from pathlib import Path
import json

import numpy as np
import mujoco

from .probe_steps import validate_reference_substitution
from .retarget_steps import lock_stance, validate_motion, ground_walking_frames, sole_gaps
from .checkpoints import checkpoint_rank
from .reference import StandReference
from .rig import Rig
from .motion_protocol import REFERENCE_SCHEMA, tracking_checks, validate_motion_reference


class StepTests(unittest.TestCase):
    def test_legacy_validation_cannot_reenter_training_as_a_grounded_reference(self):
        path = Path(__file__).parent / "assets/steps/backward_reference.json"
        metadata = json.loads(path.read_text())
        with self.assertRaises(ValueError):
            validate_motion_reference(metadata)
        metadata["schema"] = REFERENCE_SCHEMA
        metadata["checks"]["root_wrench_consistency"] = True
        metadata["checks"]["actuation_feasibility"] = True
        validate_motion_reference(metadata)
        metadata["checks"]["ground_support"] = False
        with self.assertRaises(ValueError):
            validate_motion_reference(metadata)
        metadata["checks"]["ground_support"] = True
        metadata["schema"] = "mimic_step_reference_v1"
        with self.assertRaises(ValueError):
            validate_motion_reference(metadata)
    def test_grounding_removes_floating_support_without_changing_joints(self):
        rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        poses = np.tile(rig.rest, (2, 1))
        poses[:, 2] += [.01, .02]
        grounded = ground_walking_frames(rig, poses)
        np.testing.assert_equal(poses[:, 7:], grounded[:, 7:])
        np.testing.assert_equal(poses[:, :2], grounded[:, :2])
        data = mujoco.MjData(rig.model)
        for pose in grounded:
            data.qpos[:] = pose
            mujoco.mj_kinematics(rig.model, data)
            self.assertAlmostEqual(float(sole_gaps(rig.model, data).min()), 0., places=8)

    def test_packaged_steps_pass_geometry_and_preserve_travel_direction(self):
        root = Path(__file__).resolve().parent
        rig = Rig.load(root.parent / "dummy.xml")
        for direction, sign in (("forward", 1), ("backward", -1)):
            path = root / "assets/steps" / f"{direction}_reference.npz"
            reference = StandReference(path, rig, "cpu")
            self.assertGreater(reference.duration, 3.)
            self.assertGreater(sign * float(reference.qpos[-1, 0] - reference.qpos[0, 0]), 1.)
            with np.load(path, allow_pickle=False) as data:
                checks, _ = validate_motion(rig, data["qpos"], data["qvel"], data["foot_targets"],
                                            data["contact"], float(data["dt"]))
                floating = data["qpos"].copy()
                floating[:, 2] += .02
                rejected, _ = validate_motion(rig, floating, data["qvel"], data["foot_targets"],
                                              data["contact"], float(data["dt"]))
                self.assertFalse(rejected["ground_support"])
            self.assertTrue(all(value for key, value in checks.items()
                                if key not in ("root_wrench_consistency", "actuation_feasibility")), checks)
            self.assertFalse(checks["root_wrench_consistency"], "Archived v3 clips are geometric priors, not dynamically validated training targets")
            self.assertFalse(checks["actuation_feasibility"])
            metadata = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(metadata["source"]["license"], "CC-BY-4.0")

    def test_motion_selection_does_not_confuse_survival_with_tracking(self):
        def metrics(error):
            return dict(episodes=8, successes=8, mean_survival_seconds=3., mean_root_tracking_error_m=.1,
                        motion_tracking=dict(passed=error < .1, cases=[dict(mean_foot_error_m=error,
                                                                         joint_rmse_rad=.1, foot_lift_range_m=[.06, .06]) for _ in range(8)]))
        good, poor = metrics(.05), metrics(.2)
        self.assertGreater(checkpoint_rank(good), checkpoint_rank(poor))
        poor["motion_tracking"]["passed"] = True
        with self.assertRaisesRegex(ValueError, "Inconsistent motion tracking"):
            checkpoint_rank(poor)
        good["motion_tracking"]["cases"][0]["foot_lift_range_m"] = [.005, .06]
        self.assertFalse(tracking_checks(good, good["motion_tracking"]["cases"])["foot_lift"])
        with self.assertRaisesRegex(ValueError, "Inconsistent motion tracking"):
            checkpoint_rank(good)

    def test_stance_lock_does_not_pin_swing_or_other_foot(self):
        motion = np.arange(30, dtype=float).reshape(5, 2, 3)
        original = motion.copy()
        contact = np.array([[1, 0], [1, 1], [0, 1], [1, 0], [1, 0]], dtype=bool)
        result = lock_stance(motion, contact)
        np.testing.assert_equal(motion, original)
        np.testing.assert_equal(result[1, 0], motion[0, 0])
        np.testing.assert_equal(result[2, 0], motion[2, 0])
        np.testing.assert_equal(result[2, 1], motion[1, 1])
        np.testing.assert_equal(result[4, 0], motion[3, 0])

    def test_motion_substitution_does_not_bypass_rig_or_motor_contract(self):
        old = dict(reference_sha256="old", reference_duration=6., model_sha256="rig",
                   residual_radians=.25, num_obs=362, latency_steps=2)
        new = {**old, "reference_sha256": "new", "reference_duration": 4.}
        validate_reference_substitution(old, new)
        for key in ("model_sha256", "residual_radians", "num_obs", "latency_steps"):
            with self.assertRaises(ValueError):
                validate_reference_substitution(old, {**new, key: "changed"})


if __name__ == "__main__":
    unittest.main()
