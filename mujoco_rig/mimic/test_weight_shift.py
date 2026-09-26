"""A separate task must preserve stepping admission and standing retention."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .baseline import sha256
from .checkpoints import BestCheckpoint, checkpoint_rank
from .motion_protocol import validate_motion_reference
from .reference import StandReference
from .rig import Rig
from .weight_shift import SCHEMA, PHYSICAL_CHECKS, tracking_checks, validate_reference


def metrics(error=.01, quiet_error=.02):
    rows = [dict(mean_root_error_m=.01, mean_foot_error_m=.01, joint_rmse_rad=.01,
                 horizontal_displacement_rmse_m=error, reference_displacement_rms_m=.03) for _ in range(8)]
    result = dict(episodes=8, successes=8, mean_survival_seconds=3., mean_root_tracking_error_m=.01,
                  evaluation_schema="mimic_weight_shift_tracking_v2",
                  quiet_native=dict(episodes=8, successes=8, mean_root_tracking_error_m=.02),
                  quiet_native_five_seconds=dict(episodes=8, successes=8, mean_root_tracking_error_m=quiet_error),
                  quiet_standing_passed=quiet_error <= .05)
    checks = tracking_checks(result, rows)
    result["weight_shift_tracking"] = dict(passed=all(checks.values()), checks=checks, cases=rows)
    return result


class WeightShiftTests(unittest.TestCase):
    def test_weight_shift_is_not_stepping_admission(self):
        metadata = dict(schema=SCHEMA, passed=True, checks={k: True for k in PHYSICAL_CHECKS},
                        weight_shift_admission=dict(both_feet_planted=True, horizontal_excursion=True))
        metadata["checks"]["foot_lift"] = False
        validate_reference(metadata)
        with self.assertRaises(ValueError):
            validate_motion_reference(metadata)
        for key in PHYSICAL_CHECKS:
            invalid = copy.deepcopy(metadata)
            invalid["checks"][key] = False
            with self.assertRaises(ValueError):
                validate_reference(invalid)

    def test_reference_loader_rejects_lifted_contact_even_with_valid_content_hash(self):
        rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reference.npz"
            contact = np.ones((4, 2), dtype=bool)
            contact[1, 0] = False
            np.savez(path, qpos=np.tile(rig.rest, (4, 1)), qvel=np.zeros((4, rig.model.nv)), dt=1.,
                     contact=contact, sole_contact=np.ones((4, 4), dtype=bool))
            metadata = dict(schema=SCHEMA, passed=True, checks={k: True for k in PHYSICAL_CHECKS},
                            weight_shift_admission=dict(both_feet_planted=True, horizontal_excursion=True),
                            reference_sha256=sha256(path), model_sha256=sha256(rig.path), source={"license": "CC-BY-4.0"})
            path.with_suffix(".json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "both feet"):
                StandReference(path, rig, "cpu")

    def test_survival_alone_does_not_pass_shift_and_forgetting_cannot_win(self):
        baseline = metrics(error=.03)
        self.assertFalse(baseline["weight_shift_tracking"]["passed"])
        better_shift_but_forgets = metrics(error=.001, quiet_error=.06)
        self.assertLess(checkpoint_rank(better_shift_but_forgets), checkpoint_rank(baseline))
        valid_improvement = metrics(error=.001)
        self.assertGreater(checkpoint_rank(valid_improvement), checkpoint_rank(baseline))
        forged = metrics()
        forged["quiet_standing_passed"] = False
        with self.assertRaisesRegex(ValueError, "standing"):
            checkpoint_rank(forged)
        with tempfile.TemporaryDirectory() as folder:
            selected = BestCheckpoint(Path(folder), {})
            save = lambda path: Path(path).write_bytes(b"weights")
            self.assertTrue(selected.consider(save, baseline, iteration=0, samples=0))
            self.assertFalse(selected.consider(save, better_shift_but_forgets, iteration=1, samples=32))
            self.assertTrue(selected.consider(save, valid_improvement, iteration=2, samples=64))
            self.assertEqual(selected.metadata["selection_policy"], "weight_shift_tracking_v2")

    def test_small_motion_cannot_pass_by_remaining_stationary(self):
        result = metrics()
        rows = result["weight_shift_tracking"]["cases"]
        for row in rows:
            row.update(horizontal_displacement_rmse_m=.006, reference_displacement_rms_m=.006)
        self.assertFalse(tracking_checks(result, rows)["weight_shift"])
        for row in rows:
            row["horizontal_displacement_rmse_m"] = .001
        self.assertTrue(tracking_checks(result, rows)["weight_shift"])
        result["evaluation_schema"] = "mimic_weight_shift_tracking_v1"
        with self.assertRaisesRegex(ValueError, "v2"):
            checkpoint_rank(result)

    def test_rehearsal_selection_rejects_delayed_falls_and_horizon_changes(self):
        from .weight_shift import REHEARSAL_SCHEMA
        baseline = metrics(error=.02)
        baseline.update(evaluation_schema=REHEARSAL_SCHEMA, extended_motion_seconds=8.)
        extended = copy.deepcopy(baseline)
        extended.update(episode_seconds=8., mean_survival_seconds=7.)
        baseline["extended_motion_native"] = extended
        candidate = copy.deepcopy(baseline)
        candidate["mean_survival_seconds"] = 3.1
        candidate["extended_motion_native"]["mean_survival_seconds"] = 6.
        self.assertLess(checkpoint_rank(candidate), checkpoint_rank(baseline))
        with tempfile.TemporaryDirectory() as folder:
            selected = BestCheckpoint(Path(folder), {})
            save = lambda path: Path(path).write_bytes(b"weights")
            selected.consider(save, baseline, iteration=0, samples=0)
            changed = copy.deepcopy(baseline)
            changed["extended_motion_seconds"] = changed["extended_motion_native"]["episode_seconds"] = 9.
            with self.assertRaises(ValueError):
                selected.consider(save, changed, iteration=1, samples=32)
            changed = copy.deepcopy(baseline)
            changed["reward_schema"] = "mimic_weight_shift_precision_v1"
            with self.assertRaises(ValueError):
                selected.consider(save, changed, iteration=1, samples=32)


if __name__ == "__main__":
    unittest.main()
