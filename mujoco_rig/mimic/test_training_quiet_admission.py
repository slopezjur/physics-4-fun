import copy
import tempfile
import unittest
from pathlib import Path

from .baseline import sha256
from .training_quiet_admission import validate_quiet_initialization


class QuietTrainingAdmissionTests(unittest.TestCase):
    def test_report_binds_source_and_rechecks_all_physical_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("checkpoint", "reference", "rig")]
            for path in paths:
                path.write_bytes(path.name.encode())
            case = dict(survived=True, survival_seconds=600., mean_root_error_m=0.,
                        final_stable_seconds=600., settled_fraction_after_startup=1.,
                        maximum_foot_error_after_startup_m=0., tail_root_drift_m=0.,
                        tail_foot_motion_m=0., minimum_foot_load_after_startup_n=[100., 100.])
            report = dict(zip(("checkpoint_sha256", "reference_sha256", "model_sha256"), map(sha256, paths)))
            report.update(seconds=600., passed=True, cases=[copy.deepcopy(case) for _ in range(8)])
            validate_quiet_initialization(report, *paths)
            for field in ("checkpoint_sha256", "reference_sha256", "model_sha256"):
                wrong = copy.deepcopy(report)
                wrong[field] = "wrong"
                with self.assertRaisesRegex(ValueError, "provenance"):
                    validate_quiet_initialization(wrong, *paths)
            for wrong in (dict(report, seconds=30.), dict(report, cases=report["cases"][:7]),
                          dict(report, passed=False)):
                with self.assertRaisesRegex(ValueError, "600-second"):
                    validate_quiet_initialization(wrong, *paths)
            report["cases"][3]["minimum_foot_load_after_startup_n"] = [0., 200.]
            with self.assertRaisesRegex(ValueError, "600-second"):
                validate_quiet_initialization(report, *paths)


if __name__ == "__main__":
    unittest.main()
