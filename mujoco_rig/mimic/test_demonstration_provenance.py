import tempfile
import unittest
from pathlib import Path

from .baseline import sha256
from .fit_recovery_demonstrations import validate_demonstration_provenance


class DemonstrationProvenanceTests(unittest.TestCase):
    def test_teacher_and_projectile_must_match_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("teacher", "reference", "ball8", "rig", "trace", "student", "ball5")]
            for path in paths:
                path.write_bytes(path.name.encode())
            teacher, reference, ball8, rig, trace, student, ball5 = paths
            report = dict(zip(("checkpoint_sha256", "reference_sha256", "ball_sha256", "model_sha256", "trace_sha256"),
                              map(sha256, paths[:5])))
            validate_demonstration_provenance(report, teacher, reference, ball8, rig, trace)
            with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
                validate_demonstration_provenance(report, student, reference, ball8, rig, trace)
            with self.assertRaisesRegex(ValueError, "ball_sha256"):
                validate_demonstration_provenance(report, teacher, reference, ball5, rig, trace)
            trace.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "trace changed"):
                validate_demonstration_provenance(report, teacher, reference, ball8, rig, trace)


if __name__ == "__main__":
    unittest.main()
