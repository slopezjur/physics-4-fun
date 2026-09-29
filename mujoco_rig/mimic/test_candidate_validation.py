import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from .validate_reactive_ball import checkpoint_selection_hash, validate_checkpoint_identity


class CandidateValidationTests(unittest.TestCase):
    def test_default_requires_selected_checkpoint(self):
        self.assertTrue(validate_checkpoint_identity("selected", "selected"))
        with self.assertRaisesRegex(ValueError, "selected checkpoint"):
            validate_checkpoint_identity("candidate", "selected")

    def test_explicit_candidate_requires_exact_hash_and_remains_unselected(self):
        self.assertFalse(validate_checkpoint_identity("candidate", "selected", "candidate"))
        with self.assertRaisesRegex(ValueError, "explicit SHA-256"):
            validate_checkpoint_identity("candidate", "selected", "different")

    def test_unselected_fit_requires_explicit_hash_without_best_metadata(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "candidate.pt"
            with self.assertRaisesRegex(ValueError, "selected checkpoint metadata"):
                checkpoint_selection_hash(checkpoint)
            selected = checkpoint_selection_hash(checkpoint, "candidate")
            self.assertIsNone(selected)
            self.assertFalse(validate_checkpoint_identity("candidate", selected, "candidate"))
            with self.assertRaisesRegex(ValueError, "explicit SHA-256"):
                validate_checkpoint_identity("changed", selected, "candidate")


if __name__ == "__main__":
    unittest.main()
