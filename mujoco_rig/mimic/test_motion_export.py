"""Alternate reference filenames must remain compatible with the existing Godot bundle."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from .baseline import sha256
from .export_policy import export_policy, validate_export_reference


class MotionExportTests(unittest.TestCase):
    def test_named_motion_exports_canonical_reference_with_bound_hashes(self):
        reference = Path(__file__).parent / "assets/steps/forward_reference.npz"
        actor = torch.nn.Sequential(torch.nn.Linear(362, 30), torch.nn.Tanh()).eval()
        contract = dict(num_obs=362, reference_sha256=sha256(reference))
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "export"
            error = export_policy(actor, np.zeros((2, 8, 362), dtype=np.float32), reference, contract, bundle)
            self.assertLess(error, 1e-5)
            self.assertEqual(sha256(bundle / "stand_reference.npz"), contract["reference_sha256"])
            exported = json.loads((bundle / "contract.json").read_text())
            self.assertEqual(exported["reference_frames_sha256"], sha256(bundle / "reference_frames.json"))
            self.assertTrue((bundle / "ATTRIBUTION.md").is_file())

    def test_missing_reference_export_inputs_fail_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                validate_export_reference(Path(directory) / "missing.npz")


if __name__ == "__main__":
    unittest.main()
