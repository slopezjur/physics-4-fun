"""Regression checks for source interpretation and the validated reference boundary."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch

from .bvh import BVH
from .reference import StandReference
from .rig import Rig
from .acquire_motion import verify_source

ASSETS = Path(__file__).parent / "assets"


class MotionTests(unittest.TestCase):
    def test_bvh_units_channel_order_and_child_offset(self):
        text = """HIERARCHY
ROOT Root {
 OFFSET 0 0 0
 CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation
 JOINT Child {
  OFFSET 0 100 0
  CHANNELS 3 Yrotation Xrotation Zrotation
  End Site { OFFSET 0 10 0 }
 }
}
MOTION
Frames: 1
Frame Time: 0.0166667
100 200 300 90 90 0 0 0 0
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.bvh"
            path.write_text(text)
            bvh = BVH.load(path)
        positions, rotations = bvh.world_poses()
        np.testing.assert_allclose(positions[0, 0], [1, 2, 3], atol=1e-12)
        # Ordered Y then X rotations send local +Y to world +X.
        np.testing.assert_allclose(positions[0, 1], [2, 2, 3], atol=1e-12)
        np.testing.assert_allclose(rotations[0, 0] @ [0, 1, 0], [1, 0, 0], atol=1e-12)

    def test_reference_clamps_without_loop_and_preserves_unit_quaternion(self):
        rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        reference = StandReference(ASSETS / "stand_reference.npz", rig, "cpu")
        times = torch.tensor([-1.0, 0.0, reference.duration / 2, reference.duration + 1])
        q, _ = reference.sample(times)
        torch.testing.assert_close(q[0], q[1])
        torch.testing.assert_close(q[-1], reference.qpos[-1])
        torch.testing.assert_close(torch.linalg.vector_norm(q[:, 3:7], dim=1), torch.ones(4))

    def test_reference_rejects_corrupted_asset_and_wrong_rig(self):
        rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stand_reference.npz"
            shutil.copy2(ASSETS / path.name, path)
            metadata = json.loads((ASSETS / "stand_reference.json").read_text())
            metadata["model_sha256"] = "changed"
            path.with_suffix(".json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "different rig"):
                StandReference(path, rig, "cpu")
            shutil.copy2(ASSETS / "stand_reference.json", path.with_suffix(".json"))
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "manifest"):
                StandReference(path, rig, "cpu")

    def test_download_cache_cannot_silently_relicense_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "Neutral_ID.bvh").write_text("HIERARCHY altered")
            with self.assertRaisesRegex(ValueError, "provenance"):
                verify_source(source)


if __name__ == "__main__":
    unittest.main()
