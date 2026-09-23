"""Transition isolation preserves stance identity and cannot create a training asset."""
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from .baseline import sha256
from .rig import Rig
from .motion_contacts import landing_seed_for_window
from .transition_probe import probe


class TransitionProbeTests(unittest.TestCase):
    def test_cropped_stances_inherit_their_absolute_source_offsets(self):
        contact = np.array([[True, False], [True, True], [True, True],
                            [False, True], [True, False], [True, True]])
        offsets = np.array([[.001, .002], [.003, .004], [.005, .006], [.007, .008]])
        np.testing.assert_equal(landing_seed_for_window(contact, offsets, 2, 6), offsets)
        np.testing.assert_equal(landing_seed_for_window(contact, offsets, 3, 5), offsets[[1, 2]])
        self.assertIsNone(landing_seed_for_window(contact, None, 2, 6))
        with self.assertRaisesRegex(ValueError, "source stance"):
            landing_seed_for_window(contact, offsets[:2], 2, 6)

    def test_exact_probe_records_context_and_never_produces_training_manifest(self):
        asset = Path(__file__).parent / "assets/stand_reference.npz"
        rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        with np.load(asset) as arrays:
            poses = np.tile(arrays["qpos"][0], (7, 1))
        data = mujoco.MjData(rig.model)
        data.qpos[:] = poses[0]
        mujoco.mj_kinematics(rig.model, data)
        feet = [rig.model.body(name).id for name in ("Foot_L", "Foot_R")]
        targets = np.tile(data.xpos[feet], (7, 1, 1))
        metadata = json.loads(asset.with_suffix(".json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, out = root / "reference.npz", root / "probe"
            np.savez(reference, qpos=poses, dt=.04, foot_targets=targets,
                     contact=np.ones((7, 2), dtype=bool), sole_contact=np.ones((7, 4), dtype=bool))
            metadata.update(reference_sha256=sha256(reference), force_fit=dict(stance_method="fixture"))
            reference.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
            report = probe(reference, out, 1, 6, max_evaluations=1, linear_solver="exact")
            self.assertFalse(report["admitted_for_training"])
            self.assertEqual(report["fit"]["linear_solver"], "exact")
            self.assertIsNone(report["fit"]["linear_solver_iterations"])
            self.assertEqual(report["source_frame_start"], 1)
            self.assertEqual(report["source_frame_stop_exclusive"], 6)
            self.assertTrue(report["candidate"]["checks"]["root_wrench_consistency"])
            self.assertFalse((out / "diagnostic_poses.json").exists())
            with self.assertRaises(FileExistsError):
                probe(reference, out, 1, 6, max_evaluations=1)
            with self.assertRaisesRegex(ValueError, "five source frames"):
                probe(reference, root / "invalid", 1, 4)


if __name__ == "__main__":
    unittest.main()
