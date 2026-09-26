"""Diagnostic admission and first-episode trace accounting."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .baseline import sha256
from .rig import Rig
from .track_transfer import CalibrationReference, summarize_case, tracking_checks


class TransferTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        cls.source = Path(__file__).parent / "assets/stand_reference.npz"
        with np.load(cls.source) as arrays:
            cls.pose = arrays["qpos"][0].copy()

    def test_diagnostic_loader_requires_provenance_and_fresh_physical_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            arrays = dict(qpos=np.tile(self.pose, (4, 1)), qvel=np.zeros((4, self.rig.model.nv)),
                          foot_targets=np.zeros((4, 2, 3)), contact=np.ones((4, 2), dtype=bool),
                          sole_contact=np.ones((4, 4), dtype=bool), dt=1.)
            np.savez(path / "diagnostic_poses.npz", **arrays)
            metadata = dict(schema="mimic_transfer_calibration_v1", passed=True, admitted_for_training=False,
                            model_sha256=sha256(self.rig.path), source_reference_sha256=sha256(self.source),
                            diagnostic_sha256=sha256(path / "diagnostic_poses.npz"), source={"license": "CC-BY-4.0"})
            report = path / "report.json"
            report.write_text(json.dumps(metadata))
            with patch("mujoco_rig.mimic.track_transfer.validate_motion", return_value=({"physical": True}, {})) as check:
                reference = CalibrationReference(path, self.rig)
                check.assert_called_once()
                self.assertEqual(reference.duration, 3.)
                q, _ = reference.sample(torch.tensor([0., 1.5, 3.]))
                np.testing.assert_allclose(q, np.tile(self.pose, (3, 1)), atol=1e-7)
            self.assertFalse((path / "diagnostic_poses.json").exists())
            for key, value in (("model_sha256", "bad"), ("source_reference_sha256", "bad"),
                               ("diagnostic_sha256", "bad"), ("passed", False), ("admitted_for_training", True)):
                report.write_text(json.dumps({**metadata, key: value}))
                with self.assertRaisesRegex(ValueError, "provenance"):
                    CalibrationReference(path, self.rig)
            report.write_text(json.dumps(metadata))
            with patch("mujoco_rig.mimic.track_transfer.validate_motion", return_value=({"physical": False}, {})):
                with self.assertRaisesRegex(ValueError, "physical checks"):
                    CalibrationReference(path, self.rig)
            arrays["qvel"][1, 0] = 1.
            np.savez(path / "diagnostic_poses.npz", **arrays)
            report.write_text(json.dumps({**metadata, "diagnostic_sha256": sha256(path / "diagnostic_poses.npz")}))
            with self.assertRaisesRegex(ValueError, "velocities disagree"):
                CalibrationReference(path, self.rig)

    def test_trace_excludes_resets_and_uses_post_integration_reference_times(self):
        rig = self.rig
        sampled = []
        def sample(times):
            sampled.extend(times.tolist())
            return torch.tensor(np.tile(self.pose, (len(times), 1)), dtype=torch.float32), None
        task = SimpleNamespace(rig=rig, dt=rig.decimation * rig.model.opt.timestep,
                               reference=SimpleNamespace(sample=sample))
        record = dict(qpos_after=self.pose.copy(), qvel_after=np.zeros(rig.model.nv),
                      control=np.zeros(len(rig.actuators)),
                      self_contact_max_force_n=0.)
        count = 2 * rig.decimation
        records = [{**record, "qpos_after": self.pose.copy()} for _ in range(count)]
        records.append({**record, "qpos_after": self.pose + 100.})
        row, arrays = summarize_case(task, records, 2 * task.dt, .5)
        self.assertEqual(len(arrays["qpos_after"]), count)
        np.testing.assert_allclose(sampled, .5 + np.arange(1, count + 1) * rig.model.opt.timestep, atol=1e-7)
        self.assertLess(row["max_root_error_m"], 1e-6)
        self.assertIsNone(row["first_root_error_5cm_seconds"])

    def test_full_motion_requires_lift_but_short_phase_probes_do_not(self):
        case = dict(mean_root_error_m=.01, mean_foot_error_m=.01, joint_rmse_rad=.01,
                    foot_lift_range_m=[0., 0.])
        metrics = dict(successes=1, episodes=1)
        self.assertTrue(all(tracking_checks(metrics, [case], False).values()))
        self.assertFalse(tracking_checks(metrics, [case], True)["foot_lift"])
        self.assertFalse(tracking_checks({**metrics, "successes": 0}, [case], False)["survival"])


if __name__ == "__main__":
    unittest.main()
