"""Regression tests for policy preservation, warm-start contracts and update rollback."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from .checkpoints import BestCheckpoint, validate_contract
from .update_guard import UpdateGuard


class UpdateGuardTests(unittest.TestCase):
    def setUp(self):
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.optimizer = torch.optim.SGD([self.weight], lr=0.1, momentum=0.9)
        self.guard = UpdateGuard([self.weight], self.optimizer, 0.02)

    def test_rejected_update_restores_weights_and_momentum(self):
        self.guard.step(self.weight.square().sum(), lambda: 0.001)
        weight = self.weight.detach().clone()
        momentum = self.optimizer.state[self.weight]["momentum_buffer"].clone()
        accepted, kl = self.guard.step(self.weight.square().sum(), lambda: 0.5)
        self.assertFalse(accepted)
        self.assertEqual(kl, 0.5)
        torch.testing.assert_close(self.weight, weight, rtol=0, atol=0)
        torch.testing.assert_close(self.optimizer.state[self.weight]["momentum_buffer"], momentum, rtol=0, atol=0)
        self.assertIsNone(self.weight.grad)

    def test_first_rejection_removes_new_optimizer_state(self):
        self.guard.step(self.weight.square().sum(), lambda: 0.1)
        self.assertEqual(float(self.weight.detach()), 1.0)
        self.assertEqual(len(self.optimizer.state), 0)

    def test_nonfinite_measurement_rolls_back_before_raising(self):
        with self.assertRaises(FloatingPointError):
            self.guard.step(self.weight.square().sum(), lambda: float("nan"))
        self.assertEqual(float(self.weight.detach()), 1.0)
        self.assertEqual(len(self.optimizer.state), 0)

    def test_valid_update_is_retained_and_gradient_is_bounded(self):
        accepted, _ = self.guard.step(1000 * self.weight.square().sum(), lambda: 0.01)
        self.assertTrue(accepted)
        self.assertAlmostEqual(float(self.weight.detach()), 0.9, places=6)


class CheckpointTests(unittest.TestCase):
    def test_final_regression_cannot_replace_passing_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            best = BestCheckpoint(Path(directory), {"observation_contract": "target_pd"})
            good = dict(episodes=8, successes=8, mean_survival_seconds=3, mean_root_tracking_error_m=0.03)
            bad = dict(episodes=8, successes=0, mean_survival_seconds=1.6, mean_root_tracking_error_m=0.18)
            self.assertTrue(best.consider(lambda path: Path(path).write_bytes(b"good"), good, iteration=0, samples=0))
            self.assertFalse(best.consider(lambda path: self.fail("Bad policy was saved"), bad, iteration=10, samples=40960))
            self.assertEqual(best.path.read_bytes(), b"good")
            self.assertEqual(json.loads(best.metadata_path.read_text())["iteration"], 0)
            better = {**good, "mean_root_tracking_error_m": 0.02}
            self.assertTrue(best.consider(lambda path: Path(path).write_bytes(b"better"), better, iteration=11, samples=45056))
            with self.assertRaises(ValueError):
                best.consider(lambda path: self.fail("Invalid metrics were saved"),
                              {**good, "mean_root_tracking_error_m": float("nan")}, iteration=12, samples=49152)
            self.assertEqual(best.path.read_bytes(), b"better")

    def test_rejects_changed_actuation_reference_or_observation_contract(self):
        expected = {"action_mode": "reference_relative_target_pd", "num_obs": 362,
                    "reference_sha256": "reference", "model_sha256": "model", "pd_gain_per_torque_limit": 4,
                    "deployment_status": "experimental"}
        validate_contract({**expected, "deployment_status": "exported"}, expected)
        for key in ("action_mode", "num_obs", "reference_sha256", "model_sha256", "pd_gain_per_torque_limit"):
            with self.assertRaises(ValueError, msg=key):
                validate_contract({**expected, key: "changed"}, expected)

    def test_failed_checkpoint_write_preserves_previous_best(self):
        with tempfile.TemporaryDirectory() as directory:
            best = BestCheckpoint(Path(directory), {})
            metrics = dict(episodes=8, successes=8, mean_survival_seconds=3, mean_root_tracking_error_m=0.03)
            best.consider(lambda path: Path(path).write_bytes(b"good"), metrics, iteration=0, samples=0)
            metadata = copy.deepcopy(best.metadata)
            def fail(path):
                Path(path).write_bytes(b"partial")
                raise OSError("simulated interrupted checkpoint write")
            with self.assertRaises(OSError):
                best.consider(fail, {**metrics, "mean_root_tracking_error_m": 0.01}, iteration=1, samples=4096)
            self.assertEqual(best.path.read_bytes(), b"good")
            self.assertEqual(best.metadata, metadata)


if __name__ == "__main__":
    unittest.main()
