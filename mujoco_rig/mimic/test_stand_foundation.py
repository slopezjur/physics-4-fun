"""A long-standing gate must reject delayed collapse, unloaded feet, and drift."""
import unittest
import os
from pathlib import Path

import torch

from .validate_stand_foundation import case_passed


class FoundationGateTests(unittest.TestCase):
    def setUp(self):
        self.case = dict(survived=True, survival_seconds=600., mean_root_error_m=.02,
                         final_stable_seconds=20., settled_fraction_after_startup=1.,
                         minimum_foot_load_after_startup_n=[250., 260.],
                         maximum_foot_error_after_startup_m=.005, tail_root_drift_m=.001,
                         tail_foot_motion_m=.0001)

    def test_nominal_long_standing_passes(self):
        self.assertTrue(case_passed(self.case, 600.))

    def test_missing_convergence_measurement_cannot_pass(self):
        old = self.case.copy()
        old.pop("tail_foot_motion_m")
        self.assertFalse(case_passed(old, 600.))

    def test_foot_convergence_ranking_still_retains_standing_cases(self):
        import copy
        import tempfile
        from .continuous_stand import EVALUATION_SCHEMA, GATES
        from .test_continuous_stand import cases
        from .train_stand_foundation import foundation_rank
        from .checkpoints import BestCheckpoint
        baseline = dict(evaluation_schema=EVALUATION_SCHEMA, gates=GATES, reference_sha256="test",
            passed=True, successes=8, cases=cases(), worst_tail_foot_motion_m=.002,
            worst_mean_foot_error_m=.005)
        settled = copy.deepcopy(baseline)
        settled["worst_tail_foot_motion_m"] = .001
        for case in settled["cases"]:
            case["mean_root_error_m"] = .02
        self.assertGreater(foundation_rank(settled), foundation_rank(baseline))
        with tempfile.TemporaryDirectory() as directory:
            best = BestCheckpoint(Path(directory), {}, rank_function=foundation_rank)
            save = lambda path: Path(path).write_bytes(b"test")
            self.assertTrue(best.consider(save, baseline, iteration=0, samples=0))
            self.assertTrue(best.consider(save, settled, iteration=1, samples=1))
            failed = copy.deepcopy(settled)
            failed.update(passed=False, worst_tail_foot_motion_m=0.)
            failed["cases"][3]["mean_root_error_m"] = .06
            self.assertFalse(best.consider(save, failed, iteration=2, samples=2))
            self.assertEqual(best.metadata["iteration"], 1)

    def test_delayed_fall_cannot_pass_on_good_early_metrics(self):
        for changes in (dict(survived=False), dict(survival_seconds=599.9),
                        dict(final_stable_seconds=9.9)):
            with self.subTest(changes=changes):
                self.assertFalse(case_passed({**self.case, **changes}, 600.))

    def test_survival_cannot_hide_unloading_sliding_or_drift(self):
        for changes in (dict(minimum_foot_load_after_startup_n=[0., 700.]),
                        dict(maximum_foot_error_after_startup_m=.031),
                        dict(settled_fraction_after_startup=.98),
                        dict(mean_root_error_m=.051), dict(tail_root_drift_m=.011),
                        dict(tail_foot_motion_m=.0011)):
            with self.subTest(changes=changes):
                self.assertFalse(case_passed({**self.case, **changes}, 600.))

    def test_nonfinite_measurements_do_not_pass(self):
        self.assertFalse(case_passed({**self.case,
            "minimum_foot_load_after_startup_n": [250., float("nan")]}, 600.))
        for key in ("survival_seconds", "mean_root_error_m", "final_stable_seconds",
                    "settled_fraction_after_startup", "maximum_foot_error_after_startup_m",
                    "tail_root_drift_m", "tail_foot_motion_m"):
            with self.subTest(key=key):
                self.assertFalse(case_passed({**self.case, key: float("nan")}, 600.))

    def test_export_requires_matching_complete_long_and_unseen_gates(self):
        from .export_stand_foundation import validate_evidence
        report = dict(schema="mimic_standing_foundation_validation_v1", passed=True,
                      checkpoint_sha256="actor", reference_sha256="reference", model_sha256="plant",
                      seconds=600., velocity_noise_m_s=0.,
                      cases=[{**self.case, "posture": i} for i in range(8)])
        validate_evidence(report, "actor", "reference", "plant", False)
        with self.assertRaises(ValueError):
            validate_evidence(report, "another-actor", "reference", "plant", False)
        with self.assertRaises(ValueError):
            validate_evidence(report, "actor", "reference", "plant", True)
        unseen = {**report, "seconds": 60., "velocity_noise_m_s": .04,
                  "cases": [{**self.case, "posture": i % 8} for i in range(64)]}
        validate_evidence(unseen, "actor", "reference", "plant", True)
        unseen["cases"][17]["survived"] = False
        with self.assertRaises(ValueError):
            validate_evidence(unseen, "actor", "reference", "plant", True)


class NativeValidationTests(unittest.TestCase):
    def test_late_reset_preserves_posture_loads_and_chronological_delay(self):
        if not all(os.environ.get(k) for k in ("MIMICKIT_PATH", "MIMIC_STATIONARY_REFERENCE")):
            self.skipTest("Set MimicKit and stationary-reference paths")
        import tempfile
        import numpy as np
        from .runtime import activate
        activate(Path(os.environ["MIMICKIT_PATH"]))
        from envs.base_env import EnvMode
        from .stand_foundation_task import FootAnchoredStandTask
        from .native_engine import NativeDummyEngine
        from .rig import Rig
        from .baseline import sha256
        rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        reference = Path(os.environ["MIMIC_STATIONARY_REFERENCE"])
        task = FootAnchoredStandTask(rig, reference, 32, engine_factory=NativeDummyEngine,
                                    foot_anchor_sigma_m=.02)
        for tick in range(7):
            task.step(torch.full((32, 30), .01 * tick))
        expected = task.observations().clone()
        control = task.engine.policy_control
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.npz"
            np.savez(path, posture=np.arange(8), qpos=task.engine.native_qpos()[:8].numpy(),
                qvel=task.engine._array("qvel")[:8].numpy(),
                ground=task.engine.get_ground_contact_forces(0)[:8].numpy(),
                active=control.active[:8].numpy(),
                pending=torch.stack([control.pending[(control.cursor+i)%2, :8] for i in range(2)], 1).numpy(),
                reference_sha256=np.array(sha256(reference)), model_sha256=np.array(sha256(rig.path)))
            task.load_late_states(path)
            # Bank chronology must survive a different destination ring cursor.
            control.cursor = 1 - control.cursor
            task.set_mode(EnvMode.TRAIN)
            obs, _ = task.reset()
            restored = control.active[:, 60] > 0
            self.assertTrue(restored.any())
            self.assertTrue((~restored).any())
            torch.testing.assert_close(obs[restored], expected[restored], atol=2e-6, rtol=1e-5)
            task.set_mode(EnvMode.TEST)
            task.reset()
            self.assertEqual(int(control.pending.count_nonzero()), 0)

    def test_quiet_anchor_penalizes_either_foot_equally(self):
        if not os.environ.get("MIMICKIT_PATH"):
            self.skipTest("Set MimicKit for the task interface")
        from .runtime import activate
        activate(Path(os.environ["MIMICKIT_PATH"]))
        from .stand_foundation_task import foot_anchor_multiplier
        target = torch.zeros(3, 2, 3)
        feet = target.clone()
        feet[1, 0, 0] = .02
        feet[2, 1, 0] = .02
        values = foot_anchor_multiplier(feet, target, .02)
        self.assertEqual(float(values[0]), 1.)
        torch.testing.assert_close(values[1], torch.exp(torch.tensor(-.5)))
        torch.testing.assert_close(values[1], values[2])

    def test_reward_free_validation_preserves_physics_contacts_and_termination(self):
        if not all(os.environ.get(k) for k in ("MIMICKIT_PATH", "MIMIC_STATIONARY_REFERENCE")):
            self.skipTest("Set MimicKit and stationary-reference paths for native parity")
        from .runtime import activate
        activate(Path(os.environ["MIMICKIT_PATH"]))
        from .continuous_stand_task import ContinuousStandTask
        from .native_engine import NativeDummyEngine
        from .rig import Rig
        from .validate_stand_foundation import step_native
        rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        reference = Path(os.environ["MIMIC_STATIONARY_REFERENCE"])
        ordinary = ContinuousStandTask(rig, reference, engine_factory=NativeDummyEngine)
        measured = ContinuousStandTask(rig, reference, engine_factory=NativeDummyEngine)
        ordinary.episode_seconds = measured.episode_seconds = 1.
        for _ in range(128):
            action = torch.zeros(8, 30)
            _, _, expected, _ = ordinary.step(action)
            actual = step_native(measured, action)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(measured.observations(), ordinary.observations(), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
