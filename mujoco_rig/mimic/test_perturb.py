"""Push timing, force lifetime and settled-recovery semantics."""
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
import torch

from .perturb import PushTrial, RecoveryMetrics, baseline_trials
from .native_engine import NativeDummyEngine
from .rig import Rig


class PerturbTests(unittest.TestCase):
    def test_force_window_is_half_open_and_impulse_is_explicit(self):
        trial = PushTrial("push", 0., (20., 0., 0.))
        self.assertEqual([i for i in range(70) if any(trial.force_at(i))], list(range(60, 66)))
        self.assertAlmostEqual(sum(trial.force_at(i)[0] for i in range(70)) * .016668, 2.00016)
        for trial in baseline_trials():
            trial.validate(.016668, 5.9667)
        self.assertEqual(len(baseline_trials()), 52)

    def test_invalid_or_truncated_pulses_rejected(self):
        trial = PushTrial("push", 0., (20., 0., 0.))
        for change in ({"force": (float("nan"), 0, 0)}, {"duration_steps": 0}, {"start_step": -1},
                       {"phase": 2}, {"start_step": 290}, {"duration_steps": 2.5}):
            with self.assertRaises(ValueError):
                replace(trial, **change).validate(.016668, 5.9667)

    def test_recovery_requires_post_push_window_and_survival(self):
        trial = PushTrial("push", 0., (20., 0., 0.), start_step=0, duration_steps=1, trial_steps=40)
        metric = RecoveryMetrics(trial, 1 / 60)
        q = np.array([0., 0., .9, 1., 0., 0., 0.])
        v, feet, loads = np.zeros(6), np.zeros((2, 3)), [20, 20]
        for step in range(31):
            metric.sample(step, q, v, feet, loads)
        self.assertIsNone(metric.recovery_seconds)
        metric.sample(31, q, v, feet, loads)
        self.assertAlmostEqual(metric.recovery_seconds, .5)
        self.assertFalse(metric.result(False)["recovered"])
        for step in range(32, 41):
            metric.sample(step, q, v, feet, loads)
        self.assertTrue(metric.result(False)["recovered"])
        self.assertFalse(metric.result(True)["recovered"])
        metric.sample(40, q, np.ones(6), feet, loads)
        self.assertFalse(metric.result(False)["recovered"])

    def test_contact_switches_and_foot_travel_are_measured_after_onset(self):
        metric = RecoveryMetrics(PushTrial("push", 0., (0., 0., 0.), start_step=1), .016668)
        q, v = np.array([0., 0., .9, 1., 0., 0., 0.]), np.zeros(6)
        metric.sample(0, q, v, np.zeros((2, 3)), [20, 20])
        metric.sample(1, q, v, np.ones((2, 3)), [20, 20])
        metric.sample(2, q, v, np.ones((2, 3)) + [0.1, 0, 0], [0, 20])
        result = metric.result(False)
        self.assertAlmostEqual(result["foot_travel_m"], .2)
        self.assertEqual(result["contact_switches"], 1)

    def test_native_force_expires_and_partial_reset_preserves_peer(self):
        rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
        engine = NativeDummyEngine(rig, 2)
        q, v = np.tile(rig.rest, (2, 1)), np.zeros((2, rig.model.nv))
        engine.reset_envs(torch.tensor([0, 1]), q, v)
        body = rig.model.body("Chest").id
        engine.set_external_force(body, [[10, 0, 0], [20, 0, 0]])
        engine.reset_envs(torch.tensor([0]), q[:1], v[:1])
        self.assertEqual(engine.datas[0].xfrc_applied[body, 0], 0)
        self.assertEqual(engine.datas[1].xfrc_applied[body, 0], 20)
        engine.step()
        self.assertTrue(all(np.count_nonzero(d.xfrc_applied) == 0 for d in engine.datas))
        with self.assertRaises(ValueError):
            engine.set_external_force(body, [[float("nan"), 0, 0], [0, 0, 0]])


if __name__ == "__main__":
    unittest.main()
