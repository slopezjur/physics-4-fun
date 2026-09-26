"""Stationary references must not inherit clip clocks or short-survival gates."""
import os
import json
from pathlib import Path
import tempfile
import unittest

import torch

from .continuous_stand import (CONTRACT, GATES, StationaryReference, calibrate_safeguard,
                               evaluate_retention, retention_passed)


def cases():
    return [dict(case=i, survived=True, survival_seconds=30., mean_root_error_m=.01,
                 final_stable_seconds=2.) for i in range(8)]


class ProtocolTests(unittest.TestCase):
    def reference(self):
        reference = StationaryReference.__new__(StationaryReference)
        reference.qpos = torch.arange(8 * 46).reshape(8, 46).float()
        reference.qvel = torch.zeros(8, 45)
        return reference

    def test_unlimited_time_and_future_samples_preserve_each_case(self):
        reference = self.reference()
        ids = torch.arange(8)
        q, v = reference.sample(torch.full((8,), 100000.), ids)
        torch.testing.assert_close(q, reference.qpos)
        self.assertEqual(float(v.abs().sum()), 0.)
        future, _ = reference.sample(torch.ones(8, 3) * 500., ids)
        torch.testing.assert_close(future, reference.qpos[:, None].expand(8, 3, 46))
        q[0] = 0
        self.assertNotEqual(float(reference.qpos[0].sum()), 0.)

    def test_invalid_time_fails(self):
        for value in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.reference().sample(torch.full((8,), value), torch.arange(8))

    def test_survival_requires_complete_settled_30_second_cases(self):
        self.assertTrue(retention_passed(cases()))
        for key, value in [('survived', False), ('survival_seconds', 29.99),
                           ('mean_root_error_m', .051), ('final_stable_seconds', 1.99)]:
            changed = cases()
            changed[3][key] = value
            self.assertFalse(retention_passed(changed))
        with self.assertRaises(ValueError):
            retention_passed(cases()[:-1])
        changed = cases()
        changed[0]['mean_root_error_m'] = float('nan')
        with self.assertRaises(ValueError):
            retention_passed(changed)

    def test_calibration_cannot_certify_a_failing_teacher_or_nonmonotonic_results(self):
        row = lambda name, kl, passed: dict(label=name, anchor_kl=dict(max=kl), retention=dict(passed=passed))
        result = calibrate_safeguard([row('teacher', 0., False), row('other', .4, True)])
        self.assertEqual(result['status'], 'no_passing_teacher')
        self.assertIsNone(result['recommended_kl_limit'])
        result = calibrate_safeguard([row('teacher', 0., True), row('good', 1.8, True), row('bad', 1.5, False)])
        self.assertEqual(result['status'], 'not_separable_by_kl')
        result = calibrate_safeguard([row('teacher', 0., True), row('good', .4, True), row('bad', 1.5, False)])
        self.assertEqual(result['status'], 'sampled_separation_only')
        self.assertIsNone(result['recommended_kl_limit'])


class NativeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(os.environ.get(k) for k in ('MIMICKIT_PATH', 'MIMIC_STATIONARY_REFERENCE')):
            raise unittest.SkipTest('Set MIMICKIT_PATH and MIMIC_STATIONARY_REFERENCE for native checks')
        from .runtime import activate
        activate(Path(os.environ['MIMICKIT_PATH']))
        from .continuous_stand_task import ContinuousStandTask
        from .native_engine import NativeDummyEngine
        from .rig import Rig
        cls.path = Path(os.environ['MIMIC_STATIONARY_REFERENCE'])
        cls.rig = Rig.load(Path(__file__).resolve().parents[1] / 'dummy.xml')
        cls.task = ContinuousStandTask(cls.rig, cls.path, engine_factory=NativeDummyEngine)

    def setUp(self):
        self.task.episode_seconds = 30.
        self.task.reset()

    def test_phase_future_targets_and_reward_are_time_independent(self):
        before = self.task.observations().clone()
        reward = self.task.reward().clone()
        self.task.steps[:] = 600000
        torch.testing.assert_close(self.task.observations(), before)
        torch.testing.assert_close(self.task.reward(), reward)
        self.assertEqual(float(before[:, -1].abs().sum()), 0.)
        contract = self.task.observation_contract()
        self.assertEqual(contract['observation_contract'], CONTRACT)
        self.assertIsNone(contract['reference_duration'])

    def test_partial_reset_preserves_other_worlds_and_zero_target_velocity(self):
        before = self.task.engine.native_qpos().clone()
        self.task.steps[:] = 2000
        self.task.reset(torch.tensor([1, 3]))
        self.assertEqual(self.task.steps.tolist(), [2000, 0, 2000, 0, 2000, 2000, 2000, 2000])
        torch.testing.assert_close(self.task.engine.native_qpos(), before)
        self.task.step(torch.zeros(8, 30))
        control = self.task.engine.policy_control
        packet = control.pending[(control.cursor - 1) % len(control.pending)]
        torch.testing.assert_close(packet[:, 30:60], torch.zeros(8, 30))

    def test_legacy_loader_rejects_new_semantics_and_old_phase_is_unchanged(self):
        from .reference import StandReference
        from .stand_task import StandTask
        with self.assertRaisesRegex(ValueError, 'explicit continuous-standing'):
            StandReference(self.path, self.rig, 'cpu')
        reference = StandReference(Path(__file__).parent / 'assets/stand_reference.npz', self.rig, 'cpu')
        dummy = StandTask.__new__(StandTask)
        dummy.reference = reference
        torch.testing.assert_close(dummy.reference_phase(torch.tensor([0., reference.duration])), torch.tensor([0., 1.]))

    def test_every_reset_has_explicit_foot_and_toe_support(self):
        loads = self.task.engine.get_ground_contact_forces(0)[:, self.task.allowed_ground, 2].abs()
        self.assertTrue(bool((loads > 0).all()))

    def test_nonzero_velocity_and_old_reset_semantics_are_rejected(self):
        import numpy as np
        from .baseline import sha256
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.npz'
            with np.load(self.path) as data:
                q, v = data['qpos'].copy(), data['qvel'].copy()
            v[0, 0] = .001
            np.savez(path, qpos=q, qvel=v)
            metadata = json.loads(self.path.with_suffix('.json').read_text())
            metadata['reference_sha256'] = sha256(path)
            path.with_suffix('.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'nonzero target velocity'):
                StationaryReference(path, self.rig, 'cpu')
            del metadata['semantics']['contact_seating_m']
            path.with_suffix('.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'semantics mismatch'):
                StationaryReference(path, self.rig, 'cpu')

    def test_first_failure_cannot_be_replaced_by_a_successful_reset(self):
        from unittest.mock import patch
        from envs.base_env import DoneFlags
        real_step = self.task.step
        count = 0

        def step(action):
            nonlocal count
            count += 1
            obs, reward, done, info = real_step(action)
            done[:] = DoneFlags.NULL.value
            done[0] = DoneFlags.FAIL.value if count == 1 else DoneFlags.TIME.value
            if count == 3:
                done[1:] = DoneFlags.TIME.value
            return obs, reward, done, info

        with patch.object(self.task, 'step', side_effect=step):
            report, traces = evaluate_retention(self.task, lambda obs: torch.zeros(8, 30))
        self.assertFalse(report['passed'])
        self.assertFalse(report['cases'][0]['survived'])
        self.assertEqual(int(traces['active'][:, 0].sum()), 1)
        self.assertEqual(int(traces['active'][:, 1].sum()), 3)


if __name__ == '__main__':
    unittest.main()
