"""Stationary migration, balanced critic splits and behavioral checkpoint retention."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from .checkpoints import BestCheckpoint
from .continuous_stand import EVALUATION_SCHEMA, CONTRACT, GATES, retention_passed
from .train_continuous_stand import validate_initialization, calibration_protocol, run_iterations


def metrics(passing=(0, 5, 6)):
    cases = [dict(case=i, survived=i in passing, survival_seconds=30. if i in passing else 2.,
                  mean_root_error_m=.01, final_stable_seconds=2. if i in passing else 0.) for i in range(8)]
    return dict(evaluation_schema=EVALUATION_SCHEMA, reference_sha256='same-reference', gates=GATES,
                passed=retention_passed(cases), successes=len(passing), cases=cases)


class ContinuousTrainingTests(unittest.TestCase):
    def test_migration_is_explicit_and_only_changes_reference_semantics(self):
        target = dict(observation_contract=CONTRACT, model_sha256='plant', num_obs=362,
                      pd_gain_per_torque_limit=4., reference_sha256='new', reference_duration=None,
                      reference_semantics={'phase':'constant_zero'},
                      observation_layout=[dict(name='pose', offset=0, width=361),
                                          dict(name='stationary_reference_phase_zero', offset=361, width=1)])
        source = {**target, 'observation_contract':'mimic_stand_target_pd_v2',
                  'reference_sha256':'old', 'reference_duration':6.}
        source['observation_layout'] = copy.deepcopy(target['observation_layout'])
        source['observation_layout'][-1]['name'] = 'reference_phase'
        source.pop('reference_semantics')
        with self.assertRaisesRegex(ValueError, 'explicit'):
            validate_initialization(source, target, False)
        self.assertIn('unchanged', validate_initialization(source, target, True))
        changed = {**source, 'pd_gain_per_torque_limit':5.}
        with self.assertRaisesRegex(ValueError, 'pd_gain'):
            validate_initialization(changed, target, True)
        self.assertEqual(validate_initialization(target, target, False), 'same_stationary_contract')

    def test_every_posture_has_separate_calibration_worlds(self):
        task = SimpleNamespace(ids=torch.arange(64), reference=SimpleNamespace(sha256='hash'))
        protocol = calibration_protocol(task)
        protocol.validate(64)
        for i in range(8):
            group = protocol.group_ids == i
            self.assertEqual(int((group & protocol.validation_worlds).sum()), 2)
            self.assertEqual(int((group & ~protocol.validation_worlds).sum()), 6)
        protocol.validation_worlds = task.ids % 4 == 3
        with self.assertRaisesRegex(ValueError, 'Every calibration group'):
            protocol.validate(64)

    def test_more_successes_cannot_replace_a_previously_passing_posture(self):
        with tempfile.TemporaryDirectory() as directory:
            best = BestCheckpoint(Path(directory), {})
            save = lambda p:Path(p).write_bytes(b'weights')
            self.assertTrue(best.consider(save, metrics(), iteration=0, samples=0))
            self.assertFalse(best.consider(save, metrics((0,1,2,3,5)), iteration=1, samples=32))
            self.assertEqual(best.decisions[-1]['regressions'][0]['case'], 6)
            self.assertEqual(best.decisions[-1]['reason'], 'standing_case_regression')
            self.assertTrue(best.consider(save, metrics((0,1,5,6)), iteration=2, samples=64))
            self.assertFalse(best.consider(save, metrics(), iteration=3, samples=96))
            self.assertEqual(best.decisions[-1]['regressions'][0]['case'], 1)
            mismatched = metrics((0,1,2,5,6))
            mismatched['reference_sha256'] = 'different'
            with self.assertRaisesRegex(ValueError, 'different stationary'):
                best.consider(save, mismatched, iteration=4, samples=128)

    def test_survival_without_settling_does_not_count_as_a_passing_case(self):
        with tempfile.TemporaryDirectory() as directory:
            best = BestCheckpoint(Path(directory), {})
            save = lambda p:Path(p).write_bytes(b'weights')
            best.consider(save, metrics(), iteration=0, samples=0)
            candidate = metrics((0,1,5,6))
            candidate['cases'][6]['final_stable_seconds'] = 0.
            self.assertFalse(best.consider(save, candidate, iteration=1, samples=32))

    def test_output_cadence_does_not_reset_continuous_rollouts(self):
        class Agent:
            resets = 0
            physical_steps = 0
            def _reset_envs(self):
                self.resets += 1
                self.physical_steps = 0
                return None, {}
            def _init_train(self):
                self._iter = self._sample_count = 0
            def _train_iter(self):
                self.physical_steps += 32
            def _update_sample_count(self):
                return self.physical_steps * 64
        agent = Agent()
        run_iterations(agent, 64)
        self.assertEqual(agent.resets, 1)
        self.assertEqual(agent.physical_steps, 2048)
        self.assertEqual(agent._iter, 64)
        self.assertEqual(agent._sample_count, 131072)


if __name__ == '__main__':
    unittest.main()
