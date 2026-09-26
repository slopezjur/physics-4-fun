"""Temporal gates, output-unit migration, and preservation of simulated states."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from .critic_calibration import temporal_slices, calibration_improves
from .standing_readiness import objective, initialize_value_scale, age_counts, require_coverage, prepare_episodes
from .train_continuous_stand import run_iterations


class ReadinessTests(unittest.TestCase):
    def test_objective_is_invariant_to_control_frequency(self):
        a, b = objective(1/30, 'retention_30s_v1'), objective(1/60, 'retention_30s_v1')
        self.assertAlmostEqual(a['discount'], b['discount']**2)
        self.assertAlmostEqual(a['reward_scale'], b['reward_scale']*(1+b['discount']))
        self.assertAlmostEqual(a['discount']**900, b['discount']**1800)
        self.assertAlmostEqual(b['reward_scale']/(1-b['discount']), 1.)

    def test_value_unit_migration_changes_only_head_and_clears_optimizer(self):
        head = torch.nn.Linear(2,1)
        with torch.no_grad():
            head.weight.fill_(2.)
            head.bias.fill_(3.)
        agent = SimpleNamespace(_model=SimpleNamespace(_critic_out=head),
            _critic_optimizer=SimpleNamespace(_optimizer=SimpleNamespace(state={'old':1})))
        destination = objective(1/60, 'retention_30s_v1')
        result = initialize_value_scale(agent, .99, 1., destination)
        self.assertAlmostEqual(result['scale'], .01)
        torch.testing.assert_close(head.weight, torch.full_like(head.weight, .02))
        torch.testing.assert_close(head.bias, torch.full_like(head.bias, .03))
        self.assertEqual(agent._critic_optimizer._optimizer.state, {})
        before = head.weight.detach().clone()
        result = initialize_value_scale(agent, destination['discount'], destination['reward_scale'], destination)
        self.assertAlmostEqual(result['scale'], 1.)
        torch.testing.assert_close(head.weight, before)

    def test_aggregate_improvement_cannot_hide_failure_window_regression(self):
        baseline = {'posture':{'rmse':10.}, 'before_failure_5s':{'rmse':2.}}
        candidate = {'posture':{'rmse':5.}, 'before_failure_5s':{'rmse':3.}}
        self.assertFalse(calibration_improves(candidate, baseline, baseline, ('posture',)))
        candidate['before_failure_5s']['rmse'] = 1.
        self.assertTrue(calibration_improves(candidate, baseline, baseline, ('posture',)))

    def test_failure_windows_exclude_timeouts_and_future_failure_is_not_an_input(self):
        ages = torch.tensor([0.,6.,16.,25.])
        masks = temporal_slices(ages, torch.tensor([.4,1.,4.,.2]), torch.tensor([True,True,True,False]))
        self.assertEqual(masks['before_failure_5s'].tolist(), [True,True,True,False])
        self.assertEqual(masks['before_failure_0.5s'].tolist(), [True,False,False,False])
        self.assertEqual(masks['age_15_plus'].tolist(), [False,False,True,True])

    def test_coverage_checks_each_previously_retained_posture(self):
        ids = torch.arange(256)
        counts = age_counts((ids//8).float(), ids)
        require_coverage(counts, {0,5,6})
        counts[5,2:] = 0
        with self.assertRaisesRegex(ValueError, '5'):
            require_coverage(counts, {0,5,6})

    def test_prepared_state_is_not_reset_by_training_entry(self):
        class Agent:
            def _reset_envs(self):
                raise AssertionError('Prepared physical state would be lost')
            def _init_train(self):
                self._iter = 0
            def _train_iter(self):
                self._curr_obs += 1
            def _update_sample_count(self):
                return (self._iter+1)*32
        agent = Agent()
        run_iterations(agent, 2, initial_state=(1775, {}))
        self.assertEqual(agent._curr_obs, 1777)
        self.assertEqual(agent._sample_count, 64)

    @unittest.skipUnless(os.environ.get('MIMICKIT_PATH'), 'Set MIMICKIT_PATH')
    def test_preparation_simulates_live_states_without_weight_changes(self):
        from .runtime import activate
        activate(Path(os.environ['MIMICKIT_PATH']))
        class Task:
            ids = torch.arange(256)
            device, dt = 'cpu', 1.
            ages = torch.zeros(256)
            def get_num_envs(self):
                return len(self.ids)
            def reset(self, ids=None):
                self.ages[self.ids if ids is None else ids] = 0
                return self.ages.clone(), {}
            def step(self, action):
                self.ages += 1
                return self.ages.clone(), torch.ones(256), (self.ages>=30).long()*3, {}
            def get_env_time(self):
                return self.ages
        class Agent(torch.nn.Module):
            def set_mode(self, mode):
                pass
            def _decide_action(self, obs, info):
                return torch.zeros_like(obs), {}
        task = Task()
        (obs, _), report = prepare_episodes(Agent(), task, {0,5,6})
        self.assertTrue(report['passed'])
        torch.testing.assert_close(obs, task.ages)
        self.assertGreaterEqual(float(obs.max()), 29.)
        self.assertEqual(float(obs.min()), 0.)


if __name__ == '__main__':
    unittest.main()
