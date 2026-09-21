"""Guard data leakage, action timing and preservation of the accepted behavior seed."""
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_perturb import write_json
from learn_recovery_demonstrations import choose_checkpoint, training_groups, widen_checkpoint
from recovery_demonstrations import choose_cases, sensor_pairs
from probe_recovery_feasibility import Snapshot, initial_state
from perturb_env import PerturbEnv
from ppo import ActorCritic


class RecoveryDemonstrationTests(unittest.TestCase):
    def test_whole_trajectory_split_is_fixed_before_teacher_outcomes(self):
        with tempfile.TemporaryDirectory() as temp:
            bank = Path(temp)
            (bank/'selection').mkdir()
            write_json(bank/'protocol.json', dict(repeats=4, shards=2,
                target_bones=['Head', 'Chest'], approach_sides=['+X', '+Y', '-X', '-Y']))
            for shard in range(2):
                ids = list(range(shard, 32, 2))
                write_json(bank/'selection'/f'accepted-{shard:02d}.json',
                           dict(indices=ids, survived=[i % 4 == 0 for i in ids]))
            cases = choose_cases(bank)
            self.assertEqual(len(cases), 24)
            train = {x['trial'] for x in cases if x['split'] == 'train'}
            validation = {x['trial'] for x in cases if x['split'] == 'validation'}
            self.assertEqual(len(train), 16)
            self.assertEqual(validation, set(range(3, 32, 4)))
            self.assertFalse(train & validation)

    def test_validation_states_never_enter_training_or_handback_retention(self):
        pair = lambda value: dict(obs=torch.full((8, 140), value), mean=torch.full((8, 30), value))
        data = dict(trajectories={1: dict(case=dict(trial=1, split='train'), **pair(1.)),
                                  2: dict(case=dict(trial=2, split='validation'), **pair(99.))},
                    reference=dict(quiet=pair(0.), impact=pair(2.)))
        groups = training_groups(data, 3)
        for key in ('demo', 'quiet', 'impact'):
            self.assertFalse((groups[key][0] == 99).any())
        self.assertEqual(len(groups['demo'][0]), 3)
        self.assertEqual(len(groups['impact'][0]), 13)
        self.assertTrue((groups['validation'][0] == 99).all())

    def test_widening_retains_actor_and_std_without_mutating_source(self):
        torch.manual_seed(11)
        old = ActorCritic(105, 30)
        torch.nn.init.normal_(old.actor[-1].weight, std=.01)
        source = dict(num_obs=105, num_actions=30, model=old.state_dict())
        new = widen_checkpoint(source)
        obs = torch.randn(20, 140)
        torch.testing.assert_close(new.actor(obs), old.actor(obs[:, :105]), atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(new.log_std, old.log_std, atol=0, rtol=0)
        self.assertEqual(source['model']['actor.0.weight'].shape[1], 105)
        self.assertFalse(new.actor[0].weight[:, 105:].count_nonzero())

    def test_best_fit_cannot_bypass_retention_filter(self):
        measurements = [dict(update=250, quiet=.1, impact=0., validation=.001),
                        dict(update=500, quiet=0., impact=0., validation=.005)]
        chosen, passed = choose_checkpoint(measurements, dict(quiet_mse_limit=.001, impact_mse_limit=.001))
        self.assertTrue(passed)
        self.assertEqual(chosen['update'], 500)

    def test_sensor_labels_are_pre_action_and_native_snapshot_is_exact(self):
        torch.set_num_threads(1)
        env = PerturbEnv(num_envs=1, observation_version='foundation_v2')
        env.auto_reset = False
        env.next_ball[:] = np.inf
        for _ in range(3):
            env.step(torch.zeros(1, 30))
        snapshot = pickle.loads(pickle.dumps(Snapshot(env), protocol=5))
        actions = np.random.default_rng(3).uniform(-.05, .05, (4, 30)).astype(np.float32)
        states, controls, before = [], [], []
        for action in actions:
            before.append(env.get_observations()[0].numpy())
            env.step(torch.tensor(action[None]))
            states.append(initial_state(env)); controls.append(env.datas[0].ctrl.copy())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'trace.npz'
            np.savez(path, commands=actions, state=states, applied_controls=controls)
            pairs = sensor_pairs(snapshot, path)
        np.testing.assert_array_equal(pairs['obs'], before)
        np.testing.assert_array_equal(pairs['mean'], actions)


if __name__ == '__main__':
    unittest.main()
