"""Guard scenario balance, replay integrity, and selection/aggregation semantics."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_perturb import (HEADINGS, SEEDS, TARGET_BONES, aggregate, choose_candidate,
                               make_design, run_policy, write_json)
from perturb_env import PerturbEnv
from ppo import ActorCritic


class ImpactBenchmarkTests(unittest.TestCase):
    def test_balance_reproducibility_and_split_independence(self):
        a = make_design(SEEDS['selection'], 16, 30)
        b = make_design(SEEDS['selection'], 16, 30)
        c = make_design(SEEDS['heldout'], 16, 30)
        self.assertTrue(all(np.array_equal(a[k], b[k]) for k in a))
        self.assertFalse(np.array_equal(a['noise'], c['noise']))
        self.assertFalse(np.array_equal(a['launch_time'], c['launch_time']))
        for body in range(len(TARGET_BONES)):
            for heading in range(len(HEADINGS)):
                mask = (a['body_index'] == body) & (a['heading'] == heading)
                self.assertEqual(mask.sum(), 16)
                for shard in range(8):
                    self.assertEqual((mask & (a['replicate'] % 8 == shard)).sum(), 2)
        centered = a['theta'] - a['heading'] * np.pi/2
        self.assertTrue((np.abs(centered) <= np.pi/4).all())

    def test_selection_uses_declared_priority(self):
        score = lambda recovered, survived: {'recovered': np.array(recovered), 'survived': np.array(survived)}
        data = {'accepted': score([1, 1, 1], [1, 1, 1]),
                'a': score([1, 1, 0], [1, 1, 0]), 'b': score([1, 0, 0], [1, 1, 1])}
        self.assertEqual(choose_candidate(data), 'a')
        data['b']['recovered'] = np.array([1, 1, 0])
        self.assertEqual(choose_candidate(data), 'b')
        data['a']['survived'] = np.array([1, 1, 1])
        self.assertEqual(choose_candidate(data), 'a')

    def test_aggregate_restores_ids_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_json(folder/'p-00.json', {'indices': [2, 0], 'survived': [True, False]})
            write_json(folder/'p-01.json', {'indices': [3, 1], 'survived': [False, True]})
            np.testing.assert_array_equal(aggregate(folder, 'p', 4, 2)['survived'], [False, True, True, False])
            write_json(folder/'p-01.json', {'indices': [2, 1], 'survived': [False, True]})
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                aggregate(folder, 'p', 4, 2)
            write_json(folder/'p-00.json', {'indices': [0, 0], 'survived': [False, True]})
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                aggregate(folder, 'p', 4, 2)
            write_json(folder/'p-00.json', {'indices': [-1, 0], 'survived': [False, True]})
            with self.assertRaisesRegex(ValueError, 'Invalid'):
                aggregate(folder, 'p', 4, 2)

    def test_generated_launches_replay_without_mutation(self):
        torch.set_num_threads(1)
        env = PerturbEnv(num_envs=1, episode_seconds=6)
        n = 2
        scenario = dict(body_index=np.array([0, 3]), theta=np.array([0., 2.]),
                        launch_step=np.array([91, 99]), qpos=np.repeat(env.datas[0].qpos[None], n, axis=0).astype(np.float32),
                        start=np.zeros((n, 3), np.float32), velocity=np.zeros((n, 3), np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'policy.pt'
            net = ActorCritic(env.num_obs, env.num_actions)
            torch.save(dict(model=net.state_dict(), num_obs=env.num_obs, num_actions=env.num_actions), path)
            first = run_policy(path, scenario, generate=True)
            snapshot = {k: v.copy() for k, v in scenario.items()}
            second = run_policy(path, scenario)
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])
            for key in snapshot:
                np.testing.assert_array_equal(scenario[key], snapshot[key])
            np.testing.assert_allclose(np.linalg.norm(scenario['velocity'], axis=1), 6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
