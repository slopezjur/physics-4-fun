"""Behavior retention, ablation isolation and immutable checkpoint continuation."""
import copy
import hashlib
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from policy_reference import PolicyReference, REFERENCE_VERSION
from ppo import PPO
from recovery_reward import REWARD_VERSION
from train import configure_reference, parse_args, save_checkpoint, seed_from_checkpoint
from build_policy_reference import collect


def data(obs=3, actions=2):
    return {'version': REFERENCE_VERSION, 'source': {'checkpoint': 'fixed-teacher.pt'},
            'scale': torch.ones(actions),
            'groups': {'quiet': {'obs': torch.zeros(3, obs), 'mean': torch.zeros(3, actions)},
                       'impact': {'obs': torch.ones(17, obs), 'mean': torch.ones(17, actions)}}}


class ReferenceTests(unittest.TestCase):
    def test_collection_preserves_alternating_control_phases(self):
        class Env:
            max_episode_length, num_obs, num_actions, pelvis = 40, 3, 2, 0
            dt, decimation, rest_pelvis_z = .25, 1, 1.0
            settle_hold = np.ones(1)
            datas = [SimpleNamespace(xpos=np.array([[0., 0., 1.]]), xmat=np.eye(3).reshape(1, 9))]
            def reset_all(self):
                self.t = 0
                return torch.zeros(1, 3)
            def step(self, action):
                self.t += 1
                return torch.full((1, 3), float(self.t % 2)), None, None, None
        with patch('build_policy_reference.PerturbEnv', return_value=Env()):
            result = collect(SimpleNamespace(actor=lambda obs: obs[:, :2]),
                             ball=False, num_envs=1, seconds=10, seed=101)
        self.assertEqual(set(result['obs'][:, 0].tolist()), {0., 1.})
        torch.testing.assert_close(result['mean'], result['obs'][:, :2])

    def test_equal_group_weight_and_gradient_with_frozen_targets(self):
        source = data()
        reference = PolicyReference(source)
        actor = torch.nn.Linear(3, 2, bias=False)
        torch.nn.init.zeros_(actor.weight)
        source['groups']['impact']['mean'].zero_()
        loss = reference.loss(actor)
        self.assertAlmostEqual(loss.item(), .5)
        loss.backward()
        self.assertGreater(actor.weight.grad.abs().sum().item(), 0)
        self.assertFalse(reference.scale.requires_grad)
        self.assertTrue(all(not x.requires_grad for pair in reference.groups.values() for x in pair))

    def test_sampling_does_not_consume_training_rng_and_resume_preserves_stream(self):
        reference = PolicyReference(data())
        actor = torch.nn.Linear(3, 2)
        before = torch.random.get_rng_state().clone()
        reference.loss(actor)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        saved = reference.state_dict()
        restored = PolicyReference.from_state_dict(saved)
        torch.testing.assert_close(reference.loss(actor), restored.loss(actor), rtol=0, atol=0)
        torch.testing.assert_close(reference.generator.get_state(), restored.generator.get_state())

    def test_invalid_data_fails_closed(self):
        for mutation in ('version', 'empty', 'scale', 'width', 'nan'):
            source = data()
            if mutation == 'version': source['version'] = 'unknown'
            if mutation == 'empty': source['groups']['quiet']['obs'] = torch.empty(0, 3)
            if mutation == 'scale': source['scale'][0] = 0
            if mutation == 'width': source['groups']['impact']['obs'] = torch.ones(17, 4)
            if mutation == 'nan': source['groups']['impact']['mean'][0, 0] = float('nan')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                PolicyReference(source)

    def test_reference_penalty_reduces_drift_against_same_unconstrained_update(self):
        torch.manual_seed(17)
        plain, anchored = PPO(3, 2, epochs=5, desired_kl=1), PPO(3, 2, epochs=5, desired_kl=1)
        with torch.no_grad():
            plain.net.actor[-1].bias.fill_(.2)
        anchored.net.load_state_dict(plain.net.state_dict())
        source = data()
        source['groups']['impact']['mean'].zero_()
        anchored.reference = PolicyReference(source, coefficient=100)
        obs = torch.randn(64, 3)
        act, logp, val = plain.act(obs)
        # The policy gradient deliberately rewards moving away from the teacher.
        batch = (obs, act, logp, (act - .2).sum(-1), torch.ones(64), val)
        torch.manual_seed(21)
        plain.update(batch)
        torch.manual_seed(21)
        anchored.update(batch)
        self.assertLess(sum(anchored.reference.measure(anchored.net.actor).values()),
                        sum(anchored.reference.measure(plain.net.actor).values()))

    def test_warmup_does_not_consume_reference_samples_or_move_actor(self):
        algo = PPO(3, 2)
        algo.reference = PolicyReference(data())
        obs = torch.randn(16, 3)
        act, logp, val = algo.act(obs)
        actor = copy.deepcopy(algo.net.actor.state_dict())
        rng = algo.reference.generator.get_state().clone()
        algo.update((obs, act, logp, torch.randn(16), torch.ones(16), val), critic_only=True)
        for key, value in actor.items():
            torch.testing.assert_close(value, algo.net.actor.state_dict()[key], rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, algo.reference.generator.get_state()))

    def test_zero_coefficient_is_an_exact_unconstrained_control(self):
        torch.manual_seed(4)
        plain, control = PPO(3, 2), PPO(3, 2)
        control.net.load_state_dict(plain.net.state_dict())
        control.reference = PolicyReference(data(), coefficient=0)
        obs = torch.randn(32, 3)
        act, logp, val = plain.act(obs)
        batch = (obs, act, logp, torch.randn(32), torch.ones(32), val)
        torch.manual_seed(6)
        plain.update(batch)
        torch.manual_seed(6)
        control.update(batch)
        for a, b in zip(plain.net.parameters(), control.net.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_checkpoint_keeps_original_reference_not_resumed_actor(self):
        source, target = PPO(3, 2), PPO(3, 2)
        source.reference = PolicyReference(data(), coefficient=2)
        with torch.no_grad(): source.net.actor[-1].bias.fill_(.75)
        env = SimpleNamespace(num_envs=1, num_obs=3, num_actions=2, reward_version=REWARD_VERSION)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'seed.pt'
            save_checkpoint(path, source, env, SimpleNamespace(task='perturb', ball_every=(4, 7)), 16, 2)
            seed_from_checkpoint(target, env, SimpleNamespace(task='perturb', init_from=path,
                                 reset_std=False, critic_warmup_iters=None), 'cpu')
        self.assertEqual(target.reference.coefficient, 2)
        self.assertEqual(target.reference.source['checkpoint'], 'fixed-teacher.pt')
        torch.testing.assert_close(target.reference.groups['quiet'][1], torch.zeros(3, 2))
        self.assertGreater(target.reference.measure(target.net.actor)['reference_quiet'], 0)

    def test_coefficient_requires_reference_and_wrong_plant_is_rejected(self):
        algo, env = PPO(3, 2), SimpleNamespace(num_obs=3, num_actions=2)
        with self.assertRaisesRegex(ValueError, 'requires'):
            configure_reference(algo, env, parse_args(['--reference_coef', '1']))
        algo.reference = PolicyReference(data())
        with self.assertRaisesRegex(ValueError, 'plant differs'):
            configure_reference(algo, env, parse_args([]))

    def test_repeated_reference_flag_preserves_sampling_state_and_coefficient(self):
        source = data()
        root = pathlib.Path(__file__).resolve().parents[1]
        source['source']['model_sha256'] = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ('dummy.xml', 'dummy_ball.xml')}
        algo, env = PPO(3, 2), SimpleNamespace(num_obs=3, num_actions=2)
        algo.reference = PolicyReference(source, coefficient=2)
        algo.reference.loss(algo.net.actor)
        previous = algo.reference
        rng = previous.generator.get_state().clone()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'targets.pt'
            torch.save(source, path)
            configure_reference(algo, env, parse_args(['--reference_data', str(path)]))
        self.assertIs(algo.reference, previous)
        self.assertEqual(algo.reference.coefficient, 2)
        self.assertTrue(torch.equal(rng, algo.reference.generator.get_state()))


if __name__ == '__main__':
    unittest.main()
