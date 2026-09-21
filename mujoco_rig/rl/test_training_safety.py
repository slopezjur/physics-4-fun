"""Regression tests for terminal targets, protected adaptation and seed selection."""
import copy
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from perturb_env import PerturbEnv
from ppo import PPO
from recovery_reward import REWARD_VERSION
from seed_validation import quiet_rejection
from train import (EpisodeStats, PerturbTask, collect_rollout, parse_args,
                   save_checkpoint, seed_from_checkpoint)
import overnight


def batch_for(algo, n=32):
    obs = torch.randn(n, 3)
    action, logp, value = algo.act(obs)
    return obs, action, logp, torch.randn(n), torch.full((n,), 10.0), value


class TimeoutTests(unittest.TestCase):
    def test_timeout_bootstraps_but_fall_does_not_and_neither_leaks_across_reset(self):
        algo = PPO(3, 2)
        rewards = torch.tensor([[1., 1.], [1000., 1000.]])
        values = torch.tensor([[100., 100.], [900., 900.]])
        dones = torch.tensor([[1., 1.], [0., 0.]])
        timeout_values = torch.tensor([[100., 0.], [0., 0.]])
        adv, ret = algo.compute_returns(rewards, dones, values, torch.tensor([900., 900.]),
                                        timeout_values)
        torch.testing.assert_close(ret[0], torch.tensor([100., 1.]))
        torch.testing.assert_close(adv[0], torch.tensor([0., -99.]))

    def test_real_reset_preserves_final_observation_and_fall_wins_over_timeout(self):
        env = PerturbEnv(num_envs=2)
        env.episode_length_buf[:] = env.max_episode_length - 1
        env.datas[1].qpos[2] = 0.1
        mujoco.mj_forward(env.model, env.datas[1])
        obs, _, done, extras = env.step(torch.full((2, env.num_actions), .1))
        self.assertTrue(done.all())
        self.assertTrue(extras['time_outs'][0])
        self.assertFalse(extras['time_outs'][1])
        torch.testing.assert_close(extras['terminal_observation'][0, -33:-3], torch.full((30,), .1))
        torch.testing.assert_close(obs[0, -33:-3], torch.zeros(30))

    def test_rollout_reads_terminal_value_and_keeps_raw_episode_return(self):
        class Env:
            num_envs, num_obs, num_actions = 1, 3, 2
            def step(self, action):
                return (torch.full((1, 3), 999.), torch.ones(1), torch.ones(1, dtype=torch.bool),
                        dict(time_outs=torch.ones(1, dtype=torch.bool),
                             terminal_observation=torch.full((1, 3), 5.)))
        algo = PPO(3, 2)
        with patch.object(algo.net, 'value', side_effect=lambda obs: obs[:, 0]):
            episodes = EpisodeStats(1, 'cpu')
            buf, _, _, _ = collect_rollout(Env(), algo, torch.zeros(1, 3), 1, episodes, 'cpu')
        self.assertEqual(buf['timeout_value'].item(), 5)
        self.assertEqual(buf['rew'].item(), 1)
        self.assertEqual(episodes.returns, [1.0])


class OptimizerTests(unittest.TestCase):
    def test_value_diagnostics_measure_collection_values_before_fitting(self):
        algo = PPO(3, 2, epochs=2, minibatches=2, value_clip=None)
        obs, action, logp, _, _, _ = batch_for(algo)
        target = torch.linspace(1, 32, 32)
        old_value = target + 5
        advantage = target - old_value
        stats = algo.update((obs, action, logp, advantage, target, old_value), critic_only=True)
        self.assertAlmostEqual(stats['value_ev_before'], 1.0)
        self.assertAlmostEqual(stats['value_bias_before'], 5.0)
        self.assertAlmostEqual(stats['value_rmse_before'], 5.0)
        self.assertAlmostEqual(stats['advantage_std'], 0.0)
        self.assertNotAlmostEqual(stats['explained_variance'], stats['value_ev_before'])

    def test_exploration_scale_preserves_mean_pattern_and_does_not_repeat_on_resume(self):
        source, scaled, resumed = PPO(3, 2), PPO(3, 2), PPO(3, 2)
        source.update(batch_for(source))
        with torch.no_grad():
            source.net.log_std.copy_(torch.tensor([.12, .20]).log())
        original_std = source.net.log_std.exp().detach().clone()
        env = SimpleNamespace(num_envs=1, num_obs=3, num_actions=2, reward_version=REWARD_VERSION)
        with tempfile.TemporaryDirectory() as tmp:
            initial, continued = pathlib.Path(tmp) / 'initial.pt', pathlib.Path(tmp) / 'continued.pt'
            save_checkpoint(initial, source, env, SimpleNamespace(task='perturb', ball_every=(4, 7)), 16, 6)
            args = parse_args(['--init_from', str(initial), '--exploration_scale', '.25'])
            seed_from_checkpoint(scaled, env, args, 'cpu')
            torch.testing.assert_close(scaled.net.log_std.exp(), original_std * .25)
            for left, right in zip(source.net.actor.parameters(), scaled.net.actor.parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
                for key, value in source.opt.state[left].items():
                    torch.testing.assert_close(value, scaled.opt.state[right][key], rtol=0, atol=0)
            self.assertNotIn(scaled.net.log_std, scaled.opt.state)
            self.assertEqual(scaled.critic_warmup_remaining, 50)
            save_checkpoint(continued, scaled, env, args, 16, 6)
            seed_from_checkpoint(resumed, env, parse_args(['--init_from', str(continued)]), 'cpu')
        for left, right in zip(scaled.net.parameters(), resumed.net.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(resumed.critic_warmup_remaining, 50)

    def test_identity_exploration_scale_does_not_reset_optimizer_or_warmup(self):
        source, target = PPO(3, 2), PPO(3, 2)
        source.update(batch_for(source))
        env = SimpleNamespace(num_envs=1, num_obs=3, num_actions=2, reward_version=REWARD_VERSION)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'identity.pt'
            save_checkpoint(path, source, env, SimpleNamespace(task='perturb', ball_every=(4, 7)), 16, 6)
            seed_from_checkpoint(target, env,
                                 parse_args(['--init_from', str(path), '--exploration_scale', '1']), 'cpu')
        self.assertIn(target.net.log_std, target.opt.state)
        self.assertEqual(target.critic_warmup_remaining, 0)
        torch.testing.assert_close(source.net.log_std, target.net.log_std, rtol=0, atol=0)

    def test_warmup_preserves_actor_std_and_existing_adam_state(self):
        torch.manual_seed(4)
        algo = PPO(3, 2, epochs=2, minibatches=2, value_clip=None)
        batch = batch_for(algo)
        algo.update(batch)
        actor = copy.deepcopy(algo.net.actor.state_dict())
        std = algo.net.log_std.detach().clone()
        optimizer = copy.deepcopy(algo.opt.state_dict())
        before = algo.net.value(batch[0]).detach().clone()
        stats = algo.update(batch, critic_only=True)
        for key, value in algo.net.actor.state_dict().items():
            torch.testing.assert_close(value, actor[key], rtol=0, atol=0)
        torch.testing.assert_close(algo.net.log_std, std, rtol=0, atol=0)
        for idx, state in optimizer['state'].items():
            for key, value in state.items():
                torch.testing.assert_close(algo.opt.state_dict()['state'][idx][key], value, rtol=0, atol=0)
        self.assertFalse(torch.equal(before, algo.net.value(batch[0])))
        self.assertEqual(stats['actor_updates'], 0)

    def test_critic_gradient_scale_does_not_change_actor_update(self):
        a, b = PPO(3, 2, epochs=1), PPO(3, 2, epochs=1)
        b.net.load_state_dict(a.net.state_dict())
        batch = batch_for(a)
        huge_target = (*batch[:4], torch.full_like(batch[4], 1e6), batch[5])
        torch.manual_seed(20)
        a.update(batch)
        torch.manual_seed(20)
        b.update(huge_target)
        for left, right in zip(a.actor_parameters, b.actor_parameters):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_kl_stops_actor_before_all_minibatches_are_used(self):
        torch.manual_seed(3)
        algo = PPO(3, 2, lr=.05, lr_max=.05, desired_kl=1e-5, epochs=4, minibatches=4)
        stats = algo.update(batch_for(algo, 64))
        self.assertEqual(stats['kl_stopped'], 1)
        self.assertGreater(stats['actor_updates'], 0)
        self.assertLess(stats['actor_updates'], 16)
        self.assertLess(algo.lr, .05)

    def test_actor_lr_ceiling_applies_from_first_step(self):
        algo = PPO(3, 2, lr_max=3e-5)
        self.assertEqual(algo.opt.param_groups[0]['lr'], 3e-5)

    def test_small_lr_ceiling_still_allows_reduction_after_kl_stop(self):
        algo = PPO(3, 2, lr_max=1e-6, desired_kl=1e-15)
        stats = algo.update(batch_for(algo))
        self.assertTrue(stats['kl_stopped'])
        self.assertLess(algo.lr, 1e-6)

    def test_same_version_resume_preserves_warmup_and_both_optimizers(self):
        source, target = PPO(3, 2), PPO(3, 2)
        source.update(batch_for(source))
        source.critic_warmup_remaining = 17
        env = SimpleNamespace(num_envs=1, num_obs=3, num_actions=2, reward_version=REWARD_VERSION)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'resume.pt'
            save_checkpoint(path, source, env, SimpleNamespace(task='perturb', ball_every=(4, 7)), 16, 2)
            seed_from_checkpoint(target, env, SimpleNamespace(task='perturb', init_from=path, reset_std=False,
                                 critic_warmup_iters=None), 'cpu')
        self.assertEqual(target.critic_warmup_remaining, 17)
        self.assertEqual(len(target.opt.state), len(source.opt.state))
        self.assertEqual(len(target.critic_opt.state), len(source.critic_opt.state))
        for left, right in zip(target.net.parameters(), source.net.parameters()):
            torch.testing.assert_close(left, right)

    def test_old_target_version_resets_critic_and_optimizers_but_keeps_behavior(self):
        source, target = PPO(3, 2), PPO(3, 2)
        source.update(batch_for(source))
        fresh_critic = copy.deepcopy(target.net.critic.state_dict())
        env = SimpleNamespace(num_envs=1, num_obs=3, num_actions=2, reward_version=REWARD_VERSION)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'old.pt'
            save_checkpoint(path, source, env, SimpleNamespace(task='perturb', ball_every=(4, 7)), 16, 2)
            checkpoint = torch.load(path, weights_only=False)
            checkpoint.pop('training_version')
            torch.save(checkpoint, path)
            seed_from_checkpoint(target, env, SimpleNamespace(task='perturb', init_from=path, reset_std=False,
                                 critic_warmup_iters=None), 'cpu')
        self.assertEqual(target.critic_warmup_remaining, 50)
        self.assertFalse(target.opt.state)
        self.assertFalse(target.critic_opt.state)
        for key, value in target.net.critic.state_dict().items():
            torch.testing.assert_close(value, fresh_critic[key], rtol=0, atol=0)
        for left, right in zip(target.actor_parameters, source.actor_parameters):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_incomplete_same_version_optimizer_state_refuses_resume(self):
        source, target = PPO(3, 2), PPO(3, 2)
        env = SimpleNamespace(num_envs=1, num_obs=3, num_actions=2, reward_version=REWARD_VERSION)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'incomplete.pt'
            save_checkpoint(path, source, env, SimpleNamespace(task='perturb', ball_every=(4, 7)), 16, 2)
            checkpoint = torch.load(path, weights_only=False)
            checkpoint.pop('critic_optimizer')
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'optimizer state'):
                seed_from_checkpoint(target, env, SimpleNamespace(init_from=path, reset_std=False), 'cpu')


class GateTests(unittest.TestCase):
    def test_exploration_scale_rejects_invalid_or_ambiguous_requests(self):
        cases = [['--init_from', 'seed.pt', '--exploration_scale', x]
                 for x in ('0', '-1', 'nan', 'inf')]
        cases += [['--exploration_scale', '.25'],
                  ['--init_from', 'seed.pt', '--exploration_scale', '.25', '--reset_std']]
        for args in cases:
            with self.subTest(args=args), patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                parse_args(args)

    @unittest.skipUnless(os.name == 'nt' and shutil.which('pwsh'), 'PowerShell wrapper runs on Windows')
    def test_resume_wrapper_does_not_override_saved_difficulty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            captured = root / 'args.json'
            fake_python = root / 'capture.ps1'
            fake_python.write_text(
                '$args | ConvertTo-Json -Compress | Set-Content -LiteralPath $env:P4F_TEST_ARGS\n'
                '$global:LASTEXITCODE = 0\n')
            wrapper = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'train.ps1'
            result = subprocess.run([shutil.which('pwsh'), '-NoProfile', '-File', str(wrapper),
                                     '-Task', 'perturb', '-InitFrom', 'saved-stage.pt', '-Minutes', '5'],
                                    env=dict(os.environ, P4F_MUJOCO_PYTHON=str(fake_python),
                                             P4F_TEST_ARGS=str(captured)),
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            args = json.loads(captured.read_text(encoding='utf-8-sig'))
            self.assertNotIn('--ball_speed', args)
            self.assertEqual(args[args.index('--seed') + 1], 'saved-stage.pt')
            self.assertIn('--no_promote', args)

    def test_seed_curriculum_recommendation_applies_to_first_chunk(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict('os.environ', {'P4F_NIGHT': tmp}), \
             patch.object(sys, 'argv', ['overnight', '--task', 'perturb', '--seed', 'good.pt',
                  '--sessions', '1', '--minutes', '5', '--ball_speed', '2', '--abort_at', '0',
                  '--no_promote']), \
             patch.object(overnight.POLICIES['perturb'], 'score',
                          return_value=(60, 'mastered stage', 2.2, None)), \
             patch.object(overnight, 'run') as run:
            def train(cmd, log):
                self.assertEqual(cmd[cmd.index('--speed_start') + 1], 2.2)
                self.assertEqual(cmd[cmd.index('--speed_end') + 1], 2.2)
                pathlib.Path(log).write_text('stop after checking launch arguments\n')
                return 1
            run.side_effect = train
            self.assertEqual(overnight.main(), 2)
            run.assert_called_once()

    def test_implicit_fresh_perturb_run_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'requires --init_from'):
            PerturbTask().prepare(parse_args([]))
        PerturbTask().prepare(parse_args(['--from_scratch']))

    def test_quiet_gate_fails_closed(self):
        self.assertTrue(quiet_rejection(float('nan'), 100))
        self.assertTrue(quiet_rejection(99, 100))
        self.assertTrue(quiet_rejection(100, 99))
        self.assertIsNone(quiet_rejection(100, 100))

    def test_perturb_chain_rejects_long_or_unseeded_sessions(self):
        for args in [SimpleNamespace(seed='', minutes=5), SimpleNamespace(seed='seed.pt', minutes=15),
                     SimpleNamespace(seed='seed.pt', minutes=0), SimpleNamespace(seed='seed.pt', minutes=-1)]:
            with self.assertRaises(ValueError):
                overnight.PerturbSessions().validate(args)

    def test_failed_quiet_gate_retains_seed_and_stops_repeating_failed_experiment(self):
        def train(cmd, log):
            pathlib.Path(log).write_text('it 25 return 10 ep_len 1000/1200 speed 2.00\n')
            return 0
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict('os.environ', {'P4F_NIGHT': tmp}), \
             patch.object(sys, 'argv', ['overnight', '--task', 'perturb', '--seed', 'good.pt',
                  '--sessions', '2', '--minutes', '1', '--ball_speed', '2', '--abort_at', '0', '--no_promote']), \
             patch.object(overnight, 'run', side_effect=train) as run, \
             patch.object(overnight, 'newest_run', return_value='run'), \
             patch.object(overnight, 'newest_checkpoint', side_effect=['bad1.pt', 'bad2.pt']), \
             patch.object(overnight.POLICIES['perturb'], 'score', side_effect=[
                 (60, 'seed', 2, None), (90, 'candidate', 3, 'quiet-room gate failed'),
                 (95, 'candidate', 3, 'quiet-room gate failed')]):
            self.assertEqual(overnight.main(), 2)
            self.assertEqual(run.call_count, 1)
            self.assertIn('REJECTED: quiet-room gate failed',
                          (pathlib.Path(tmp) / 'night_perturb.md').read_text())
            for call in run.call_args_list:
                cmd = call.args[0]
                self.assertEqual(cmd[cmd.index('--init_from') + 1], 'good.pt')

    def test_failed_or_empty_trainer_cannot_reuse_an_old_checkpoint(self):
        for code in (0, 1):
            with self.subTest(exit_code=code), tempfile.TemporaryDirectory() as tmp, \
                 patch.dict('os.environ', {'P4F_NIGHT': tmp}), \
                 patch.object(sys, 'argv', ['overnight', '--task', 'perturb', '--seed', 'good.pt',
                      '--sessions', '2', '--minutes', '1', '--ball_speed', '2', '--abort_at', '0']), \
                 patch.object(overnight, 'run') as run, \
                 patch.object(overnight, 'newest_checkpoint') as checkpoint, \
                 patch.object(overnight.subprocess, 'run') as promote, \
                 patch.object(overnight.POLICIES['perturb'], 'score', return_value=(60, 'seed', 2, None)):
                def train(cmd, log):
                    pathlib.Path(log).write_text('No usable training output.\n')
                    return code
                run.side_effect = train
                self.assertEqual(overnight.main(), 2)
                self.assertEqual(run.call_count, 1)
                checkpoint.assert_not_called()
                promote.assert_not_called()


if __name__ == '__main__':
    unittest.main()
