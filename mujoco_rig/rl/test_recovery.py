"""Behavioural regressions for perturb reward, recovery gates and policy learning."""
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from perturb_env import PerturbEnv
from ppo import PPO
from recovery_metrics import RecoveryMetrics
from recovery_reward import CONFIG, REWARD_VERSION, contact_transition, reward_terms, support_state
from scoring import PerturbScorer
from train import Curriculum, EpisodeStats, parse_args, seed_from_checkpoint


def fixture():
    return dict(up=np.array(1.0), pelvis_z=np.array(1.0), rest_pelvis_z=1.0,
                com=np.array([0.0, 0.0, 1.0]), velocity=np.zeros(3),
                feet=np.array([[0.0, 0.15, 0.04], [0.0, -0.15, 0.04]]),
                foot_velocity=np.zeros((2, 3)), slip_speed_sq=np.zeros(2),
                grounded=np.ones(2, dtype=bool),
                pose_error=np.array(0.0), width=np.array(0.3), split=np.array(0.0),
                rest_width=0.3, limit_excess=np.array(0.0), action=np.zeros(30),
                previous_action=np.zeros(30), rapid_replants=np.array(0.0))


class RewardTests(unittest.TestCase):
    def test_airborne_foot_never_expands_support(self):
        f = fixture()
        f['grounded'][0] = False
        before = support_state(np, f['com'], f['velocity'], f['feet'], f['grounded'])
        f['feet'][0] = [2.0, 2.0, 1.0]
        after = support_state(np, f['com'], f['velocity'], f['feet'], f['grounded'])
        np.testing.assert_allclose(before, after)

    def test_capture_placement_only_helps_when_grounded(self):
        f = fixture()
        f['com'][0] = 0.30
        f['feet'][0, :2] = f['com'][:2]
        f['grounded'][0] = False
        hover = sum(reward_terms(np, **f).values())
        f['foot_velocity'][0, 0] = 4.0
        self.assertLessEqual(sum(reward_terms(np, **f).values()), hover)
        f['foot_velocity'][:] = 0
        f['grounded'][0] = True
        self.assertGreater(sum(reward_terms(np, **f).values()), hover + 1.0)

    def test_sliding_and_ankle_reversals_cost_reward(self):
        f = fixture()
        rest = sum(reward_terms(np, **f).values())
        f['foot_velocity'][0, 0] = 0.5
        f['slip_speed_sq'][0] = 0.25
        self.assertLess(sum(reward_terms(np, **f).values()), rest)
        f['previous_action'][0], f['action'][0] = -1, 1
        self.assertAlmostEqual(reward_terms(np, **f)['action_rate'], -0.4)

    def test_staggered_stance_does_not_require_cosmetic_step(self):
        f = fixture()
        rest = reward_terms(np, **f)['stance']
        f['split'] = np.array(0.15)
        self.assertEqual(reward_terms(np, **f)['stance'], rest)

    def test_no_support_earns_no_settling_or_support(self):
        f = fixture()
        f['grounded'][:] = False
        terms = reward_terms(np, **f)
        self.assertEqual(terms['support'], 0)
        self.assertEqual(terms['settling'], 0)

    def test_numpy_torch_reward_parity(self):
        f = fixture()
        f['velocity'][:] = [0.4, -0.2, 0.1]
        f['foot_velocity'][1] = [0.5, 0.2, -0.8]
        f['slip_speed_sq'][1] = 0.29
        f['action'][:] = np.linspace(-1, 1, 30)
        expected = reward_terms(np, **f)
        actual = reward_terms(torch, **{k: torch.as_tensor(v) for k, v in f.items()})
        for key in expected:
            self.assertAlmostEqual(float(expected[key]), actual[key].item(), places=5, msg=key)

    def test_contact_hysteresis_and_repeated_landings(self):
        ground = np.ones((1, 2), dtype=bool)
        elapsed = np.full((1, 2), CONFIG.replant_interval)
        for heights, rapid_expected in [([.02, 0], False), ([.04, 0], False),
                                         ([.02, 0], False), ([0, 0], False),
                                         ([.04, 0], False), ([0, 0], True)]:
            h = np.array([heights])
            expected = contact_transition(np, h, ground, elapsed, 1 / 60)
            actual = contact_transition(torch, torch.tensor(h), torch.tensor(ground),
                                        torch.tensor(elapsed), 1 / 60)
            for a, b in zip(expected, actual):
                np.testing.assert_allclose(a, b.numpy())
            ground, elapsed, rapid = expected
            self.assertEqual(bool(rapid[0] > 0), rapid_expected)

    def test_history_resets_and_reward_reads_are_pure(self):
        env = PerturbEnv(num_envs=2)
        env.since_landing[:] = 0
        env.rapid_replants[:] = 1
        env.grounded[:] = False
        env.reset_idx([0])
        self.assertTrue(env.grounded[0].all())
        self.assertEqual(env.rapid_replants[0], 0)
        self.assertEqual(env.rapid_replants[1], 1)
        elapsed = env.since_landing.copy()
        action = np.zeros(env.num_actions)
        self.assertEqual(env.reward(0, action), env.reward(0, action))
        np.testing.assert_array_equal(elapsed, env.since_landing)


class LearningTests(unittest.TestCase):
    def test_curriculum_waits_for_settled_episode_endings(self):
        env = SimpleNamespace(ball=0, model=SimpleNamespace(body_mass=[8.0]),
                              max_episode_length=120, ball_speed=2.0)
        args = parse_args(['--stage_min_episodes', '1', '--stage_min_iters', '1'])
        curriculum = Curriculum(args, env)
        curriculum.stage_iters = curriculum.stage_episodes = 100
        episodes = EpisodeStats(1, 'cpu')
        episodes.lengths = [120.0] * 100
        episodes.recoveries = [0.0] * 100
        curriculum.maybe_promote(100, env, 16, episodes)
        self.assertEqual(env.ball_speed, 2.0)
        episodes.recoveries = [1.0] * 100
        curriculum.maybe_promote(101, env, 16, episodes)
        self.assertAlmostEqual(env.ball_speed, 2.2)
        self.assertEqual(episodes.recoveries, [])

    def test_terminal_metrics_are_captured_before_auto_reset(self):
        env = PerturbEnv(num_envs=1)
        env.episode_length_buf[:] = env.max_episode_length - 1
        env.shots_fired[:] = 1
        env.next_ball[:] = 1000
        def stable_end(i, d):
            env.settle_hold[i] = 1.0
        with patch.object(env, '_after_physics', side_effect=stable_end):
            _, _, done, extras = env.step(torch.zeros(1, env.num_actions))
        self.assertTrue(done[0])
        self.assertTrue(extras['recovered'][0])
        self.assertEqual(env.settle_hold[0], 0)

    def test_quiet_partial_episode_cannot_promote_impact_curriculum(self):
        env = PerturbEnv(num_envs=1)
        env.settle_hold[:] = 2
        env.episode_length_buf[:] = 600
        self.assertFalse(env._step_extras()['recovered'][0])
        env.shots_fired[:] = 1
        env.ball_flight[:] = 20
        self.assertFalse(env._step_extras()['recovered'][0])
        env.ball_flight[:] = 1
        self.assertTrue(env._step_extras()['recovered'][0])

    def test_temporal_loss_ignores_resets_and_exploration(self):
        algo = PPO(3, 2)
        algo.net.actor = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            algo.net.actor.weight.fill_(1.0)
        obs = torch.zeros(2, 3)
        following = torch.tensor([[1.0, 0, 0], [100.0, 0, 0]])
        loss = algo.temporal_loss(obs, following, torch.tensor([1.0, 0]))
        self.assertAlmostEqual(loss.item(), 1.0)
        loss.backward()
        self.assertIsNone(algo.net.log_std.grad)
        self.assertGreater(algo.net.actor.weight.grad.abs().sum().item(), 0)
        self.assertEqual(algo.temporal_loss(obs, following, torch.zeros(2)).item(), 0)

    def test_old_reward_seed_keeps_actor_but_resets_values_and_optimizer(self):
        source, target = PPO(3, 2), PPO(3, 2)
        with torch.no_grad():
            for parameter in source.net.parameters():
                parameter.fill_(0.1)
        initial_critic = {k: v.clone() for k, v in target.net.critic.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'old.pt'
            torch.save(dict(model=source.net.state_dict(), optimizer=source.opt.state_dict(),
                            num_obs=3, num_actions=2, lr=0.01), path)
            seed_from_checkpoint(target, SimpleNamespace(num_obs=3, num_actions=2,
                                 reward_version=REWARD_VERSION),
                                 SimpleNamespace(task='perturb', init_from=path, reset_std=False), 'cpu')
        for k, v in target.net.actor.state_dict().items():
            torch.testing.assert_close(v, source.net.actor.state_dict()[k])
        for k, v in target.net.critic.state_dict().items():
            torch.testing.assert_close(v, initial_critic[k])
        self.assertEqual(target.lr, 3e-4)


class RecoveryGateTests(unittest.TestCase):
    def test_survival_alone_and_preimpact_hold_do_not_count(self):
        m = RecoveryMetrics(2, 1 / 60, 0.4)
        f = fixture()
        for _ in range(60):
            m.record(0, f, np.zeros(3), True, 0, np.zeros(30))
        self.assertEqual(m.result(np.ones(2, bool), True)['recovered'], 0)
        for _ in range(35):
            m.record(0, f, np.zeros(3), True, 1, np.zeros(30))
        self.assertEqual(m.result(np.ones(2, bool), True)['recovered'], 0)
        for _ in range(25):
            m.record(0, f, np.zeros(3), True, 1, np.zeros(30))
        self.assertEqual(m.result(np.ones(2, bool), True)['recovered'], 50)
        f['foot_velocity'][0, 0] = 0.4
        m.record(0, f, np.zeros(3), True, 1, np.zeros(30))
        self.assertEqual(m.result(np.ones(2, bool), True)['recovered'], 0)

    def test_fallen_world_never_recovers_for_free(self):
        m = RecoveryMetrics(1, 1 / 60, 0.4)
        for _ in range(120):
            m.record(0, fixture(), np.zeros(3), False, 1, np.zeros(30))
        self.assertEqual(m.result(np.array([False]), True)['recovered'], 0)

    def test_quality_cannot_trade_away_survival(self):
        scorer = PerturbScorer()
        self.assertTrue(scorer.regression(dict(recovered=90, survived=90),
                                          dict(recovered=80, survived=95)))
        self.assertIsNone(scorer.regression(dict(survived=95), dict(survived=95)))
        self.assertTrue(np.isnan(scorer.metric(dict(survived=100))))


if __name__ == '__main__':
    unittest.main()
