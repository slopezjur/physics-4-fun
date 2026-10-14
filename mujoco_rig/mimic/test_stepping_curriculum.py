"""Behavioral regression checks for commanded stepping and terminal credit."""
import unittest

import torch

from .stepping_curriculum import AUTHORITY_INDEX, GOAL_WIDTH, StepConfig, StepCurriculum
from .stepping_skill_core import GOAL_START, OBSERVATION_WIDTH, PREVIOUS_LATENT_START, SteppingHead
from .train_stepping_skill import advantages


def measurement(count=2):
    feet = torch.tensor([[[0., -.15, 0.], [0., .15, 0.]]]).repeat(count, 1, 1)
    return dict(com=torch.tensor([[0., 0., 1.]]).repeat(count, 1),
                velocity=torch.zeros(count, 3), feet=feet,
                clearance=torch.zeros(count, 2), loads=torch.full((count, 2), 300.),
                pelvis_height=torch.ones(count), pelvis_up=torch.ones(count),
                angular=torch.zeros(count, 3), finite=torch.ones(count, dtype=torch.bool),
                stand=torch.ones(count, dtype=torch.bool))


class SteppingCreditTests(unittest.TestCase):
    def test_all_terminals_remove_bootstrap_and_cut_the_following_episode(self):
        reward = torch.zeros(64, 12)
        reward[1] = 1000.
        terminal = torch.zeros(64, 12, dtype=torch.bool)
        terminal[0] = True
        value = torch.full_like(reward, 2.)
        next_value = torch.full_like(reward, 500.)
        batch = dict(reward=reward, value=value, next_value=next_value, terminal=terminal)
        advantage, returns, _ = advantages(batch)
        torch.testing.assert_close(advantage[0], torch.full((12,), -2.))
        torch.testing.assert_close(returns[0], torch.zeros(12))
        self.assertGreater(float(advantage[1].min()), 1000.)

    def test_zero_head_preserves_stand_and_memory_remains_per_joint(self):
        actor = SteppingHead()
        x = torch.zeros(3, OBSERVATION_WIDTH)
        x[:, GOAL_START+AUTHORITY_INDEX] = 1.
        self.assertFalse(bool(actor(x).any()))
        x[0, PREVIOUS_LATENT_START] = .3
        x[1, PREVIOUS_LATENT_START+1] = -.2
        output = actor(x)
        self.assertAlmostEqual(float(output[0, 0]), .27, places=6)
        self.assertAlmostEqual(float(output[1, 1]), -.18, places=6)
        self.assertEqual(int(torch.count_nonzero(output)), 2)
        x[:, GOAL_START+AUTHORITY_INDEX] = 0.
        self.assertFalse(bool(actor(x).any()))

    def test_critic_is_not_gated_by_task_authority(self):
        critic = SteppingHead(outputs=1)
        with torch.no_grad():
            critic.net[-1].bias.fill_(2.)
        self.assertTrue(torch.equal(critic(torch.zeros(2, OBSERVATION_WIDTH)), torch.full((2,), 2.)))


class SteppingCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.dt = .02
        self.state = measurement()
        self.curriculum = StepCurriculum(2, self.dt, 'cpu', config=StepConfig(episode_seconds=12.))
        self.reset()

    def reset(self, level=0, ids=None, onset=0., enabled=True):
        ids = torch.arange(2) if ids is None else ids
        self.curriculum.reset(ids, self.state, side=torch.tensor([0, 1])[ids],
                              distance=torch.full((len(ids),), .04 if level == 2 and enabled else 0.),
                              onset_seconds=torch.full((len(ids),), onset),
                              enabled=torch.full((len(ids),), enabled),
                              level=torch.full((len(ids),), level, dtype=torch.long))

    def tick(self, count=1, active=None, failed=None):
        active = torch.ones(2, dtype=torch.bool) if active is None else active
        failed = torch.zeros(2, dtype=torch.bool) if failed is None else failed
        result = None
        for _ in range(count):
            result = self.curriculum.advance(self.state, failed, active)
        return result

    def test_clock_alone_cannot_complete_transfer_or_task(self):
        _, terminal, info = self.tick(300)
        self.assertFalse(bool(terminal.any()))
        self.assertTrue(torch.equal(info['phase_after'], torch.full((2,), 2)))
        self.assertFalse(bool(info['task_success'].any()))

    def test_inactive_world_keeps_all_goal_memory_and_has_zero_reward(self):
        before = self.curriculum.observations().clone()
        self.state['com'][1, 1] = -.04
        self.state['loads'][1] = torch.tensor([500., 100.])
        reward, terminal, _ = self.tick(10, active=torch.tensor([True, False]))
        torch.testing.assert_close(self.curriculum.observations()[1], before[1], rtol=0, atol=0)
        self.assertEqual(float(reward[1]), 0.)
        self.assertFalse(bool(terminal[1]))

    def test_reset_one_world_does_not_erase_another_worlds_progress(self):
        self.tick(8)
        before = self.curriculum.observations()[1].clone()
        self.reset(ids=torch.tensor([0]), onset=1.)
        torch.testing.assert_close(self.curriculum.observations()[1], before, rtol=0, atol=0)
        self.assertEqual(float(self.curriculum.observations()[0, AUTHORITY_INDEX]), 0.)

    def test_nonfinite_state_and_collapse_are_terminal(self):
        self.state['finite'][0] = False
        self.state['pelvis_height'][1] = .4
        reward, terminal, info = self.tick(failed=torch.tensor([False, True]))
        self.assertTrue(bool(terminal.all()))
        self.assertFalse(bool(info['task_success'].any()))
        self.assertTrue(bool(torch.isfinite(reward).all()))

    def test_onset_enables_authority_but_does_not_complete_a_phase(self):
        self.reset(onset=.1)
        observations = self.curriculum.observations()
        self.assertEqual(observations.shape, (2, GOAL_WIDTH))
        self.assertFalse(bool(observations[:, AUTHORITY_INDEX].any()))
        _, _, info = self.tick(6)
        self.assertTrue(bool(self.curriculum.observations()[:, AUTHORITY_INDEX].all()))
        self.assertTrue(torch.equal(info['phase_after'], torch.full((2,), 2)))

    def transfer(self, level=1):
        self.reset(level=level)
        self.state['com'][:, 1] = torch.tensor([.04, -.04])
        self.state['loads'][:] = torch.tensor([[100., 500.], [500., 100.]])
        return self.tick(5)

    def lift(self):
        self.state['stand'][:] = False
        self.state['com'][:, 1] = torch.tensor([.15, -.15])
        self.state['loads'][:] = torch.tensor([[0., 600.], [600., 0.]])
        self.state['clearance'][0, 0] = .025
        self.state['clearance'][1, 1] = .025
        return self.tick(3)

    def test_transfer_dwell_resets_after_one_invalid_interval(self):
        self.state['com'][:, 1] = torch.tensor([.04, -.04])
        self.state['loads'][:] = torch.tensor([[100., 500.], [500., 100.]])
        self.tick(4)
        self.state['loads'][:] = 300.
        self.tick()
        self.state['loads'][:] = torch.tensor([[100., 500.], [500., 100.]])
        _, _, info = self.tick(4)
        self.assertFalse(bool(info['transfer_completed'].any()))
        _, _, info = self.tick()
        self.assertTrue(bool(info['transfer_completed'].all()))

    def test_single_support_is_admissible_and_requires_continuous_clearance(self):
        _, _, info = self.transfer()
        self.assertTrue(torch.equal(info['phase_after'], torch.full((2,), 3)))
        self.state['stand'][:] = False
        self.state['com'][:, 1] = torch.tensor([.15, -.15])
        self.state['loads'][:] = torch.tensor([[0., 600.], [600., 0.]])
        self.state['clearance'][0, 0] = .025
        self.state['clearance'][1, 1] = .025
        self.tick(2)
        self.state['clearance'].zero_()
        self.tick()
        self.state['clearance'][0, 0] = .025
        self.state['clearance'][1, 1] = .025
        _, terminal, info = self.tick(2)
        self.assertFalse(bool(terminal.any()))
        self.assertFalse(bool(info['lift_completed'].any()))
        _, terminal, info = self.tick()
        self.assertTrue(bool(info['lift_completed'].all()))
        self.assertTrue(torch.equal(info['phase_after'], torch.full((2,), 5)))
        self.assertFalse(bool(terminal.any()))

    def test_loaded_slide_cannot_replace_lift_or_place(self):
        self.transfer(level=2)
        self.state['feet'][0, 0, 0] = .04
        self.state['feet'][1, 1, 0] = .04
        _, _, info = self.tick(20)
        self.assertFalse(bool(info['lift_completed'].any()))
        self.assertFalse(bool(info['place_completed'].any()))
        self.assertTrue(torch.equal(info['phase_after'], torch.full((2,), 3)))

    def test_step_sequence_never_skips_phase_and_stance_slip_is_latched(self):
        self.transfer(level=2)
        _, _, info = self.lift()
        self.assertTrue(torch.equal(info['phase_after'], torch.full((2,), 4)))
        self.state['feet'][0, 0, 0] = .04
        self.state['feet'][1, 1, 0] = .04
        _, _, info = self.tick(3)
        self.assertTrue(torch.equal(info['phase_after'], torch.full((2,), 5)))
        self.assertFalse(bool(info['land_completed'].any()))
        self.state['feet'][0, 1, 0] = .03
        self.tick()
        self.state['feet'][0, 1, 0] = 0.
        self.state['loads'][:] = 300.
        self.state['clearance'].zero_()
        _, _, info = self.tick(6)
        self.assertFalse(bool(info['land_completed'][0]))
        self.assertTrue(bool(info['land_completed'][1]))

    def test_success_is_not_early_termination_and_final_settle_cannot_be_lost(self):
        self.transfer(level=0)
        self.state['com'][:, 1] = 0.
        self.state['loads'][:] = 300.
        _, terminal, info = self.tick(102)
        self.assertTrue(bool(info['task_success'].all()))
        self.assertFalse(bool(terminal.any()))
        self.state['stand'][:] = False
        _, terminal, info = self.tick()
        self.assertTrue(bool(info['success_latched'].all()))
        self.assertFalse(bool(info['task_success'].any()))
        self.assertFalse(bool(terminal.any()))

    def test_stage_and_side_are_observable_and_idle_completion_reward_is_zero(self):
        self.reset(level=2)
        x = self.curriculum.observations()
        self.assertTrue(torch.equal(x[:, 7:10], torch.tensor([[0., 0., 1.], [0., 0., 1.]])))
        self.assertTrue(torch.equal(x[:, 10:12], torch.eye(2)))
        self.assertTrue(torch.equal(x[:, 14], torch.ones(2)))
        reward, _, info = self.tick(20)
        self.assertFalse(bool(info['reward_completion'].any()))
        self.assertTrue(bool((reward < 0.).all()))


if __name__ == '__main__':
    unittest.main()
