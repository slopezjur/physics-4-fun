"""Quiet-path isolation, command visibility and reference reset/hold contracts."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from .motion_tracker import MotionTracker, quiet_input, CORE_WIDTH, migrate_feedback_checkpoint
from .runtime import activate


class Source(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(CORE_WIDTH, 30)
        self.obs_norm = SimpleNamespace(_mean=torch.zeros(CORE_WIDTH), _std=torch.ones(CORE_WIDTH))

    def forward(self, observation):
        return self.linear(observation).tanh()


class TrackerPolicyTests(unittest.TestCase):
    def test_selection_preserves_survival_and_can_retain_initial_checkpoint(self):
        from .train_motion_tracker import native_selection_score
        case = dict(survived=True, final_stable_seconds=2., displacement_rmse_m=.01,
                    displacement_limit_m=.002)
        metrics = dict(quiet={'passed': True}, quiet_max_qpos_difference=0.,
                       motion={'cases': [case]}, fixed={'cases': [dict(case)]})
        self.assertEqual(native_selection_score(metrics),5.)
        metrics['fixed']['cases'][0]['survived'] = False
        self.assertIsNone(native_selection_score(metrics))

    def test_continuation_allows_data_changes_but_rejects_control_changes(self):
        from .train_motion_tracker import validate_continuation_contract
        old = dict(num_obs=643, action_mode='reference_relative_target_pd',
            motion_reference_hashes=['a'], stationary_reference_sha256='quiet',
            conditioning=dict(lookahead_seconds=[.1,.3,.6]))
        new = dict(old, motion_reference_hashes=['a','b'],
            conditioning=dict(lookahead_seconds=(.1,.3,.6)))
        self.assertEqual(set(validate_continuation_contract(old,new)), {'motion_reference_hashes'})
        for field, value in (('num_obs',644), ('action_mode','torque'), ('stationary_reference_sha256','other')):
            with self.assertRaises(ValueError):
                validate_continuation_contract(old, dict(new, **{field:value}))

    def test_training_motion_branch_cannot_change_quiet_actions_or_source(self):
        source = Source()
        tracker = MotionTracker(source)
        core = torch.randn(8, CORE_WIDTH)
        observation = quiet_input(core)
        torch.testing.assert_close(tracker(observation), source(core), rtol=0, atol=0)
        observation[:, -1] = 1.
        frozen_correction = {k:v.clone() for k,v in tracker.correction.state_dict().items()}
        optimizer = torch.optim.Adam(tracker.feedback.parameters(), lr=.01)
        optimizer.zero_grad()
        tracker(observation).square().mean().backward()
        optimizer.step()
        torch.testing.assert_close(tracker(quiet_input(core)), source(core), rtol=0, atol=0)
        self.assertTrue(all(torch.equal(p, tracker.source.state_dict()[k]) for k,p in source.state_dict().items()))
        self.assertTrue(all(torch.equal(p, tracker.correction.state_dict()[k]) for k,p in frozen_correction.items()))

    def test_zero_feedback_preserves_trained_v1_actions(self):
        tracker = MotionTracker(Source())
        with torch.no_grad():
            tracker.correction[-1].weight.normal_(std=.05)
            tracker.correction[-1].bias.normal_(std=.05)
        obs = quiet_input(torch.randn(32, CORE_WIDTH))
        obs[:, -1] = 1.
        core = obs[:, :CORE_WIDTH]
        legacy = tracker.source(core) + tracker.correction(torch.cat((
            ((core[:,:95]-tracker.state_mean)/tracker.state_std).clamp(-10,10),
            obs[:,CORE_WIDTH:].clamp(-10,10)), -1)).tanh()
        torch.testing.assert_close(tracker(obs), legacy, rtol=0, atol=0)

    def test_feedback_observes_goal_error_and_masks_uninitialized_targets(self):
        tracker = MotionTracker(Source())
        core = torch.zeros(2, CORE_WIDTH)
        core[:, 95:217] = 100.
        core[:, [155,216]] = 0.
        core[:, 217] = .03
        features = tracker.feedback_features(core)
        self.assertTrue((features[:, :122] == 0).all())
        torch.testing.assert_close(features[:,122], torch.ones(2))
        core[1,155] = 1.
        self.assertTrue((tracker.feedback_features(core)[1,:60] == 10).all())
        self.assertEqual(float(tracker.feedback_features(core)[1,60]),1.)

    def test_checkpoint_migration_rejects_missing_old_state(self):
        tracker = MotionTracker(Source())
        prefix = '_model._actor_layers.'
        initialized = {prefix+k:v for k,v in tracker.state_dict().items()}
        added = ('queue_mean','queue_std','pose_error_scale','feedback.')
        saved = {k:v for k,v in initialized.items() if not any(k[len(prefix):].startswith(a) for a in added)}
        migrated, changed = migrate_feedback_checkpoint(saved,initialized)
        self.assertTrue(changed)
        self.assertEqual(set(migrated),set(initialized))
        broken = dict(saved)
        broken.pop(prefix+'correction.0.weight')
        with self.assertRaises(ValueError):
            migrate_feedback_checkpoint(broken,initialized)


class TrackerTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        activate(Path(os.environ.get('MIMICKIT_PATH', 'D:/Proyectos/Juegos/Tools/MimicKit')))
        from .motion_tracker_task import MotionTrackingTask
        from .balance_observation import CaptureObservationMixin
        from .continuous_stand_task import ContinuousStandTask
        from .native_engine import NativeDummyEngine
        from .rig import Rig
        from envs.base_env import EnvMode
        cls.mode = EnvMode
        cls.rig = Rig.load(Path('mujoco_rig/dummy.xml'))
        stationary = Path('logs/mimickit-steps/continuous-stand-02/stationary_reference.npz')
        motion = Path('logs/mimickit-steps/weight-shift-reference-01/weight_shift_reference.npz')
        cls.task = MotionTrackingTask(cls.rig, stationary, [motion], 8, engine_factory=NativeDummyEngine)
        cls.fixed_task = MotionTrackingTask(cls.rig, stationary, [motion], 8,
            fixed_fraction=.25, engine_factory=NativeDummyEngine)
        class Quiet(CaptureObservationMixin, ContinuousStandTask):
            pass
        cls.quiet = Quiet(cls.rig, stationary, 8, engine_factory=NativeDummyEngine)

    def test_quiet_core_and_physical_targets_match_source_task(self):
        task, quiet = self.task, self.quiet
        task.set_mode(self.mode.TRAIN)
        task.reset(); quiet.reset()
        for _ in range(8):
            torch.testing.assert_close(task.observations()[:2,:CORE_WIDTH], quiet.observations()[:2], rtol=0, atol=0)
            self.assertTrue((task.observations()[:2,CORE_WIDTH:] == 0).all())
            action = torch.full((8,30), .05)
            task.step(action); quiet.step(action)
            torch.testing.assert_close(task.engine.native_qpos()[:2], quiet.engine.native_qpos()[:2], rtol=0, atol=0)

    def test_reference_holds_final_pose_and_zero_velocity_without_phase(self):
        task = self.task
        task.set_mode(self.mode.TEST);task.reset()
        q, v = task.sample_reference(torch.full((8,), 300.))
        reference = task.bank.references[0]
        torch.testing.assert_close(q, reference.qpos[-1].expand(8,-1))
        self.assertTrue((v == 0).all())
        self.assertTrue((task.observations()[:,361] == 0).all())

    def test_partial_reset_preserves_peer_queues_and_reference_commands(self):
        task = self.task
        task.set_mode(self.mode.TRAIN); task.reset()
        task.step(torch.full((8,30), .05))
        previous = task.observations()[1:].clone()
        task.reset(torch.tensor([0]))
        torch.testing.assert_close(task.observations()[1:], previous, rtol=0, atol=0)

    def test_core_future_targets_match_controller_motion_reference(self):
        from .motion_tracker import encode_pose
        task = self.task
        task.set_mode(self.mode.TEST);task.reset()
        time = task.offset[:, None] + task.dt * torch.tensor([1.,2.,3.])
        desired = task.sample_reference(time)[0]
        torch.testing.assert_close(task.observations()[:,217:361], encode_pose(desired).flatten(1), rtol=0, atol=0)

    def test_double_support_resets_have_real_loads_and_no_false_capture_risk(self):
        task = self.task
        task.set_mode(self.mode.TEST);task.reset()
        observation = task.observations()
        self.assertTrue((observation[:,93:95] * 1000. >= 5.).all())
        self.assertTrue((observation[:,362] == 0).all())

    def test_fixed_curriculum_targets_do_not_move_and_partial_reset_preserves_peers(self):
        task = self.fixed_task
        task.set_mode(self.mode.TRAIN);task.reset()
        q0,v0 = task.sample_reference(task.offset)
        q1,v1 = task.sample_reference(task.offset + 100.)
        held = task.fixed_mask()
        self.assertEqual(int(held.sum()),2)
        torch.testing.assert_close(q0[held],q1[held],rtol=0,atol=0)
        self.assertTrue((v0[held] == 0).all() and (v1[held] == 0).all())
        phases=task.held_phase.clone()
        task.reset(torch.tensor([2]))
        torch.testing.assert_close(phases[3:],task.held_phase[3:],rtol=0,atol=0)
        self.assertTrue((task.observations()[:2,CORE_WIDTH:] == 0).all())

    def test_fixed_evaluation_covers_phases_with_real_double_support(self):
        task = self.fixed_task
        task.fixed_pose_eval=True
        task.set_mode(self.mode.TEST);task.reset()
        q,v=task.sample_reference(torch.zeros(8))
        q_later,v_later=task.sample_reference(torch.full((8,),30.))
        torch.testing.assert_close(q,q_later,rtol=0,atol=0)
        self.assertTrue((v == 0).all() and (v_later == 0).all())
        self.assertTrue((task.observations()[:,93:95]*1000. >= 5.).all())
        self.assertEqual(float(task.held_phase[0]),0.)
        self.assertAlmostEqual(float(task.held_phase[-1]),task.bank.duration)

    def test_dense_accuracy_reward_ranks_errors_without_flat_far_penalty(self):
        from .motion_tracker_task import root_accuracy_reward
        errors=torch.tensor([[.002,0.,0.],[.005,0.,0.],[.020,0.,0.]],requires_grad=True)
        score=root_accuracy_reward(errors,'inverse_huber',.03,.005)
        self.assertTrue((score > 0).all() and (score <= 1).all())
        self.assertGreater(float(score[0].detach()),float(score[1].detach()))
        self.assertGreater(float(score[1].detach()),float(score[2].detach()))
        score.sum().backward()
        self.assertLess(float(errors.grad[-1,0]),-1.)


if __name__ == '__main__':
    unittest.main()
