"""Short native regressions for the commanded stepping observation/control contract."""
from pathlib import Path
import json
import unittest

import torch

from .runtime import activate


class SteppingSkillCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
        from .rig import Rig
        from .run_learned_support import PARENT
        from .run_support_skill_study import FOUNDATION
        from .stand_experiment_evidence import load_actor
        from .stepping_skill_core import StepSkillEnvironment, SteppingHead
        torch.set_num_threads(1)
        cls.environment_type, cls.head_type = StepSkillEnvironment, SteppingHead
        cls.rig = Rig.load(FOUNDATION / 'dummy.xml', policy_exclude=())
        protocol = json.loads((FOUNDATION / 'protocol.json').read_text(encoding='utf-8'))
        cls.anchor = load_actor(cls.rig, FOUNDATION / 'stationary_reference.npz', protocol, PARENT)
        cls.source_weights = {key: value.clone() for key, value in cls.anchor.state_dict().items()}

    @classmethod
    def tearDownClass(cls):
        for key, value in cls.anchor.state_dict().items():
            if not torch.equal(value, cls.source_weights[key]):
                raise AssertionError('Native checks modified the frozen Stand actor')
        if cls.anchor.training or any(value.requires_grad for value in cls.anchor.parameters()):
            raise AssertionError('Frozen Stand actor no longer has evaluation-only parameters')

    def setUp(self):
        self.env = self.environment_type(self.rig, self.anchor)
        self.active = torch.ones(16, dtype=torch.bool)

    def test_initial_head_and_observation_reads_are_behavior_preserving(self):
        env = self.env
        before = env.engine.native_qpos().clone()
        history = env.history.clone()
        steps = env.task.steps.clone()
        first, second = env.features(), env.features()
        self.assertEqual(first.shape, (16, 528))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(before, env.engine.native_qpos()))
        self.assertTrue(torch.equal(history, env.history))
        self.assertTrue(torch.equal(steps, env.task.steps))
        self.assertTrue(torch.equal(history[:, 0], history[:, 1]))
        self.assertTrue(torch.equal(history[:, 1], history[:, 2]))
        actor = self.head_type(input_width=first.shape[-1]).eval()
        self.assertFalse(actor(first).any())
        self.assertFalse(env.anchor.training)
        self.assertFalse(any(value.requires_grad for value in env.anchor.parameters()))

    def test_history_and_partial_reset_preserve_other_worlds(self):
        env = self.env
        env.reset(onset=0., level=2, distance=.04)
        active = self.active.clone()
        active[15] = False
        before_history = env.history.clone()
        latent = torch.zeros(16, 33)
        left = env.motor_names.index('Foot_L_rx')
        latent[:, left] = .3
        _, _, info = env.step(latent, active)
        self.assertTrue(torch.equal(env.history[active, 0], info['balance_features'][active]))
        self.assertTrue(torch.equal(env.history[active, 1:], before_history[active, :2]))
        self.assertTrue(torch.equal(env.history[~active], before_history[~active]))
        self.assertFalse(env.last_latent[~active].any())
        env.step(latent, active)
        reset_ids = torch.tensor([0, 8])
        keep = torch.ones(16, dtype=torch.bool)
        keep[reset_ids] = False
        features, qpos = env.features().clone(), env.engine.native_qpos().clone()
        previous, remembered, history = env.previous.clone(), env.last_latent.clone(), env.history.clone()
        control = env.engine.policy_control
        pending, active_target, cursor = control.pending.clone(), control.active.clone(), control.cursor
        curriculum = {key: value.clone() for key, value in vars(env.curriculum).items()
                      if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 16}
        env.reset(reset_ids, level=0, onset=1., enabled=True)
        self.assertTrue(torch.equal(env.features()[keep], features[keep]))
        self.assertTrue(torch.equal(env.engine.native_qpos()[keep], qpos[keep]))
        self.assertTrue(torch.equal(env.previous[keep], previous[keep]))
        self.assertTrue(torch.equal(env.last_latent[keep], remembered[keep]))
        self.assertTrue(torch.equal(env.history[keep], history[keep]))
        self.assertTrue(torch.equal(control.pending[:, keep], pending[:, keep]))
        self.assertTrue(torch.equal(control.active[keep], active_target[keep]))
        self.assertEqual(control.cursor, cursor)
        for key, value in curriculum.items():
            self.assertTrue(torch.equal(getattr(env.curriculum, key)[keep], value[keep]), key)
        self.assertFalse(env.previous[reset_ids].any())
        self.assertFalse(env.last_latent[reset_ids].any())
        self.assertFalse(env.curriculum.completed[reset_ids].any())
        self.assertFalse(env.curriculum.age[reset_ids].any())
        self.assertTrue(torch.equal(env.history[reset_ids, 0], env.history[reset_ids, 1]))
        self.assertTrue(torch.equal(env.history[reset_ids, 1], env.history[reset_ids, 2]))

    def test_independent_leg_commands_and_real_two_interval_delay(self):
        env = self.env
        env.reset(onset=0.)
        left, right = [env.motor_names.index(f'Foot_{side}_rx') for side in ('L', 'R')]
        self.assertEqual(len(env.leg_indices), 16)
        leg_mask = torch.zeros(33, dtype=torch.bool)
        leg_mask[env.leg_indices] = True
        self.assertTrue(torch.equal(env.radius[leg_mask], torch.full((16,), .5)))
        self.assertTrue(torch.equal(env.radius[~leg_mask], torch.full((17,), .1)))
        latent = torch.zeros(16, 33)
        latent[0, left] = .8
        latent[8, right] = -.8
        issued = []
        for frame in range(3):
            previous = env.previous.clone()
            _, _, info = env.step(latent, self.active)
            desired = env.radius * latent.tanh()
            expected_addition = previous + (desired - previous).clamp(-.02, .02)
            self.assertTrue(torch.equal(info['desired_addition'], desired))
            self.assertTrue(torch.equal(info['addition'], expected_addition))
            self.assertTrue((info['addition'].abs() <= env.radius + 1e-7).all())
            self.assertTrue(((info['addition'] - previous).abs() <= .02 + 1e-7).all())
            control = env.engine.policy_control
            expected_target = (info['nominal'] + expected_addition).clamp(control.lower, control.upper)
            self.assertTrue(torch.equal(info['issued'], expected_target))
            self.assertEqual(torch.count_nonzero(info['addition']).item(), 2)
            self.assertGreater(float(info['addition'][0, left]), 0.)
            self.assertEqual(float(info['addition'][0, right]), 0.)
            self.assertLess(float(info['addition'][8, right]), 0.)
            self.assertEqual(float(info['addition'][8, left]), 0.)
            if frame < 2:
                self.assertFalse(info['active_target'].any())
            else:
                self.assertTrue(torch.equal(info['active_target'][:, :33], issued[frame - 2]))
                self.assertTrue((info['active_target'][:, 66] == 1).all())
            issued.append(info['issued'].clone())

    def test_wait_and_disabled_sampled_noise_have_zero_authority(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                env = self.env
                env.reset(onset=1., enabled=enabled)
                for frame in range(3):
                    latent = torch.linspace(-3., 3., 33).expand(16, -1).clone() * (frame + 1)
                    _, _, info = env.step(latent, self.active)
                    self.assertFalse(info['authority'].any())
                    self.assertFalse(info['latent'].any())
                    self.assertFalse(env.last_latent.any())
                    self.assertFalse(info['addition'].any())
                    self.assertFalse(info['force'].any())
                    self.assertTrue(torch.equal(info['issued'], info['nominal']))
                    self.assertTrue(torch.equal(info['sampled_latent'], latent))


if __name__ == '__main__':
    unittest.main()
