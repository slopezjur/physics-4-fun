"""Reward discrimination, cumulative retention and terminal calibration semantics."""
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from .shift_reward import precision_reward, contract
from .update_guard import UpdateGuard
from .quiet_anchor import QuietAnchor, valid_observations
from .critic_calibration import discounted_targets


class ShiftLearningTests(unittest.TestCase):
    def test_reward_resolves_small_errors_and_is_bounded(self):
        target = torch.zeros(5, 3)
        actual = target.clone()
        actual[:, 0] = torch.tensor([0., .002, .005, .01, .02])
        reward = precision_reward(torch.ones(5), actual, target, target, target)
        self.assertEqual(float(reward[0]), 1.)
        self.assertTrue(bool((reward[1:] < reward[:-1]).all()))
        self.assertLess(float(reward[3]), .61)
        self.assertTrue(bool(((reward >= 0) & (reward <= 1)).all()))
        self.assertNotEqual(contract('reference')['schema'], contract('precision_v1')['schema'])

    def test_rejection_restores_weights_and_momentum_even_if_rollout_kl_passes(self):
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = torch.optim.SGD([parameter], lr=.1, momentum=.9)
        guard = UpdateGuard([parameter], optimizer, .02)
        guard.step(parameter.square().sum(), lambda: 0.)
        old = parameter.detach().clone()
        momentum = copy.deepcopy(optimizer.state_dict())
        kept, _ = guard.step(parameter.square().sum(), lambda: 0., accept=lambda: False)
        self.assertFalse(kept)
        torch.testing.assert_close(parameter, old, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state_dict()['state'][0]['momentum_buffer'], momentum['state'][0]['momentum_buffer'])
        def invalid():
            raise FloatingPointError('anchor')
        with self.assertRaises(FloatingPointError):
            guard.step(parameter.square().sum(), lambda: 0., accept=invalid)
        torch.testing.assert_close(parameter, old, rtol=0, atol=0)

    def test_anchor_excludes_post_termination_frames(self):
        states = np.arange(12).reshape(3, 2, 2).astype(np.float32)
        selected = valid_observations({'observations':states}, {'survival_seconds':[1.,3.]}, 1., stride=1)
        torch.testing.assert_close(selected, torch.tensor(states[[0,0,1,2],[0,1,1,1]]))

    def test_calibration_failure_zero_bootstrap_and_timeout_bootstrap(self):
        rewards = torch.tensor([[1.,1.],[2.,1.],[0.,1.]])
        ends = torch.tensor([[False,False],[True,False],[False,True]])
        bootstraps = torch.tensor([[0.,0.],[0.,0.],[0.,10.]])
        torch.testing.assert_close(discounted_targets(rewards,ends,bootstraps,.5,1),
                                   torch.tensor([[2.,3.],[2.,4.],[0.,6.]]))
        torch.testing.assert_close(discounted_targets(rewards,ends,bootstraps,.5,2),
                                   torch.tensor([[2.,3.],[0.,6.]]))

    def test_anchor_thinning_keeps_every_phase(self):
        states = np.tile(np.arange(8)[None,:,None],(9,1,1)).astype(np.float32)
        selected = valid_observations({'observations':states}, {'survival_seconds':[9.]*8}, 1., stride=4)
        self.assertEqual(selected.shape, (24,1))
        self.assertEqual(torch.bincount(selected[:,0].long()).tolist(), [3]*8)

    @unittest.skipUnless(os.environ.get('MIMICKIT_PATH'), 'Set MIMICKIT_PATH')
    def test_anchor_stays_fixed_across_successive_updates(self):
        from .runtime import activate
        activate(Path(os.environ['MIMICKIT_PATH']))
        from learning.distribution_gaussian_diag import DistributionGaussianDiag
        parameter = torch.nn.Parameter(torch.tensor([0.]))
        class Model:
            def eval_actor(self, obs):
                return DistributionGaussianDiag(parameter.expand(len(obs),1), torch.zeros(len(obs),1))
        agent = SimpleNamespace(_device='cpu',_obs_norm=SimpleNamespace(normalize=lambda x:x),_model=Model())
        anchor = QuietAnchor(agent,torch.zeros(2,1),.02)
        with torch.no_grad():
            parameter.fill_(.15)
        self.assertTrue(anchor.accepts(agent))
        with torch.no_grad():
            parameter.fill_(.25)
        self.assertFalse(anchor.accepts(agent))
        torch.testing.assert_close(anchor.teacher.mean,torch.zeros(2,1))


if __name__ == '__main__':
    unittest.main()
