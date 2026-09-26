"""Backtracking must preserve constraints, useful descent and optimizer transactions."""
import unittest
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from .update_guard import UpdateGuard


class BacktrackingTests(unittest.TestCase):
    def setup_sgd(self, backtracks=8):
        p = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = torch.optim.SGD([p], lr=1., momentum=.9)
        optimizer.state[p]['momentum_buffer'] = torch.tensor([.5])
        return p, optimizer, UpdateGuard([p], optimizer, .5, backtracks)

    def test_retries_restore_momentum_and_base_rate(self):
        p, optimizer, guard = self.setup_sgd()
        kept, kl = guard.step(p.square().sum(), lambda: float((p-1).square().sum().detach()),
                              accept=lambda: abs(float(p.detach())-1) <= .2,
                              measure_loss=lambda: p.square().sum())
        self.assertTrue(kept)
        self.assertGreater(kl, .5)  # Report all attempted KL, not only the accepted trial.
        self.assertEqual(guard.last_search['trials'], 4)
        self.assertEqual(guard.last_search['scale'], .125)
        torch.testing.assert_close(p, torch.tensor([1-.125*1.45]))
        torch.testing.assert_close(optimizer.state[p]['momentum_buffer'], torch.tensor([1.45]))
        self.assertEqual(optimizer.param_groups[0]['lr'], 1.)
        self.assertGreater(guard.last_search['loss_decrease'], 0)

    def test_exhaustion_restores_everything(self):
        p, optimizer, guard = self.setup_sgd(3)
        kept, _ = guard.step(p.square().sum(), lambda: 0., accept=lambda: False,
                              measure_loss=lambda: p.square().sum())
        self.assertFalse(kept)
        self.assertEqual(guard.last_search['trials'], 4)
        self.assertEqual(guard.last_search['scale'], 0.)
        torch.testing.assert_close(p, torch.tensor([1.]), rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[p]['momentum_buffer'], torch.tensor([.5]), rtol=0, atol=0)
        self.assertEqual(optimizer.param_groups[0]['lr'], 1.)

    def test_feasible_uphill_momentum_is_not_progress(self):
        p, optimizer, guard = self.setup_sgd(4)
        optimizer.state[p]['momentum_buffer'].fill_(100.)
        kept, _ = guard.step((p-2).square().sum(), lambda: 0., measure_loss=lambda: (p-2).square().sum())
        self.assertFalse(kept)
        self.assertEqual(guard.last_search['loss_rejections'], 5)
        torch.testing.assert_close(p, torch.tensor([1.]), rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[p]['momentum_buffer'], torch.tensor([100.]))

    def test_zero_change_is_not_accepted(self):
        p = torch.nn.Parameter(torch.tensor([0.]))
        guard = UpdateGuard([p], torch.optim.SGD([p], lr=.1), .02, 2)
        kept, _ = guard.step((p*0+1).sum(), lambda: 0., measure_loss=lambda: (p*0+1).sum())
        self.assertFalse(kept)
        self.assertEqual(guard.last_search['loss_rejections'], 3)

    def test_retry_exception_restores_weights_momentum_and_rate(self):
        p, optimizer, guard = self.setup_sgd()
        def fail():
            raise RuntimeError('loss evaluation failed')
        with self.assertRaisesRegex(RuntimeError, 'loss evaluation'):
            guard.step(p.square().sum(), lambda: float((p-1).square().sum().detach()), measure_loss=fail)
        self.assertGreater(guard.last_search['trials'], 1)
        torch.testing.assert_close(p, torch.tensor([1.]))
        torch.testing.assert_close(optimizer.state[p]['momentum_buffer'], torch.tensor([.5]))
        self.assertEqual(optimizer.param_groups[0]['lr'], 1.)

    def test_adam_counter_advances_once_not_once_per_trial(self):
        p = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = torch.optim.Adam([p], lr=.1)
        p.square().sum().backward()
        optimizer.step()
        before = p.detach().clone()
        step = optimizer.state[p]['step'].clone()
        guard = UpdateGuard([p], optimizer, .02, 4)
        kept, _ = guard.step(p.square().sum(), lambda: 0.,
            accept=lambda: float((p.detach()-before).abs().max()) < .03,
            measure_loss=lambda: p.square().sum())
        self.assertTrue(kept)
        self.assertEqual(guard.last_search['trials'], 3)
        torch.testing.assert_close(optimizer.state[p]['step'], step+1)
        self.assertEqual(optimizer.param_groups[0]['lr'], .1)

    def test_invalid_budget_or_missing_loss_cannot_change_state(self):
        p, optimizer, guard = self.setup_sgd()
        with self.assertRaises(ValueError):
            guard.step(p.square().sum(), lambda: 0.)
        for budget in (-1, 13, 1.5, True):
            with self.assertRaises(ValueError):
                UpdateGuard([p], optimizer, .02, budget)
        torch.testing.assert_close(p, torch.tensor([1.]))

    @unittest.skipUnless(os.environ.get('MIMICKIT_PATH'), 'Set MIMICKIT_PATH')
    def test_projection_finds_descent_where_halving_cannot(self):
        from .runtime import activate
        from .quiet_anchor import QuietAnchor
        activate(Path(os.environ['MIMICKIT_PATH']))
        from learning.distribution_gaussian_diag import DistributionGaussianDiag
        p = torch.nn.Parameter(torch.zeros(2))
        class Model:
            def eval_actor(self, obs):
                return DistributionGaussianDiag(p[0].expand(len(obs),1), torch.zeros(len(obs),1))
        agent = SimpleNamespace(_device='cpu', _obs_norm=SimpleNamespace(normalize=lambda x:x), _model=Model())
        anchor = QuietAnchor(agent, torch.zeros(3,1), .02)
        with torch.no_grad():
            p[0] = .199999
        before = p.detach().clone()
        optimizer = torch.optim.SGD([p], lr=.01, momentum=.9)
        guard = UpdateGuard([p], optimizer, .02, 8)
        loss = lambda: (p-1).square().sum()
        kept, _ = guard.step(loss(), lambda: 0., accept=lambda: anchor.accepts(agent), measure_loss=loss)
        self.assertFalse(kept)
        torch.testing.assert_close(p,before,rtol=0,atol=0)
        kept, _ = guard.step(loss(), lambda: 0., accept=lambda: anchor.accepts(agent), measure_loss=loss,
                            project_direction=lambda d: anchor.project_direction(agent,[p],d))
        self.assertTrue(kept)
        self.assertTrue(guard.last_search['projected'])
        self.assertTrue(anchor.accepts(agent))
        self.assertLess(float(p[0].detach()), float(before[0]))
        self.assertGreater(float(p[1].detach()), 0.)
        self.assertGreater(guard.last_search['loss_decrease'], 0.)
        self.assertEqual(optimizer.param_groups[0]['lr'], .01)

    def test_projection_cannot_bypass_actual_constraints(self):
        p, optimizer, guard = self.setup_sgd(2)
        kept, _ = guard.step(p.square().sum(), lambda: 0., accept=lambda: False,
            measure_loss=lambda: p.square().sum(), project_direction=lambda d: [torch.tensor([-.1])])
        self.assertFalse(kept)
        self.assertEqual(guard.last_search['projected_trials'], 3)
        torch.testing.assert_close(p,torch.tensor([1.]),rtol=0,atol=0)
        torch.testing.assert_close(optimizer.state[p]['momentum_buffer'],torch.tensor([.5]),rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
