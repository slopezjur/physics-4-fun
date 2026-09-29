import unittest

import torch

from .gated_recovery_net import GatedRecoveryNet, recovery_gate, quiet_rehearsal_indices, support_recovery_gate, motion_recovery_gate


class GatedRecoveryTests(unittest.TestCase):
    def test_damping_starts_identical_and_trains_only_supported_moving_states(self):
        torch.manual_seed(7)
        source = GatedRecoveryNet(torch.nn.Linear(362, 512), 363)
        source.leg_mask[14:] = 1
        candidate = GatedRecoveryNet(torch.nn.Linear(362, 512), 363, damping_activation=True)
        state = candidate.state_dict()
        state.update(source.state_dict())
        candidate.load_state_dict(state)
        obs = torch.randn(4, 363)
        obs[:, 93:95] = .3
        obs[:, 9:11] = torch.tensor([[.05, 0.], [.25, 0.], [.25, 0.], [.25, 0.]])
        obs[:, 362] = torch.tensor([0., 0., .02, 0.])
        obs[3, 93] = 0.
        before = source(obs).detach()
        torch.testing.assert_close(candidate(obs), before, atol=0, rtol=0)
        parameters = [(name, value) for name, value in candidate.named_parameters() if value.requires_grad]
        self.assertEqual([name for name, _ in parameters], ["damping.weight"])
        optimizer = torch.optim.SGD([value for _, value in parameters], lr=.01)
        candidate(obs[1:2]).sum().backward()
        optimizer.step()
        after = candidate(obs)
        self.assertFalse(torch.equal(after[1, 14:], before[1, 14:]))
        torch.testing.assert_close(after[:, :14], before[:, :14], atol=0, rtol=0)
        torch.testing.assert_close(after[[0, 2, 3]], before[[0, 2, 3]], atol=0, rtol=0)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(candidate.state_dict()[key], value, atol=0, rtol=0)

    def test_velocity_activation_permits_gradients_without_capture_error(self):
        network = GatedRecoveryNet(torch.nn.Linear(362, 512), 363, motion_activation=True)
        network.leg_mask[14:] = 1
        obs = torch.zeros(2, 363)
        obs[1, 9] = .225
        quiet = network(obs[:1]).detach().clone()
        network(obs[1:]).sum().backward()
        self.assertGreater(float(network.recovery[-1].bias.grad[14:].abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in network.base.parameters()))
        with torch.no_grad():
            network.recovery[-1].bias.fill_(.2)
        torch.testing.assert_close(network(obs[:1]), quiet, atol=0, rtol=0)
        self.assertIn("motion_gate_thresholds", network.state_dict())
        restored = GatedRecoveryNet(torch.nn.Linear(362, 512), 363, motion_activation=True)
        restored.load_state_dict(network.state_dict())
        torch.testing.assert_close(restored(obs), network(obs), atol=0, rtol=0)
        with self.assertRaises(RuntimeError):
            GatedRecoveryNet(torch.nn.Linear(362, 512), 363).load_state_dict(network.state_dict())

    def test_velocity_activation_uses_current_motion_and_capture(self):
        obs = torch.zeros(5, 363)
        obs[:, 9] = torch.tensor([0., .15, .225, .30, 0.])
        obs[-1, 362] = .02
        torch.testing.assert_close(motion_recovery_gate(obs), torch.tensor([0., 0., .5, 1., 1.]))
        observations = torch.zeros(1000, 363)
        observations[17, 9] = .225
        self.assertIn(17, quiet_rehearsal_indices(observations, 32, motion_activation=True).tolist())

    def test_support_gate_is_bilateral_and_requires_support(self):
        obs = torch.zeros(4, 363)
        obs[:, 93:95] = torch.tensor([[.3, .3], [0., .3], [.3, 0.], [0., 0.]])
        torch.testing.assert_close(support_recovery_gate(obs), torch.tensor([0., 1., 1., 0.]))

    def test_support_rehearsal_keeps_unloading_without_capture_error(self):
        obs = torch.zeros(1000, 363)
        obs[:, 93:95] = .3
        obs[17, 93] = 0.
        obs[723, 94] = 0.
        picks = quiet_rehearsal_indices(obs, 32, support_activation=True)
        self.assertIn(17, picks.tolist())
        self.assertIn(723, picks.tolist())
        self.assertEqual(len(picks.unique()), 32)

    def test_support_version_expands_branch_without_changing_quiet_base(self):
        network = GatedRecoveryNet(torch.nn.Linear(362, 512), 363,
                                   support_activation=True, correction_limit=2.)
        network.leg_mask[:] = 1
        with torch.no_grad():
            network.recovery[-1].bias.fill_(1.5)
        obs = torch.zeros(2, 363)
        obs[:, 93:95] = torch.tensor([[.3, .3], [0., .3]])
        base = network.base_mean(network.base(obs[:, :362]))
        torch.testing.assert_close(network(obs)[0], base[0], atol=0, rtol=0)
        torch.testing.assert_close(network(obs)[1] - base[1], torch.full((30,), 1.5))

    def test_capture_gate_preserves_quiet_even_with_torso_motion(self):
        network = GatedRecoveryNet(torch.nn.Linear(362, 512), 363)
        network.leg_mask[:] = 1
        with torch.no_grad():
            network.velocity.weight.fill_(.1)
        obs = torch.zeros(3, 363)
        obs[:, 9] = .5
        obs[:, 362] = torch.tensor([0., .001, .020])
        base = network.base_mean(network.base(obs[:, :362]))
        result = network(obs)
        torch.testing.assert_close(result[:2], base[:2], atol=0, rtol=0)
        torch.testing.assert_close(result[2] - base[2], torch.full((30,), .1))

    def test_quiet_rehearsal_includes_brief_active_states_and_quiet_coverage(self):
        obs = torch.zeros(1000, 362)
        obs[17, 9] = .3
        obs[723, 12] = .8
        indices = quiet_rehearsal_indices(obs, 32)
        self.assertEqual(len(indices), 32)
        self.assertEqual(len(indices.unique()), 32)
        self.assertIn(17, indices.tolist())
        self.assertIn(723, indices.tolist())
        self.assertEqual(int(indices[0]), 0)
        self.assertEqual(int(indices[-1]), 999)
        self.assertEqual(quiet_rehearsal_indices(torch.zeros(1, 362), 1).tolist(), [0])

    def test_motion_gate_is_continuous_and_has_no_time_input(self):
        obs = torch.zeros(5, 362)
        obs[:, 9] = torch.tensor([0., .2, .3, .4, 0.])
        obs[-1, 12] = .75
        torch.testing.assert_close(recovery_gate(obs), torch.tensor([0., 0., .5, 1., .5]))

    def test_training_cannot_change_quiet_base_or_upper_body(self):
        torch.manual_seed(1)
        network = GatedRecoveryNet(torch.nn.Sequential(torch.nn.Linear(362, 512), torch.nn.ReLU()), 362)
        network.leg_mask[14:] = 1
        obs = torch.randn(16, 362)
        obs[:, 9:15] = 0
        before = network(obs).detach().clone()
        hit = obs.clone()
        hit[:, 9] = .5
        original_hit = network(hit).detach().clone()
        optimizer = torch.optim.Adam(network.parameters(), lr=.01)
        for _ in range(3):
            optimizer.zero_grad()
            network(hit).square().mean().backward()
            optimizer.step()
        torch.testing.assert_close(network(obs), before, rtol=0, atol=0)
        torch.testing.assert_close(network(hit)[:, :14], original_hit[:, :14], rtol=0, atol=0)
        self.assertFalse(torch.equal(network(hit)[:, 14:], original_hit[:, 14:]))
        self.assertTrue(all(p.grad is None for p in network.base.parameters()))
        self.assertTrue(all(p.grad is None for p in network.base_mean.parameters()))


if __name__ == "__main__":
    unittest.main()
