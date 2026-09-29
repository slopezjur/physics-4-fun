import math
import unittest

import torch

from .risk_exploration import conditioned_logstd


class RiskExplorationTests(unittest.TestCase):
    def test_support_gate_preserves_quiet_and_explores_either_unloaded_foot(self):
        obs = torch.zeros(5, 363)
        obs[:, 93:95] = .3
        obs[1, 93] = 0
        obs[2, 94] = 0
        obs[3, 362] = .02
        obs[4, 93:95] = 0  # No supported foot and no capture risk.
        base = torch.full((5, 30), math.log(.05))
        legs = torch.arange(30) >= 14
        result = conditioned_logstd(obs, base, legs, .2, "support")
        torch.testing.assert_close(result[[0, 4]], base[[0, 4]], rtol=0, atol=0)
        torch.testing.assert_close(result[:, ~legs], base[:, ~legs], rtol=0, atol=0)
        torch.testing.assert_close(result[1:4, legs].exp(), torch.full((3, 16), .2))
        with self.assertRaisesRegex(ValueError, "363-value"):
            conditioned_logstd(obs[:, :362], base, legs, .2, "support")

    def test_quiet_distribution_is_exactly_preserved_and_only_legs_expand(self):
        obs = torch.zeros(3, 362)
        obs[1, 9] = .25
        obs[2, 10] = -.4
        base = torch.full((3, 30), math.log(.05))
        legs = torch.arange(30) >= 14
        result = conditioned_logstd(obs, base, legs, .2)
        torch.testing.assert_close(result[0], base[0], rtol=0, atol=0)
        torch.testing.assert_close(result[:, ~legs], base[:, ~legs], rtol=0, atol=0)
        torch.testing.assert_close(result[1, legs].exp(), torch.full((16,), .125))
        torch.testing.assert_close(result[2, legs].exp(), torch.full((16,), .2))

    def test_state_conditioning_has_consistent_normal_log_probability(self):
        obs = torch.zeros(3, 362)
        obs[:, 9] = torch.tensor([0., .25, .5])
        base = torch.full((3, 30), math.log(.05))
        logstd = conditioned_logstd(obs, base, torch.arange(30) >= 14, .2)
        mean = torch.zeros_like(logstd)
        action = torch.full_like(logstd, .07)
        manual = -.5 * ((action-mean) / logstd.exp()).square().sum(-1)
        manual -= .5 * 30 * math.log(2*math.pi) + logstd.sum(-1)
        expected = torch.distributions.Normal(mean, logstd.exp()).log_prob(action).sum(-1)
        torch.testing.assert_close(manual, expected)


if __name__ == "__main__":
    unittest.main()
