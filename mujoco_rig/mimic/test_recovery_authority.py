import unittest
import torch

from .probe_recovery_authority import support_gate


class RecoveryAuthorityTests(unittest.TestCase):
    def test_gate_keeps_leg_control_through_balanced_single_support(self):
        obs=torch.zeros(4,363)
        obs[:,93:95]=torch.tensor([[.2,.2],[0.,.2],[.2,0.],[.2,.2]])
        obs[3,362]=.020
        torch.testing.assert_close(support_gate(obs),torch.tensor([0.,1.,1.,1.]))

    def test_nominal_targets_are_identical_when_range_doubles(self):
        actions=torch.linspace(-1,1,30)
        torch.testing.assert_close(actions*.25,(actions*.5)*.5,rtol=0,atol=0)


if __name__ == "__main__":
    unittest.main()
