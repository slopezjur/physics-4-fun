"""Actor-direction diagnostics preserve controls and paired trial identity."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from .probe_actor_update import frozen_state, assert_frozen, paired_summary, surrogate


class ActorProbeTests(unittest.TestCase):
    def test_paired_noop_is_exactly_zero(self):
        values = np.arange(32,dtype=float)
        result = paired_summary(values,values)
        self.assertEqual(result['mean_delta'],0.)
        self.assertEqual(result['interval_95'],[0.,0.])

    def test_paired_constant_change_does_not_conflate_posture_differences(self):
        values = np.tile(np.arange(8,dtype=float)*100,4)
        result = paired_summary(values,values+2.)
        self.assertEqual(result['mean_delta'],2.)
        self.assertEqual(result['interval_95'],[2.,2.])
        self.assertEqual(result['by_posture'],[2.]*8)

    def test_invalid_pairing_and_nonfinite_values_rejected(self):
        for left,right in [(np.ones(17),np.ones(17)),(np.ones(16),np.ones(24)),
                           (np.ones(16),np.full(16,np.nan))]:
            with self.assertRaises(ValueError):
                paired_summary(left,right)

    def test_only_optimizer_owned_actor_parameters_can_change(self):
        agent = torch.nn.Module()
        agent.actor = torch.nn.Linear(2,1)
        agent.critic = torch.nn.Linear(2,1)
        agent.register_buffer('normalization',torch.ones(2))
        agent._actor_optimizer = SimpleNamespace(_param_list=list(agent.actor.parameters()))
        before = frozen_state(agent)
        with torch.no_grad():
            agent.actor.weight.add_(1)
        assert_frozen(agent,before)
        with torch.no_grad():
            agent.critic.bias.add_(1)
        with self.assertRaisesRegex(RuntimeError,'critic'):
            assert_frozen(agent,before)
        agent.critic.bias.data.copy_(before['critic.bias'])
        agent.normalization.add_(1)
        with self.assertRaisesRegex(RuntimeError,'normalization'):
            assert_frozen(agent,before)

    def test_surrogate_chunking_weights_short_last_chunk_correctly(self):
        class Agent:
            def _compute_actor_loss(self,batch):
                return {'actor_loss':batch['obs'].mean()}
        batch = {'obs':torch.arange(11,dtype=torch.float32)[:,None]}
        self.assertAlmostEqual(surrogate(Agent(),batch,chunk=4)['actor_loss'],5.)


if __name__ == '__main__':
    unittest.main()
