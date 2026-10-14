from pathlib import Path
import unittest

import torch

from .recovery_motor import RecoveryTargetControl, authority_gate
from .rig import Rig
from .target_control import DelayedTargetControl


class RecoveryMotorTests(unittest.TestCase):
    def setUp(self):
        self.rig=Rig.load(Path(__file__).resolve().parents[1]/'dummy.xml')
        self.motor=RecoveryTargetControl(self.rig,2,'cpu')
        self.q=torch.tensor(self.rig.rest,dtype=torch.float32).repeat(2,1)
        self.v=torch.zeros(2,self.rig.model.nv)
        self.motor.set_reference(self.q,self.v)

    def observation(self):
        obs=torch.zeros(2,363)
        obs[:,93:95]=.15
        return obs

    def test_capture_and_supported_unloading_gates_are_bilateral(self):
        obs=self.observation()
        self.assertEqual(authority_gate(obs).tolist(),[0.,0.])
        obs[:,93:95]=0.
        self.assertEqual(authority_gate(obs).tolist(),[0.,0.])
        obs[:,9]=.35
        obs[0,93:95]=torch.tensor([.005,.15])
        obs[1,93:95]=torch.tensor([.15,.005])
        torch.testing.assert_close(authority_gate(obs),torch.ones(2))
        obs=self.observation();obs[:,362]=.08
        torch.testing.assert_close(authority_gate(obs),torch.ones(2))

    def test_closed_gate_preserves_original_queue_and_torque_exactly(self):
        original=DelayedTargetControl(self.rig,2,'cpu')
        original.set_reference(self.q,self.v)
        generator=torch.Generator().manual_seed(42)
        for _ in range(12):
            action=torch.rand(2,30,generator=generator)*2-1
            original.apply(action);self.motor.observe(self.observation());self.motor.apply(action)
            self.assertTrue(torch.equal(self.motor.pending,original.pending))
            self.assertTrue(torch.equal(self.motor.active,original.active))
            self.assertTrue(torch.equal(self.motor.physics_control(self.q,self.v),
                                         original.physics_control(self.q,self.v)))

    def test_expansion_is_slew_limited_to_hips_and_knees_with_visible_origin(self):
        c=self.motor;obs=self.observation();obs[:,362]=.08;c.observe(obs)
        action=torch.ones(2,30);action[:,c.leg_mask]=-.6
        previous=self.q[:,c.q_indices]
        for _ in range(24):
            c.apply(action)
            issued=c.pending[(c.cursor-1)%len(c.pending),:,:30]
            desired=(self.q[:,c.q_indices]+action*1.25).clamp(c.lower,c.upper)
            extension=c.leg_mask & ((desired-self.q[:,c.q_indices]).abs()>.25+1e-6)
            self.assertTrue(bool(((issued-previous).abs()[extension]<=2*c.dt+1e-6).all()))
            torch.testing.assert_close(issued[:,~c.leg_mask],
                (self.q[:,c.q_indices]+action*.25).clamp(c.lower,c.upper)[:,~c.leg_mask])
            self.assertTrue(bool((issued<=c.upper).all() & (issued>=c.lower).all()))
            previous=issued.clone()
        self.assertTrue(bool(((issued-self.q[:,c.q_indices])[:,c.leg_mask].abs()>.25).any()))
        torque=c.physics_control(self.q,self.v)
        self.assertTrue(bool((torque[:,c.indices].abs()<=c.scale).all()))
        self.assertEqual(c.contract()['action_mode'],'support_gated_target_pd_v2')

    def test_partial_reset_clears_gate_and_queue_without_touching_peer(self):
        obs=self.observation();obs[:,362]=.08;self.motor.observe(obs)
        self.motor.apply(torch.ones(2,30))
        peer=self.motor.pending[:,1].clone()
        self.motor.reset([0])
        torch.testing.assert_close(self.motor.gate,torch.tensor([0.,1.]))
        self.assertFalse(bool(self.motor.pending[:,0].any()))
        torch.testing.assert_close(self.motor.pending[:,1],peer)

    def test_invalid_observation_and_bounds_are_rejected(self):
        for obs in (torch.zeros(2,362),torch.full((2,363),float('nan'))):
            with self.assertRaises(ValueError):self.motor.observe(obs)
        with self.assertRaises(ValueError):RecoveryTargetControl(self.rig,2,'cpu',maximum_radians=2.)

    def test_action_unit_bridge_preserves_original_requested_targets(self):
        from types import SimpleNamespace
        from .capture_step_teacher import bridge_source_action
        task=SimpleNamespace(engine=SimpleNamespace(policy_control=self.motor))
        observation=self.observation();observation[:,362]=.08
        self.motor.observe(observation)
        original=torch.linspace(-.8,.8,30).repeat(2,1)
        adapted=bridge_source_action(task,original,observation)
        radius=.25+(1.25-.25)*self.motor.gate[:,None]
        actual=torch.where(self.motor.leg_mask,adapted*radius,adapted*.25)
        torch.testing.assert_close(actual,original*.25)
        for _ in range(4):self.motor.apply(adapted)
        expected=(self.q[:,self.motor.q_indices]+original*.25).clamp(self.motor.lower,self.motor.upper)
        torch.testing.assert_close(self.motor.active[:,:30],expected)


if __name__=='__main__':
    unittest.main()
