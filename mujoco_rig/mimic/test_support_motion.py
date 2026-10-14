import unittest

import torch

from .capture_reward import rotate_xyzw
from .step_reward import PlacementRecoveryReward, swing_quality
from .support_motion import UsefulPlacement, SoleSlip
from .rig import Rig


class SupportMotionTests(unittest.TestCase):
    def setUp(self):
        self.points=torch.zeros(1,2,2,3)
        self.legs=torch.zeros(1,2,4)
        self.loads=torch.tensor([[100.,100.]])
        self.com=torch.tensor([[.15,0.,1.]])
        self.velocity=torch.zeros(1,3)
        self.error=torch.tensor([.1])
        self.hit=torch.tensor([True])

    def tick(self,tracker,points,legs,loads,count=1):
        events=[]
        for _ in range(count):
            events.append(tracker.update(points,legs,loads,self.com,self.velocity,self.error,self.hit))
        return sum(float(e) for e in events)

    def test_grounded_shuffling_earns_no_swing_or_placement(self):
        tracker=UsefulPlacement(self.points,self.legs,.02)
        moved=self.points.clone(); moved[:,0,:,0]=.1
        self.assertEqual(self.tick(tracker,moved,self.legs,torch.tensor([[50.,150.]]),10),0.)
        self.assertEqual(float(swing_quality(self.com,self.velocity,moved,self.points[...,2],torch.tensor([[50.,150.]]))),0.)

    def test_either_foot_useful_placement_is_credited_only_once_and_reset_clears(self):
        for foot in (0,1):
            tracker=UsefulPlacement(self.points,self.legs,.02)
            points=self.points.clone(); points[:,foot,:,2]=.02
            loads=self.loads.clone(); loads[:,foot]=0.
            self.tick(tracker,points,self.legs,loads,3)
            points[:,foot,:,0]=.08
            legs=self.legs.clone(); legs[:,foot,3]=.1
            self.tick(tracker,points,legs,loads)
            points[:,foot,:,2]=0.
            self.assertEqual(self.tick(tracker,points,legs,self.loads,3),1.)
            self.assertEqual(self.tick(tracker,points,legs,self.loads,20),0.)
            tracker.reset(torch.tensor([0]),self.points,self.legs)
            self.assertFalse(bool(tracker.used[0]))

    def test_wrong_direction_and_no_leg_motion_earn_no_event(self):
        for placement,leg in ((-.08,.1),(.08,0.)):
            tracker=UsefulPlacement(self.points,self.legs,.02)
            points=self.points.clone();points[:,0,:,2]=.02
            loads=torch.tensor([[0.,100.]])
            self.tick(tracker,points,self.legs,loads,3)
            points[:,0,:,0]=placement
            legs=self.legs.clone();legs[:,0,3]=leg
            self.tick(tracker,points,legs,loads)
            self.assertEqual(self.tick(tracker,points,legs,self.loads,3),0.)

    def test_peak_articulation_recognizes_flexion_then_extension(self):
        for peak,expected in ((False,0.),(True,1.)):
            tracker=UsefulPlacement(self.points,self.legs,.02,peak_articulation=peak)
            points=self.points.clone();points[:,0,:,2]=.02
            loads=torch.tensor([[0.,100.]])
            self.tick(tracker,points,self.legs,loads,3)
            legs=self.legs.clone();legs[:,0,3]=.3
            points[:,0,:,0]=.08
            self.tick(tracker,points,legs,loads,3)
            points[:,0,:,2]=0.
            self.assertEqual(self.tick(tracker,points,self.legs,self.loads,3),expected)
            self.assertEqual(int(tracker.last_diagnostics['endpoint_missed_peak'].sum()),1)

    def test_signal_window_counts_events_across_transitions_and_drain(self):
        from .recovery_signals import RecoverySignals
        window=RecoverySignals('cpu')
        diagnostics={key:torch.tensor([[True,False],[True,True]]) for key in window.KEYS[3:9]}
        diagnostics['useful_events']=torch.tensor([True,True])
        window.record(torch.tensor([True,False]),diagnostics,torch.tensor([.1,.2]))
        diagnostics['useful_events'][:]=False
        window.record(torch.tensor([True,False]),diagnostics,torch.tensor([.3,.2]))
        report=window.drain()
        self.assertEqual(report['credited_events'],1)
        self.assertEqual(report['measured_placements'],2)
        self.assertEqual(report['post_hit_world_steps'],2)
        self.assertAlmostEqual(report['mean_post_hit_slip_proxy_m_s'],.2,places=6)
        self.assertEqual(window.drain()['credited_events'],0)

    def test_slip_proxy_distinguishes_slide_from_rolling_material_point(self):
        from pathlib import Path
        rig=Rig.load(Path(__file__).resolve().parents[1]/'dummy.xml')
        proxy=SoleSlip(rig,'cpu',.02)
        positions=torch.zeros(1,rig.model.nbody-1,3)
        rotations=torch.zeros(1,rig.model.nbody-1,4);rotations[...,3]=1.
        proxy.reset(torch.tensor([0]),positions,rotations)
        moved=positions.clone();moved[:,:,0]+=.004
        cost,speed=proxy.update(moved,rotations,torch.tensor([[200.,0.,0.,0.]]))
        self.assertAlmostEqual(float(speed),.2,places=5)
        self.assertGreater(float(cost),.8)
        proxy.reset(torch.tensor([0]),positions,rotations)
        body=int(proxy.bodies[0]);anchor=proxy.local[0,0].clone()
        # Positive Y rotation: the +X bottom edge remains the lowest contact.
        anchor[0]=proxy.local[0,:,0].amax()
        rotation=rotations.clone(); rotation[:,body,1]=torch.sin(torch.tensor(.02/2));rotation[:,body,3]=torch.cos(torch.tensor(.02/2))
        pivot=positions.clone();pivot[:,body]=anchor-rotate_xyzw(rotation[:,body],anchor)
        _,speed=proxy.update(pivot,rotation,torch.tensor([[200.,0.,0.,0.]]))
        self.assertLess(float(speed),.02)

    def test_standing_reward_unchanged_without_hit_cost_or_event(self):
        from .step_reward import TransferRecoveryReward
        args=(torch.ones(1),torch.ones(1),torch.ones(1),torch.ones(1),torch.zeros(1,3),torch.zeros(1,3),torch.zeros(1),torch.zeros(1),self.loads)
        torch.testing.assert_close(PlacementRecoveryReward()(*args,placement=torch.zeros(1),slip=torch.zeros(1)),TransferRecoveryReward()(*args))


if __name__=='__main__':
    unittest.main()
