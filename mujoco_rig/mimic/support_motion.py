"""Physical support diagnostics and bounded useful-placement credit."""
import itertools

import torch

from .capture_reward import rotate_xyzw
from .recovery_steps import RecoveryStepTracker


class SoleSlip:
    """Material-point velocity near the lowest box corners; a contact-slip proxy."""
    def __init__(self, rig, device, dt):
        self.dt = dt
        self.bodies = torch.tensor([rig.model.body(f'{body}_{side}').id-1
            for side in ('L','R') for body in ('Foot','Toe')], device=device)
        corners = []
        signs = torch.tensor(list(itertools.product((-1.,1.), repeat=3)), device=device)
        for side in ('L','R'):
            for body in ('Foot','Toe'):
                geom = rig.model.geom(f'g_{body}_{side}').id
                if not (rig.model.geom_quat[geom] == [1,0,0,0]).all():
                    raise ValueError('Sole proxy requires body-aligned box geoms')
                center = torch.tensor(rig.model.geom_pos[geom], device=device, dtype=torch.float32)
                size = torch.tensor(rig.model.geom_size[geom], device=device, dtype=torch.float32)
                corners.append(center+signs*size)
        self.local = torch.stack(corners)
        self.positions = self.rotations = None

    def reset(self, ids, positions, rotations):
        p, r = positions[:,self.bodies], rotations[:,self.bodies]
        if self.positions is None:
            self.positions, self.rotations = p.clone(), r.clone()
        else:
            self.positions[ids], self.rotations[ids] = p[ids], r[ids]

    def update(self, positions, rotations, loads):
        p, r = positions[:,self.bodies], rotations[:,self.bodies]
        current = p[:,:,None] + rotate_xyzw(r[:,:,None].expand(-1,-1,8,-1), self.local)
        previous = self.positions[:,:,None] + rotate_xyzw(self.rotations[:,:,None].expand(-1,-1,8,-1), self.local)
        velocity = (current-previous)/self.dt
        # Keep the same material corner in both states: rolling changes contact
        # location, while true sliding moves the contacting material tangentially.
        weights = torch.softmax(-(current[...,2]-current[...,2].amin(-1,keepdim=True))/.003, -1)
        speed = (torch.linalg.vector_norm(velocity[...,:2],dim=-1)*weights).sum(-1)
        speed = speed.reshape(-1,2,2)
        load = loads.reshape(-1,2,2)
        weighted = (speed*load).sum((1,2))/load.sum((1,2)).clamp_min(5.)
        self.positions, self.rotations = p.clone(), r.clone()
        cost = ((weighted-.02).clamp_min(0)/.2).clamp_max(1)
        return cost, weighted


class UsefulPlacement:
    """One bounded event per episode: a loaded landing toward lost support."""
    def __init__(self, points, legs, dt, *, peak_articulation=False):
        self.tracker = RecoveryStepTracker(points, dt)
        self.leg_start = legs.clone()
        self.target = points[:,:,:,:2].mean(2).clone()
        self.target_distance = torch.zeros(points.shape[:2], device=points.device)
        self.eligible = torch.zeros_like(self.target_distance,dtype=torch.bool)
        self.used = torch.zeros(len(points),device=points.device,dtype=torch.bool)
        self.peak_articulation = peak_articulation
        self.leg_peak = torch.zeros_like(self.target_distance)
        self.last_diagnostics = {}

    def reset(self, ids, points, legs):
        fresh = RecoveryStepTracker(points, self.tracker.dt)
        for key,value in vars(fresh).items():
            if isinstance(value,torch.Tensor):
                getattr(self.tracker,key)[ids] = value[ids]
        self.eligible[ids] = False
        self.used[ids] = False
        self.leg_start[ids] = legs[ids]
        self.leg_peak[ids] = 0.

    def update(self, points, legs, loads, com, velocity, error, hit):
        xy = points[:,:,:,:2].mean(2)
        starting = hit[:,None] & (loads<5.) & ~self.tracker.in_swing
        ground = points[...,2].amin((1,2))
        target = com[:,:2]+velocity[:,:2]*((com[:,2]-ground).clamp_min(.05)/9.81).sqrt()[:,None]
        self.target = torch.where(starting[:,:,None],target[:,None],self.target)
        self.target_distance = torch.where(starting,torch.linalg.vector_norm(xy-target[:,None],dim=-1),self.target_distance)
        self.leg_start = torch.where(starting[:,:,None],legs,self.leg_start)
        self.leg_peak = torch.where(starting,0.,self.leg_peak)
        displacement = torch.linalg.vector_norm(legs-self.leg_start,dim=-1)
        self.leg_peak = torch.maximum(self.leg_peak,torch.where(self.tracker.in_swing | starting,displacement,0.))
        self.eligible = torch.where(starting,(error>=.03)[:,None] & (loads.flip(-1)>=30.),self.eligible)
        self.eligible &= (loads.flip(-1)>=30.) | ~self.tracker.in_swing
        before = self.tracker.steps.clone()
        self.tracker.update(points,loads,torch.ones_like(hit),hit)
        placed = self.tracker.steps>before
        progress = self.target_distance-torch.linalg.vector_norm(xy-self.target,dim=-1)
        articulated = (self.leg_peak if self.peak_articulation else displacement)>=.08
        useful = placed & self.eligible & (progress>=.02) & articulated & (loads>=30.)
        event = useful.any(-1) & ~self.used
        self.last_diagnostics = dict(measured_placements=placed,
            rejected_support_or_initial_risk=placed & ~self.eligible,
            rejected_capture_progress=placed & (progress<.02),
            rejected_articulation=placed & ~articulated,
            rejected_landing_load=placed & (loads<30.),
            endpoint_missed_peak=placed & (displacement<.08) & (self.leg_peak>=.08),
            useful_events=event)
        self.used |= event
        return event.float()

    @staticmethod
    def contract(peak_articulation=False):
        return dict(**RecoveryStepTracker.contract(), initial_capture_error_m=.03,
            opposite_support_n=30., landing_support_n=30., capture_target_progress_m=.02,
            hip_knee_displacement_rad=.08,
            articulation_measurement='maximum during swing' if peak_articulation else 'landing versus takeoff',
            maximum_credited_events_per_episode=1,
            objective='Either foot; lift, articulate and land toward the capture point recorded at takeoff')
