"""Opt-in motor authority using current balance state and visible target history."""
import math

import torch

from .target_control import DelayedTargetControl


def authority_gate(observation):
    if observation.shape[-1] != 363 or not torch.isfinite(observation).all():
        raise ValueError('Recovery authority requires finite capture observations')
    capture=((observation[...,362]-.03)/.05).clamp(0,1)
    loads=observation[...,93:95]*1000.
    unloading=((15.-loads.amin(-1))/10.).clamp(0,1)
    support=(loads.amax(-1)/50.).clamp(0,1)
    speed=torch.linalg.vector_norm(observation[...,9:11],dim=-1)
    moving=((speed-.15)/.20).clamp(0,1)
    return torch.maximum(capture,unloading*support*moving)


class RecoveryTargetControl(DelayedTargetControl):
    def __init__(self,rig,num_envs,device,*,maximum_radians=1.25,target_speed_rad_s=2.):
        if not math.isfinite(maximum_radians) or not .25 < maximum_radians <= 1.5:
            raise ValueError('Recovery target radius must be within (0.25,1.5] rad')
        if not math.isfinite(target_speed_rad_s) or not 0. < target_speed_rad_s <= 4.:
            raise ValueError('Require a bounded positive target speed')
        super().__init__(rig,num_envs,device)
        self.maximum_radians=maximum_radians
        self.target_speed_rad_s=target_speed_rad_s
        self.dt=rig.model.opt.timestep*rig.decimation
        self.leg_mask=torch.tensor([rig.model.joint(int(rig.model.actuator_trnid[a,0])).name
            .startswith(('Thigh_','Shin_')) for a in rig.actuators],device=device)
        self.gate=torch.zeros(num_envs,device=device)

    def observe(self,observation):
        if observation.shape != (len(self.ctrl),363):
            raise ValueError('Recovery motor observation batch mismatch')
        self.gate.copy_(authority_gate(observation))

    def reset(self,env_ids=None):
        super().reset(env_ids)
        self.gate[slice(None) if env_ids is None else env_ids]=0.

    def target_positions(self,action):
        quiet=super().target_positions(action)
        enabled=(self.gate>0)[:,None] & self.leg_mask
        radius=self.residual_radians+(self.maximum_radians-self.residual_radians)*self.gate[:,None]
        desired=(self.reference_q[:,self.q_indices]+action.clamp(-1,1)*radius).clamp(self.lower,self.upper)
        # The newest issued target and validity flag are already in actor observations.
        # Read before apply overwrites the oldest queue slot.
        previous=self.pending[(self.cursor-1)%len(self.pending)]
        origin=torch.where(previous[:,60:61]>0,previous[:,:30],self.reference_q[:,self.q_indices])
        limit=self.target_speed_rad_s*self.dt
        limited=origin+(desired-origin).clamp(-limit,limit)
        # Keep the trained corrective path as well as quiet standing intact.
        # Only requested targets beyond the original envelope need the new limiter.
        extension=(desired-self.reference_q[:,self.q_indices]).abs()>self.residual_radians+1e-6
        recovery=torch.where(extension,limited.clamp(self.lower,self.upper),desired)
        return torch.where(enabled,recovery,quiet)

    def contract(self):
        result=super().contract()
        result.update(action_mode='support_gated_target_pd_v2',recovery_motor=dict(
            schema='observable_hip_knee_authority_v2',maximum_residual_radians=self.maximum_radians,
            joints='Either leg: Thigh rx/ry/rz and Shin rx; other motors retain 0.25 rad',
            capture_gate_m=[.03,.08],unload_gate_n=[15.,5.],opposite_support_full_n=50.,
            unloading_speed_gate_m_s=[.15,.35],target_speed_rad_s=self.target_speed_rad_s,
            slew_origin='Newest issued absolute target; reference target before queue initialization',
            slew_scope='Only requested targets outside the original 0.25-radian interval',
            activation='Maximum of capture error and supported unloading while moving',
            quiet='Exactly original target mapping when gate is zero',
            observation_width=363,extra_hidden_state=False,clock=False))
        return result
