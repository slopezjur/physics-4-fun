"""Frozen amplitude, output-noise, timing, elbow and long-quiet challenges."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

from .probe_arm_proposal_foundations import ROOT,PLAN
from .probe_arm_execution_distribution import setup,seal,episode
from .train_arm_relative_proposal import load_head
from .train_arm_correction_ppo import CorrectionEnvironment
from .run_support_skill_study import digest,save

HEADS={'all-711028':ROOT/'step-15/inherited','all-721028':ROOT/'step-21/model',
       'active-711028':ROOT/'step-22/model','active-721028':ROOT/'step-23/model'}


class OutputProbe(nn.Module):
    def __init__(self,base,env,*,std=0.,seed=710081,elbows=False):
        super().__init__();self.base,self.env,self.std,self.elbows=base,env,std,elbows
        self.register_buffer('mean',base.mean.clone());self.register_buffer('scale',base.scale.clone())
        names=[env.rig.model.joint(int(j)).name for j in env.rig.model.actuator_trnid[env.rig.actuators,0]]
        self.register_buffer('mask',torch.tensor([n.startswith('Forearm_') for n in names]));assert self.mask.sum()==2
        self.generator=torch.Generator().manual_seed(seed);self.key=None;self.noise=None;self.noises=[]

    @torch.no_grad()
    def forward(self,x):
        value=self.base(x)
        if self.elbows:value=value.masked_fill(self.mask,0)
        if self.std:
            key=tuple(self.env.task.steps.tolist())
            if key!=self.key:
                self.noise=torch.randn(value.shape,generator=self.generator)*self.std
                self.noises.append(self.noise.numpy().copy());self.key=key
            value=value+self.noise
        return value


def adapter_for(env,anchor,proposal,poses,*,timeline=None,quiet_seconds=None):
    adapter=CorrectionEnvironment(env,anchor,proposal,poses,quiet=quiet_seconds is not None)
    if timeline is None and quiet_seconds is None:return adapter
    from .predictive_target_support import blend,blend_pose
    p=adapter.physical
    if timeline is not None:adapter.onset,adapter.ramp,adapter.hold=timeline
    adapter.hold_end=adapter.onset+adapter.ramp+adapter.hold
    adapter.seconds=quiet_seconds if quiet_seconds is not None else adapter.hold_end+3.5
    p.frames=int(np.ceil(adapter.seconds/env.dt));goals=[];coms=[];amounts=[];data=mujoco.MjData(env.rig.model)
    for frame in range(p.frames+1):
        now=frame*env.dt;future=now+2*env.dt;row=[];com=[]
        for pose in poses:
            ref,clear=pose[:2]
            if quiet_seconds is not None or now<adapter.onset:q=ref
            elif future<adapter.onset+adapter.ramp:q=blend_pose(ref,clear,future-adapter.onset,adapter.ramp)
            elif future<adapter.hold_end:q=clear
            else:q=blend_pose(clear,ref,future-adapter.hold_end,1.5)
            row.append(q);data.qpos[:]=q;mujoco.mj_forward(env.rig.model,data);com.append(data.subtree_com[1].copy())
        goals.append(row);coms.append(com);amounts.append(float(blend(0.,1.,now-adapter.onset,1.)))
    p.goals=torch.tensor(np.array(goals),dtype=torch.float32);p.target_com=torch.tensor(np.array(coms),dtype=torch.float32)
    p.authority=torch.tensor(amounts,dtype=torch.float32);p.gate.frames=p.frames;p.gate.hold_end=adapter.hold_end
    return adapter


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,choices=(24,25,26,28,29,30),required=True);args=parser.parse_args()
    step=args.step;out=ROOT/f'step-{step:02d}';selected=None
    if step in (29,30):selected=json.loads((ROOT/'step-27/report.json').read_text())['selected_recipe']
    extra=[PLAN,Path(__file__).with_name('ARM_PROPOSAL_30_FINAL_BLOCK.md')]+[p/'final.pt' for p in HEADS.values()]
    if selected is not None:extra.append(ROOT/'step-27/report.json')
    inputs=seal(out,extra);env,anchor,base,original_poses,actors=setup()
    heads={k:load_head(p) for k,p in HEADS.items()}
    if step==26:heads['original']=base
    if selected is not None:heads={k:v for k,v in heads.items() if k.startswith(selected+'-')}
    if step==29:
        from .run_single_support_feedback import make_environment
        env=make_environment(env.rig,anchor,80.);heads={'anchor-baseline':base,**heads}
    configs=[]
    for label in heads:
        for variant in ((False,True) if step==28 else (False,)):
            for seed in ((710081,710082) if step==25 else (710081,)):
                configs.append(dict(actor=label,mode='mean',variant=variant,seed=seed,quiet=step==29,
                    amplitude=.5 if step==24 else 1.,proposal_std=.005 if step==25 else 0.,elbows_zero=step==28,
                    timeline=[1.5,2.,4.] if step==26 else [.8,1.2,2.5] if step==30 else None))
    save(out/'protocol.json',dict(step=step,inputs=inputs,configs=configs,training_updates=0,
        selected_recipe=selected,noise_placement='Raw proposal output before original authority/fade/slew'))
    results=[];quiet_baseline=None
    for index,c in enumerate(configs):
        poses=original_poses.copy()
        if c['amplitude']!=1:
            from .predictive_target_support import blend_pose
            for i in range(16):poses[i,1]=blend_pose(poses[i,0],poses[i,1],c['amplitude'],1.)
        proposal=OutputProbe(heads[c['actor']],env,std=c['proposal_std'],seed=c['seed'],elbows=c['elbows_zero'])
        if c['variant']:adapter=CorrectionEnvironment(env,anchor,proposal,poses,variant=True)
        else:adapter=adapter_for(env,anchor,proposal,poses,timeline=c['timeline'],quiet_seconds=60. if step==29 else None)
        folder=f'{index:02d}-{c["actor"]}-{c["seed"]}'
        r=episode(adapter,actors['zero'],c,out/folder)
        if proposal.noises:np.savez_compressed(out/folder/'noise.npz',noise=np.stack(proposal.noises))
        if step==29:
            z=dict(np.load(out/folder/'trace.npz'))
            if quiet_baseline is None:quiet_baseline=z
            else:assert all(np.array_equal(z[k],quiet_baseline[k]) for k in ('qpos','issued','alive'))
            r['quiet_exact']=True
        results.append(dict(folder=folder,**r));save(out/'progress.json',results)
    assert all(digest(Path(p))==h for p,h in inputs.items())
    save(out/'report.json',dict(step=step,results=results,inputs_unchanged=True,arena_changed=False,
        outputs={str(p):digest(p) for p in out.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
