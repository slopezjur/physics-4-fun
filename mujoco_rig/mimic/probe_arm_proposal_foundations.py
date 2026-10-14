"""Proposal ablations with unchanged physical command execution."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .probe_arm_execution_distribution import setup,seal,episode
from .train_arm_correction_ppo import CorrectionEnvironment,OUT as PPO
from .target_control33 import DelayedTargetControl33
from .run_support_skill_study import digest,save

ROOT=Path('logs/mimickit-training/arm-proposal-30-20261008-01')
TEACHER=Path('logs/mimickit-training/arm-hold-learning-20261008-01')
PLAN=Path(__file__).with_name('ARM_PROPOSAL_30_PLAN.md')


class ProposalVariant(nn.Module):
    def __init__(self,base,env,anchor,poses,kind):
        super().__init__()
        self.base,self.env,self.anchor,self.kind=base,env,anchor,kind
        self.register_buffer('mean',base.mean.clone());self.register_buffer('scale',base.scale.clone())
        c=env.engine.policy_control;model=env.rig.model
        names=[model.joint(int(j)).name for j in model.actuator_trnid[env.rig.actuators,0]]
        self.register_buffer('arm_mask',torch.tensor(['Arm_' in name for name in names]))
        self.left=names.index('UpperArm_L_ry')
        self.register_buffer('reference',torch.tensor(poses[:,0,c.q_indices],dtype=torch.float32))
        self.register_buffer('delta',torch.tensor(poses[:,1,c.q_indices]-poses[:,0,c.q_indices],dtype=torch.float32))
        if kind in ('static','arm_static'):
            from .static_support import solve_target_support
            ff=[]
            for pose in poses:
                row=[]
                for goal in pose[:2]:
                    result=solve_target_support(env.rig,goal,minimum_norm=True)
                    assert result['feasible']
                    row.append(np.array(result['normalized_motor_torque'])/c.gain)
                ff.append(row)
            self.register_buffer('ff',torch.tensor(np.array(ff),dtype=torch.float32))

    @torch.no_grad()
    def forward(self,x):
        if self.kind=='inherited':return self.base(x)
        c=self.env.engine.policy_control;goal=x[:,374:420][:,c.q_indices]
        zero=torch.zeros((len(x),33),dtype=x.dtype,device=x.device)
        if self.kind=='goal':return zero
        if self.kind in ('static','arm_static'):
            amount=((goal-self.reference)*self.delta).sum(-1)/self.delta.square().sum(-1).clamp_min(1e-12)
            value=self.ff[:,0]+amount.clamp(0,1)[:,None]*(self.ff[:,1]-self.ff[:,0])
            if self.kind=='static':return value
        elif self.kind=='arm_goal':value=zero
        elif self.kind=='left_goal':
            value=self.base(x).clone();value[:,self.left]=0;return value
        elif self.kind=='drop_arms':return torch.where(self.arm_mask,zero,self.base(x))
        elif self.kind.startswith('cap_'):return self.base(x).clamp(-float(self.kind[4:]),float(self.kind[4:]))
        else:raise ValueError(self.kind)
        source=DelayedTargetControl33.target_positions(c,self.anchor(x[:,:374]))
        return torch.where(self.arm_mask,value,source-goal)


def specs(step):
    modes={2:['inherited'],3:['goal'],4:['static'],6:['arm_goal'],7:['arm_static'],
           8:['left_goal'],9:['drop_arms'],10:['cap_0.05','cap_0.1']}
    return [(kind,variant) for kind in modes[step] for variant in (False,True)]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,required=True);args=parser.parse_args()
    out=ROOT/f'step-{args.step:02d}';inputs=seal(out,[PLAN])
    spec=specs(args.step);save(out/'protocol.json',dict(step=args.step,inputs=inputs,specs=spec,training_updates=0,
        role='Frozen learned proposal or explicit scripted diagnostic; no new learned brain'))
    env,anchor,base,poses,actors=setup();results=[]
    for kind,variant in spec:
        proposal=ProposalVariant(base,env,anchor,poses,kind)
        adapter=CorrectionEnvironment(env,anchor,proposal,poses,variant=variant)
        folder=kind+('-varied' if variant else '-nominal')
        c=dict(actor=kind,mode='mean',variant=variant,seed=710081)
        result=episode(adapter,actors['zero'],c,out/folder,keep_features=True)
        if kind=='inherited' and not variant:
            old=np.load(PPO/'zero-nominal/trace.npz');new=np.load(out/folder/'trace.npz')
            assert all(np.array_equal(old[k],new[k]) for k in ('qpos','issued','alive'))
        results.append(dict(folder=folder,**result));save(out/'progress.json',results)
    assert all(digest(Path(p))==h for p,h in inputs.items())
    save(out/'report.json',dict(step=args.step,results=results,inputs_unchanged=True,
        outputs={str(p):digest(p) for p in out.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
