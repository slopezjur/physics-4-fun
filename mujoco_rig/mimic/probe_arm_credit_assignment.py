"""Common-state physical branches to audit frozen correction critics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .run_support_skill_study import digest,save
from .train_arm_correction_ppo import CorrectionEnvironment,CorrectionNetwork,OUT as PPO,BASELINE
from .probe_arm_execution_distribution import ROOT,setup,seal
from .probe_arm_action_response import snapshot,fingerprint

PHASES={11:('ramp',90),12:('hold',210),13:('return',360)}
BRANCHES=[('zero',0.,0),('plus-pulse',.005,1),('minus-pulse',-.005,1),
          ('plus-held',.005,12),('minus-held',-.005,12),
          ('learned-521028',1.,12),('opposite-521028',-1.,12),
          ('learned-531028',1.,12),('opposite-531028',-1.,12)]


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--step',type=int,required=True);args=p.parse_args()
    phase,start_frame=PHASES[args.step];out=ROOT/f'step-{args.step:02d}'
    critic_paths=[PPO/f'{seed}/critic-final.pt' for seed in (521028,531028)]
    inputs=seal(out,critic_paths+[BASELINE])
    save(out/'protocol.json',dict(step=args.step,phase=phase,start_frame=start_frame,horizon=120,
        branches=BRANCHES,critics=[521028,531028],inputs=inputs,training_updates=0,
        continuation='zero correction after each bounded intervention; critics evaluated off-policy diagnostically'))
    env,anchor,proposal,poses,actors=setup();adapter=CorrectionEnvironment(env,anchor,proposal,poses)
    critics=[]
    for path in critic_paths:
        critic=CorrectionNetwork(1);critic.load_state_dict(torch.load(path,weights_only=True))
        critics.append(critic.eval().requires_grad_(False))
    qi=adapter.physical.qi;model=env.rig.model
    motor=int(np.flatnonzero(qi.numpy()==model.jnt_qposadr[model.joint('UpperArm_L_ry').id])[0])
    archived=np.load(BASELINE);results=[];common=None
    for name,magnitude,duration in BRANCHES:
        adapter.reset();adapter.physical.control.cursor=0;active=torch.arange(16)<8
        for frame in range(start_frame):
            _,terminal,info=adapter.step(torch.zeros((16,33)),active)
            for key in ('qpos','issued'):assert np.array_equal(info[key][:8].numpy(),archived[key][frame])
            assert np.array_equal(active[:8].numpy(),archived['alive'][frame])
            active&=~terminal
        state=snapshot(adapter.physical,active);stamp=fingerprint(state)
        if common is None:
            common=stamp;np.savez_compressed(out/'start-state.npz',**state)
        assert stamp==common
        rows=[]
        for offset in range(120):
            x,*_=adapter.observation();action=torch.zeros((16,33))
            if offset<duration:
                if name.startswith(('learned','opposite')):action=actors[name.split('-')[1]](x)*magnitude
                else:action[:,motor]=magnitude
            value=torch.stack([c(x) for c in critics],-1)
            before=active.clone();reward,terminal,info=adapter.step(action,active)
            nx,*_=adapter.observation();nextvalue=torch.stack([c(nx) for c in critics],-1)
            values=dict(alive=before,reward=reward,value=value,next_value=nextvalue,terminal=terminal,
                failed=info['failed'],support=info['supported'],stand=info['stand'],qpos=info['qpos'],
                issued=info['issued'],addition=info['addition'],nominal=info['nominal_addition'],
                correction=action,components=info['components'],com=env.last['com_m'],xy=env.last['foot_xy_m'])
            rows.append({k:v[:8].numpy().copy() for k,v in values.items()});active&=~terminal
        z={k:np.stack([row[k] for row in rows]) for k in rows[0]}
        if name=='zero':
            for key in ('qpos','issued','alive'):assert np.array_equal(z[key],archived[key][start_frame:start_frame+120])
        np.savez_compressed(out/f'{name}.npz',**z)
        row=dict(name=name,start_sha256=stamp,survivors=np.flatnonzero(z['alive'][-1]&~z['failed'][-1]).tolist())
        results.append(row);save(out/'progress.json',results)
        print('BRANCH',phase,row,flush=True)
    assert all(digest(Path(p))==h for p,h in inputs.items())
    save(out/'report.json',dict(step=args.step,phase=phase,results=results,inputs_unchanged=True,
        outputs={str(p):digest(p) for p in out.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
