"""Common-data loss, physical gate failures and prospective recipe selection."""
import json
from pathlib import Path

import numpy as np
import torch

from .probe_arm_proposal_foundations import ROOT,TEACHER,PLAN
from .probe_arm_execution_distribution import setup,seal
from .train_arm_relative_proposal import load_head
from .challenge_arm_proposals import HEADS
from .run_support_skill_study import digest,save


def physical_results():
    first=json.loads((ROOT/'step-16/report.json').read_text())['results'][0]
    varied=json.loads((ROOT/'step-17/report.json').read_text())['results'][0]
    return {'all-711028':[(ROOT/'step-16'/first['folder'],first),(ROOT/'step-17'/varied['folder'],varied)],
        **{label:[(ROOT/f'step-{step:02d}'/r['folder'],r) for r in json.loads((ROOT/f'step-{step:02d}/report.json').read_text())['results']]
           for label,step in (('all-721028',21),('active-711028',22),('active-721028',23))}}


@torch.no_grad()
def main():
    out=ROOT/'step-27';folders={**HEADS,'relative-711028':ROOT/'step-15/relative','relative-721028':ROOT/'step-18/relative'}
    paths=[p/'final.pt' for p in folders.values()]+[TEACHER/'training-samples.npz',PLAN]
    physical=physical_results()
    for cases in physical.values():paths += [folder/'trace.npz' for folder,_ in cases]
    inputs=seal(out,paths);env,_,_,poses,_=setup();qi=env.engine.policy_control.q_indices.numpy()
    data=dict(np.load(TEACHER/'training-samples.npz'));x=torch.tensor(data['x']);target=data['raw_target']-data['goal']
    regression=[]
    for label,path in folders.items():
        net=load_head(path);prediction=net(x).numpy()
        for heldout in (False,True):
            for quiet in (False,True):
                use=(data['validation']==heldout)&(data['quiet']==quiet)
                regression.append(dict(actor=label,heldout=heldout,quiet=quiet,samples=int(use.sum()),
                    rmse=float(np.sqrt(np.mean((prediction[use]-target[use])**2)))))
    comparisons=[];scores={}
    model=env.rig.model;names=[model.joint(int(j)).name for j in model.actuator_trnid[env.rig.actuators,0]]
    for label,cases in physical.items():
        for folder,r in cases:
            z=np.load(folder/'trace.npz');holding=(z['time']>r['hold_end']-2)&(z['time']<=r['hold_end'])
            live=z['alive']&holding[:,None]
            error=(z['qpos'][:,:,qi]-poses[:8,1,qi])[live];command=(z['issued']-poses[:8,1,qi])[live]
            actual=np.sqrt(np.mean(error**2,axis=0));issued=np.sqrt(np.mean(command**2,axis=0));order=np.argsort(actual)[-5:][::-1]
            gate=[]
            for c in r['cases']:
                reasons=[]
                if not c['stable_hold']:reasons.append('support')
                if c['peak_motor']>.05:reasons.append('motor')
                if c['peak_com']>.02:reasons.append('COM')
                if c['peak_feet']>.02:reasons.append('feet')
                if not c['returned_stand']:reasons.append('return')
                gate.append(dict(pose=c['pose'],complete=c['complete'],reasons=reasons,
                    peak_motor=c['peak_motor'],peak_com=c['peak_com'],peak_feet=c['peak_feet']))
            phases=[];onset,ramp=(1.2,1.7) if r['config'].get('variant') else (1.,1.5)
            for name,start,end in (('ramp',onset,onset+ramp),('hold',onset+ramp,r['hold_end']),('return',r['hold_end'],r['hold_end']+1.5),('settle',r['hold_end']+1.5,r['seconds']+.01)):
                mask=(z['time']>start)&(z['time']<=end);alive=z['alive'][mask]
                phases.append(dict(phase=name,alive_frames=int(alive.sum()),unsupported_frames=int((alive&~z['support'][mask]).sum()),
                    failed_cases=np.flatnonzero((alive&z['done'][mask]).any(0)).tolist()))
            comparisons.append(dict(actor=label,variant=r['config'].get('variant',False),holds=r['stable_holds'],returned=r['returned'],complete=r['complete'],
                gates=gate,phases=phases,worst_joints=[dict(joint=names[j],actual_rms=float(actual[j]),issued_rms=float(issued[j])) for j in order]))
    for recipe in ('all','active'):
        group=[c for c in comparisons if c['actor'].startswith(recipe+'-')]
        scores[recipe]=[min(c['complete'] for c in group),sum(c['complete'] for c in group),sum(c['returned'] for c in group)]
    assert scores['all']!=scores['active'],'Tie: preserve inconclusive selection before long challenge'
    selected=max(scores,key=lambda k:scores[k])
    assert all(digest(Path(p))==h for p,h in inputs.items())
    save(out/'report.json',dict(step=27,inputs=inputs,regression=regression,physical=comparisons,scores=scores,
        selected_recipe=selected,selection='Predeclared worst nominal/varied full score, sum full scores, then returns',
        interpretation='Goal tracking with fixed feet. No learned stepping or impact response measured.'))
    print('SELECTED',selected,scores,flush=True)


if __name__=='__main__':main()
