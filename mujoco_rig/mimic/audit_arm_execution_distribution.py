"""Independent trace gates and same-state distribution-bias measurements."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .run_support_skill_study import FOUNDATION,digest,save
from .run_learned_handoff import PRIOR
from .probe_arm_execution_distribution import ROOT,projected_mean


def audit(root):
    from .rig import Rig
    rig=Rig.load(FOUNDATION/'dummy.xml',policy_exclude=())
    model=rig.model;dt=model.opt.timestep*rig.decimation
    qi=model.jnt_qposadr[model.actuator_trnid[rig.actuators,0]]
    bounds=model.jnt_range[model.actuator_trnid[rig.actuators,0]]
    poses=np.load(PRIOR/'goals.npz')['poses'];data=mujoco.MjData(model)
    report=json.loads((root/'report.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    assert all(digest(Path(p))==h for p,h in protocol['inputs'].items())
    assert all(digest(Path(p))==h for p,h in report['outputs'].items())
    for r in report['results']:
        with np.load(root/r['folder']/'trace.npz') as z:
            assert np.allclose(z['time'],(np.arange(len(z['time']))+1)*dt,atol=1e-12,rtol=0)
            assert len(z['time'])==int(np.ceil(r['seconds']/dt))
            scaled=z['correction']*z['scale'][:,:,None]
            projected=np.clip(z['previous']+np.clip(z['nominal']+scaled-z['previous'],-.02,.02),-1,1)
            expected=np.where(z['correction']!=0,projected,z['nominal'])
            expected[z['quiet']|~z['alive']]=0
            # Float multiplication association differs from physical amount*fade.
            assert np.allclose(expected,z['addition'],atol=2e-7,rtol=0)
            live=z['alive']&~z['quiet']
            assert abs(z['addition'][live]-z['previous'][live]).max(initial=0)<=.0200001
            assert abs(z['addition']).max()<=1.0000001
            assert (z['issued']>=bounds[:,0]-1e-6).all() and (z['issued']<=bounds[:,1]+1e-6).all()
            assert np.isfinite(z['qpos']).all()
            hold=(z['time']>r['hold_end']-2)&(z['time']<=r['hold_end'])
            for i,c in enumerate(r['cases']):
                data.qpos[:]=poses[i,0];mujoco.mj_forward(model,data)
                xy=data.xpos[[model.body('Foot_L').id,model.body('Foot_R').id],:2].copy()
                data.qpos[:]=poses[i,1];mujoco.mj_forward(model,data);com=data.subtree_com[1].copy()
                motor=np.sqrt(np.mean((z['qpos'][:,i,qi]-poses[i,1,qi])**2,axis=-1))
                stable=bool(z['alive'][hold,i].all() and z['support'][hold,i].all())
                tracked=bool(stable and (motor[hold]<=.05).all()
                    and (np.linalg.norm(z['com'][hold,i,:2]-com[:2],axis=-1)<=.02).all()
                    and (np.linalg.norm(z['xy'][hold,i]-xy,axis=-1)<=.02).all())
                survived=bool(z['alive'][-1,i] and not z['done'][-1,i])
                returned=bool(survived and z['stand'][-int(np.ceil(2/dt)):,i].all())
                assert (stable,tracked,returned,tracked and returned)==(c['stable_hold'],c['tracked_hold'],c['returned_stand'],c['complete'])
    return dict(step=protocol.get('step'),cases=sum(len(r['cases']) for r in report['results']),gates_and_projection_passed=True)


def distribution():
    out=ROOT/'step-06';out.mkdir(exist_ok=False)
    inputs={};rows=[]
    report=json.loads((ROOT/'step-01/report.json').read_text())
    quantile=torch.distributions.Normal(0.,1.).icdf((torch.arange(4096,dtype=torch.float64)+.5)/4096)
    for r in report['results']:
        if r['config']['mode']!='mean':continue
        path=ROOT/'step-01'/r['folder']/'trace.npz';inputs[str(path)]=digest(path)
        with np.load(path) as z:
            mask=z['alive']&~z['quiet']&(z['scale']>0)
            mean,scale,previous,nominal=(torch.tensor(z[k][mask]) for k in ('raw_mean','scale','previous','nominal'))
            expected=projected_mean(mean,scale,previous,nominal,.005)
            deterministic=(previous+(nominal+mean*scale[:,None]-previous).clamp(-.02,.02)).clamp(-1,1)
            bias=(expected-deterministic).numpy()
            indices=np.linspace(0,len(mean)-1,64,dtype=int)
            m=nominal[indices].double()+mean[indices].double()*scale[indices,None].double()
            noisy=m[None]+quantile[:,None,None]*.005*scale[None,indices,None].double()
            lo=(previous[indices].double()-.02).clamp_min(-1);hi=(previous[indices].double()+.02).clamp_max(1)
            quadrature=noisy.clamp(lo,hi).mean(0).float()
            error=float((quadrature-expected[indices]).abs().max())
            assert error<2e-6
            rows.append(dict(actor=r['config']['actor'],states=len(mean),bias_rms=float(np.sqrt(np.mean(bias**2))),
                bias_max=float(abs(bias).max()),bias_mean_abs=float(abs(bias).mean()),raw_mean_rms=float(mean.square().mean().sqrt()),
                coordinates_over_half_milliradian=float((abs(bias)>.0005).mean()),quadrature_max_error=error))
    save(out/'report.json',dict(step=6,inputs=inputs,results=rows,physical_evaluations=0,
        interpretation='Same-state rate/envelope projection before final joint clipping; not a causal skill-success result.'))
    print(json.dumps(rows,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--distribution',action='store_true');args=p.parse_args()
    torch.set_num_threads(2)
    if args.distribution:distribution()
    rows=[]
    for folder in sorted(ROOT.glob('step-*')):
        if (folder/'protocol.json').exists() and (folder/'report.json').exists():rows.append(audit(folder))
    save(ROOT/'execution-audit.json',dict(results=rows))
    print(json.dumps(rows))


if __name__=='__main__':main()
