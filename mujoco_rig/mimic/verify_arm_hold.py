"""Check exact supervision, holdout separation, and physical arm-hold outcomes."""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .run_support_skill_study import FOUNDATION,digest,save
from .run_learned_handoff import PRIOR


def main():
    from .rig import Rig
    root=Path('logs/mimickit-training/arm-hold-learning-20261008-01')
    out=Path('logs/mimickit-training/arm-hold-review-20261008-01');out.mkdir(exist_ok=False)
    protocol=json.loads((root/'protocol.json').read_text());report=json.loads((root/'report.json').read_text())
    assert all(digest(Path(p))==h for p,h in protocol['inputs'].items())
    assert all(digest(Path(p))==h for p,h in report['output_hashes'].items())
    rig=Rig.load(FOUNDATION/'dummy.xml',policy_exclude=());model=rig.model;dt=model.opt.timestep*rig.decimation
    qi=model.jnt_qposadr[model.actuator_trnid[rig.actuators,0]]
    lower,upper=model.jnt_range[model.actuator_trnid[rig.actuators,0]].T
    poses=np.load(PRIOR/'goals.npz')['poses'];private=mujoco.MjData(model)
    dataset_checks=[];cases=[]
    for folder in sorted(root.iterdir()):
        if not (folder/'trace.npz').exists():continue
        r=json.loads((folder/'report.json').read_text());z=np.load(folder/'trace.npz')
        if folder.name=='quiet30':
            baseline=np.load('logs/mimickit-training/learned-support-review-20261008-01/zero-addition-quiet30/trace.npz')
            assert np.array_equal(z['qpos'],baseline['qpos']) and np.array_equal(z['issued'],baseline['issued'])
            continue
        hold=(z['time']>r['hold_end']-2)&(z['time']<=r['hold_end'])
        assert np.allclose(z['time'],(np.arange(len(z['time']))+1)*dt,atol=1e-12,rtol=0)
        complete=0
        for i,c in enumerate(r['cases']):
            alive=z['alive'][:,i];last=np.flatnonzero(alive)[-1]
            assert alive[:last+1].all() and not alive[last+1:].any()
            private.qpos[:]=poses[i,0];mujoco.mj_forward(model,private)
            xy=private.xpos[[model.body('Foot_L').id,model.body('Foot_R').id],:2].copy()
            private.qpos[:]=poses[i,1];mujoco.mj_forward(model,private);com=private.subtree_com[1].copy()
            error=np.sqrt(np.mean((z['qpos'][:,i,qi]-poses[i,1,qi])**2,axis=-1))
            assert np.allclose(error,z['tracking_rmse'][:,i],atol=1e-8,rtol=0)
            full=hold.sum()>=int(2/dt)-1 and z['alive'][hold,i].all()
            stable=bool(full and z['support'][hold,i].all())
            tracked=bool(stable and (error[hold]<=.05).all() and (np.linalg.norm(z['com'][hold,i,:2]-com[:2],axis=-1)<=.02).all()
                and (np.linalg.norm(z['xy'][hold,i]-xy,axis=-1)<=.02).all())
            survived=len(z['time'])==int(np.ceil(r['seconds']/dt)) and alive[-1] and z['done'][-1,i]!=1
            returned=bool(survived and z['stand'][-int(np.ceil(2/dt)):,i].all())
            assert (stable,tracked,returned,tracked and returned)==(c['stable_hold'],c['tracked_hold'],c['returned_stand'],c['complete'])
            complete+=int(tracked and returned)
        assert complete==r['complete'];cases.append(dict(condition=folder.name,complete=complete,teacher=r['teacher']))
        if r['teacher']:
            d=np.load(folder/'samples.npz');world=d['world'];frame=d['frame'];x=d['x']
            assert np.array_equal(d['validation'],world>=6)
            assert all(r['cases'][i]['complete'] for i in np.unique(world))
            ref=poses[world,0].astype(np.float32)
            quiet=np.max(abs(x[:,374:420]-ref),axis=-1)<=1e-6
            assert np.array_equal(quiet,d['quiet'])
            amount=np.clip(np.max(abs(x[:,374:420][:,qi]-ref[:,qi]),axis=-1)/.05,0,1)*x[:,420]
            desired=np.clip(d['raw_target']-d['source'],-1,1)*amount[:,None]
            addition=d['previous']+np.clip(desired-d['previous'],-.02,.02);addition[quiet]=0
            issued=np.clip(d['source']+addition,lower,upper)
            maximum=float(abs(issued-d['issued']).max());assert maximum<2e-6
            assert np.array_equal(d['issued'],z['issued'][frame,world])
            later=frame>0
            assert np.array_equal(x[later,:3],z['qpos'][frame[later]-1,world[later],:3])
            assert np.array_equal(x[later,15:54],z['qpos'][frame[later]-1,world[later],7:])
            for lag,start in ((2,95),(1,162)):
                ready=frame>=lag
                assert np.array_equal(x[ready,start:start+33],z['issued'][frame[ready]-lag,world[ready]])
                assert (x[ready,start+66]==1).all() and (x[~ready,start:start+67]==0).all()
            dataset_checks.append(dict(condition=folder.name,examples=len(x),train_examples=int((world<6).sum()),
                heldout_examples=int((world>=6).sum()),maximum_composition_error_rad=maximum))
    initial=torch.load(PRIOR/'goal-421028/final.pt',weights_only=True)
    for name in ('replay','hold','replica'):
        path=root/name/'initial.pt'
        if path.exists():
            weights=torch.load(path,weights_only=True)
            assert weights.keys()==initial.keys() and all(torch.equal(weights[k],v) for k,v in initial.items())
    save(out/'report.json',dict(inputs_and_outputs_unchanged=True,exact_issue_state_and_queue_verified=True,
        target_composition_verified=True,quiet_trace_exactly_matches_anchor=True,initial_weights_identical=True,
        dataset_checks=dataset_checks,physical_results=cases,arena_changed=False,source_sha256=digest(Path(__file__))))
    print(json.dumps(dict(dataset_checks=dataset_checks,physical_results=cases),indent=2))


if __name__=='__main__':main()
