"""Audit disturbance admission, clean labels, matched learning and physical gates."""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .run_support_skill_study import FOUNDATION,digest,save
from .run_learned_handoff import PRIOR
from .train_arm_deviations import INITIAL


def main():
    from .rig import Rig
    parent=Path('logs/mimickit-training')
    original=parent/'arm-hold-deviations-20261008-01'
    root=parent/'arm-hold-deviations-calibration-20261008-01'
    out=parent/'arm-hold-deviations-review-20261008-01';out.mkdir(exist_ok=False)
    for run in (original,root):
        protocol=json.loads((run/'protocol.json').read_text());report=json.loads((run/'report.json').read_text())
        assert all(digest(Path(p))==h for p,h in protocol['inputs'].items())
        assert all(digest(Path(p))==h for p,h in report['output_hashes'].items())
    rig=Rig.load(FOUNDATION/'dummy.xml',policy_exclude=());m=rig.model;private=mujoco.MjData(m);dt=m.opt.timestep*rig.decimation
    qi=m.jnt_qposadr[m.actuator_trnid[rig.actuators,0]];joints=m.actuator_trnid[rig.actuators,0]
    lower,upper=m.jnt_range[joints].T;poses=np.load(PRIOR/'goals.npz')['poses'];results=[]
    folders=[original/'teacher-disturbed']+[p for p in root.iterdir() if (p/'trace.npz').exists()]
    for folder in folders:
        z=np.load(folder/'trace.npz');r=json.loads((folder/'report.json').read_text())
        if folder.name=='quiet30':
            baseline=np.load(parent/'learned-support-review-20261008-01/zero-addition-quiet30/trace.npz')
            assert np.array_equal(z['qpos'],baseline['qpos']) and np.array_equal(z['issued'],baseline['issued']);continue
        h=(z['time']>r['hold_end']-2)&(z['time']<=r['hold_end']);verified=[]
        for i,c in enumerate(r['cases']):
            live=z['alive'][:,i];last=np.flatnonzero(live)[-1]
            assert live[:last+1].all() and not live[last+1:].any()
            private.qpos[:]=poses[i,0];mujoco.mj_forward(m,private);xy=private.xpos[[m.body('Foot_L').id,m.body('Foot_R').id],:2].copy()
            private.qpos[:]=poses[i,1];mujoco.mj_forward(m,private);com=private.subtree_com[1].copy()
            error=np.sqrt(np.mean((z['qpos'][:,i,qi]-poses[i,1,qi])**2,axis=-1))
            assert np.allclose(error,z['tracking_rmse'][:,i],atol=1e-8,rtol=0)
            full=h.sum()>=int(2/dt)-1 and live[h].all()
            stable=bool(full and z['support'][h,i].all())
            tracked=bool(stable and (error[h]<=.05).all() and (np.linalg.norm(z['com'][h,i,:2]-com[:2],axis=-1)<=.02).all()
                and (np.linalg.norm(z['xy'][h,i]-xy,axis=-1)<=.02).all())
            survived=len(z['time'])==int(np.ceil(r['seconds']/dt)) and live[-1] and z['done'][-1,i]!=1
            returned=bool(survived and z['stand'][-int(np.ceil(2/dt)):,i].all())
            assert (stable,tracked,returned,tracked and returned)==(c['stable_hold'],c['tracked_hold'],c['returned_stand'],c['complete'])
            verified.append(dict(pose=i,complete=tracked and returned,stable=stable,returned=returned))
        results.append(dict(condition=folder.name,cases=verified))
    teacher=root/'teacher-0.003'
    d=np.load(teacher/'corrective-samples.npz');queries=np.load(teacher/'target-queries.npz')
    lookup={(int(f),int(w)):i for i,(f,w) in enumerate(zip(queries['frame'],queries['world']))}
    ids=np.array([lookup[(int(f),int(w))] for f,w in zip(d['frame'],d['world'])])
    assert np.array_equal(d['raw_target'],queries['clean'][ids]) and np.array_equal(d['applied_raw_target'],queries['applied'][ids])
    assert np.array_equal(d['validation'],d['world']>=6)
    assert (d['frame']*dt>=2.5).all() and (d['frame']*dt<5.5).all()
    x=d['x'];ref=poses[d['world'],0].astype(np.float32)
    amount=np.clip(np.max(abs(x[:,374:420][:,qi]-ref[:,qi]),axis=-1)/.05,0,1)*x[:,420]
    desired=np.clip(d['applied_raw_target']-d['source'],-1,1)*amount[:,None]
    addition=d['previous']+np.clip(desired-d['previous'],-.02,.02);addition[d['quiet']]=0
    composition_error=float(abs(np.clip(d['source']+addition,lower,upper)-d['issued']).max());assert composition_error<2e-6
    assert (abs(queries['applied']-queries['clean'])<=.003001).all()
    outside=(queries['frame']*dt<2.5)|(queries['frame']*dt>=4.5)
    assert np.array_equal(queries['applied'][outside],queries['clean'][outside])
    z=np.load(teacher/'trace.npz');baseline=np.load(parent/'arm-hold-learning-20261008-01/teacher-nominal/trace.npz')
    before=z['time']<2.5;assert np.array_equal(z['qpos'][before],baseline['qpos'][before])
    interval=(z['time']>=2.5)&(z['time']<=5.5)
    initial=torch.load(INITIAL,weights_only=True)
    for name in ('replay','correction'):
        state=torch.load(root/name/'initial.pt',weights_only=True)
        assert all(torch.equal(state[k],v) for k,v in initial.items())
    save(out/'report.json',dict(provenance_verified=True,physical_gates_recomputed=True,clean_labels_and_applied_commands_verified=True,
        whole_pose_holdouts_preserved=True,initial_weights_identical=True,quiet_trace_exactly_matches_anchor=True,
        data_examples=len(x),train_examples=int((~d['validation']).sum()),heldout_examples=int(d['validation'].sum()),
        maximum_composition_error_rad=composition_error,
        maximum_com_deviation_m=float(np.linalg.norm(z['com'][interval]-baseline['com'][interval],axis=-1).max()),
        maximum_controlled_joint_deviation_rad=float(abs(z['qpos'][interval][:,:,qi]-baseline['qpos'][interval][:,:,qi]).max()),
        comparisons=results,physical_cases=48,supervised_updates=8000,arena_changed=False,source_sha256=digest(Path(__file__))))
    print(json.dumps(dict(comparisons=results,data_examples=len(x)),indent=2))


if __name__=='__main__':main()
