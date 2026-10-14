"""Recompute fixed-goal gates from compact evidence and verify source provenance."""
import json
from pathlib import Path

import mujoco
import numpy as np

from .run_support_skill_study import FOUNDATION, digest, save


def main():
    root=Path('logs/mimickit-training/fixed-support-goals-20261008-01')
    out=Path('logs/mimickit-training/fixed-support-goals-review-20261008-01');out.mkdir(exist_ok=False)
    protocol=json.loads((root/'protocol.json').read_text());report=json.loads((root/'report.json').read_text())
    assert all(digest(Path(p))==h for p,h in protocol['inputs'].items())
    assert all(digest(Path(p))==h for p,h in report['output_hashes'].items())
    model=mujoco.MjModel.from_xml_path(str(FOUNDATION/'dummy.xml'));data=mujoco.MjData(model)
    # Use the recorded rig's complete motor contract.
    from .rig import Rig
    rig=Rig.load(FOUNDATION/'dummy.xml',policy_exclude=())
    qi=model.jnt_qposadr[model.actuator_trnid[rig.actuators,0]]
    poses=np.load(root/'goals.npz')['poses'];summaries=[];dt=model.opt.timestep*rig.decimation
    for key,r in report['results'].items():
        z=np.load(root/key/'trace.npz');hold_end,seconds=(5.5,9.) if r['kind']=='arms' else (7.5,11.5)
        hold=(z['time']>hold_end-2)&(z['time']<=hold_end)
        assert np.allclose(z['time'],(np.arange(len(z['time']))+1)*dt,atol=1e-12,rtol=0)
        checks=[]
        for col,c in enumerate(r['cases']):
            world=c['world'];target=poses[world,1 if r['kind']=='arms' else 2]
            data.qpos[:]=poses[world,0];mujoco.mj_forward(model,data)
            xy=data.xpos[[model.body('Foot_L').id,model.body('Foot_R').id],:2].copy()
            data.qpos[:]=target;mujoco.mj_forward(model,data);com=data.subtree_com[1].copy()
            live=z['alive'][:,col];last=np.flatnonzero(live)[-1]
            assert live[:last+1].all() and not live[last+1:].any()
            error=np.sqrt(np.mean((z['qpos'][:,col,qi]-target[qi])**2,axis=-1))
            assert np.allclose(error,z['tracking_rmse'][:,col],atol=1e-8,rtol=0)
            full_hold=hold.sum()>=int(2/dt)-1 and z['alive'][hold,col].all()
            stable=bool(full_hold and z['support'][hold,col].all())
            tracked=bool(stable and (error[hold]<=.05).all() and
                (np.linalg.norm(z['com'][hold,col,:2]-com[:2],axis=-1)<=.02).all() and
                (np.linalg.norm(z['xy'][hold,col]-xy,axis=-1)<=.02).all())
            survived=bool(len(z['time'])==int(np.ceil(seconds/dt)) and live[-1] and z['done'][-1,col]!=1)
            returned=bool(survived and z['stand'][-int(np.ceil(2/dt)):,col].all())
            assert (stable,tracked,returned,tracked and returned)==(c['stable_hold'],c['tracked_hold'],c['returned_stand'],c['complete'])
            checks.append(dict(world=world,stable=stable,tracked=tracked,returned=returned))
        if r['controller']!='teacher_original':assert z['slew'][z['alive']].max()<=.020001
        summaries.append(dict(condition=key,cases=checks,complete=r['complete'],
            maximum_live_addition_rad=float(abs(z['addition'][z['alive']]).max())))
    result=dict(protected_sources_and_outputs_unchanged=True,gate_recomputation_passed=True,
        physical_cases=sum(len(r['worlds']) for r in report['results'].values()),training_updates=0,
        comparisons=summaries,arena_changed=False,source_sha256=digest(Path(__file__)))
    save(out/'report.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
