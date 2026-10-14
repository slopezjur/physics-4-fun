"""Describe whether frozen repeated-hit recoveries include a lifted corrective step."""
from pathlib import Path

import mujoco
import numpy as np

from .audit_stand_failures import digest, read, save
from .rig import Rig
from .run_perturb_preview import sealed

BASE=Path('logs/mimickit-training')
OUT=BASE/'perturb-response-foot-audit-20261008-01'


def run():
    parent=BASE/'perturb-preview-evaluation-20261008-01'
    sources={'parent':(parent,parent/'repeated/hit-repeat-clear-native')}
    for index in (1,2):
        root=BASE/f'perturb-response-evaluation-20261008-{index:02d}'
        for arm in ('control','long_return'):
            sources[f'{index}-{arm}']=(root,root/arm/'hit-repeat-clear-native')
    rig=Rig.load(BASE/'perturb-preview-learning-20261008-01/dummy.xml',policy_exclude=())
    data=mujoco.MjData(rig.model)
    foot_ids=[rig.model.body(f'Foot_{side}').id for side in ('L','R')]
    geometry=[]
    for side,foot_id in zip(('L','R'),foot_ids):
        bodies=[foot_id,rig.model.body(f'Toe_{side}').id]
        geoms=[g for g,b in enumerate(rig.model.geom_bodyid) if b in bodies]
        assert geoms and all(rig.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX for g in geoms)
        geometry.append(geoms)
    corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    inputs={str(rig.path):digest(rig.path),str(Path(__file__)):digest(Path(__file__))}
    results={}
    for name,(root,folder) in sources.items():
        sealed(root,[folder/'trace.npz',folder/'report.json'])
        for filename in ('trace.npz','report.json'):
            inputs[str(folder/filename)]=digest(folder/filename)
        with np.load(folder/'trace.npz') as z:
            q=z['qpos_after']; loads=z['foot_loads_n']; live=z['alive_before']; times=z['time_s']
        report=read(folder/'report.json')
        cases=[]
        for case,trial in enumerate(report['cases']):
            start=int(np.searchsorted(times,trial['first_contact_seconds']))
            end=int(live[:,case].sum())
            positions=[];clearances=[]
            for frame in range(start,end):
                data.qpos[:]=q[frame,case]
                mujoco.mj_forward(rig.model,data)
                positions.append(data.xpos[foot_ids].copy())
                clearances.append([min(float((data.geom_xpos[g]+(corners*rig.model.geom_size[g])@data.geom_xmat[g].reshape(3,3).T)[:,2].min()) for g in gs) for gs in geometry])
            positions,clearances=np.asarray(positions),np.asarray(clearances)
            displacement=np.linalg.norm(positions[:,:,:2]-positions[0,:,:2],axis=-1)
            support=loads[start:end,case]
            signatures=[]
            for foot in range(2):
                lifted=np.flatnonzero((support[:,foot]<5)&(clearances[:,foot]>.015))
                landings=[]
                if len(lifted):
                    landings=np.flatnonzero((np.arange(len(support))>lifted[0])&(support[:,foot]>=5)&(displacement[:,foot]>=.03))
                signatures.append(bool(trial['recovery_pass'] and len(lifted) and len(landings)))
            cases.append(dict(case=case,recovery_pass=trial['recovery_pass'],
                maximum_foot_displacement_m=displacement.max(0).tolist(),
                maximum_foot_toe_clearance_m=clearances.max(0).tolist(),minimum_foot_load_n=support.min(0).tolist(),
                recovered_step_signature=signatures))
        passing=[c for c in cases if c['recovery_pass']]
        results[name]=dict(cases=cases,recoveries=len(passing),
            recovered_step_cases=sum(any(c['recovered_step_signature']) for c in cases),
            maximum_displacement_in_recoveries_m=max(max(c['maximum_foot_displacement_m']) for c in passing) if passing else None)
        print(name,{k:v for k,v in results[name].items() if k!='cases'},flush=True)
    OUT.mkdir(exist_ok=False)
    save(OUT/'report.json',dict(inputs=inputs,results=results,new_physical_rollouts=0,
        signature='Post-contact load <5 N and all foot/toe box corners >15 mm, followed by loaded placement displaced >=30 mm and successful trial recovery.',
        limitation='Descriptive signature, not a naturalness score. Failed falls cannot count as recovered steps.'))


if __name__=='__main__':
    run()
