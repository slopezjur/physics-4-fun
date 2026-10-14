"""Phase and joint tracking comparisons for the learned arms-clear prerequisite."""
import json

import numpy as np

from .probe_arm_execution_distribution import ROOT
from .run_support_skill_study import FOUNDATION,digest,save
from .run_learned_handoff import PRIOR


def main():
    from .rig import Rig
    rig=Rig.load(FOUNDATION/'dummy.xml',policy_exclude=())
    model=rig.model;joints=model.actuator_trnid[rig.actuators,0];qi=model.jnt_qposadr[joints]
    poses=np.load(PRIOR/'goals.npz')['poses'];out=ROOT/'step-27';out.mkdir(exist_ok=False)
    paths=[]
    first=json.loads((ROOT/'step-01/report.json').read_text())
    for r in first['results']:
        if r['config']['mode']=='mean':paths.append((r['config']['actor'],ROOT/'step-01'/r['folder'],r))
    for step in (24,25):
        r=json.loads((ROOT/f'step-{step:02d}/report.json').read_text())['results'][0]
        paths.append((f'warm-{step}',ROOT/f'step-{step:02d}/nominal',r))
    results=[];inputs={}
    for label,folder,r in paths:
        path=folder/'trace.npz';inputs[str(path)]=digest(path);z=dict(np.load(path))
        hold=(z['time']>r['hold_end']-2)&(z['time']<=r['hold_end'])
        mask=z['alive']&hold[:,None]
        errors=z['qpos'][:,:,qi]-poses[:8,1,qi]
        command=z['issued']-poses[:8,1,qi]
        rms=np.sqrt(np.mean(errors[mask]**2,axis=0));issued=np.sqrt(np.mean(command[mask]**2,axis=0))
        worst=np.argsort(rms)[-5:][::-1]
        phases=[]
        for phase,start,end in (('ramp',1.,2.5),('hold',2.5,5.5),('return',5.5,7.),('settle',7.,9.1)):
            selected=(z['time']>start)&(z['time']<=end);live=z['alive'][selected]
            phases.append(dict(phase=phase,alive_frames=int(live.sum()),
                unsupported_alive_frames=int((live&~z['support'][selected]).sum()),
                failures=int((live&z['done'][selected]).sum()),
                mean_reward=float(z['reward'][selected][live].mean())))
        results.append(dict(actor=label,stable_holds=r['stable_holds'],returned=r['returned'],complete=r['complete'],
            cases=r['cases'],phases=phases,worst_hold_joints=[dict(joint=model.joint(int(joints[j])).name,
                actual_goal_rms_rad=float(rms[j]),issued_goal_rms_rad=float(issued[j])) for j in worst]))
    save(out/'report.json',dict(step=27,inputs=inputs,results=results,
        interpretation='Arms-clear prerequisite with fixed-foot admission. Foot drift is an error here, not evidence of learned stepping.'))
    print(json.dumps([{k:v for k,v in r.items() if k not in ('cases','phases')} for r in results],indent=2))


if __name__=='__main__':main()
