"""Separate observation/command reconstruction error from closed-loop imitation failure."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from .runtime import activate
from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_support import PARENT, TEACHER, Learner


@torch.no_grad()
def main():
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from .run_single_support_feedback import make_environment
    from .target_control33 import DelayedTargetControl33
    torch.set_num_threads(2)
    root=Path('logs/mimickit-training/learned-support-20261008-01')
    out=Path('logs/mimickit-training/learned-support-audit-20261008-03')
    out.mkdir(exist_ok=False)
    rig=Rig.load(FOUNDATION/'dummy.xml',policy_exclude=())
    anchor=load_actor(rig,FOUNDATION/'stationary_reference.npz',json.loads((FOUNDATION/'protocol.json').read_text()),PARENT)
    env=make_environment(rig,anchor,40.)
    z=dict(np.load(TEACHER/'entry-feedback/trace.npz'))
    ds=dict(np.load(root/'dataset.npz'))
    control=env.engine.policy_control
    # Replay stored targets through the real delayed actuator path. This is a transport
    # diagnostic only, with no learned skill or robustness claim.
    errors=[];obs_errors=[];target_errors=[];start=time.perf_counter();envelope_failure=None;first_error=None
    for frame in range(len(z['time'])):
        obs=env.task.observations()
        rows=np.flatnonzero(ds['frame']==frame)
        obs_errors.append(float(np.max(abs(obs.numpy()[ds['world'][rows]]-ds['x'][rows,:374]))) if len(rows) else 0.)
        if first_error is None and obs_errors[-1]>1e-5:
            delta=abs(obs.numpy()[ds['world'][rows]]-ds['x'][rows,:374])
            r,c=np.unravel_index(delta.argmax(),delta.shape)
            first_error=dict(frame=frame,world=int(ds['world'][rows[r]]),column=int(c),actual=float(obs[ds['world'][rows[r]],c]),reconstructed=float(ds['x'][rows[r],c]))
        source=anchor(obs)
        control.set_reference(*env.task.sample_reference(env.task.offset+env.task.steps*env.dt))
        base=DelayedTargetControl33.target_positions(control,source)
        offset=torch.tensor(z['issued'][frame])-base
        if offset[z['alive'][frame]].abs().max()>1.:
            envelope_failure=dict(frame=frame,offset=float(offset[z['alive'][frame]].abs().max()))
            break
        env.engine.active=z['alive'][frame].copy()
        env.step(offset.clamp(-1,1))
        issued=control.pending[(control.cursor-1)%2,:,:33].numpy()
        target_errors.append(float(abs(issued-z['issued'][frame])[z['alive'][frame]].max()))
        errors.append(float(abs(env.engine.native_qpos().numpy()-z['qpos'][frame])[z['alive'][frame]].max()))
    result=dict(scope='Recorded-target playback diagnostic, not a learned policy evaluation',frames=len(errors),
        maximum_qpos_error=max(errors),maximum_observation_error=max(obs_errors),maximum_issued_error=max(target_errors),
        first_observation_error_above_1e5=first_error,envelope_failure=envelope_failure,
        first_qpos_error_above_1e4=next((i for i,e in enumerate(errors) if e>1e-4),None),wall_seconds=time.perf_counter()-start)
    # Compare one-step errors on teacher states with actual loss of balance timing.
    summaries=[]
    for mode,seed in (('source',421028),('goal',421028),('source',431028),('goal',431028)):
        key=f'{mode}-{seed}'
        state=torch.load(root/key/'final.pt',weights_only=True)
        net=Learner(state['mean'],state['scale'],state['output_scale']);net.load_state_dict(state);net.eval()
        pred=net(torch.tensor(ds['x'])).numpy()+ds[mode]
        err=pred-ds['issued']
        early=ds['frame']<120
        rows=[]
        for condition in ('onset1','onset3','onset9','quiet30'):
            trace=np.load(root/f'{key}-{condition}'/'trace.npz')
            times=[float(trace['time'][np.flatnonzero(trace['alive'][:,i])[-1]]) for i in range(16)]
            rows.append(dict(condition=condition,minimum_end_s=min(times),maximum_end_s=max(times)))
        summaries.append(dict(learner=key,early_teacher_state_rmse_rad=float(np.sqrt(np.mean(err[early]**2))),end_times=rows))
    result['learners']=summaries
    result['passed_transport']=envelope_failure is None and max(errors)<1e-4 and max(obs_errors)<2e-4 and max(target_errors)<2e-6
    result['source_hashes']={str(p):digest(p) for p in [Path(__file__),root/'dataset.npz',root/'final-verification.json',PARENT]}
    np.savez_compressed(out/'errors.npz',qpos=errors,observation=obs_errors,issued=target_errors)
    save(out/'report.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
