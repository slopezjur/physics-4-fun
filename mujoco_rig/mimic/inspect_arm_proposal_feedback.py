"""Issue-state command decomposition, local sensitivity and matched teacher queries."""
import json
from pathlib import Path

import numpy as np
import torch

from .probe_arm_proposal_foundations import ROOT,TEACHER,PLAN
from .probe_arm_execution_distribution import setup,seal
from .train_arm_correction_ppo import CorrectionEnvironment
from .target_control33 import DelayedTargetControl33
from .run_support_skill_study import digest,save
from .probe_arm_action_response import snapshot,fingerprint


@torch.no_grad()
def main():
    out=ROOT/'step-11';inputs=seal(out,[PLAN,TEACHER/'training-samples.npz'])
    query_frames=[90,150,210,300,360]
    save(out/'protocol.json',dict(step=11,inputs=inputs,query_frames=query_frames,queries_per_frame=8,
        role='Read-only physical teacher queries on inherited-policy states; not recovery demonstrations'))
    env,anchor,proposal,poses,actors=setup();adapter=CorrectionEnvironment(env,anchor,proposal,poses)
    from .interval_target_support import IntervalTargetSupport
    teacher=IntervalTargetSupport(env.task,objective='interval_com')
    p=adapter.physical;c=p.control;qi=c.q_indices;names=[env.rig.model.joint(int(j)).name for j in env.rig.model.actuator_trnid[env.rig.actuators,0]]
    left=names.index('UpperArm_L_ry');active=torch.arange(16)<8;adapter.reset();rows=[];queries=[];sensitivity=[]
    for frame in range(p.frames):
        x,goals,amount,quiet,_=p.observation();x=x.clone()
        c.set_reference(*env.task.sample_reference(env.task.offset+env.task.steps*env.dt))
        source=DelayedTargetControl33.target_positions(c,anchor(x[:,:374]));residual=proposal(x);raw=goals[:,qi]+residual
        fade=((goals[:,qi]-p.reference[:,qi]).abs().amax(-1)/.05).clamp(0,1)
        desired=(raw-source).clamp(-1,1)*amount[:,None]*fade[:,None]
        previous=p.previous.clone()
        if frame in query_frames:
            stamp=fingerprint(snapshot(p,active))
            for i in range(8):
                if not active[i]:continue
                target,_=teacher.target(i,goals[i].double().numpy(),source[i].numpy())
                queries.append(dict(frame=frame,pose=i,x=x[i].numpy(),goal=goals[i,qi].numpy(),
                    source=source[i].numpy(),raw=raw[i].numpy(),teacher=target,
                    teacher_residual=target-goals[i,qi].numpy()))
            assert fingerprint(snapshot(p,active))==stamp
            with torch.enable_grad():
                probe=x.clone().requires_grad_(True)
                value=proposal(probe)[:,left]
                grad=torch.autograd.grad(value.sum(),probe)[0].numpy()
            for i in range(8):
                sensitivity.append(dict(frame=frame,pose=i,heldout=i>=6,
                    d_left_target_d_left_angle=float(grad[i,int(qi[left])+8]),
                    d_left_target_d_left_goal=float(1+grad[i,374+int(qi[left])]),
                    state_position_gradient_norm=float(np.linalg.norm(grad[i,15:54])),
                    state_velocity_gradient_norm=float(np.linalg.norm(grad[i,54:93])),
                    delayed_queue_gradient_norm=float(np.linalg.norm(grad[i,95:229]))))
        before=active.clone();_,terminal,info=adapter.step(torch.zeros((16,33)),active)
        values=dict(x=x,alive=before,quiet=quiet,goal=goals[:,qi],source=source,residual=residual,raw=raw,
            desired=desired,previous=previous,addition=p.previous.clone(),issued=info['issued'],qpos=info['qpos'])
        rows.append({k:v[:8].numpy().copy() for k,v in values.items()});active&=~terminal
    z={k:np.stack([r[k] for r in rows]) for k in rows[0]}
    original=np.load(ROOT/'step-02/inherited-nominal/trace.npz')
    assert all(np.array_equal(z[k],original[k]) for k in ('qpos','issued','alive'))
    np.savez_compressed(out/'trace.npz',**z)
    q={k:np.stack([v[k] for v in queries]) for k in queries[0]};np.savez_compressed(out/'queries.npz',**q)
    query_summary=[]
    for frame in query_frames:
        use=q['frame']==frame
        query_summary.append(dict(frame=frame,cases=int(use.sum()),
            raw_teacher_rmse=float(np.sqrt(np.mean((q['raw'][use]-q['teacher'][use])**2))),
            left_raw_teacher_rmse=float(np.sqrt(np.mean((q['raw'][use,left]-q['teacher'][use,left])**2))),
            left_proposal_mean=float((q['raw'][use,left]-q['goal'][use,left]).mean()),
            left_teacher_mean=float(q['teacher_residual'][use,left].mean())))
    assert all(digest(Path(path))==h for path,h in inputs.items())
    save(out/'report.json',dict(step=11,results=query_summary,queries=len(queries),live_state_unchanged_by_teacher=True,
        inputs_unchanged=True,outputs={str(path):digest(path) for path in out.rglob('*') if path.is_file() and 'sources' not in path.parts}))
    # A different research question on the same exact issue-state trace.
    decomposition=[]
    for start,end,name in ((60,150,'ramp'),(150,330,'hold'),(330,420,'return')):
        use=z['alive'][start:end]&~z['quiet'][start:end]
        row=dict(phase=name,samples=int(use.sum()))
        for field in ('raw','issued'):
            error=(z[field]-z['goal'])[start:end][use]
            row[field+'_goal_rms']=float(np.sqrt(np.mean(error**2)))
            row[field+'_left_goal_rms']=float(np.sqrt(np.mean(error[:,left]**2)))
        row['source_envelope_clipped_fraction']=float((abs(z['raw'][start:end]-z['source'][start:end])[use]>1).mean())
        row['slew_clipped_fraction']=float((abs(z['desired'][start:end]-z['previous'][start:end])[use]>.02).mean())
        row['final_joint_clip_fraction']=float((abs(z['issued'][start:end]-z['source'][start:end]-z['addition'][start:end])[use]>1e-6).mean())
        decomposition.append(row)
    output=ROOT/'step-12';output.mkdir()
    save(output/'report.json',dict(step=12,trace_hash=digest(out/'trace.npz'),results=decomposition))
    output=ROOT/'step-13';output.mkdir()
    teacher_data=np.load(TEACHER/'training-samples.npz');teacher_norm=(teacher_data['x']-proposal.mean.numpy())/proposal.scale.numpy()
    own_norm=(z['x']-proposal.mean.numpy())/proposal.scale.numpy()
    # Exclude intentionally masked command-history coordinates from this comparison.
    teacher_norm[:,95:229]=0;own_norm[:,:,95:229]=0
    save(output/'report.json',dict(step=13,trace_hash=digest(out/'trace.npz'),results=sensitivity,
        teacher_normalized_clip_fraction=float((abs(teacher_norm)>=20).mean()),
        own_normalized_clip_fraction=float((abs(own_norm[z['alive']&~z['quiet']])>=20).mean()),
        interpretation='Local raw-target derivatives, before action projection. Values alone are not a closed-loop stability proof.'))
    print('QUERY',query_summary,flush=True);print('DECOMPOSITION',decomposition,flush=True)


if __name__=='__main__':main()
