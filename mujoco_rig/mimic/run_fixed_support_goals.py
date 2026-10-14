"""Isolate intermediate-goal stability from learned feedback and handoff rules."""
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
import torch

from .runtime import activate
from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_support import Learner, PARENT
from .run_learned_handoff import PRIOR

OUT=Path('logs/mimickit-training/fixed-support-goals-20261008-01')


def goal_at(bank, kind, time):
    from .predictive_target_support import blend_pose
    reference,clear,transfer=bank
    if time<1.:return reference
    if time<2.5:return blend_pose(reference,clear,time-1.,1.5)
    if kind=='arms':
        if time<5.5:return clear
        return blend_pose(clear,reference,time-5.5,1.5)
    if time<4.5:return blend_pose(clear,transfer,time-2.5,2.)
    if time<7.5:return transfer
    return blend_pose(transfer,reference,time-7.5,2.)


@torch.no_grad()
def run(env,anchor,net,bank,kind,controller,out):
    from .predictive_target_support import blend
    from .interval_target_support import IntervalTargetSupport
    from .run_single_support_feedback import flags
    from .target_control33 import DelayedTargetControl33
    from .support_teacher_dataset import policy_input
    env.reset()
    worlds=[0,6] if kind=='arms' else [0,6,8,14]
    live=np.zeros(16,dtype=bool);live[worlds]=True
    control=env.engine.policy_control
    predictor=IntervalTargetSupport(env.task,objective='interval_com') if controller.startswith('teacher') else None
    previous=torch.zeros((16,33));rows=[];query_count=0
    initial_xy=env.measurements()['foot_xy_m'].numpy().copy()
    hold_start,hold_end,seconds=(2.5,5.5,9.) if kind=='arms' else (4.5,7.5,11.5)
    reference=torch.tensor(np.array([bank[i][0] for i in range(16)]),dtype=torch.float32)
    target=np.array([bank[i][1 if kind=='arms' else 2] for i in range(16)])
    start=time.perf_counter();last_update=start
    target_com=[]
    private=mujoco.MjData(env.rig.model)
    for q in target:
        private.qpos[:]=q;mujoco.mj_forward(env.rig.model,private);target_com.append(private.subtree_com[1].copy())
    target_com=np.array(target_com)
    for frame in range(int(np.ceil(seconds/env.dt))):
        now=frame*env.dt;obs=env.task.observations()
        control.set_reference(*env.task.sample_reference(env.task.offset+env.task.steps*env.dt))
        source=DelayedTargetControl33.target_positions(control,anchor(obs))
        goals=np.array([goal_at(bank[i],kind,now+2*env.dt) if now>=1 else bank[i][0] for i in range(16)])
        authority=np.full(16,float(blend(0.,1.,now-1.,1.)),dtype=np.float32)
        addition=torch.zeros((16,33))
        quiet=(torch.tensor(goals,dtype=torch.float32)-reference).abs().amax(-1)<=1e-6
        if predictor:
            for i in worlds:
                if not live[i] or authority[i]==0 or quiet[i]:continue
                q=env.engine.datas[i].qpos.copy();v=env.engine.datas[i].qvel.copy();pending=control.pending.clone()
                desired,_=predictor.target(i,goals[i],source[i].numpy())
                assert np.array_equal(q,env.engine.datas[i].qpos) and np.array_equal(v,env.engine.datas[i].qvel)
                assert torch.equal(pending,control.pending)
                addition[i]=torch.tensor(desired,dtype=torch.float32)-source[i];query_count+=1
        else:
            x=torch.tensor(policy_input(obs.numpy(),goals,authority))
            addition=torch.tensor(goals[:,control.q_indices.numpy()],dtype=torch.float32)+net(x)-source
        addition=addition.clamp(-1,1)*torch.tensor(authority)[:,None]
        if controller!='teacher_original':
            displacement=(torch.tensor(goals,dtype=torch.float32)[:,control.q_indices]-reference[:,control.q_indices]).abs().amax(-1)
            addition*= (displacement/.05).clamp(0,1)[:,None]
            addition=previous+(addition-previous).clamp(-.02,.02)
        addition[quiet]=0;addition[~torch.tensor(live)]=0
        slew=(addition-previous).abs().amax(-1);previous.copy_(addition)
        pre=env.measurements();prestand,presupport=flags(env,pre)
        before=live.copy();env.engine.active=before.copy()
        _,_,done,_=env.step(addition)
        post=env.last;stand,supported=flags(env,post)
        q=env.engine.native_qpos().numpy()
        assert np.isfinite(q[before]).all() and not env.engine.solver_warning[before].any()
        issued=control.pending[(control.cursor-1)%2,:,:33]
        assert (issued-(source+addition).clamp(control.lower,control.upper)).abs().max()<2e-6
        assert (control.ctrl[:,env.rig.actuators].abs()<=torch.tensor(env.rig.action_scale)+1e-4).all()
        error=np.sqrt(np.mean((q[:,control.q_indices.numpy()]-target[:,control.q_indices.numpy()])**2,axis=-1))
        rows.append(dict(time=now+env.dt,alive=before[worlds],qpos=q[worlds].copy(),
            qvel=np.stack([env.engine.datas[i].qvel.copy() for i in worlds]),
            stand=(stand&prestand)[worlds],support=(supported&presupport)[worlds],
            tracking_rmse=error[worlds],loads=post['foot_loads_n'].numpy()[worlds],
            com=post['com_m'].numpy()[worlds],xy=post['foot_xy_m'].numpy()[worlds],
            addition=addition.numpy()[worlds].copy(),slew=slew.numpy()[worlds].copy(),
            issued=issued.numpy()[worlds].copy(),done=done.numpy()[worlds].copy()))
        live &= done.numpy()!=1
        if not live.any():break
        if time.perf_counter()-last_update>20:
            print('PHYSICAL',out.name,round(now,2),'live',int(live.sum()),'queries',query_count,flush=True)
            last_update=time.perf_counter()
    z={k:np.stack([r[k] for r in rows]) for k in rows[0]};out.mkdir();np.savez_compressed(out/'trace.npz',**z)
    cases=[];hold=(z['time']>hold_end-2)&(z['time']<=hold_end)
    for col,world in enumerate(worlds):
        valid=z['alive'][:,col];last=np.flatnonzero(valid)[-1];survived=bool(live[world])
        complete_hold=int(hold.sum())>=int(2/env.dt)-1 and z['alive'][hold,col].all()
        stable=bool(complete_hold and z['support'][hold,col].all())
        track=bool(complete_hold and (z['tracking_rmse'][hold,col]<=.05).all())
        com_error=np.linalg.norm(z['com'][:,col,:2]-target_com[world,:2],axis=-1)
        placed=bool(complete_hold and (com_error[hold]<=.02).all() and
            (np.linalg.norm(z['xy'][hold,col]-initial_xy[world],axis=-1)<=.02).all())
        final=bool(survived and z['stand'][-int(np.ceil(2/env.dt)):,col].all())
        bad=np.flatnonzero(valid&~z['support'][:,col])
        cases.append(dict(world=world,heldout=world%8==6,survived=survived,
            stable_hold=stable,tracked_hold=bool(stable and track and placed),returned_stand=final,
            complete=bool(stable and track and placed and final),
            first_support_failure_s=float(z['time'][bad[0]]) if len(bad) else None,
            end_s=float(z['time'][last]),hold_mean_tracking_rmse_rad=float(z['tracking_rmse'][hold,col].mean()) if hold.any() and complete_hold else None))
    report=dict(kind=kind,controller=controller,worlds=worlds,cases=cases,
        stable_holds=sum(c['stable_hold'] for c in cases),tracked_holds=sum(c['tracked_hold'] for c in cases),
        returned_stand=sum(c['returned_stand'] for c in cases),complete=sum(c['complete'] for c in cases),
        teacher_queries=query_count,wall_seconds=time.perf_counter()-start,
        label='Scripted physical-controller diagnostic; never a learned result' if predictor else 'Frozen neural feedback with handoff guard',
        maximum_live_slew_rad=float(z['slew'][z['alive']].max()))
    save(out/'report.json',report);print('RESULT',out.name,{k:v for k,v in report.items() if k!='cases'},flush=True)
    return report


def main():
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from .run_single_support_feedback import make_environment
    from .support_skill_core import fit_goal
    torch.set_num_threads(2);OUT.mkdir(exist_ok=False);(OUT/'sources').mkdir()
    paths=list(Path(__file__).parent.glob('*.py'))
    inputs={str(p):digest(p) for p in paths+[PARENT,PARENT.with_suffix('.amp.pt'),PRIOR/'goal-421028/final.pt',
        PRIOR/'goals.npz',FOUNDATION/'dummy.xml',FOUNDATION/'stationary_reference.npz']}
    for p in paths:shutil.copy2(p,OUT/'sources'/p.name)
    save(OUT/'protocol.json',dict(inputs=inputs,training_updates=0,
        panel='Pose 0 (training), pose 6 (held out); both shift signs for transfer',
        arms='Original to arms-clear over 1.5 s, hold 3 s, return 1.5 s, settle 2 s',
        transfer='After arm clearing, 4 cm COM shift / 4 degree lean over 2 s, hold 3 s, return 2 s, settle 2 s',
        hold_gate='Final 2 s of hold continuously supported, motor pose RMSE <=0.05 rad, COM horizontal error <=0.02 m, foot drift <=0.02 m',
        return_gate='Final 2 s original Stand',controllers=['learner_guarded','teacher_guarded'],
        conditional='If guarded teacher fails any complete sequence, compare original teacher handoff; no promotion or training'))
    rig=Rig.load(FOUNDATION/'dummy.xml',policy_exclude=())
    anchor=load_actor(rig,FOUNDATION/'stationary_reference.npz',json.loads((FOUNDATION/'protocol.json').read_text()),PARENT)
    env=make_environment(rig,anchor,20.)
    poses=np.load(PRIOR/'goals.npz')['poses'];bank={};fits={}
    for i in range(16):
        reference,clear=poses[i,:2]
        transfer,check=fit_goal(rig.model,clear,shift=.04*(1 if i//8==0 else -1),lift=0.,distance=0.,swing=i//8,lean_radians=np.deg2rad(4.))
        assert check['foot_error_m']<=.001 and check['com_error_m']<=.001 and check['penetration_m']<=.0002
        bank[i]=(reference,clear,transfer);fits[str(i)]=check
    save(OUT/'goal-fit.json',fits);np.savez_compressed(OUT/'goals.npz',poses=np.array(list(bank.values())))
    state=torch.load(PRIOR/'goal-421028/final.pt',weights_only=True)
    net=Learner(state['mean'],state['scale'],state['output_scale']);net.load_state_dict(state);net.eval()
    results={}
    for kind in ('arms','transfer'):
        for controller in ('learner_guarded','teacher_guarded'):
            key=f'{kind}-{controller}'
            results[key]=run(env,anchor,net,bank,kind,controller,OUT/key)
            save(OUT/'progress.json',results)
        teacher=results[f'{kind}-teacher_guarded']
        if teacher['complete']<len(teacher['worlds']):
            key=f'{kind}-teacher_original';results[key]=run(env,anchor,net,bank,kind,'teacher_original',OUT/key)
            save(OUT/'progress.json',results)
    assert all(digest(Path(p))==h for p,h in inputs.items())
    save(OUT/'report.json',dict(results=results,inputs_unchanged=True,arena_changed=False,
        output_hashes={str(p):digest(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
