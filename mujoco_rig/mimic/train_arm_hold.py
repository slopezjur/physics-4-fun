"""Exact unblended target supervision for an isolated sustained support skill."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
import torch

from .runtime import activate
from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_support import Learner, PARENT, evaluate
from .run_learned_handoff import GuardedLearner, PRIOR

OUT=Path('logs/mimickit-training/arm-hold-learning-20261008-01')


@torch.no_grad()
def episode(env,anchor,net,poses,out,teacher=False,variant=False):
    from .interval_target_support import IntervalTargetSupport
    from .predictive_target_support import blend,blend_pose
    from .target_control33 import DelayedTargetControl33
    from .support_teacher_dataset import policy_input
    from .run_single_support_feedback import flags
    onset,ramp,hold=(1.2,1.7,3.5) if variant else (1.,1.5,3.)
    hold_end=onset+ramp+hold;seconds=hold_end+3.5
    env.reset();control=env.engine.policy_control;qi=control.q_indices.numpy()
    reference=torch.tensor(poses[:,0],dtype=torch.float32)
    live=np.zeros(16,dtype=bool);live[:8]=True
    previous=torch.zeros((16,33));rows=[];samples=[];start=time.perf_counter();last_print=start
    predictor=IntervalTargetSupport(env.task,objective='interval_com') if teacher else None
    initial_xy=env.measurements()['foot_xy_m'].numpy()[:8].copy()
    private=mujoco.MjData(env.rig.model);target_com=[]
    for i in range(8):
        private.qpos[:]=poses[i,1];mujoco.mj_forward(env.rig.model,private);target_com.append(private.subtree_com[1].copy())
    target_com=np.array(target_com)
    for frame in range(int(np.ceil(seconds/env.dt))):
        now=frame*env.dt;future=now+2*env.dt
        obs=env.task.observations()
        control.set_reference(*env.task.sample_reference(env.task.offset+env.task.steps*env.dt))
        source=DelayedTargetControl33.target_positions(control,anchor(obs))
        goals=[]
        for i in range(16):
            ref,clear=poses[i,:2]
            if now<onset:q=ref
            elif future<onset+ramp:q=blend_pose(ref,clear,future-onset,ramp)
            elif future<hold_end:q=clear
            else:q=blend_pose(clear,ref,future-hold_end,1.5)
            goals.append(q)
        goals=np.array(goals)
        authority=np.full(16,float(blend(0.,1.,now-onset,1.)),dtype=np.float32)
        x=torch.tensor(policy_input(obs.numpy(),goals,authority))
        quiet=(torch.tensor(goals,dtype=torch.float32)-reference).abs().amax(-1)<=1e-6
        if teacher:
            raw=source.clone()
            for i in range(8):
                if not live[i] or quiet[i] or authority[i]==0:continue
                q=env.engine.datas[i].qpos.copy();v=env.engine.datas[i].qvel.copy();pending=control.pending.clone()
                desired,_=predictor.target(i,goals[i],source[i].numpy())
                assert np.array_equal(q,env.engine.datas[i].qpos) and np.array_equal(v,env.engine.datas[i].qvel)
                assert torch.equal(pending,control.pending)
                raw[i]=torch.tensor(desired,dtype=torch.float32)
        else:
            raw=torch.tensor(goals[:,qi],dtype=torch.float32)+net(x)
        displacement=(torch.tensor(goals[:,qi],dtype=torch.float32)-reference[:,qi]).abs().amax(-1)
        desired=(raw-source).clamp(-1,1)*torch.tensor(authority)[:,None]*(displacement/.05).clamp(0,1)[:,None]
        before_addition=previous.clone()
        addition=previous+(desired-previous).clamp(-.02,.02)
        addition[quiet]=0;addition[~torch.tensor(live)]=0;previous.copy_(addition)
        pre=env.measurements();prestand,presupport=flags(env,pre)
        before=live.copy();env.engine.active=before.copy()
        _,_,done,_=env.step(addition)
        post=env.last;stand,supported=flags(env,post);q=env.engine.native_qpos().numpy()
        issued=control.pending[(control.cursor-1)%2,:,:33]
        assert (issued-(source+addition).clamp(control.lower,control.upper)).abs().max()<2e-6
        assert np.isfinite(q[before]).all() and not env.engine.solver_warning[before].any()
        assert (control.ctrl[:,env.rig.actuators].abs()<=torch.tensor(env.rig.action_scale)+1e-4).all()
        rows.append(dict(time=now+env.dt,alive=before[:8],qpos=q[:8].copy(),stand=(stand&prestand)[:8],
            support=(supported&presupport)[:8],loads=post['foot_loads_n'].numpy()[:8],com=post['com_m'].numpy()[:8],
            xy=post['foot_xy_m'].numpy()[:8],issued=issued.numpy()[:8].copy(),addition=addition.numpy()[:8].copy(),
            done=done.numpy()[:8].copy(),tracking_rmse=np.sqrt(np.mean((q[:8,qi]-poses[:8,1,qi])**2,axis=-1))))
        if teacher:
            samples.append(dict(x=x.numpy()[:8].copy(),raw_target=raw.numpy()[:8].copy(),source=source.numpy()[:8].copy(),
                previous=before_addition.numpy()[:8].copy(),goal=goals[:8,qi].astype(np.float32),quiet=quiet.numpy()[:8]))
        live &= done.numpy()!=1
        if not live.any():break
        if time.perf_counter()-last_print>20:
            print('ROLLOUT',out.name,round(now,2),'live',int(live.sum()),flush=True);last_print=time.perf_counter()
    z={k:np.stack([r[k] for r in rows]) for k in rows[0]}
    hold_mask=(z['time']>hold_end-2)&(z['time']<=hold_end);cases=[]
    for i in range(8):
        complete_hold=hold_mask.sum()>=int(2/env.dt)-1 and z['alive'][hold_mask,i].all()
        stable=bool(complete_hold and z['support'][hold_mask,i].all())
        tracking=bool(stable and (z['tracking_rmse'][hold_mask,i]<=.05).all() and
            (np.linalg.norm(z['com'][hold_mask,i,:2]-target_com[i,:2],axis=-1)<=.02).all() and
            (np.linalg.norm(z['xy'][hold_mask,i]-initial_xy[i],axis=-1)<=.02).all())
        returned=bool(live[i] and z['stand'][-int(np.ceil(2/env.dt)):,i].all())
        cases.append(dict(pose=i,heldout=i>=6,stable_hold=stable,tracked_hold=tracking,returned_stand=returned,
            complete=tracking and returned))
    out.mkdir();np.savez_compressed(out/'trace.npz',**z)
    if teacher:
        collected={k:np.stack([r[k] for r in samples]) for k in samples[0]}
        accepted=np.array([c['complete'] for c in cases])
        use=z['alive']&accepted[None]
        frames,worlds=np.nonzero(use)
        dataset={k:v[use] for k,v in collected.items()}
        dataset.update(issued=z['issued'][use],world=worlds,frame=frames,validation=worlds>=6)
        np.savez_compressed(out/'samples.npz',**dataset)
    result=dict(teacher=teacher,variant=variant,onset=onset,ramp=ramp,hold=hold,hold_end=hold_end,seconds=seconds,
        cases=cases,complete=sum(c['complete'] for c in cases),stable_holds=sum(c['stable_hold'] for c in cases),
        returned=sum(c['returned_stand'] for c in cases),wall_seconds=time.perf_counter()-start)
    save(out/'report.json',result);print('RESULT',out.name,{k:v for k,v in result.items() if k!='cases'},flush=True)
    return result


def train(initial,data,folder,seed=451028):
    torch.manual_seed(seed);rng=torch.Generator(device='cuda').manual_seed(seed)
    net=deepcopy(initial).cuda().train();optimizer=torch.optim.Adam(net.parameters(),lr=.0001)
    tr=~data['validation'];x=torch.tensor(data['x'][tr],device='cuda');y=torch.tensor((data['raw_target']-data['goal'])[tr],device='cuda')
    assert len(x)>0 and data['validation'].any()
    folder.mkdir();torch.save(net.cpu().state_dict(),folder/'initial.pt');net.cuda()
    start=time.perf_counter()
    for _ in range(4000):
        ids=torch.randint(len(x),(512,),device='cuda',generator=rng)
        loss=((net(x[ids])-y[ids])/net.output_scale).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    net.eval()
    with torch.no_grad():
        vx=torch.tensor(data['x'][~tr],device='cuda');vy=torch.tensor((data['raw_target']-data['goal'])[~tr],device='cuda')
        result=dict(seed=seed,updates=4000,train_examples=len(x),heldout_examples=len(vx),wall_seconds=time.perf_counter()-start,
            train_rmse_rad=float((net(x)-y).square().mean().sqrt()),heldout_rmse_rad=float((net(vx)-vy).square().mean().sqrt()))
    torch.save(net.cpu().state_dict(),folder/'final.pt');save(folder/'report.json',result)
    print('TRAINED',folder.name,result,flush=True);return net,result


def main():
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from .run_single_support_feedback import make_environment
    torch.set_num_threads(2);OUT.mkdir(exist_ok=False);(OUT/'sources').mkdir()
    paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('ARM_HOLD_LEARNING_PLAN.md')]
    inputs={str(p):digest(p) for p in paths+[PARENT,PARENT.with_suffix('.amp.pt'),PRIOR/'goal-421028/final.pt',PRIOR/'goals.npz',
        PRIOR/'dataset.npz',FOUNDATION/'dummy.xml',FOUNDATION/'stationary_reference.npz']}
    for p in paths:shutil.copy2(p,OUT/'sources'/p.name)
    save(OUT/'protocol.json',dict(inputs=inputs,training_updates_per_arm=4000,seed=451028,
        label='Exact issue-time observation and raw target before authority, displacement fade and slew',
        whole_pose_holdouts=[6,7],arena_promotion=False))
    rig=Rig.load(FOUNDATION/'dummy.xml',policy_exclude=())
    anchor=load_actor(rig,FOUNDATION/'stationary_reference.npz',json.loads((FOUNDATION/'protocol.json').read_text()),PARENT)
    env=make_environment(rig,anchor,40.)
    poses=np.load(PRIOR/'goals.npz')['poses'];state=torch.load(PRIOR/'goal-421028/final.pt',weights_only=True)
    initial=Learner(state['mean'],state['scale'],state['output_scale']);initial.load_state_dict(state);initial.eval()
    results={};datasets=[]
    for variant,label in ((False,'teacher-nominal'),(True,'teacher-varied')):
        results[label]=episode(env,anchor,initial,poses,OUT/label,teacher=True,variant=variant)
        datasets.append(dict(np.load(OUT/label/'samples.npz')));save(OUT/'progress.json',results)
    data={k:np.concatenate([d[k] for d in datasets]) for k in datasets[0]}
    np.savez_compressed(OUT/'training-samples.npz',**data)
    results['unchanged']=episode(env,anchor,initial,poses,OUT/'unchanged')
    old=dict(np.load(PRIOR/'dataset.npz'));old['raw_target']=old['issued']
    # Replay control deliberately retains its historical blended-label semantics.
    for key,dataset in (('replay',old),('hold',data)):
        net,training=train(initial,dataset,OUT/key)
        results[key]=dict(training=training,evaluation=episode(env,anchor,net,poses,OUT/f'{key}-nominal'))
        if key=='hold':
            bank={i:dict(poses=poses[i]) for i in range(16)}
            results['quiet']=evaluate(env,anchor,bank,GuardedLearner(net,anchor,env),'goal',1.,True,OUT/'quiet30')
            assert results['quiet']['whole_interval_stand']==16
            if results[key]['evaluation']['complete']==8:
                results['varied']=episode(env,anchor,net,poses,OUT/'hold-varied',variant=True)
                replica,tr=train(initial,data,OUT/'replica',seed=461028)
                results['replica']=dict(training=tr,evaluation=episode(env,anchor,replica,poses,OUT/'replica-nominal'))
        save(OUT/'progress.json',results)
    assert all(digest(Path(p))==h for p,h in inputs.items())
    save(OUT/'report.json',dict(results=results,inputs_unchanged=True,arena_changed=False,
        output_hashes={str(p):digest(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
