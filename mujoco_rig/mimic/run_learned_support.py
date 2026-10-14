"""Bounded neural distillation and closed-loop support prerequisite experiments."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
import torch
from torch import nn

from .runtime import activate
from .run_support_skill_study import FOUNDATION, digest, longest, save

TEACHER = Path('logs/mimickit-training/single-support-all-poses-20261007-01')
PARENT = Path('logs/mimickit-training/perturb-preview-learning-20261008-01/single_ball/final/model.pt')
ARMS = [(mode, seed) for seed in (421028, 431028) for mode in ('source', 'goal')]


def name(mode, seed):
    return f'{mode}-{seed}'


class Learner(nn.Module):
    def __init__(self, mean, scale, output_scale):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        self.register_buffer('output_scale', output_scale)
        self.net = nn.Sequential(nn.Linear(421, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 33))

    def forward(self, x):
        return self.net(((x-self.mean)/self.scale).clamp(-20, 20))*self.output_scale


def reconstruct(env, anchor, out):
    """Issue-time reconstruction: only the preceding saved physical state is used."""
    from .torque_stand_task import quat_to_tan_norm
    from .capture_reward import rotate_xyzw
    from .predictive_target_support import blend
    z = dict(np.load(TEACHER/'entry-feedback/trace.npz', allow_pickle=False))
    report = json.loads((TEACHER/'entry-feedback/report.json').read_text())
    worlds = np.array(report['worlds'])
    accepted = np.array([c['entry_hold_return'] for c in report['cases']])
    assert accepted.sum() == 13 and np.array_equal(worlds[~accepted], [10, 11, 12])
    initial = env.task.observations().numpy()
    n = len(z['time'])
    obs = np.tile(initial, (n, 1, 1))
    q = torch.tensor(z['qpos'][:-1], dtype=torch.float32)
    v = torch.tensor(z['qvel'][:-1], dtype=torch.float32)
    rot = q[..., [4, 5, 6, 3]]
    obs[1:, :, :3] = q[..., :3]
    obs[1:, :, 3:9] = quat_to_tan_norm(rot).numpy()
    obs[1:, :, 9:12] = v[..., :3]
    obs[1:, :, 12:15] = rotate_xyzw(rot, v[..., 3:6]).numpy()
    obs[1:, :, 15:54] = q[..., 7:]
    obs[1:, :, 54:93] = v[..., 6:]*.1
    obs[1:, :, 93:95] = z['loads'][:-1]*.001
    obs[:, :, 95:229] = 0
    for lag, start in ((2, 95), (1, 162)):
        obs[lag:, :, start:start+33] = z['issued'][:-lag]
        obs[lag:, :, start+66] = 1
    amounts = np.array([float(blend(0., 1., t-env.dt-1., 1.)) for t in z['time']], np.float32)
    x = np.concatenate((obs, z['requested_qpos'].astype(np.float32), np.broadcast_to(amounts[:, None, None], (n, 16, 1))), -1)
    control = env.engine.policy_control
    qi = control.q_indices.numpy()
    reference = env.task.reference.qpos[torch.arange(16)%8].numpy()[:, qi]
    with torch.no_grad():
        actor = torch.cat([anchor(chunk) for chunk in torch.tensor(obs.reshape(-1, 374)).split(1024)]).numpy().reshape(n,16,33)
    source = np.clip(reference[None]+.25*actor, control.lower.numpy(), control.upper.numpy())
    offset = z['issued']-source
    use = z['alive'] & accepted[None]
    frames, cols = np.nonzero(use)
    error = np.max(abs(np.clip(source+offset, control.lower.numpy(), control.upper.numpy())-z['issued'])[use])
    assert error < 2e-6 and abs(offset[use]).max() <= 1.
    # Independent native orientation conversion check on saved states; no rollout is admitted by this check.
    data = mujoco.MjData(env.rig.model)
    angular_error = 0.
    for frame in (100, 350, 600):
        for world in (0, 6, 9, 15):
            data.qpos[:] = z['qpos'][frame, world]
            data.qvel[:] = z['qvel'][frame, world]
            mujoco.mj_forward(env.rig.model, data)
            angular_error = max(angular_error, float(np.max(abs(data.xmat[1].reshape(3,3)@data.qvel[3:6]-obs[frame+1,world,12:15]))))
    assert angular_error < 2e-6
    dataset = dict(x=x[use], source=source[use], issued=z['issued'][use], goal=z['requested_qpos'][use][:,qi].astype(np.float32),
                   world=worlds[cols], frame=frames, validation=worlds[cols]%8>=6)
    np.savez_compressed(out/'dataset.npz', **dataset)
    save(out/'dataset-contract.json', dict(accepted_worlds=worlds[accepted].tolist(), excluded_worlds=worlds[~accepted].tolist(),
        examples=len(frames), train_examples=int((~dataset['validation']).sum()), validation_examples=int(dataset['validation'].sum()),
        split='Whole poses 0-5 train, 6-7 held out; normalization fitted only on train',
        observation='Approximate reconstruction from preceding float32 position, native velocity and summed loads; original exact observations unavailable',
        maximum_target_reconstruction_error=float(error), maximum_source_offset=float(abs(offset[use]).max()),
        angular_conversion_error=angular_error, old_export_gate_passed=False, current_anchor=digest(PARENT)))
    return dataset


def train(dataset, mode, seed, out):
    torch.manual_seed(seed)
    rng = torch.Generator(device='cuda').manual_seed(seed)
    train_mask = ~dataset['validation']
    x = torch.tensor(dataset['x'], device='cuda')
    base = torch.tensor(dataset[mode], device='cuda')
    y = torch.tensor(dataset['issued'], device='cuda')-base
    ids = torch.tensor(np.flatnonzero(train_mask), device='cuda')
    val = torch.tensor(np.flatnonzero(~train_mask), device='cuda')
    mean, scale = x[ids].mean(0), x[ids].std(0).clamp_min(.02)
    output_scale = y[ids].square().mean(0).sqrt().clamp_min(.02)
    net = Learner(mean, scale, output_scale).cuda()
    out.mkdir()
    torch.save(net.state_dict(), out/'initial.pt')
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    start = time.perf_counter()
    for update in range(4000):
        index = ids[torch.randint(len(ids), (512,), device='cuda', generator=rng)]
        loss = ((net(x[index])-y[index])/output_scale).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (update+1)%1000 == 0:
            print('train',out.name,update+1,float(loss),flush=True)
    net.eval()
    with torch.no_grad():
        pred = net(x)
        result = dict(mode=mode, seed=seed, updates=4000, wall_seconds=time.perf_counter()-start,
            train_rmse_rad=float((pred[ids]-y[ids]).square().mean().sqrt()),
            heldout_rmse_rad=float((pred[val]-y[val]).square().mean().sqrt()),
            heldout_max_error_rad=float((pred[val]-y[val]).abs().max()))
    torch.save(net.cpu().state_dict(), out/'final.pt')
    save(out/'report.json', result)
    print('TRAIN_COMPLETE',result,flush=True)
    return net, result


@torch.no_grad()
def evaluate(env, anchor, bank, net, mode, onset, quiet, out):
    from .predictive_target_support import SupportSequence, blend, capture_margin
    from .target_control33 import DelayedTargetControl33
    from .run_single_support_feedback import flags
    from .support_teacher_dataset import policy_input
    env.reset()
    seq = {i:SupportSequence(*bank[i]['poses'],env.dt,'capture') for i in range(16)}
    for s in seq.values():
        s.started = onset
    live = np.ones(16,dtype=bool)
    reason = ['duration']*16
    initial_xy = env.measurements()['foot_xy_m'].numpy().copy()
    rows=[]
    start=time.perf_counter()
    control=env.engine.policy_control
    seconds=30. if quiet else onset+24.
    for frame in range(int(np.ceil(seconds/env.dt))):
        now=frame*env.dt
        obs=env.task.observations()
        source=anchor(obs)
        control.set_reference(*env.task.sample_reference(env.task.offset+env.task.steps*env.dt))
        source_targets=DelayedTargetControl33.target_positions(control,source)
        pre=env.measurements(); prestand,presupport=flags(env,pre)
        loads=pre['foot_loads_n'].numpy()
        goals=np.stack([bank[i]['poses'][0] for i in range(16)])
        authority=np.zeros(16,dtype=np.float32)
        for i in range(16):
            if not live[i] or now<onset:continue
            side=i//8; s=seq[i]
            strict=bool(presupport[i] and loads[i,side]<5 and loads[i,1-side]>=30 and pre['clearance_m'][i,side]>=.015
                        and np.linalg.norm(pre['foot_xy_m'][i,1-side].numpy()-initial_xy[i,1-side])<=.02)
            if not quiet:
                s.update(now,loads[i,1-side]/max(1.,loads[i].sum()),float(pre['horizontal_speed_m_s'][i]),
                         presupport[i],strict,loads[i].min()>=30,capture_margin(env.rig.model,env.engine.datas[i],1-side))
                if s.finished:
                    live[i]=False;reason[i]=s.reason;continue
                goals[i]=s.goal(now+2*env.dt)
            authority[i]=float(blend(0.,1.,now-onset,1.))
        if not live.any():break
        x=torch.tensor(policy_input(obs.numpy(),goals,authority))
        base=source_targets if mode=='source' else torch.tensor(goals[:,control.q_indices.numpy()],dtype=torch.float32)
        target=base+net(x)
        addition=(target-source_targets).clamp(-1,1)
        addition[torch.tensor(authority==0)]=0
        env.engine.active=live.copy();before=live.copy()
        _,_,done,_=env.step(addition)
        post=env.last;stand,supported=flags(env,post)
        q=env.engine.native_qpos().numpy()
        assert np.isfinite(q[before]).all() and not env.engine.solver_warning[before].any()
        issued=control.pending[(control.cursor-1)%2,:,:33]
        assert torch.max(abs(issued-(source_targets+addition).clamp(control.lower,control.upper)))<2e-6
        assert (abs(control.ctrl[:,env.rig.actuators])<=torch.tensor(env.rig.action_scale)+1e-4).all()
        rows.append(dict(time=now+env.dt,alive=before,stand=stand&prestand,supported=supported&presupport,
            loads=post['foot_loads_n'].numpy().copy(),clearance=post['clearance_m'].numpy().copy(),
            xy=post['foot_xy_m'].numpy().copy(),qpos=q.copy(),issued=issued.numpy().copy(),
            done=done.numpy().copy(),phase=np.array([s.phase for s in seq.values()])))
        for i in range(16):
            if not before[i]:continue
            if done[i]==1:live[i]=False;reason[i]='physical_failure'
            elif not quiet and seq[i].phase=='settle' and now-seq[i].started>=2:
                live[i]=False;reason[i]='sequence_complete'
    z={k:np.stack([r[k] for r in rows]) for k in rows[0]}
    cases=[]
    for i in range(16):
        side=i//8; alive=z['alive'][:,i];last=np.flatnonzero(alive)[-1]
        strict=alive&z['supported'][:,i]&(z['loads'][:,i,side]<5)&(z['loads'][:,i,1-side]>=30)&(z['clearance'][:,i,side]>=.015)
        strict &= np.linalg.norm(z['xy'][:,i,1-side]-initial_xy[i,1-side],axis=-1)<=.02
        count=int(np.ceil(2/env.dt));final=last+1>=count and z['stand'][last+1-count:last+1,i].all() and reason[i]!='physical_failure'
        duration=longest(strict)*env.dt
        cases.append(dict(world=i,pose=i%8,side=side,heldout=i%8>=6,excluded_teacher=i in (10,11,12),
            reason=reason[i],single_support_seconds=duration,final_stand=bool(final),
            whole_interval_stand=bool(alive.all() and z['stand'][:,i].all() and reason[i]!='physical_failure'),
            completed=bool(duration>=1 and final and reason[i]=='sequence_complete'),
            peak_clearance_m=float(z['clearance'][alive,i,side].max()),
            peak_foot_displacement_m=float(np.linalg.norm(z['xy'][alive,i,side]-initial_xy[i,side],axis=-1).max())))
    out.mkdir();np.savez_compressed(out/'trace.npz',**z)
    result=dict(onset=onset,quiet=quiet,wall_seconds=time.perf_counter()-start,cases=cases,
        completed=sum(c['completed'] for c in cases),final_stand=sum(c['final_stand'] for c in cases),
        whole_interval_stand=sum(c['whole_interval_stand'] for c in cases),
        valid_lift_50ms=sum(c['single_support_seconds']>=.05 for c in cases),
        runtime='Neural joint target output; measured-support command scheduler; no physical optimizer or live pose writes')
    save(out/'report.json',result)
    print('EVAL_COMPLETE',out.name,{k:v for k,v in result.items() if k!='cases'},flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from .run_single_support_feedback import make_environment,target_bank
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir()
    paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('LEARNED_SUPPORT_PLAN.md')]
    inputs={str(p):digest(p) for p in paths+[PARENT,PARENT.with_suffix('.amp.pt'),FOUNDATION/'dummy.xml',FOUNDATION/'stationary_reference.npz',
        TEACHER/'entry-feedback/trace.npz',TEACHER/'entry-feedback/report.json']}
    for p in paths:shutil.copy2(p,out/'sources'/p.name)
    save(out/'protocol.json',dict(inputs=inputs,arms=ARMS,steps=20,native_only=True,arena_promotion=False))
    rig=Rig.load(FOUNDATION/'dummy.xml',policy_exclude=())
    protocol=json.loads((FOUNDATION/'protocol.json').read_text())
    anchor=load_actor(rig,FOUNDATION/'stationary_reference.npz',protocol,PARENT)
    env=make_environment(rig,anchor,40.)
    dataset=reconstruct(env,anchor,out)
    print('DATASET_READY',len(dataset['x']),flush=True)
    bank=target_bank(rig,env,range(16),com_shift=.12,lean_degrees=12.)
    np.savez_compressed(out/'goals.npz',poses=np.array([bank[i]['poses'] for i in range(16)]))
    results=[];nets={}
    for mode,seed in ARMS:
        key=name(mode,seed)
        nets[key],result=train(dataset,mode,seed,out/key)
        results.append(dict(step=len(results)+1,experiment=key,kind='training',result=result))
        save(out/'progress.json',results)
    for onset,quiet in ((1.,False),(3.,False),(9.,False),(1.,True)):
        for mode,seed in ARMS:
            key=name(mode,seed);label=f'{key}-'+('quiet30' if quiet else f'onset{onset:g}')
            result=evaluate(env,anchor,bank,nets[key],mode,onset,quiet,out/label)
            results.append(dict(step=len(results)+1,experiment=label,kind='physical_evaluation',result=result))
            save(out/'progress.json',results)
    assert len(results)==20 and all(digest(Path(p))==h for p,h in inputs.items())
    save(out/'final-verification.json',dict(completed_steps=20,training_updates=16000,physical_cases=256,
        input_hashes_unchanged=True,arena_changed=False,neural_models=list(nets),
        output_hashes={str(p):digest(p) for p in out.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
