"""Frozen-actor robustness assessments; execution requires sustained-Stand admission."""
import argparse
import copy
import traceback
from pathlib import Path
import numpy as np
import torch
from .replicate_stand_representations import read_state, cpu, pending_commands, gates, tensor_hash, verify_control_trace
from .evaluate_stand_sustained import duration_summary
from .verify_stand_learning_curve import verify_gates
from .audit_stand_failures import read, save
from .stand_experiment_evidence import protect, seal, load_actor
from .runtime import activate
from .rig import Rig


def initialize_scenario(task, backend, scenario):
    q,v=read_state(task,backend)
    translation=np.asarray(scenario.get('initial_translation_m',[0.,0.,0.]), dtype=np.float32)
    assert translation.shape==(3,) and translation[2]==0 and np.linalg.norm(translation)<=.1
    q[:,:3]+=translation
    angle=float(scenario.get('initial_yaw_degrees',0.))
    assert abs(angle)<=10.
    c,s=np.cos(np.deg2rad(angle)/2),np.sin(np.deg2rad(angle)/2)
    w,x,y,z=q[:,3:7].T.copy()
    q[:,3:7]=np.stack((c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w),axis=1)
    velocity=np.asarray(scenario.get('initial_velocity_m_s',[0.,0.,0.]),dtype=np.float32)
    assert velocity.shape==(3,) and velocity[2]==0 and np.linalg.norm(velocity)<=.2
    v[:,:3]+=velocity
    offsets=scenario.get('initial_joint_offsets_degrees',{})
    for name,degrees in offsets.items():
        assert abs(degrees)<=5.
        joint=task.rig.model.joint(name).id
        assert task.rig.model.jnt_type[joint]==3
        index=task.rig.model.jnt_qposadr[joint]
        q[:,index]+=np.deg2rad(degrees)
        lower,upper=task.rig.model.jnt_range[joint]
        assert ((q[:,index]>=lower)&(q[:,index]<=upper)).all()
    if np.any(translation) or angle or np.any(velocity) or offsets:
        task.engine.reset_envs(task.ids,torch.as_tensor(q,device=task.device),torch.as_tensor(v,device=task.device))
        task.previous_com=task.center_of_mass().clone()


def force_schedule(scenario,frame,dt):
    force=np.zeros(3,dtype=np.float32)
    for pulse in scenario.get('pulses',[]):
        start=int(np.ceil(pulse['onset_seconds']/dt))
        count=int(np.ceil(pulse['duration_seconds']/dt))
        if start<=frame<start+count:
            force+=np.asarray(pulse['force_world_n'],dtype=np.float32)
    return force


def apply_disturbance(task,backend,scenario,frame,active):
    force=np.broadcast_to(force_schedule(scenario,frame,task.dt),(8,3)).copy()
    force[~active]=0
    body=task.rig.model.body(scenario.get('body','Chest')).id
    task.engine.set_external_force(body,force)
    if backend=='native':
        actual=np.stack([data.xfrc_applied[body,:3].copy() for data in task.engine.datas])
    else:
        actual=cpu(task.engine._sim_state.body_force[0][:,body-1,:3])
    assert np.array_equal(actual,force)
    return actual


def challenge_summary(trace,dt,seconds,scenario):
    result=duration_summary(trace,dt,seconds)
    pulses=scenario.get('pulses',[])
    last_end=max((int(np.ceil(p['onset_seconds']/dt))+int(np.ceil(p['duration_seconds']/dt)))*dt for p in pulses) if pulses else 0.
    recovery=[]
    for c,row in enumerate(result['cases']):
        valid=trace['balance'][:,c]&trace['posture'][:,c]&trace['pre_balance'][:,c]&trace['pre_posture'][:,c]
        live=trace['alive_before'][:,c]
        final=live&(trace['time_s']>trace['time_s'][int(live.sum())-1]-2.)
        starts=[]
        required=int(np.ceil(2./dt))
        for index in np.flatnonzero(live&(trace['time_s']>=last_end)):
            if index+required<=len(live) and live[index:index+required].all() and valid[index:index+required].all():
                starts.append(index)
                break
        recovered_at=float(trace['time_s'][starts[0]]) if starts else None
        onset_frame=min((int(np.ceil(p['onset_seconds']/dt)) for p in pulses),default=0)
        pre=live&(np.arange(len(live))<onset_frame)
        prequalified=bool(valid[pre].all())
        per_pulse=[]
        ordered=sorted(pulses,key=lambda p:p['onset_seconds'])
        for pulse_index,pulse in enumerate(ordered):
            end=(int(np.ceil(pulse['onset_seconds']/dt))+int(np.ceil(pulse['duration_seconds']/dt)))*dt
            next_start=int(np.ceil(ordered[pulse_index+1]['onset_seconds']/dt))*dt if pulse_index+1<len(ordered) else seconds+dt
            first=None
            for index in np.flatnonzero(live&(trace['time_s']>=end)&(trace['time_s']<=end+2.+dt)):
                stop=index+required
                if stop<=len(live) and trace['time_s'][stop-1]<=next_start+1e-9 and live[index:stop].all() and valid[index:stop].all():
                    first=float(trace['time_s'][index])
                    break
            per_pulse.append(dict(pulse=pulse_index,recovery_window_start=first,passed=first is not None))
        recovered=bool(row['survived'] and prequalified and valid[final].all() and recovered_at is not None and recovered_at<=last_end+2.+dt and all(p['passed'] for p in per_pulse))
        recovery.append(dict(case=c,recovery_pass=recovered,pre_disturbance_valid=prequalified,
            first_two_second_valid_window_start=recovered_at,last_disturbance_end_seconds=last_end,per_pulse_recovery=per_pulse))
    result.update(recovery_cases=recovery,recovery_passes=sum(row['recovery_pass'] for row in recovery),
        recovery_scope='Survive; valid before first push; after each push begin two continuous valid seconds within two seconds (one control-frame timestamp tolerance) and before the next push; valid final two seconds')
    return result


@torch.no_grad()
def evaluate_challenge(rig, reference, mode, backend, actor, out, *, seconds, scenario):
    from envs.base_env import EnvMode
    from .representation_task import RepresentationTask
    device = 'cpu' if backend == 'native' else 'cuda:0'
    task = RepresentationTask(rig, reference, 8, device, representation=mode, trace=True,
        reward_mode='reference_v9', episode_seconds=seconds)
    task.set_mode(EnvMode.TEST)
    task.reset()
    initialize_scenario(task, backend, scenario)
    out.mkdir()
    contract = task.observation_contract()
    contract['challenge'] = scenario
    save(out/'contract.json', contract)
    before = None
    if actor is not None:
        actor.to(device).eval()
        before = tensor_hash(actor.state_dict())
    active, rows = np.ones(8, dtype=bool), []
    joints = rig.model.actuator_trnid[rig.actuators, 0]
    qindices = rig.model.jnt_qposadr[joints]
    for frame in range(int(np.ceil(seconds/task.dt))):
        if backend == 'native':
            task.engine.active = active.copy()
        obs = task.observations()
        q, v = read_state(task, backend)
        c = task.engine.policy_control
        pending = cpu(pending_commands(c))
        flags_before, measurements_before = gates(task)
        assert torch.isfinite(obs[torch.as_tensor(active, device=device)]).all()
        action = torch.zeros((8, 33), device=device) if actor is None else actor(obs)
        action[torch.as_tensor(~active, device=device)] = 0
        assert torch.isfinite(action).all() and (action.abs() <= 1).all()
        reference_q, _ = task.sample_reference(task.offset+task.steps*task.dt)
        applied = apply_disturbance(task, backend, scenario, frame, active)
        _, _, done, _ = task.step(action)
        flags, measurements = gates(task)
        warnings = task.engine.solver_warning.copy() if backend == 'native' else np.zeros(8, dtype=bool)
        done[torch.as_tensor(warnings, device=device)] = 1
        after_q, after_v = read_state(task, backend)
        row = dict(external_force_world_n=applied, alive_before=active.copy(), observation=cpu(obs), action=cpu(action),
            qpos_before=q, qvel_before=v, qpos_after=after_q, qvel_after=after_v,
            pending_before=pending, issued_entry=cpu(c.pending[(c.cursor-1)%2]),
            reference_motor_q=cpu(reference_q[:, qindices]),
            substep_q=cpu(task.engine.substep_q), substep_v=cpu(task.engine.substep_v),
            substep_torque=cpu(task.engine.substep_torque), done=cpu(done), solver_warning=warnings,
            balance=cpu(flags['balance']), posture=cpu(flags['posture']),
            pre_balance=cpu(flags_before['balance']), pre_posture=cpu(flags_before['posture']))
        if mode == 'target_pd':
            row['active_targets'] = cpu(c.active)
        row.update({key: cpu(value) for key, value in measurements.items()})
        row.update({f'pre_{key}': cpu(value) for key, value in measurements_before.items()})
        rows.append(row)
        if warnings[active].any() or not np.isfinite(after_q[active]).all() or not np.isfinite(after_v[active]).all():
            np.savez_compressed(out/'failed-trace.npz', **{key: np.stack([r[key] for r in rows]) for key in rows[0]})
            raise RuntimeError('Native warning/nonfinite admission failure')
        active &= cpu(done) == 0
        if not active.any():
            break
    arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    arrays['time_s'] = (np.arange(len(rows))+1)*task.dt
    np.savez_compressed(out/'trace.npz', **arrays)
    formula_error = verify_control_trace(arrays, contract)
    reset_result = None
    if actor is not None:
        assert tensor_hash(actor.state_dict()) == before
        actor.cpu()
    report = dict(**challenge_summary(arrays, task.dt, seconds, scenario), representation=mode, backend=backend,
        zero_action_baseline=actor is None, actor_hash=before, passive_springs=False,
        active_pose_feedback=mode == 'target_pd', inverse_contact_teacher=False, external_support=False,
        live_control_intervals=int(arrays['alive_before'].sum()), max_control_formula_error_nm=formula_error,
        solver_warnings=0 if backend == 'native' else None, partial_reset=reset_result)
    save(out/'report.json', report)
    print(dict(evaluation=str(out), passes=report['standing_passes'], mean_survival=report['mean_survival_seconds']), flush=True)
    return report


def verify_challenge(folder, seconds, scenario):
    contract, report = read(folder/'contract.json'), read(folder/'report.json')
    assert contract['challenge'] == scenario
    with np.load(folder/'trace.npz', allow_pickle=False) as data:
        trace = {key: data[key] for key in data.files}
    live, dt = trace['alive_before'], contract['control_timestep']
    assert live.shape[1] == 8 and len(live) <= int(np.ceil(seconds/dt))
    assert contract['representation'] == 'target_pd' and contract['num_obs'] == 374
    assert contract['passive_joint_stiffness'] == 0 and contract['standing_pose_servo']
    with np.load(folder.parent/'stationary_reference.npz') as reference:
        q,v=reference['qpos'].copy(),reference['qvel'].copy()
    q[:,:3]+=np.asarray(scenario.get('initial_translation_m',[0.,0.,0.]))
    angle=np.deg2rad(scenario.get('initial_yaw_degrees',0.))/2
    w,x,y,z=q[:,3:7].T.copy()
    c,s=np.cos(angle),np.sin(angle)
    q[:,3:7]=np.stack((c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w),axis=1)
    for name,degrees in scenario.get('initial_joint_offsets_degrees',{}).items():
        index=7+contract['joint_names'].index(name)
        q[:,index]+=np.deg2rad(degrees)
    v[:,:3]+=np.asarray(scenario.get('initial_velocity_m_s',[0.,0.,0.]))
    assert np.allclose(trace['qpos_before'][0],q,rtol=0,atol=2e-6)
    assert np.allclose(trace['qvel_before'][0],v,rtol=0,atol=2e-6)
    assert np.array_equal(trace['time_s'], (np.arange(len(live))+1)*dt)
    assert np.array_equal(trace['qpos_before'][1:], trace['qpos_after'][:-1])
    assert np.array_equal(trace['qvel_before'][1:], trace['qvel_after'][:-1])
    assert all(np.isfinite(trace[key][live]).all() for key in
        ('observation','action','qpos_before','qvel_before','qpos_after','qvel_after'))
    assert not trace['solver_warning'][live].any()
    verify_gates(trace, '')
    verify_gates(trace, 'pre_')
    assert verify_control_trace(trace, contract) == report['max_control_formula_error_nm']
    assert all(report[key] == value for key,value in challenge_summary(trace,dt,seconds,scenario).items())
    expected=np.stack([np.broadcast_to(force_schedule(scenario,f,dt),(8,3)).copy() for f in range(len(live))])
    expected[~live]=0
    assert np.array_equal(expected,trace['external_force_world_n'])
    for case in range(8):
        count=int(live[:,case].sum())
        assert live[:count,case].all() and not live[count:,case].any()
        assert (trace['done'][:count-1,case]==0).all() and trace['done'][count-1,case] in (1,3)
        if trace['done'][count-1,case]==3:
            assert count==int(np.ceil(seconds/dt))
    return trace, report


def run(args):
    specification=read(args.specification)
    assert 12 <= specification['seconds'] <= 120
    scenarios=specification['scenarios']
    assert 1 <= len(scenarios) <= 32
    names=[scenario['name'] for scenario in scenarios]
    assert len(set(names)) == len(names)
    for scenario in scenarios:
        assert scenario['name'].replace('_','').replace('-','').isalnum()
        pulses=sorted(scenario.get('pulses',[]),key=lambda p:p['onset_seconds'])
        previous_end=0.
        for pulse in pulses:
            onset,duration=pulse['onset_seconds'],pulse['duration_seconds']
            assert 0 < duration <= .5 and onset >= previous_end
            assert onset+duration+4. <= specification['seconds']
            force=np.asarray(pulse['force_world_n'])
            assert force.shape==(3,) and np.isfinite(force).all() and np.linalg.norm(force)<=200
            previous_end=onset+duration
    admissions=specification['admission_roots']
    seeds=[]
    for name in admissions:
        root=Path(name)
        report=read(root/'report.json')
        assert report['sustained_stand_verified'], 'Two independently trained sustained actors are required'
        assert read(root/'final-verification.json')['sustained_stand_verified']
        seeds.append(read(root/'protocol.json')['seed'])
    assert len(set(seeds)) >= 2
    activate(args.mimickit)
    torch.set_num_threads(1)
    inputs,sources=protect(args.prior,args.out,[Path(__file__)],extra_inputs=(args.specification,))
    canonical={str(Path(p).resolve()) for p in inputs}
    assert all(str((Path(p)/'final-verification.json').resolve()) in canonical for p in admissions)
    prior=read(args.prior/'protocol.json')
    checkpoint=Path(specification['checkpoint'])
    assert str(checkpoint.resolve()) in canonical
    protocol=dict(inputs=inputs,sources=sources,config=prior['config'],demonstrations=prior['demonstrations'],
        model_sha256=prior['model_sha256'],reference_sha256=prior['reference_sha256'],specification=specification,
        maximum_cases=16*len(specification['scenarios']),training_updates=0,
        pulse_discretization='Ceil onset and duration to control frames; force is held for every physics substep',
        recovery_timestamp_tolerance='One control frame beyond the two-second onset deadline')
    save(args.out/'protocol.json',protocol)
    rig,reference=Rig.load(args.out/'dummy.xml',policy_exclude=()),args.out/'stationary_reference.npz'
    actor=load_actor(rig,reference,protocol,checkpoint)
    results={}
    for scenario in specification['scenarios']:
        assert scenario['name'].replace('_','').replace('-','').isalnum()
        assert scenario['name'] not in results
        for pulse in scenario.get('pulses',[]):
            assert 0 < pulse['duration_seconds'] <= .5 and 0 <= pulse['onset_seconds'] < specification['seconds']-4
            force=np.asarray(pulse['force_world_n'])
            assert force.shape==(3,) and np.isfinite(force).all() and np.linalg.norm(force)<=200
        pair={}
        for backend in ('native','gpu'):
            folder=args.out/f"{scenario['name']}-{backend}"
            pair[backend]=evaluate_challenge(rig,reference,'target_pd',backend,copy.deepcopy(actor),folder,
                seconds=specification['seconds'],scenario=scenario)
            _, verified=verify_challenge(folder,specification['seconds'],scenario)
            assert verified==pair[backend]
        results[scenario['name']]=pair
        save(args.out/'progress.json',dict(completed_scenarios=list(results),results=results))
    save(args.out/'report.json',dict(results=results,training_updates=0,cases=16*len(results),
        all_recovery_passed=all(r['recovery_passes']==8 for pair in results.values() for r in pair.values()),
        all_continuous_stand_passed=all(r['standing_passes']==8 for pair in results.values() for r in pair.values()),
        arena_changed=False))
    seal(args.out,inputs,sources,verified=True,cases=16*len(results),training_updates=0,
        force_schedule_and_commands_verified=True,physical_gates_and_delayed_controls_verified=True,
        actor_and_normalizers_unchanged=True)
    print({name:{b:dict(stand=r['standing_passes'],recovery=r['recovery_passes'],mean_survival=r['mean_survival_seconds'])
        for b,r in pair.items()} for name,pair in results.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--specification',type=Path,required=True)
    parser.add_argument('--mimickit',type=Path,default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args=parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            save(args.out/'failure.json',dict(error=traceback.format_exc()))
        raise

