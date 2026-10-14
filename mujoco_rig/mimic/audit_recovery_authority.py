"""Native motor-contract experiment; frozen source actor, no policy updates."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .baseline import sha256
from .runtime import activate


def run(args):
    if args.support_actuation!='target_pd' and not (args.support_probe and args.support_only):
        raise ValueError('Direct torque is restricted to standalone diagnostic support probes')
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .balance_observation import task_classes
    from .checkpoints import validate_contract
    from .collect_recovery_examples import example_checks
    from .continuous_stand import evaluate_retention,GATES
    from .policy import StandPolicy
    from .reactive_ball_eval import evaluate,selection_cases
    from .recovery_motor import authority_gate
    from .recovery_quality import RecoveryQuality
    from .rig import Rig
    torch.set_num_threads(1)
    mp_util.init(0,1,'cpu',None)
    rig=Rig.load(Path(__file__).resolve().parents[1]/'dummy.xml')
    contract=json.loads((args.checkpoint.parent/'contract.json').read_text())
    _,Task=task_classes(contract)
    source_task=Task(rig,args.reference,8,'cpu',ball_path=args.ball_model,recovery_mode='step_v7')
    validate_contract(contract,source_task.observation_contract())
    agent=PPOAgent(yaml.safe_load((args.checkpoint.parent/'agent.yaml').read_text()),source_task,'cpu')
    checkpoint_hash=sha256(args.checkpoint)
    agent.load(str(args.checkpoint))
    actor=StandPolicy(agent).eval()
    args.out.mkdir(parents=True,exist_ok=False)
    sources=('audit_recovery_authority.py','recovery_motor.py','capture_step_teacher.py','full_body_teacher.py',
             'dynamic_support_teacher.py','inverse_contact_control.py','probe_dynamic_support.py','target_control.py','reactive_ball_task.py',
             'reactive_ball_eval.py','recovery_quality.py','collect_recovery_examples.py')
    source_hashes={name:sha256(Path(__file__).with_name(name)) for name in sources}
    for name in sources:(args.out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(schema='native_motor_authority_pilot_v1',checkpoint_sha256=checkpoint_hash,
        character_model_sha256=sha256(rig.path),reference_sha256=sha256(args.reference),
        ball_model_sha256=sha256(args.ball_model),source_hashes=source_hashes,variants=[],
        optimizer_updates=0,arena_changed=False,
        scope='Source actor is frozen; expanded motor semantics are an explicit diagnostic substitution, '
              'not an admitted checkpoint migration. Fixed development impacts; no unseen selection.')
    def save():
        if sha256(args.checkpoint)!=checkpoint_hash:raise RuntimeError('Source checkpoint changed')
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def adapted_actor(task, radius):
        if radius is None or args.actor_adapter=='raw':return actor,None
        if args.actor_adapter=='dynamic_support':
            from .dynamic_support_teacher import DynamicSupportTeacher
            teacher=DynamicSupportTeacher(task,actor,contact_damping=args.contact_damping,
                                          contact_tracking_weight=args.contact_tracking_weight,
                                          all_motors=args.dynamic_all_motors,force_model=args.dynamic_force_model,
                                          balance_tracking_weight=args.balance_tracking_weight)
            return teacher,teacher
        if args.actor_adapter=='full_body':
            from .full_body_teacher import FullBodyTeacher
            teacher=FullBodyTeacher(task,actor,correction_gain=args.teacher_gain,
                                    composition=args.teacher_composition,release_settled=args.teacher_release_settled,
                                    rearm_quiet=args.teacher_rearm_quiet)
            return teacher,teacher
        if args.actor_adapter=='capture_step':
            from .capture_step_teacher import CaptureStepTeacher
            teacher=CaptureStepTeacher(task,actor)
            return teacher,teacher
        from .capture_step_teacher import bridge_source_action
        return lambda obs:bridge_source_action(task,actor(obs),obs),None
    baseline_poses=None
    if args.baseline_report:
        baseline=json.loads(args.baseline_report.read_text())
        for key in ('checkpoint_sha256','reference_sha256','character_model_sha256','ball_model_sha256'):
            if baseline[key]!=report[key]:raise ValueError('Baseline identity mismatch: '+key)
        baseline_poses=np.load(args.baseline_report.parent/'legacy-quiet.npz')['qpos_after']
        report['baseline_report_sha256']=sha256(args.baseline_report)
    for radius in ([] if args.baseline_report else [None])+args.radii:
        label='legacy' if radius is None else f'radius-{radius:g}'
        started=time.monotonic()
        task=Task(rig,args.reference,8,'cpu',ball_path=args.ball_model,recovery_mode='step_v7',
                  recovery_motor_radians=radius)
        quiet_policy,quiet_teacher=adapted_actor(task,radius)
        def quiet_actor(obs):
            task.shot_enabled.zero_();task.second_shot_enabled.zero_()
            return quiet_policy(obs)
        quiet,trace=evaluate_retention(task,quiet_actor)
        if baseline_poses is None:baseline_poses=trace['qpos_after']
        same_length=len(trace['qpos_after'])==len(baseline_poses)
        quiet_error=float(np.max(np.abs(trace['qpos_after']-baseline_poses[:len(trace['qpos_after'])])))
        row=dict(label=label,radius_radians=radius,contract=task.observation_contract(),quiet=quiet,
            quiet_gate_active_frames=int((authority_gate(torch.from_numpy(trace['observations']))>0).sum()),
            quiet_same_length=same_length,quiet_max_qpos_difference=quiet_error)
        if quiet_teacher is not None and hasattr(quiet_teacher,'diagnostics'):
            row['quiet_teacher_diagnostics']=quiet_teacher.diagnostics()
        report['variants'].append(row)
        np.savez_compressed(args.out/f'{label}-quiet.npz',**trace)
        save()
        if not quiet['passed']:
            row['stopped']='quiet_gate_failed';save();print(json.dumps(dict(label=label,quiet_passed=False)),flush=True)
            continue
        if args.support_probe:
            if radius is None:raise ValueError('Support probe requires experimental target motor bounds')
            from .probe_dynamic_support import evaluate_support
            support_task=Task(rig,args.reference,8,'cpu',ball_path=args.ball_model,
                             recovery_mode='step_v7',recovery_motor_radians=radius)
            support, support_trace=evaluate_support(support_task,actor,contact_damping=args.contact_damping,
                                                   contact_tracking_weight=args.contact_tracking_weight,
                                                   all_motors=args.dynamic_all_motors,actuation=args.support_actuation,
                                                   force_model=args.dynamic_force_model,balance_tracking_weight=args.balance_tracking_weight)
            row['forced_support']=support
            np.savez_compressed(args.out/f'{label}-forced-support.npz',**support_trace)
            row['forced_support_trace_sha256']=sha256(args.out/f'{label}-forced-support.npz')
            save()
            print(json.dumps(dict(forced_support_passed=support['passed'],
                survival_seconds=[c['survival_seconds'] for c in support['cases']],
                infeasible_solves=support['teacher_diagnostics']['infeasible_solves'])),flush=True)
            if not support['passed'] or args.support_only:
                row['stopped']='forced_support_failed' if not support['passed'] else 'support_probe_only'
                save();continue
        cases=[dict(c,name=f'{speed:g}-{c["name"]}') for speed in args.speeds
            for c in selection_cases(speed,args.seconds,rig.model.opt.timestep*rig.decimation)
            if c['direction'] in args.directions]
        count=len(cases)
        task=Task(rig,args.reference,count,'cpu',ball_path=args.ball_model,recovery_mode='step_v7',
            recovery_motor_radians=radius,speed_min=2.,speed_max=3.,episode_seconds=30.)
        row['development_protocol']=dict(speeds=args.speeds,directions=args.directions,seconds=args.seconds)
        hit_actor,teacher=adapted_actor(task,radius)
        quality=RecoveryQuality(task,hit_actor)
        actions,poses,velocities,plant,active_trace=[],[],[],[],[]
        gate_seconds=torch.zeros(count)
        rate_peak=torch.zeros(count)
        last_target=None
        @torch.no_grad()
        def policy(obs):
            action=quality(obs);actions.append(action.numpy().copy());return action
        def record(task,step,active,done):
            nonlocal last_target
            quality.record(task,step,active,done)
            gate=authority_gate(task.observations())
            gate_seconds.add_((gate>0)*active*task.dt)
            control=task.engine.policy_control
            target=control.pending[(control.cursor-1)%len(control.pending),:,:30].clone()
            if last_target is not None:
                change=((target-last_target)/task.dt).abs().amax(-1)
                rate_peak[:]=torch.maximum(rate_peak,change*active)
            last_target=target
            poses.append(task.engine.native_qpos().numpy().copy())
            velocities.append(task.engine._array('qvel')[:,:rig.model.nv].numpy().copy())
            plant.append(task.engine._array('qpos').numpy().copy());active_trace.append(active.numpy().copy())
            if teacher is not None and hasattr(teacher,'phase') and (step+1)%300==0:
                print(json.dumps(dict(control_step=step+1,active=int(active.sum()),
                    teacher_phase_worlds=np.bincount(teacher.phase,minlength=6).tolist())),flush=True)
        metrics,_=evaluate(task,policy,cases,record)
        metrics['motion_quality']=quality.report()
        rows=[dict(case=i,checks=example_checks(c,metrics['motion_quality']['cases'][i]))
              for i,c in enumerate(metrics['cases'])]
        for item in rows:item['accepted']=all(item['checks'].values())
        row.update(hits=metrics,candidates=rows,accepted_cases=[c['case'] for c in rows if c['accepted']],
            gate_active_seconds_by_world=gate_seconds.tolist(),
            maximum_all_motor_issued_target_speed_rad_s_by_world=rate_peak.tolist(),
            reward_signals=task.recovery_signals.drain(),wall_seconds=time.monotonic()-started)
        row['actor_adapter']=args.actor_adapter
        if teacher is not None:row['teacher_triggers_by_world_and_foot']=teacher.triggers.tolist()
        if teacher is not None and hasattr(teacher,'diagnostics'):
            row['teacher_contract']=teacher.contract()
            row['teacher_diagnostics']=teacher.diagnostics()
        np.savez_compressed(args.out/f'{label}-trace.npz',action=np.asarray(actions),qpos=np.asarray(poses),
            qvel=np.asarray(velocities),plant_qpos=np.asarray(plant),active=np.asarray(active_trace))
        row['trace_sha256']=sha256(args.out/f'{label}-trace.npz')
        save()
        print(json.dumps(dict(label=label,quiet_passed=quiet['passed'],quiet_error=quiet_error,
            recovered=metrics['recovered_hits'],survived=metrics['survived_hits'],stepped=metrics['recovered_with_step'],
            accepted=row['accepted_cases'])),flush=True)
    report['complete']=True
    save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('mimickit','checkpoint','reference','ball-model','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--radii',type=float,nargs='+',default=[.5,.85,1.25])
    parser.add_argument('--actor-adapter',choices=('raw','bridge','capture_step','full_body','dynamic_support'),default='raw')
    parser.add_argument('--teacher-gain',type=float,default=.5)
    parser.add_argument('--teacher-composition',choices=('blend','source_offset'),default='blend')
    parser.add_argument('--teacher-release-settled',action='store_true')
    parser.add_argument('--teacher-rearm-quiet',action='store_true')
    parser.add_argument('--contact-damping',type=float,default=18.)
    parser.add_argument('--contact-tracking-weight',type=float,default=0.)
    parser.add_argument('--balance-tracking-weight',type=float,default=1.)
    parser.add_argument('--dynamic-all-motors',action='store_true')
    parser.add_argument('--dynamic-force-model',choices=('independent_rays','mujoco_inverse_contacts','mujoco_forward_refined'),default='independent_rays')
    parser.add_argument('--support-probe',action='store_true')
    parser.add_argument('--support-only',action='store_true')
    parser.add_argument('--support-actuation',choices=('target_pd','direct_physics_torque'),default='target_pd')
    parser.add_argument('--speeds',type=float,nargs='+',default=[2.5,3.])
    parser.add_argument('--directions',type=int,nargs='+',choices=(0,1,2,3),default=[0,1,2,3])
    parser.add_argument('--seconds',type=float,default=33.)
    parser.add_argument('--baseline-report',type=Path)
    run(parser.parse_args())
