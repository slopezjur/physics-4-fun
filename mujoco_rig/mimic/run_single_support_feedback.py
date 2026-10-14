"""Small native comparison of equilibrium and predictive joint-target support."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time

import mujoco
import numpy as np
import torch

from .runtime import activate
from .run_support_skill_study import FOUNDATION, PARENTS, digest, longest, save
from .target_control33 import DelayedTargetControl33


def flags(env, state):
    values = {**env.balance_criteria(state), **env.posture_criteria(state)}
    stand = torch.stack(list(values.values())).all(0).numpy()
    values.pop('two_foot_support')
    support = torch.stack(list(values.values())).all(0).numpy()
    support &= state['foot_loads_n'].amax(-1).numpy() >= 30
    return stand, support


def make_environment(rig, anchor, seconds):
    from envs.base_env import EnvMode
    from .support_skill_core import SkillEnvironment, Goal
    env = SkillEnvironment(rig, FOUNDATION/'stationary_reference.npz', deepcopy(anchor),
        [Goal(side=side, shift=.11, lift=.02) for side in (0, 1) for _ in range(8)],
        'cpu', seconds=seconds, action_mode='full', radius=1.)
    env.set_mode(EnvMode.TEST)
    env.reset()
    return env


def check_transport(env):
    from .predictive_target_support import PredictiveTargetSupport
    control = env.engine.policy_control
    control.set_reference(*env.task.sample_reference(env.task.offset))
    q = control.reference_q[:, control.q_indices]
    control.pending[:, :, :33] = q
    control.pending[:, :, 66] = 1
    control.pending[1, :, :33] += .001
    before = [d.qpos.copy() for d in env.engine.datas]
    pending = control.pending.clone()
    predictor = PredictiveTargetSupport(env.task)
    predicted = predictor.predict(0)
    expected = (predicted.qpos.copy(), predicted.qvel.copy())
    assert all(np.array_equal(q, d.qpos) for q, d in zip(before, env.engine.datas))
    assert torch.equal(pending, control.pending)
    env.engine.active = np.arange(16) == 0
    for _ in range(2):
        control.apply(torch.zeros((16, 33)))
        env.engine.step()
    live = env.engine.datas[0]
    errors = dict(qpos=float(np.max(abs(expected[0]-live.qpos))), qvel=float(np.max(abs(expected[1]-live.qvel))))
    assert errors['qpos'] < 3e-6 and errors['qvel'] < 3e-5, errors
    return dict(passed=True, maximum_prediction_errors=errors, private_prediction_did_not_mutate_live_state=True)


def target_bank(rig, env, worlds, *, com_shift=.11, lean_degrees=0.):
    from .support_skill_core import fit_goal
    from .support_skill_physical_teacher import clear_arms
    from .static_support import solve_target_support
    from .screen_support_geometry import pose_metrics
    bank = {}
    for world in worlds:
        side = world//8
        reference = env.task.reference.qpos[world%8].numpy().astype(float)
        cleared = clear_arms(rig.model, reference, .10)
        poses, feedforward, reports = [reference, cleared], [], []
        for lift in (0., .02):
            pose, fit = fit_goal(rig.model, cleared, shift=com_shift*(1 if side == 0 else -1), lift=lift, distance=0., swing=side,
                                 lean_radians=np.deg2rad(lean_degrees))
            names = tuple(f'{body}_{leg}' for body in ('Foot','Toe') for index,leg in enumerate(('L','R')) if not lift or index != side)
            support = solve_target_support(rig, pose, support_names=names)
            metrics = pose_metrics(rig.model,pose,reference,side)
            admitted = (support['feasible'] and fit['penetration_m'] <= .0002
                and fit['foot_error_m'] <= .001 and fit['com_error_m'] <= .001
                and metrics['posture_pass'] and metrics['stance_ankle_margin_rad'] >= .04
                and (not lift or metrics['whole_foot_clearance_m'][side] >= .018))
            if not admitted:
                raise RuntimeError(f'Target-PD support admission failed for world {world}: {fit}, {support}')
            poses.append(pose);reports.append(dict(fit=fit, static=support,metrics=metrics,admitted=True))
        for pose in poses[:-1]:
            support = solve_target_support(rig, pose)
            if not support['feasible']:
                raise RuntimeError('Reference/clear-arm equilibrium target infeasible')
            feedforward.append(np.asarray(support['normalized_motor_torque'])/4.)
        # The airborne equilibrium may use only the designated support foot.
        feedforward.append(np.asarray(reports[-1]['static']['normalized_motor_torque'])/4.)
        bank[world] = dict(poses=poses, feedforward=feedforward, reports=reports)
    return bank


@torch.no_grad()
def evaluate(rig, anchor, kind, bank, out, *, quiet=False, release_gate='load', objective='pose', prepared=False, prepared_queue='cold', poses=(0,1), collect_demonstrations=False):
    from .predictive_target_support import PredictiveTargetSupport, SupportSequence, blend, capture_margin
    from .interval_target_support import IntervalTargetSupport
    from . import support_teacher_dataset
    env = make_environment(rig, anchor, 4. if quiet or prepared else 24.)
    worlds = list(poses) if quiet else [pose+8*side for side in (0,1) for pose in poses]
    live = np.zeros(16,dtype=bool);live[worlds] = True
    controller_type = IntervalTargetSupport if objective in ('interval','interval_com','interval_task') else PredictiveTargetSupport
    predictor = controller_type(env.task,objective=objective)
    sequences = {i:SupportSequence(*bank[i]['poses'], env.dt, release_gate) for i in worlds}
    if prepared:
        env.engine.reset_envs(torch.tensor(worlds),np.stack([bank[i]['poses'][3] for i in worlds]),np.zeros((len(worlds),rig.model.nv)))
        for sequence in sequences.values():
            sequence.phase,sequence.started = 'hold',0.
    initial_queue = []
    if prepared_queue == 'equilibrium':
        if not prepared:
            raise ValueError('Equilibrium queue is only a prepared-reset diagnostic')
        control = env.engine.policy_control
        control.set_reference(*env.task.sample_reference(env.task.offset))
        source = anchor(env.task.observations())
        source_targets = DelayedTargetControl33.target_positions(control,source).numpy()
        for world in worlds:
            target = bank[world]['poses'][3][control.q_indices.numpy()] + bank[world]['feedforward'][3]
            if (target < control.lower.numpy()).any() or (target > control.upper.numpy()).any():
                raise RuntimeError('Initial support target exceeds physical joint limits')
            if np.max(abs(target-source_targets[world])) > 1.:
                raise RuntimeError('Initial support target exceeds source-relative envelope')
            control.pending[:,world,:33] = torch.tensor(target,dtype=torch.float32)
            control.pending[:,world,33:66] = 0
            control.pending[:,world,66] = 1
            initial_queue.append(dict(world=world,target=target.tolist(),
                maximum_source_offset=float(np.max(abs(target-source_targets[world])))))
        control.active.copy_(control.pending[control.cursor])
        control.physics_control(env.engine._array('qpos'),env.engine._array('qvel'))
        for world in worlds:
            data = env.engine.datas[world]
            data.ctrl[:] = control.ctrl[world].numpy()
            mujoco.mj_forward(rig.model,data)
            env.engine._ground_force_cache[world] = env.engine._read_ground_forces(data)
    anchors = env.measurements()['foot_xy_m'].numpy().copy()
    rows = []; started = time.perf_counter(); diagnostics = []; teacher_rows = []
    seconds = 4. if quiet or prepared else 24.
    final_reason = {i:'duration' for i in worlds}
    for frame in range(int(np.ceil(seconds/env.dt))):
        now = frame*env.dt
        obs = env.task.observations()
        source = anchor(obs).numpy()
        control = env.engine.policy_control
        control.set_reference(*env.task.sample_reference(env.task.offset+env.task.steps*env.dt))
        source_targets = DelayedTargetControl33.target_positions(control,torch.tensor(source)).numpy()
        offset = np.zeros((16,33),dtype=np.float32)
        authority = np.zeros(16,dtype=np.float32)
        margins = np.full(16,-1.)
        requested = np.stack([env.task.reference.qpos[i%8].numpy() for i in range(16)])
        pre = env.measurements();prestand,presupport = flags(env,pre)
        loads = pre['foot_loads_n'].numpy()
        for world in worlds:
            if not live[world]:continue
            sequence = sequences[world];side = world//8
            support_share = loads[world,1-side]/max(1.,loads[world].sum())
            strict = (presupport[world] and loads[world,side]<5 and loads[world,1-side]>=30
                and pre['clearance_m'][world,side]>=.015
                and np.linalg.norm(pre['foot_xy_m'][world,1-side].numpy()-anchors[world,1-side])<=.02)
            if not quiet and not prepared:
                margins[world] = capture_margin(rig.model,env.engine.datas[world],1-side)
                sequence.update(now,support_share,float(pre['horizontal_speed_m_s'][world]),presupport[world],bool(strict),loads[world].min()>=30,margins[world])
                if sequence.finished:
                    live[world]=False;final_reason[world]=sequence.reason;continue
            if kind == 'source' or (now < 1. and not prepared):continue
            future = now + 2*env.dt
            goal = bank[world]['poses'][0] if quiet else sequence.goal(future)
            requested[world] = goal
            if kind == 'feedback':
                target,info = predictor.target(world,goal,source_targets[world])
                diagnostics.append(info)
            else:
                ff = bank[world]['feedforward']
                elapsed = future-sequence.started
                if sequence.phase=='clear':feed = blend(ff[0],ff[1],elapsed,1.5)
                elif sequence.phase=='transfer':feed = blend(ff[1],ff[2],elapsed,4.)
                elif sequence.phase=='lift':feed = blend(ff[2],ff[3],elapsed,1.)
                elif sequence.phase=='hold':feed = ff[3]
                elif sequence.phase=='land':feed = blend(ff[3],ff[2],elapsed,1.)
                elif sequence.phase=='return':feed = blend(ff[2],ff[0],elapsed,4.)
                else:feed = ff[0]
                target = goal[control.q_indices.numpy()]+feed
            amount = 1. if prepared else float(blend(0.,1.,now-1.,1.))
            authority[world] = amount
            offset[world] = np.clip(amount*(target-source_targets[world]),-1.,1.)
        if not live.any():break
        env.engine.active = live.copy()
        before = live.copy()
        if collect_demonstrations:
            learner_observation = support_teacher_dataset.policy_input(obs.numpy()[worlds],requested[worlds],authority[worlds])
        _,_,done,_ = env.step(torch.tensor(offset))
        post = env.last;stand,supported = flags(env,post)
        stand &= prestand;supported &= presupport
        qpos = env.engine.native_qpos().numpy().copy()
        qvel = np.stack([d.qvel.copy() for d in env.engine.datas])
        if env.engine.solver_warning[live].any() or not np.isfinite(qpos[live]).all():
            raise RuntimeError('Physical numerical failure')
        issued = control.pending[(control.cursor-1)%2,:,:33].numpy()
        expected = np.clip(source_targets+offset,control.lower.numpy(),control.upper.numpy())
        assert np.max(abs(issued-expected))<2e-6
        assert (abs(control.ctrl.numpy()[:,rig.actuators])<=rig.action_scale+1e-4).all()
        if collect_demonstrations:
            teacher_rows.append(dict(observation=learner_observation,action=offset[worlds].copy(),
                source_targets=source_targets[worlds].copy(),issued_targets=issued[worlds].copy()))
        rows.append(dict(time=now+env.dt,alive=before[worlds].copy(),qpos=qpos[worlds],qvel=qvel[worlds],
            loads=post['foot_loads_n'].numpy()[worlds],clearance=post['clearance_m'].numpy()[worlds],
            xy=post['foot_xy_m'].numpy()[worlds],com=post['com_m'].numpy()[worlds],stand=stand[worlds],
            supported=supported[worlds],issued=issued[worlds].copy(),phase=np.array([sequences[i].phase for i in worlds]),
            capture_margin_before=margins[worlds],requested_qpos=requested[worlds],
            done=done.numpy()[worlds].copy()))
        for world in worlds:
            if done[world]==1:
                final_reason[world]='physical_failure';live[world]=False
            elif not quiet and sequences[world].phase=='settle' and now-sequences[world].started>=2.:
                final_reason[world]='sequence_complete';live[world]=False
        if frame and frame%120==0:
            print(out.name,round(now,2),{i:sequences[i].phase for i in worlds if live[i]},flush=True)
        if not live.any():break
    z={k:np.stack([r[k] for r in rows]) for k in rows[0]}
    out.mkdir();np.savez_compressed(out/'trace.npz',**z)
    cases=[]
    for index,world in enumerate(worlds):
        side=world//8;alive=z['alive'][:,index]
        valid=alive&z['supported'][:,index]&(z['loads'][:,index,side]<5)&(z['loads'][:,index,1-side]>=30)
        valid &= z['clearance'][:,index,side]>=.015
        valid &= np.linalg.norm(z['xy'][:,index,1-side]-anchors[world,1-side],axis=-1)<=.02
        last=np.flatnonzero(alive)[-1]
        settled=last+1>=int(np.ceil(2/env.dt)) and z['stand'][last+1-int(np.ceil(2/env.dt)):last+1,index].all()
        final=bool(settled and final_reason[world]!='physical_failure')
        duration=longest(valid)*env.dt
        cases.append(dict(pose=world%8,swing_side=side,reason=final_reason[world],final_stand=final,
            sequence_completed=final_reason[world]=='sequence_complete',
            whole_interval_stand=bool(z['stand'][alive,index].all()),single_support_seconds=duration,
            reached_hold=sequences[world].reached_hold,
            entry_hold_return=bool(duration>=1 and final and final_reason[world]=='sequence_complete'),
            prepared_hold=bool(prepared and duration>=1 and final_reason[world]!='physical_failure'),
            peak_support_share=float((z['loads'][alive,index,1-side]/np.maximum(z['loads'][alive,index].sum(-1),1)).max())))
    dataset = support_teacher_dataset.export(out,teacher_rows,z['alive'],cases,worlds,control.lower.numpy(),control.upper.numpy()) if collect_demonstrations else None
    report=dict(controller=kind,quiet=quiet,prepared_reset=prepared,release_gate=release_gate,objective=objective,cases=cases,wall_seconds=time.perf_counter()-started,
        worlds=worlds,teacher_dataset=dataset,
        prepared_queue=prepared_queue,initial_queue=initial_queue,
        interval_corrections=getattr(predictor,'corrections',0),
        maximum_interval_cost_ratio=getattr(predictor,'maximum_cost_ratio',None),
        prepared_hold_passes=sum(c['prepared_hold'] for c in cases),
        successes=sum(c['entry_hold_return'] for c in cases),final_stand=sum(c['final_stand'] for c in cases),
        private_prediction_steps=predictor.private_steps,diagnostic_queries=len(diagnostics),
        max_conversion_error=max((d['conversion_error'] for d in diagnostics),default=0.),trace_sha256=digest(out/'trace.npz'))
    save(out/'report.json',report);print(out.name,report,flush=True)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--release-gate',choices=('load','capture'),default='load')
    parser.add_argument('--quiet-evidence',type=Path)
    parser.add_argument('--equilibrium-evidence',type=Path)
    parser.add_argument('--objective',choices=('pose','task','interval','interval_com','interval_task'),default='pose')
    parser.add_argument('--prepared',action='store_true',help='Diagnostic lifted-pose resets, not entry trials')
    parser.add_argument('--prepared-queue',choices=('cold','equilibrium'),default='cold')
    parser.add_argument('--com-shift',type=float,default=.11)
    parser.add_argument('--lean-degrees',type=float,default=0.)
    parser.add_argument('--geometry-evidence',type=Path)
    parser.add_argument('--feedback-only',action='store_true',help='Skip equilibrium entry comparator; retain fresh quiet gates')
    parser.add_argument('--all-poses',action='store_true')
    parser.add_argument('--collect-demonstrations',action='store_true')
    args=parser.parse_args();activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    if args.prepared_queue!='cold' and not args.prepared:
        parser.error('--prepared-queue equilibrium requires --prepared')
    if args.collect_demonstrations and (not args.all_poses or not args.feedback_only or args.prepared):
        parser.error('--collect-demonstrations requires --all-poses --feedback-only and ordinary resets')
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from . import predictive_target_support, support_skill_physical_teacher, static_support, interval_target_support, screen_support_geometry, support_teacher_dataset
    torch.set_num_threads(1)
    rig=Rig.load(FOUNDATION/'dummy.xml',policy_exclude=())
    original=json.loads((FOUNDATION/'protocol.json').read_text())
    checkpoint=PARENTS/'301027/varied/final/model.pt'
    anchor=load_actor(rig,FOUNDATION/'stationary_reference.npz',original,checkpoint).eval().requires_grad_(False)
    env=make_environment(rig,anchor,24.)
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'sources').mkdir()
    root=Path.cwd().resolve();sources={}
    for module in tuple(sys.modules.values()):
        path=Path(getattr(module,'__file__','') or '').resolve()
        if path.is_file() and path.suffix=='.py' and path.is_relative_to(root) and '.venv' not in path.parts:
            rel=path.relative_to(root);sources[str(rel)]=digest(path)
            target=args.out/'sources'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    inputs=[rig.path,FOUNDATION/'stationary_reference.npz',FOUNDATION/'stationary_reference.json',checkpoint,checkpoint.with_suffix('.amp.pt')]
    if args.geometry_evidence:
        geometry=json.loads((args.geometry_evidence/'report.json').read_text())
        selected=geometry['selected']
        if not geometry['completed'] or selected is None or selected['shift_m']!=args.com_shift or selected['lean_degrees']!=args.lean_degrees:
            raise RuntimeError('Requested geometry does not match the admitted selection')
        for path,h in json.loads((args.geometry_evidence/'protocol.json').read_text())['inputs'].items():
            if digest(path)!=h:raise RuntimeError('Geometry input changed')
        inputs.extend(args.geometry_evidence/name for name in ('protocol.json','report.json','final-verification.json'))
    prior=None
    equilibrium=None
    if args.quiet_evidence:
        prior=json.loads((args.quiet_evidence/'report.json').read_text())
        assert prior['completed'] and prior['quiet_admitted']
        previous=json.loads((args.quiet_evidence/'protocol.json').read_text())
        for path,h in previous['inputs'].items():assert digest(path)==h
        inputs.extend([args.quiet_evidence/name for name in ('protocol.json','report.json','final-verification.json')])
    if args.equilibrium_evidence:
        equilibrium=json.loads((args.equilibrium_evidence/'report.json').read_text())
        previous=json.loads((args.equilibrium_evidence/'protocol.json').read_text())
        assert equilibrium['completed'] and previous['release_gate']==args.release_gate
        for path,h in previous['inputs'].items():assert digest(path)==h
        inputs.extend([args.equilibrium_evidence/name for name in ('protocol.json','report.json','final-verification.json')])
    poses=list(range(8)) if args.all_poses else [0,1]
    worlds=[pose+8*side for side in (0,1) for pose in poses]
    quiet_cases=0 if prior or args.prepared else 2*len(poses)
    entry_arms=1 if args.prepared or args.feedback_only or equilibrium else 2
    protocol=dict(schema='single_support_feedback_pilot_v1',parent=301027,poses=poses,sides=[0,1],
        native_only=True,training_updates=0,
        maximum_physical_cases=quiet_cases+entry_arms*len(worlds),
        collect_demonstrations=args.collect_demonstrations,
        feedback_only=args.feedback_only,control_radius_rad=1.,
        prepared_reset=args.prepared,
        prepared_queue=args.prepared_queue,
        target_admission='Necessary quasi-static support under clipped position-target torque bounds at zero velocity; source-relative bounds may be narrower; does not certify dynamic stability',
        prepared_queue_contract='cold: two invalid slots; equilibrium: both slots initialized from admitted support torque targets, no time advance, normal delay retained for new commands',
        objective=args.objective,equilibrium_evidence=str(args.equilibrium_evidence) if equilibrium else None,
        release_gate=args.release_gate,quiet_evidence=str(args.quiet_evidence) if prior else None,
        capture_release='Actual stance-floor contact hull margin >=5 mm; support share >=2/3; other timing/posture gates unchanged',
        arm_clearance_m=.10,com_shift_m=args.com_shift,lean_degrees=args.lean_degrees,
        geometry_evidence=str(args.geometry_evidence) if args.geometry_evidence else None,
        whole_foot_clearance_gate_m=.015,hold_gate_seconds=1.,
        support_drift_gate_m=.02,source_warmup_seconds=1.,controller_blend_seconds=1.,
        controller_objective='Pose: one-Hz generalized acceleration feedback, root weight 4/motors 0.6; task adds COM (weight 10, 1 Hz), both feet positions (weight 8, 2 Hz) and orientations (weight 1, 2 Hz); two forward refinements',
        interval_objective='One actual four-substep PD rollout; finite target derivatives 1e-4 rad; one bounded 0.05 rad correction with forward backtracking; original pose objective and motor constraints',
        interval_com_objective='Same interval refinement plus COM acceleration tracking, one Hz/weight 10, including Jacobian velocity bias; no added foot tracking',
        interval_task_objective='Same interval refinement with the full task objective: COM and both foot positions/orientations with Jacobian velocity bias',
        gate='All quiet feedback cases must retain final and whole-interval Stand before entry trials',
        sources=sources,inputs={str(p):digest(p) for p in inputs})
    save(args.out/'protocol.json',protocol)
    save(args.out/'transport-check.json',check_transport(env))
    env.reset();bank=target_bank(rig,env,worlds,com_shift=args.com_shift,lean_degrees=args.lean_degrees)
    if args.geometry_evidence:
        selected_variant=next(v for v in geometry['variants'] if v['shift_m']==args.com_shift and v['lean_degrees']==args.lean_degrees)
        for case in selected_variant['cases']:
            world=case['pose']+8*case['swing_side']
            if not case['admitted'] or not np.allclose(bank[world]['poses'][3],case['qpos'],rtol=0,atol=2e-6):
                raise RuntimeError('Generated lifted pose differs from admitted geometry')
    save(args.out/'target-admission.json',{str(i):v['reports'] for i,v in bank.items()})
    results={}
    if not prior and not args.prepared:
        for kind in ('source','feedback'):
            results['quiet-'+kind]=evaluate(rig,anchor,kind,bank,args.out/('quiet-'+kind),quiet=True,objective=args.objective,poses=poses)
    quiet=prior['results'] if prior else results
    admitted=not args.prepared and all(c['final_stand'] and c['whole_interval_stand'] for k,r in quiet.items() if k.startswith('quiet-') for c in r['cases'])
    if admitted or args.prepared:
        for kind in ('equilibrium','feedback'):
            if kind=='equilibrium' and (equilibrium or args.prepared or args.feedback_only):continue
            label=('prepared-' if args.prepared else 'entry-')+kind
            results[label]=evaluate(rig,anchor,kind,bank,args.out/label,release_gate=args.release_gate,objective=args.objective,prepared=args.prepared,prepared_queue=args.prepared_queue,
                poses=poses,collect_demonstrations=args.collect_demonstrations)
    for path,h in {**protocol['sources'],**protocol['inputs']}.items():
        if digest(path)!=h:raise RuntimeError(f'Source/input changed: {path}')
    save(args.out/'report.json',dict(completed=True,quiet_admitted=admitted if not args.prepared else None,prepared_reset=args.prepared,results=results,training_updates=0,arena_changed=False))
    save(args.out/'final-verification.json',dict(inputs_unchanged=True,outputs={str(p.relative_to(args.out)):digest(p)
        for p in sorted(args.out.rglob('*')) if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
