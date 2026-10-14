"""Bounded physical transfer examples through the frozen stepping interface."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .rig import Rig
from .run_learned_support import PARENT
from .run_support_skill_study import FOUNDATION, digest, save
from .stand_experiment_evidence import load_actor
from .stepping_curriculum import Phase
from .stepping_skill_core import StepSkillEnvironment
from .stepping_transfer_teacher import SteppingTransferTeacher, fit_transfer_goal
from .predictive_target_support import blend_pose
from .target_control33 import DelayedTargetControl33

OUT = Path('logs/mimickit-training/stepping-transfer-examples-20261009-01')
PILOT = [0, 1, 8, 9]
TRAIN = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]


def setup(out):
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=False)
    (out/'sources').mkdir()
    excluded = {'verify_stepping_transfer_examples.py', 'test_stepping_transfer_teacher.py'}
    sources = [p for p in sorted(Path(__file__).parent.glob('*.py')) if p.name not in excluded]
    sources.append(Path(__file__).with_name('STEPPING_TRANSFER_EXAMPLES_PLAN.md'))
    for path in sources:
        shutil.copy2(path, out/'sources'/path.name)
    protocol = json.loads((FOUNDATION/'protocol.json').read_text())
    inputs = sources + [PARENT, PARENT.with_suffix('.amp.pt'), FOUNDATION/'dummy.xml',
        FOUNDATION/'protocol.json', FOUNDATION/'stationary_reference.npz',
        FOUNDATION/'stationary_reference.json',
        Path('Scenes/RL/Isaac3/MuJoCo/MimicPerturbArena.tscn')]
    for item in protocol['demonstrations']:
        path = Path(item['path'])
        assert digest(path) == item['sha256']
        inputs.append(path)
    hashes = {str(path): digest(path) for path in inputs}
    save(out/'inputs.json', hashes)
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    anchor = load_actor(rig, FOUNDATION/'stationary_reference.npz', protocol, PARENT)
    anchor.eval().requires_grad_(False)
    return rig, anchor, hashes


@torch.no_grad()
def panel(rig, anchor, folder, worlds, ramp, onset=1.):
    folder.mkdir()
    env = StepSkillEnvironment(rig, anchor)
    env.reset(level=0, side=torch.arange(16)//8, distance=0., onset=onset)
    teacher = SteppingTransferTeacher(env.task)
    reference = env.task.reference.qpos[torch.arange(16)%8].numpy().astype(float)
    goals, fits = {}, {}
    for world in worlds:
        goals[world], fits[world] = fit_transfer_goal(rig.model, reference[world],
            env.curriculum.transfer_target[world].numpy(), lean_radians=np.deg2rad(4.))
        assert fits[world]['foot_error_m'] <= .001, fits[world]
        assert fits[world]['com_error_m'] <= .001, fits[world]
        assert fits[world]['penetration_m'] <= .0002, fits[world]
    save(folder/'goals.json', dict(fits=fits, goals={str(k):v.tolist() for k,v in goals.items()},
        target_xy=env.curriculum.transfer_target.numpy().tolist()))
    np.savez_compressed(folder/'initial.npz', qpos=env.engine.native_qpos().numpy(),
        qvel=env.engine._array('qvel').numpy(), features=env.features().numpy(),
        raw_observations=env.task.observations().numpy(),
        **{key:value.numpy() for key,value in env.measurement().items()})
    active = torch.zeros(16, dtype=torch.bool)
    active[worlds] = True
    rows, started = [], time.perf_counter()
    max_mapping_error = 0.
    for frame in range(int(np.ceil(12/env.dt))):
        before = active.clone()
        features = env.features().clone()
        raw = env.task.observations().clone()
        source = anchor(raw)
        control = env.engine.policy_control
        # Set the same current reference that task.step will install; this is
        # normal controller input, not a change to the task's reference motion.
        control.set_reference(*env.task.sample_reference(env.task.offset+env.task.steps*env.dt))
        nominal = DelayedTargetControl33.target_positions(control, source)
        latent = torch.zeros(16, 33)
        planned = (nominal+env.previous+(0-env.previous).clamp(-.02,.02)).clamp(control.lower, control.upper)
        planned_exact = planned.numpy().astype(np.float64)
        requested_residual = np.zeros((16,33), dtype=np.float64)
        residual_lower = np.zeros_like(requested_residual)
        residual_upper = np.zeros_like(requested_residual)
        teacher_metrics = np.zeros((16,4), dtype=np.float64)
        queried = np.zeros(16, dtype=bool)
        requested = reference.copy()
        before_q = np.stack([data.qpos.copy() for data in env.engine.datas])
        before_v = np.stack([data.qvel.copy() for data in env.engine.datas])
        for world in worlds:
            if not active[world]:
                continue
            phase = int(env.curriculum.phase[world])
            age = float(env.curriculum.phase_age[world])
            if phase == Phase.TRANSFER:
                requested[world] = blend_pose(reference[world], goals[world], age+2*env.dt, ramp)
            elif phase == Phase.SETTLE and age < ramp:
                requested[world] = blend_pose(goals[world], reference[world], age+2*env.dt, ramp)
            else:
                continue
            value, details = teacher.command(world, requested[world], nominal[world].numpy(),
                                             env.previous[world].numpy(), env.radius.numpy())
            latent[world] = torch.from_numpy(value)
            planned[world] = torch.tensor(details['planned_target'], dtype=torch.float32)
            planned_exact[world] = details['planned_target']
            requested_residual[world] = details['requested_residual']
            residual_lower[world] = details['residual_lower']
            residual_upper[world] = details['residual_upper']
            teacher_metrics[world] = [details['tracking_residual'], details['saturation'],
                details['interval_initial_cost'], details['interval_final_cost']]
            queried[world] = True
        reward, terminal, info = env.step(latent, active)
        assert torch.equal(source[active], info['source_action'][active])
        assert torch.equal(nominal[active], info['nominal'][active])
        mapping_error = float((info['issued'][active]-planned[active]).abs().max()) if active.any() else 0.
        max_mapping_error = max(max_mapping_error, mapping_error)
        assert mapping_error <= 2e-6, mapping_error
        rows.append(dict(time=np.array((frame+1)*env.dt), alive=before.numpy().copy(),
            reward=reward.numpy().copy(), terminal=terminal.numpy().copy(),
            features=features.numpy(), raw_observations=raw.numpy(), source_before=source.numpy(),
            proposed_target=planned_exact, requested_residual=requested_residual,
            residual_lower=residual_lower, residual_upper=residual_upper,
            teacher_metrics=teacher_metrics, queried=queried, goal_qpos=requested,
            pre_native_qpos=before_q, pre_native_qvel=before_v,
            **{key:value.numpy().copy() for key,value in info.items()
               if isinstance(value,torch.Tensor) and key not in ('reward','terminal')}))
        active &= ~terminal
        if (frame+1) % 120 == 0:
            print('PHYSICAL', folder.name, round((frame+1)*env.dt,3),
                  'alive', int(active.sum()), 'transferred', int(env.curriculum.completed[:,0].sum()), flush=True)
    z = {key:np.stack([row[key] for row in rows]) for key in rows[0]}
    tail = int(np.ceil(2/env.dt))
    cases = []
    for world in worlds:
        live = z['alive'][:,world]
        survived = bool((live & ~z['physical_failed'][:,world]).all())
        final_stand = bool(survived and z['settled'][-tail:,world].all())
        success = bool(survived and final_stand and z['task_success'][-1,world])
        failed_frames = np.flatnonzero(live & z['physical_failed'][:,world])
        transfer_frames = np.flatnonzero(z['phase_after'][:,world] == int(Phase.SETTLE))
        cases.append(dict(case=world, pose=world%8, side=world//8, heldout=False,
            survived=survived, final_stand=final_stand, success=success,
            transfer_completed=bool(len(transfer_frames)),
            transfer_time_seconds=float(z['time'][transfer_frames[0]]) if len(transfer_frames) else None,
            physical_failure_seconds=float(z['time'][failed_frames[0]]) if len(failed_frames) else None,
            phase_max=int(z['phase_after'][:,world].max())))
    report = dict(worlds=worlds, cases=cases, level=0, quiet=False, onset_seconds=onset,
        distance_m=0., com_transfer_m=.04, ramp_seconds=ramp, seconds=12., dt=env.dt,
        successes=sum(row['success'] for row in cases), survived=sum(row['survived'] for row in cases),
        final_stand=sum(row['final_stand'] for row in cases),
        by_side=[sum(row['success'] for row in cases if row['side']==side) for side in (0,1)],
        teacher_queries=int(z['queried'].sum()), private_steps=int(teacher.private_steps),
        active_world_intervals=int(z['alive'].sum()), maximum_mapping_error_rad=max_mapping_error,
        wall_seconds=time.perf_counter()-started, controller='private feedback teacher, not neural',
        live_query_state_unchanged=True)
    np.savez_compressed(folder/'trace.npz', **z)
    save(folder/'report.json', report)
    print('PANEL', folder.name, {k:v for k,v in report.items() if k!='cases'}, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    rig, anchor, hashes = setup(args.out)
    start = time.perf_counter()
    env = StepSkillEnvironment(rig, anchor)
    save(args.out/'contract.json', dict(task=env.contract(), curriculum=asdict(env.curriculum.config),
        pilot_worlds=PILOT, training_worlds=TRAIN, ramps_seconds=[2.,4.],
        maximum_teacher_cases=44, reserved_poses_used=False))
    reports = [panel(rig, anchor, args.out/'pilot-ramp2', PILOT, 2.)]
    if reports[0]['successes'] != len(PILOT):
        reports.append(panel(rig, anchor, args.out/'pilot-ramp4', PILOT, 4.))
    qualified = [r for r in reports if min(r['by_side']) >= 1]
    status = 'physical_pilot_rejected'
    training_ready = False
    if qualified:
        chosen = max(qualified, key=lambda row:row['successes'])
        expanded = panel(rig, anchor, args.out/'training-panel', TRAIN, chosen['ramp_seconds'])
        reports.append(expanded)
        training_ready = expanded['successes'] >= 9 and min(expanded['by_side']) >= 4
        status = 'physical_admitted_ready_for_exact_label_fit' if training_ready else 'physical_expansion_rejected'
    for path, expected in hashes.items():
        assert digest(path) == expected, path
    save(args.out/'report.json', dict(status=status, physical_panels=reports,
        conditional_training_ready=training_ready, neural_training_ran=False,
        skip_reason=None if training_ready else 'No admitted teacher examples for both sides under the fixed task gate',
        wall_seconds=time.perf_counter()-start, arena_changed=False))
    outputs = {str(p.relative_to(args.out)):digest(p) for p in sorted(args.out.rglob('*'))
               if p.is_file() and 'sources' not in p.relative_to(args.out).parts and p.name!='outputs.json'}
    save(args.out/'outputs.json', outputs)
    print('RESULT', status, 'training_ready', training_ready, flush=True)


if __name__ == '__main__':
    main()
