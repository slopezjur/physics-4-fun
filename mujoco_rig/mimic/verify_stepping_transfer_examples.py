"""Independent saved-data checks for bounded physical transfer demonstrations."""
import argparse
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
import torch
from torch.nn import functional as F

from util.torch_util import quat_to_tan_norm
from .run_learned_support import PARENT
from .run_support_skill_study import FOUNDATION, digest, save
from .verify_stepping_skill import arrays, check_curriculum, check_geometry, close, equal, read


OUT = Path('logs/mimickit-training/stepping-transfer-examples-20261009-01')
PILOT = [0, 1, 8, 9]
TRAIN = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]


def controls(z, report, contract, initial):
    task, count = contract['task'], len(z['time'])
    live = z['alive']
    expected_first = np.zeros(16, bool)
    expected_first[report['worlds']] = True
    equal(live[0], expected_first, 'selected physical worlds')
    equal(live[1:], live[:-1] & ~z['terminal'][:-1], 'termination chronology')
    equal(z['active'], live, 'native activity')
    close(z['time'], (np.arange(count)+1)*report['dt'], 'control clock', 1e-10)
    assert live.dtype == np.bool_ and not z['force'].any() and not z['warning'].any()
    radius = np.full(33, .1, np.float32)
    radius[task['leg_indices']] = .5
    equal(task['radius_rad'], radius, 'unchanged motor envelopes')
    control = task['controller']
    lower, upper = (np.asarray(control[key], np.float32) for key in
                    ('target_lower_radians', 'target_upper_radians'))
    authority = z['command_before'][..., 13].astype(bool)
    equal(z['authority'], authority, 'commanded authority')
    latent = np.where(authority[..., None], z['sampled_latent'], 0)
    latent[~live] = z['previous_latent'][~live]
    equal(z['latent'], latent, 'realized latent')
    desired = radius*np.tanh(latent)
    close(z['desired_addition'], desired, 'latent to desired correction', 8e-8)
    addition = z['previous']+np.clip(desired-z['previous'], -.02, .02)
    addition[~live] = z['previous'][~live]
    close(z['addition'], addition, 'bounded correction', 1e-7)
    assert (np.abs(z['addition']) <= radius+1e-7).all()
    assert (np.abs(z['addition']-z['previous']) <= .0200001).all()
    close(z['issued'], np.clip(z['nominal']+z['addition'], lower, upper), 'issued target', 1e-7)
    equal(z['source_before'], z['source_action'], 'same issue-time source action')
    assert not z['previous'][0].any() and not z['previous_latent'][0].any()
    equal(z['previous'][1:], z['addition'][:-1], 'correction memory')
    equal(z['previous_latent'][1:], z['latent'][:-1], 'latent memory')
    history = np.concatenate((z['balance_features'][:-1, :, None], z['history_before'][:-1, :, :2]), 2)
    history[~live[:-1]] = z['history_before'][:-1][~live[:-1]]
    equal(z['history_before'][1:], history, 'measured history')
    equal(z['history_before'][0, :, 0], z['history_before'][0, :, 2], 'reset history')
    assert not z['active_target'][:2].any()
    equal(z['active_target'][2:, :, :33], z['issued'][:-2], 'two-interval issue delay')
    assert not z['active_target'][..., 33:66].any()
    assert z['active_target'][2:, :, 66].all()
    equal(z['pre_native_qpos'][0].astype(np.float32), initial['qpos'], 'initial physical pose')
    equal(z['pre_native_qvel'][0].astype(np.float32), initial['qvel'], 'initial physical speed')
    equal(z['pre_native_qpos'][1:].astype(np.float32), z['qpos'][:-1], 'native pose chronology')
    equal(z['pre_native_qvel'][1:].astype(np.float32), z['qvel'][:-1], 'native speed chronology')
    for key, pre in (('qpos', 'pre_native_qpos'), ('qvel', 'pre_native_qvel')):
        equal(z[key][~live], z[pre].astype(np.float32)[~live], key+' inactive freeze')
    assert not z['addition'][live & ~authority].any()
    cost = .005*((z['addition']/radius).astype(np.float64)**2).mean(-1)*live
    close(z['action_cost'], cost, 'action cost', 2e-8)
    assert report['active_world_intervals'] == int(live.sum())
    return dict(active_world_intervals=int(live.sum()),
                maximum_correction_rad=float(np.abs(z['addition'][live]).max()),
                maximum_slew_rad=float(np.abs(z['addition']-z['previous']).max()))


@torch.no_grad()
def observations(z, initial, task, state, reference, model):
    raw = torch.from_numpy(z['raw_observations'])
    assert raw.shape == (*z['alive'].shape, 374) and z['features'].shape == (*z['alive'].shape, 528)
    normalized = ((raw-state['_obs_norm._mean'])/state['_obs_norm._std']).clamp(-10, 10)
    expected = torch.cat((normalized,
        torch.from_numpy(z['history_before']).flatten(-2).clamp(-20, 20),
        torch.from_numpy(z['previous'])/torch.tensor(task['radius_rad']),
        torch.from_numpy(z['previous_latent']), torch.from_numpy(z['command_before'])), -1)
    equal(z['features'], expected.numpy(), 'exact issue-time 528 inputs')
    equal(z['features'][0], initial['features'], 'initial learner inputs')
    equal(z['raw_observations'][0], initial['raw_observations'], 'initial source inputs')
    q = torch.from_numpy(z['pre_native_qpos'].astype(np.float32))
    v = torch.from_numpy(z['pre_native_qvel'].astype(np.float32))
    equal(raw[..., :3].numpy(), q[..., :3].numpy(), 'raw root position')
    equal(raw[..., 3:9].numpy(), quat_to_tan_norm(q[..., [4, 5, 6, 3]]).numpy(), 'raw root orientation')
    equal(raw[..., 9:12].numpy(), v[..., :3].numpy(), 'raw root translation velocity')
    angular = np.concatenate((initial['angular'][None], z['angular'][:-1]), 0)
    equal(raw[..., 12:15].numpy(), angular, 'raw world angular velocity')
    equal(raw[..., 15:54].numpy(), q[..., 7:].numpy(), 'raw joint positions')
    equal(raw[..., 54:93].numpy(), (v[..., 6:]*.1).numpy(), 'raw joint velocities')
    loads = np.concatenate((initial['loads'][None], z['loads'][:-1]), 0)
    close(raw[..., 93:95].numpy(), loads*np.float32(.001), 'raw solved contact loads', 2e-7)
    slots = np.concatenate((z['issued'], np.zeros_like(z['issued']), np.ones((*z['alive'].shape, 1), np.float32)), -1)
    oldest = np.concatenate((np.zeros((2, 16, 67), np.float32), slots[:-2]), 0)
    newest = np.concatenate((np.zeros((1, 16, 67), np.float32), slots[:-1]), 0)
    equal(raw[..., 95:229].numpy(), np.concatenate((oldest, newest), -1), 'observed ordered queue')
    reference = torch.from_numpy(reference.astype(np.float32))[torch.arange(16) % 8]
    future = torch.cat((reference[:, :3], quat_to_tan_norm(reference[:, [4, 5, 6, 3]]), reference[:, 7:]), -1).repeat(1, 3)
    equal(raw[..., 229:373].numpy(), np.broadcast_to(future.numpy(), raw[..., 229:373].shape), 'immutable future references')
    assert not raw[..., 373].any()
    control = task['controller']
    target = reference[:, control['target_qpos_indices']].numpy()
    nominal = np.clip(target[None]+np.float32(.25)*np.clip(z['source_before'], -1, 1),
        np.asarray(control['target_lower_radians'], np.float32), np.asarray(control['target_upper_radians'], np.float32))
    equal(z['nominal'], nominal, 'source-relative target')
    maximum_source_error = 0.
    for frame in range(len(raw)):
        value = normalized[frame]
        for layer in (0, 2):
            prefix = f'_model._actor_layers.{layer}'
            value = F.relu(F.linear(value, state[prefix+'.weight'], state[prefix+'.bias']))
        value = F.linear(value, state['_model._action_dist._mean_net.weight'], state['_model._action_dist._mean_net.bias'])
        value = (value*state['_a_norm._std']+state['_a_norm._mean']).clamp(-1, 1).numpy()
        close(z['source_before'][frame], value, 'frozen source forward', 3e-6, 2e-6)
        maximum_source_error = max(maximum_source_error, float(np.abs(value-z['source_before'][frame]).max()))
    center = z['feet'][..., :2].mean(2)
    height = np.maximum(z['com'][..., 2]-z['feet'][..., 2].min(2), .05)
    capture = z['com'][..., :2]+z['velocity'][..., :2]*np.sqrt(height/9.81)[..., None]
    body_mass = np.float32(model.body_mass[1:].sum())
    bf = np.concatenate(((z['com'][..., :2]-center)/.1, z['velocity'][..., :2]/.5,
        z['angular']/2, (capture-center)/.1, z['loads']/(body_mass*9.81),
        ((z['pelvis_height']-reference[:, 2].numpy())/.15)[..., None], z['pelvis_up'][..., None],
        ((z['feet'][..., :2]-z['com'][..., None, :2])/.2).reshape(*z['alive'].shape, 4)), -1)
    close(z['balance_features'], bf, 'physical balance features', 4e-6, 2e-6)
    return dict(exact_issue_time_features=True, raw_queue_and_state_verified=True,
                maximum_source_forward_error=maximum_source_error)


def teacher_labels(z, report, goals, task):
    active = z['alive']
    phase = np.argmax(z['command_before'][..., :7], -1)
    phase_age = np.concatenate((np.zeros((1, 16), np.float32), z['phase_age'][:-1]), 0)
    queried = active & ((phase == 2) | ((phase == 6) & (phase_age < report['ramp_seconds'])))
    equal(z['queried'], queried, 'teacher query eligibility')
    assert not z['sampled_latent'][~queried].any()
    assert queried.sum() == report['teacher_queries'] and report['live_query_state_unchanged'] is True
    radius = np.asarray(task['radius_rad'], np.float64)
    previous, nominal = z['previous'].astype(np.float64), z['nominal'].astype(np.float64)
    lo = np.maximum(-radius*(1-1e-6), previous-.02)
    hi = np.minimum(radius*(1-1e-6), previous+.02)
    lower, upper = (np.asarray(task['controller'][key], np.float32) for key in
                    ('target_lower_radians', 'target_upper_radians'))
    target_lower = np.clip(nominal+lo, lower, upper)
    target_upper = np.clip(nominal+hi, lower, upper)
    planned = z['proposed_target']
    assert planned.dtype == np.float64, 'Preserve private target precision for inverse labels'
    assert (planned[queried] >= target_lower[queried]-1e-7).all()
    assert (planned[queried] <= target_upper[queried]+1e-7).all()
    residual = np.clip(planned-nominal, lo, hi)
    close(z['requested_residual'][queried], residual[queried], 'target inversion residual', 1e-12)
    label = np.arctanh(np.clip(residual/radius, -1+1e-6, 1-1e-6)).astype(np.float32)
    equal(z['sampled_latent'][queried], label[queried], 'exact issued finite latent label')
    errors = np.abs(z['issued'][active].astype(float)-planned[active])
    maximum = float(errors.max())
    assert maximum <= 2e-6
    close(report['maximum_mapping_error_rad'], maximum, 'reported target mapping error', 1e-12)
    reference = arrays(FOUNDATION/'stationary_reference.npz')['qpos'][np.arange(16) % 8].astype(float)
    expected_goal = np.broadcast_to(reference, z['goal_qpos'].shape).copy()
    for world in report['worlds']:
        transfer = np.asarray(goals['goals'][str(world)], float)
        for frame in np.flatnonzero(queried[:, world]):
            start, end = (reference[world], transfer) if phase[frame, world] == 2 else (transfer, reference[world])
            end = end.copy()
            if np.dot(start[3:7], end[3:7]) < 0:
                end[3:7] *= -1
            fraction = np.clip((float(phase_age[frame, world])+2*report['dt'])/report['ramp_seconds'], 0, 1)
            fraction = fraction**3*(10+fraction*(-15+6*fraction))
            pose = start+(end-start)*fraction
            pose[3:7] /= np.linalg.norm(pose[3:7])
            expected_goal[frame, world] = pose
    close(z['goal_qpos'], expected_goal, 'private goal schedule', 2e-12)
    return dict(queries=int(queried.sum()), exact_finite_latent_labels=True,
                maximum_target_mapping_error_rad=maximum,
                private_solver_nonmutation='Sealed runtime assertions; no private integration rerun')


def outcomes(z, report):
    tail = int(np.ceil(2/report['dt']))
    rows = []
    assert len(report['cases']) == len(report['worlds'])
    for world, case in zip(report['worlds'], report['cases']):
        assert case['case'] == world and case['pose'] == world % 8 and case['side'] == world//8
        assert not case['heldout'] and world % 8 < 6
        survived = bool((z['alive'][:, world] & ~z['physical_failed'][:, world]).all())
        final = bool(survived and z['settled'][-tail:, world].all())
        success = bool(survived and final and z['task_success'][-1, world])
        transferred = np.flatnonzero(z['phase_after'][:, world] == 6)
        failure = np.flatnonzero(z['alive'][:, world] & z['physical_failed'][:, world])
        expected = dict(survived=survived, final_stand=final, success=success,
            transfer_completed=bool(len(transferred)),
            transfer_time_seconds=float(z['time'][transferred[0]]) if len(transferred) else None,
            physical_failure_seconds=float(z['time'][failure[0]]) if len(failure) else None,
            phase_max=int(z['phase_after'][:, world].max()))
        assert all(case[key] == value for key, value in expected.items()), (world, expected)
        rows.append(dict(case=world, **expected))
    for key, singular in (('successes', 'success'), ('survived', 'survived'), ('final_stand', 'final_stand')):
        assert report[key] == sum(row[singular] for row in rows)
    assert report['by_side'] == [sum(row['success'] for row in rows if row['case']//8 == side) for side in (0, 1)]
    return rows


def provenance(out):
    inputs, outputs = read(out/'inputs.json'), read(out/'outputs.json')
    for path, expected in inputs.items():
        path = Path(path)
        assert digest(path) == expected, ('input', path)
        if path.suffix == '.py' or path.name == 'STEPPING_TRANSFER_EXAMPLES_PLAN.md':
            assert digest(out/'sources'/path.name) == expected, ('source snapshot', path)
    for path, expected in outputs.items():
        target = (out/path).resolve()
        assert target.is_relative_to(out.resolve())
        assert digest(target) == expected, ('output', path)
    assert not list(out.rglob('*.pt')), 'Physical examples must not introduce checkpoints'
    for path in (PARENT, PARENT.with_suffix('.amp.pt'), FOUNDATION/'dummy.xml', FOUNDATION/'stationary_reference.npz'):
        assert str(path) in inputs
    for row in read(FOUNDATION/'protocol.json')['demonstrations']:
        assert inputs[str(Path(row['path']))] == row['sha256']
    return inputs, outputs


def audit(out):
    inputs, outputs = provenance(out)
    contract, summary = read(out/'contract.json'), read(out/'report.json')
    task = contract['task']
    assert task['observation_width'] == 528 and task['goal_width'] == 37
    assert task['authority_goal_index'] == 13 and task['goal_start'] == 491
    assert task['persistent_latent_rho'] == .9 and task['slew_rad_per_interval'] == .02
    assert task['source_actor_frozen'] and not task['external_force'] and not task['autonomous_hit_policy']
    assert task['terminal_bootstrap'] == 0 and task['episode_seconds'] == 12.
    assert contract['pilot_worlds'] == PILOT and contract['training_worlds'] == TRAIN
    assert contract['ramps_seconds'] == [2., 4.] and contract['reserved_poses_used'] is False
    control = task['controller']
    assert control['motor_count'] == 33 and control['residual_radians'] == .25
    assert control['pd_gain_per_torque_limit'] == 4. and control['pd_damping_time'] == .02
    model = mujoco.MjModel.from_xml_path(str(FOUNDATION/'dummy.xml'))
    assert not model.jnt_stiffness.any()
    assert task['dt'] == 4*model.opt.timestep
    names = [model.joint(int(joint)).name for joint in model.actuator_trnid[:, 0]]
    assert task['motor_names'] == names
    assert task['leg_indices'] == [i for i, name in enumerate(names) if name.startswith(('Thigh_', 'Shin_', 'Foot_', 'Toe_'))]
    state = torch.load(PARENT, map_location='cpu', weights_only=True)
    reference = arrays(FOUNDATION/'stationary_reference.npz')['qpos']
    reports, checks = {}, {}
    for path in sorted(out.glob('*/trace.npz')):
        z, report = arrays(path), read(path.with_name('report.json'))
        initial, goals = arrays(path.with_name('initial.npz')), read(path.with_name('goals.json'))
        assert all(np.isfinite(value).all() for value in z.values()), path
        assert report['dt'] == task['dt'] and report['seconds'] == 12. and report['onset_seconds'] == 1.
        assert report['level'] == 0 and report['quiet'] is False and report['distance_m'] == 0.
        assert report['com_transfer_m'] == .04 and len(z['time']) == int(np.ceil(12/task['dt']))
        check = controls(z, report, contract, initial)
        check_geometry(z, model)
        curriculum = check_curriculum(z, report, contract['curriculum'])
        features = observations(z, initial, task, state, reference, model)
        labels = teacher_labels(z, report, goals, task)
        cases = outcomes(z, report)
        reports[path.parent.name] = report
        checks[path.parent.name] = dict(controls=check, curriculum=curriculum, features=features, labels=labels, cases=cases)
    order = ['pilot-ramp2']
    assert reports['pilot-ramp2']['worlds'] == PILOT and reports['pilot-ramp2']['ramp_seconds'] == 2.
    if reports['pilot-ramp2']['successes'] != 4:
        order.append('pilot-ramp4')
        assert reports['pilot-ramp4']['worlds'] == PILOT and reports['pilot-ramp4']['ramp_seconds'] == 4.
    qualified = [reports[name] for name in order if min(reports[name]['by_side']) >= 1]
    ready, status = False, 'physical_pilot_rejected'
    if qualified:
        selected = max(qualified, key=lambda report: report['successes'])
        order.append('training-panel')
        expanded = reports['training-panel']
        assert expanded['worlds'] == TRAIN and expanded['ramp_seconds'] == selected['ramp_seconds']
        ready = expanded['successes'] >= 9 and min(expanded['by_side']) >= 4
        status = 'physical_admitted_ready_for_exact_label_fit' if ready else 'physical_expansion_rejected'
    assert set(reports) == set(order)
    assert summary['physical_panels'] == [reports[name] for name in order]
    assert summary['status'] == status and summary['conditional_training_ready'] == ready
    assert summary['neural_training_ran'] is False and summary['arena_changed'] is False
    cases = sum(len(report['worlds']) for report in reports.values())
    assert cases <= contract['maximum_teacher_cases']
    return dict(passed=True, inputs=len(inputs), sealed_outputs=len(outputs), traces=len(reports),
        physical_cases=cases, checks=checks, status=status, conditional_training_ready=ready,
        training_steps=0, physics_integration_steps=0, arena_changed=False,
        scope='Saved physical examples and issue-time latent labels; independent controls, geometry, '
              'curriculum and frozen-source inference. Private teacher queries are not rerun; '
              'their nonmutation is supported by the sealed runtime checks. No neural capability claim.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    out = parser.parse_args().out
    assert not (out/'audit.json').exists()
    torch.set_num_threads(1)
    source, archive = Path(__file__), out/'audit-harness'
    archive.mkdir(exist_ok=False)
    shutil.copy2(source, archive/source.name)
    source_hash = digest(source)
    try:
        report = audit(out)
        assert digest(source) == source_hash
        report['audit_source_sha256'] = source_hash
    except Exception as error:
        save(out/'audit.json', dict(passed=False, error_type=type(error).__name__, error=str(error),
                                  audit_source_sha256=source_hash))
        raise
    save(out/'audit.json', report)
    print(json.dumps({key:report[key] for key in ('passed', 'physical_cases', 'status')}), flush=True)


if __name__ == '__main__':
    main()
