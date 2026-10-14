"""Independent saved-data audit of the commanded stepping curriculum."""
from pathlib import Path
import json
import shutil

import mujoco
import numpy as np
import torch

from .corrective_step_audit import foot_geometry, measure_steps
from .run_support_skill_study import FOUNDATION, digest, save


OUT = Path('logs/mimickit-training/stepping-skill-20261009-01')
TRAIN_WORLDS = np.array([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def close(actual, expected, label, atol=2e-6, rtol=0):
    assert np.allclose(actual, expected, atol=atol, rtol=rtol), label


def equal(actual, expected, label):
    assert np.array_equal(actual, expected), label


def longest(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return int(np.max(np.flatnonzero(edges == -1)-np.flatnonzero(edges == 1), initial=0))


def check_controls(z, contract):
    """Reconstruct commanded targets and memory without evaluating either actor."""
    env = contract['task']
    dt = env['dt']
    count = len(z['time'])
    close(z['time'], (np.arange(count)+1)*dt, 'control clock', 1e-10)
    active = z['alive']
    equal(z['active'], active, 'active mask')
    assert active.dtype == np.bool_ and active[0].all()
    equal(active[1:], active[:-1] & ~z['terminal'][:-1], 'terminal chronology')
    assert not z['force'].any() and not z['warning'].any()
    radius = np.asarray(env['radius_rad'], np.float32)
    assert radius.shape == (33,) and len(env['leg_indices']) == 16
    expected_radius = np.full(33, .1, np.float32)
    expected_radius[env['leg_indices']] = .5
    equal(radius, expected_radius, 'motor authority')
    assert env['slew_rad_per_interval'] == .02
    control = env['controller']
    lower = np.asarray(control['target_lower_radians'], np.float32)
    upper = np.asarray(control['target_upper_radians'], np.float32)
    command_before = z['command_before']
    authority = command_before[:, :, env['authority_goal_index']].astype(bool)
    equal(z['authority'], authority, 'authority observation')
    latent = np.where(authority[..., None], z['sampled_latent'], 0)
    latent[~active] = z['previous_latent'][~active]
    equal(z['latent'], latent, 'authority and inactive latent')
    desired = radius*np.tanh(latent)
    close(z['desired_addition'], desired, 'desired addition', 8e-8)
    addition = z['previous']+np.clip(desired-z['previous'], -.02, .02)
    addition[~active] = z['previous'][~active]
    close(z['addition'], addition, 'slew mapping', 1e-7)
    assert np.max(np.abs(z['addition'])-radius) <= 1e-7
    assert np.max(np.abs(z['addition']-z['previous'])) <= .0200001
    expected_cost = .005*((z['addition']/radius).astype(np.float64)**2).mean(-1)*active
    close(z['action_cost'], expected_cost, 'joint action cost', 2e-8)
    close(z['issued'], np.clip(z['nominal']+z['addition'], lower, upper), 'issued targets', 1e-7)
    assert not z['previous'][0].any() and not z['previous_latent'][0].any()
    equal(z['previous'][1:], z['addition'][:-1], 'addition memory')
    memory = np.where(active[:-1, :, None], latent[:-1], z['previous_latent'][:-1])
    equal(z['previous_latent'][1:], memory, 'latent memory')
    history = np.concatenate((z['balance_features'][:-1, :, None], z['history_before'][:-1, :, :2]), axis=2)
    history[~active[:-1]] = z['history_before'][:-1][~active[:-1]]
    equal(z['history_before'][1:], history, 'balance history')
    equal(z['history_before'][0, :, 0], z['history_before'][0, :, 2], 'initial history')
    assert not z['active_target'][:2, :, -1].any()
    equal(z['active_target'][2:, :, :33], z['issued'][:-2], 'two-interval target delay')
    assert not z['addition'][active & ~authority].any()
    for world in range(16):
        stopped = np.flatnonzero(~active[:, world])
        if len(stopped):
            first = stopped[0]
            for key in ('qpos', 'qvel'):
                equal(z[key][first:, world], np.broadcast_to(z[key][first-1, world], z[key][first:, world].shape), key+' inactive freeze')
    return dict(intervals=count, active_world_intervals=int(active.sum()),
                maximum_addition_rad=float(np.abs(z['addition']).max()),
                maximum_slew_rad=float(np.abs(z['addition']-z['previous']).max()))


def check_geometry(z, model):
    """Kinematics and independent Stand predicates; no integration or policy calls."""
    xy, clearance = foot_geometry(model, z['qpos'])
    close(z['xy'], xy, 'foot positions')
    close(z['clearance'], clearance, 'whole-foot clearance')
    close(z['feet'][..., :2], xy, 'measurement feet')
    close(z['pelvis_height'], z['qpos'][..., 2], 'pelvis height', 0)
    q = z['qpos']
    up = 1-2*(q[..., 4]**2+q[..., 5]**2)
    close(z['pelvis_up'], up, 'pelvis orientation', 2e-7)
    reference = arrays(FOUNDATION/'stationary_reference.npz')['qpos'].astype(np.float32)
    data = mujoco.MjData(model)
    feet = [model.body(f'Foot_{side}').id for side in ('L', 'R')]
    reset_feet, reset_com = np.empty((16, 2, 3)), np.empty((16, 3))
    for world in range(16):
        data.qpos[:] = reference[world % 8]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        reset_feet[world], reset_com[world] = data.xpos[feet], data.subtree_com[1]
    close(z['initial_feet'][0], reset_feet, 'native reset foot reference')
    close(z['initial_com'][0], reset_com, 'native reset COM reference')
    stand = ((z['pelvis_height'] >= .75) & (z['pelvis_up'] >= np.cos(np.deg2rad(15)))
             & (z['horizontal_speed_m_s'] <= .2) & (z['angular_speed_rad_s'] <= 1.)
             & (z['loads'].min(-1) >= 5.) & (z['non_foot_ground_contacts_n'].max(-1) <= 1.)
             & z['finite'] & (z['upper_body_tilt_degrees'] <= [20., 20., 30.]).all(-1)
             & (z['upper_body_clearance_deficit_m'] <= [.05, .06, .1]).all(-1))
    equal(z['measurement_stand'], stand, 'independent Stand predicates')
    equal(z['pre_stand'][1:], stand[:-1], 'pre-state Stand chronology')
    equal(z['stand'], stand & z['pre_stand'], 'whole-interval Stand')
    settled = stand & (z['loads'].min(-1) >= 30.)
    if len(settled) > 1:
        equal(z['settled'][1:], settled[1:] & settled[:-1], 'whole-interval settled')
    return xy, clearance


def check_collection(path, sigma, gamma=.995, lam=.98):
    z = arrays(path)
    for key in ('x', 'action', 'mean', 'logp', 'value', 'reward', 'terminal', 'failed', 'next_value', 'advantage'):
        assert z[key].shape[:2] == (64, 12), (path, key)
        assert np.isfinite(z[key]).all(), (path, key)
    assert z['action'].shape == z['mean'].shape == (64, 12, 33)
    assert z['terminal'].dtype == z['failed'].dtype == np.bool_
    assert not (z['failed'] & ~z['terminal']).any()
    equal(z['worlds'], TRAIN_WORLDS, 'training world identities')
    if 'sigma' in z:
        assert float(z['sigma']) == sigma
    assert np.isclose(sigma, .3*np.sqrt(1-.9**2), atol=1e-12, rtol=0)
    lp = (-.5*((z['action'].astype(float)-z['mean'])/sigma)**2-np.log(sigma)-.5*np.log(2*np.pi)).sum(-1)
    close(z['logp'], lp, '33-dimensional conditional density', 2e-5, 2e-6)
    carry = np.zeros(12)
    advantage = np.empty((64, 12))
    for frame in range(63, -1, -1):
        continuation = ~z['terminal'][frame]
        delta = z['reward'][frame].astype(float)+gamma*z['next_value'][frame]*continuation-z['value'][frame]
        carry = delta+gamma*lam*continuation*carry
        advantage[frame] = carry
    close(z['advantage'], advantage, 'terminal-zero-bootstrap GAE', 2e-4, 3e-6)
    close(z['returns'], (z['advantage']+z['value']).reshape(z['returns'].shape), 'GAE returns', 0)
    stored = z['advantage'].astype(float)
    normal = (stored-stored.mean())/max(stored.std(ddof=1), 1e-6)
    close(z['normalized_advantage'], normal, 'whole-batch advantage normalization', 3e-6, 2e-6)
    return dict(samples=768, terminal_samples=int(z['terminal'].sum()),
                failures=int(z['failed'].sum()), maximum_gae_error=float(np.abs(z['advantage']-advantage).max()))


def state_dict(path):
    return torch.load(path, weights_only=True, map_location='cpu')


def actor_forward(state, features):
    x = np.asarray(features, np.float32)
    value = x
    for layer in (0, 2, 4):
        value = value@state[f'net.{layer}.weight'].numpy().T+state[f'net.{layer}.bias'].numpy()
        if layer != 4:
            value = np.where(value >= 0, value, np.expm1(np.minimum(value, 0)))
    return x[..., 504:505]*(np.float32(.9)*x[..., 458:491]+np.float32(.1)*value)


def check_fit(folder, contract, previous=None):
    initial, final = state_dict(folder/'initial.pt'), state_dict(folder/'final.pt')
    assert initial.keys() == final.keys()
    assert all(torch.isfinite(value).all() for value in final.values())
    if previous is None:
        assert not initial['net.4.weight'].any() and not initial['net.4.bias'].any()
    else:
        previous_state = state_dict(previous/'final.pt')
        assert all(torch.equal(initial[key], previous_state[key]) for key in initial)
    assert initial['net.0.weight'].shape == (128, contract['input_width'])
    assert initial['net.4.weight'].shape == (33, 128)
    report, history = read(folder/'report.json'), read(folder/'history.json')
    equal(report['train_worlds'], TRAIN_WORLDS, 'fit training identities')
    assert report['collections'] == 64 and report['transitions'] == 49152
    assert report['episodic_terminal_bootstrap'] is False
    assert report['initial_hash'] == digest(folder/'initial.pt')
    assert report['final_hash'] == digest(folder/'final.pt')
    assert len(history) == 64
    accepted = rejected = 0
    for update, row in enumerate(history, 1):
        assert row['update'] == update and row['transitions'] == update*768
        assert 0 <= row['actor_updates'] <= 9 and row['rolled_back'] in (0, 1)
        assert row['actor_updates']+row['rolled_back'] <= 9
        assert len(row['accepted_kl']) == row['actor_updates']
        assert all(np.isfinite(kl) and 0 <= kl <= .01 for kl in row['accepted_kl'])
        assert np.isfinite(row['reward']) and np.isfinite(row['critic_loss'])
        assert 0 <= row['failures'] <= row['completed'] <= 768
        accepted += row['actor_updates']
        rejected += row['rolled_back']
    assert report['accepted_actor_updates'] == accepted and report['actor_rollbacks'] == rejected
    checks = {tag: check_collection(folder/f'{tag}-collection.npz', contract['sigma'],
                                   contract['gamma'], contract['gae_lambda']) for tag in ('first', 'last')}
    first = arrays(folder/'first-collection.npz')
    prediction = actor_forward(initial, first['x'].reshape(768, 528)).reshape(64, 12, 33)
    close(first['mean'], prediction, 'independent initial policy forward', 2e-6, 2e-6)
    return dict(transitions=49152, collections=64, accepted_actor_updates=accepted,
                rejected_actor_updates=rejected, nominal_actor_slots=576,
                critic_minibatches_from_sealed_loop=576, collections_checked=checks,
                initial_policy_prediction_max_error=float(np.abs(first['mean']-prediction).max()),
                optimizer_replayed=False, intermediate_collections_replayed=False)


def check_report(z, report, xy, clearance):
    dt = report['dt']
    assert len(z['time']) == int(np.ceil(report['seconds']/dt))
    tail = int(np.ceil(2/dt))
    mechanical = settled_steps = 0
    assert len(report['cases']) == 16
    for world, case in enumerate(report['cases']):
        assert case['case'] == world and case['pose'] == world % 8
        assert case['side'] == world//8 and case['heldout'] == (world % 8 >= 6)
        assert case['level'] == report['level']
        alive = z['alive'][:, world] & ~z['physical_failed'][:, world]
        survived = bool(alive.all())
        final_stand = bool(survived and z['settled'][-tail:, world].all())
        events = measure_steps(z['time'], xy[:, world], clearance[:, world], z['loads'][:, world],
                               alive, z['settled'][:, world],
                               hit_time=None if report['quiet'] else report['onset_seconds'])
        assert events == case['events']
        recovered = bool(final_stand and any(event['settled_window_start'] is not None for event in events))
        success = bool(survived and final_stand and z['task_success'][-1, world])
        if report['level'] == 2 and not report['quiet']:
            success &= recovered
        assert case['survived'] == survived and case['success'] == success
        assert case['final_stand'] == final_stand and case['recovered_with_step'] == recovered
        assert case['whole_stand'] == bool(survived and z['stand'][:, world].all())
        assert case['phase_max'] == int(z['phase_after'][:, world].max())
        mechanical += len(events)
        settled_steps += sum(event['settled_window_start'] is not None for event in events)
    for key in ('survived', 'whole_stand', 'final_stand', 'recovered_with_step'):
        assert report[key] == sum(case[key] for case in report['cases'])
    assert report['successes'] == sum(case['success'] for case in report['cases'])
    return dict(cases=16, measured_steps=mechanical, settled_steps=settled_steps,
                claimed_successes=report['successes'])


def gate(reports):
    rows = []
    for report in reports:
        train = [case for case in report['cases'] if case['pose'] < 6]
        counts = [sum(case['success'] for case in train if case['side'] == side) for side in (0, 1)]
        rows.append(dict(training_successes=sum(counts), by_side=counts,
                         passed=sum(counts) >= 9 and min(counts) >= 4))
    return dict(passed=all(row['passed'] for row in rows), conditions=rows, reserved_cases_used=False)


def check_stages(out, contract, reports):
    summary = read(out/'summary.json')
    assert summary['arena_changed'] is False
    levels = summary['levels']
    assert 1 <= len(levels) <= 3
    expected_evaluations = {'parity-zero'}
    fits = {}
    for level, stage in enumerate(levels):
        assert stage['level'] == level
        if level:
            assert levels[level-1]['advanced']
        signs = ('positive', 'negative') if level == 2 else ('positive',)
        baseline = []
        for sign in signs:
            name = f'eval-baseline-l{level}-t0-d{sign}'
            baseline.append(reports[name])
            expected_evaluations.add(name)
        assert stage['baseline'] == baseline
        quiet_name = f'quiet-baseline-l{level}'
        assert stage['baseline_quiet'] == reports[quiet_name]
        expected_evaluations.add(quiet_name)
        assert set(stage['candidates']) == set(map(str, contract['seeds']))
        nominal_passes = []
        for seed in contract['seeds']:
            folder = out/f'fit-{seed}'/f'level-{level}'
            fits[str(folder.relative_to(out))] = check_fit(folder, contract,
                None if level == 0 else folder.with_name(f'level-{level-1}'))
            candidate = stage['candidates'][str(seed)]
            names = [f'eval-{seed}-l{level}-t0-d{sign}' for sign in signs]
            expected_evaluations.update(names)
            nominal = [reports[name] for name in names]
            assert candidate['nominal'] == nominal
            quiet_name = f'quiet-{seed}-l{level}'
            expected_evaluations.add(quiet_name)
            assert candidate['quiet'] == reports[quiet_name]
            assert reports[quiet_name]['quiet'] and reports[quiet_name]['seconds'] == 60.
            admitted = gate(nominal)
            admitted['passed'] &= reports[quiet_name]['whole_stand'] == 16
            assert candidate['nominal_gate'] == admitted
            nominal_passes.append(admitted['passed'])
        assert stage['nominal_passed'] == all(nominal_passes)
        varied_passes = []
        if stage['nominal_passed']:
            for seed in contract['seeds']:
                candidate = stage['candidates'][str(seed)]
                names = [f'eval-{seed}-l{level}-t1-d{sign}' for sign in signs]
                expected_evaluations.update(names)
                varied = [reports[name] for name in names]
                assert candidate['varied'] == varied
                assert all(item['onset_seconds'] == 1.7 for item in varied)
                admitted = gate(varied)
                assert candidate['varied_gate'] == admitted
                varied_passes.append(admitted['passed'])
        else:
            assert all('varied' not in value for value in stage['candidates'].values())
        assert stage['advanced'] == bool(stage['nominal_passed'] and all(varied_passes))
    assert set(reports) == expected_evaluations
    assert summary['total_transitions'] == len(levels)*2*49152
    expected_status = ('commanded_step_pilot_passed_pending_independent_audit' if len(levels) == 3 and levels[-1]['advanced']
                       else 'stopped_at_failed_curriculum_gate')
    assert summary['status'] == expected_status
    return fits, summary


def check_curriculum(z, report, config):
    """Rebuild physical phase admission, consecutive dwell and rewards in NumPy."""
    c = dict(config, episode_seconds=report['seconds'])
    dt, n = np.float32(report['dt']), len(z['time'])
    worlds = np.arange(16)
    side, opposite = worlds//8, 1-worlds//8
    initial_feet, initial_com = z['initial_feet'][0], z['initial_com'][0]
    equal(z['initial_feet'], np.broadcast_to(initial_feet, z['initial_feet'].shape), 'fixed initial feet')
    equal(z['initial_com'], np.broadcast_to(initial_com, z['initial_com'].shape), 'fixed initial COM')
    distance = np.float32(report['distance_m'] if report['level'] == 2 and not report['quiet'] else 0.)
    for key, value in (('side', side), ('level', np.full(16, report['level'])),
                       ('enabled', np.full(16, not report['quiet'])),
                       ('distance', np.full(16, distance, np.float32)),
                       ('onset_seconds', np.full(16, report['onset_seconds'], np.float32))):
        equal(z[key], np.broadcast_to(value, z[key].shape), 'fixed task command '+key)
    initial_swing, initial_stance = initial_feet[worlds, side, :2], initial_feet[worlds, opposite, :2]
    direction = initial_stance-initial_com[:, :2]
    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-6)
    target = initial_com[:, :2]+np.float32(c['transfer_m'])*direction
    end = initial_swing.copy()
    end[:, 0] += distance
    settle_com = initial_com[:, :2].copy()
    settle_com[:, 0] += .5*distance
    close(z['transfer_target'], np.broadcast_to(target, z['transfer_target'].shape), 'transfer target', 1e-7)
    close(z['end_target'], np.broadcast_to(end, z['end_target'].shape), 'placement target', 1e-7)
    phase = np.full(16, 0 if report['quiet'] else 1, dtype=int)
    dwell = np.zeros((16, 5), dtype=np.float32)
    age, phase_age, maximum_drift = (np.zeros(16, np.float32) for _ in range(3))
    intervals, phase_intervals = (np.zeros(16, np.int64) for _ in range(2))
    dwell_intervals = np.zeros((16, 5), np.int64)
    completed = np.zeros((16, 4), bool)
    latched, physical_latch, timeout_latch, terminal = (np.zeros(16, bool) for _ in range(4))
    required_seconds = np.asarray([c[key] for key in ('transfer_dwell_seconds', 'lift_dwell_seconds',
        'place_dwell_seconds', 'landing_dwell_seconds', 'settling_dwell_seconds')])
    required = np.ceil(required_seconds/report['dt']).astype(np.int64)
    previous_quality = np.zeros(16)
    references = arrays(FOUNDATION/'stationary_reference.npz')['qpos']
    initial_height = references[worlds % 8, 2]
    close(z['initial_pelvis_height'], np.broadcast_to(initial_height, z['initial_pelvis_height'].shape), 'reset height', 1e-7)
    initial_height = z['initial_pelvis_height'][0]
    gate_keys = ('transfer_gate', 'lift_gate', 'place_gate', 'land_gate', 'settle_gate')
    maximum_reward_error = 0.

    def metrics(frame, current_phase):
        feet, loads = z['feet'][frame], z['loads'][frame]
        swing_xy, stance_xy = feet[worlds, side, :2], feet[worlds, opposite, :2]
        swing_load, stance_load = loads[worlds, side], loads[worlds, opposite]
        clearance = z['clearance'][frame, worlds, side]
        share = stance_load/np.maximum(loads.sum(-1), 1.)
        drift = np.linalg.norm(stance_xy-initial_stance, axis=-1)
        transfer_error = np.linalg.norm(z['com'][frame, :, :2]-target, axis=-1)
        progress = ((z['com'][frame, :, :2]-initial_com[:, :2])*direction).sum(-1)
        support_error = np.linalg.norm(z['com'][frame, :, :2]-initial_stance, axis=-1)
        placement_error = np.linalg.norm(swing_xy-end, axis=-1)
        world = np.linalg.norm(swing_xy-initial_swing, axis=-1)
        relative = np.linalg.norm((swing_xy-stance_xy)-(initial_swing-initial_stance), axis=-1)
        settle_error = np.linalg.norm(z['com'][frame, :, :2]-settle_com, axis=-1)
        safe = (z['finite'][frame] & (z['pelvis_height'][frame] >= c['min_pelvis_height_m'])
                & (z['pelvis_up'][frame] >= np.cos(np.deg2rad(c['max_pelvis_tilt_degrees']))))
        supported = safe & (stance_load >= c['support_n']) & (maximum_drift <= c['stance_drift_m']) & (drift <= c['stance_drift_m'])
        lifted = supported & (swing_load < c['unload_n']) & (clearance >= c['clearance_m']) & (support_error <= c['support_com_tolerance_m'])
        translated = (world >= c['minimum_placement_m']) & (relative >= c['minimum_placement_m'])
        gates = np.column_stack((
            supported & (share >= c['transfer_stance_share']) & (transfer_error <= c['transfer_tolerance_m']) & (progress >= c['transfer_min_progress_m']),
            lifted,
            lifted & completed[:, 1] & translated & (placement_error <= c['placement_tolerance_m']),
            supported & completed[:, 1] & (swing_load >= c['support_n']) & (clearance <= c['landing_clearance_m'])
                & (placement_error <= c['placement_tolerance_m']) & ((report['level'] < 2) | (completed[:, 2] & translated)),
            supported & z['measurement_stand'][frame] & (swing_load >= c['support_n'])
                & (placement_error <= c['settling_foot_tolerance_m']) & (settle_error <= c['settling_com_tolerance_m'])
                & completed[:, 0] & ((report['level'] == 0) | completed[:, 3]) & ((report['level'] < 2) | translated)))
        posture = np.exp(-((z['pelvis_height'][frame]-initial_height)/.15)**2
                         -((1-np.clip(z['pelvis_up'][frame], -1, 1))/.2)**2)
        support = np.clip(stance_load/c['support_n'], 0, 1)*safe
        stable = np.clip(1-(drift/c['stance_drift_m'])**2, 0, 1)
        share_quality = np.clip((share-.5)/(c['transfer_stance_share']-.5), 0, 1)
        lift_quality = np.clip(clearance/c['lift_m'], 0, 1)*np.exp(-(np.maximum(clearance-c['lift_m'], 0)/c['lift_m'])**2)
        lifted_quality = np.clip(1-swing_load/c['unload_n'], 0, 1)*lift_quality
        com_quality = np.exp(-(support_error/c['support_com_tolerance_m'])**2)
        place_quality = np.exp(-(placement_error/c['placement_tolerance_m'])**2)
        bilateral = np.clip(loads.min(-1)/c['support_n'], 0, 1)
        qualities = np.column_stack((
            .6*np.exp(-(transfer_error/.025)**2)+.4*share_quality,
            .4*com_quality+.6*lifted_quality,
            .3*com_quality+.25*lifted_quality+.45*lifted_quality*place_quality,
            completed[:, 1]*(.4*bilateral+.3*place_quality+.3*np.exp(-(clearance/c['landing_clearance_m'])**2)),
            .4*z['measurement_stand'][frame]+.3*np.exp(-(settle_error/c['settling_com_tolerance_m'])**2)+.3*place_quality))
        qualities *= (posture*support*stable)[:, None]
        quality = np.where(current_phase >= 2, np.clip(qualities[worlds, np.clip(current_phase-2, 0, 4)], 0, 1), 0.)
        values = dict(world_placement=world, relative_placement=relative, transfer_progress=progress,
                      transfer_error=transfer_error, support_com_error=support_error,
                      placement_error=placement_error, settle_com_error=settle_error)
        return gates, quality, values

    for frame in range(n):
        active = z['alive'][frame] & ~terminal
        equal(z['phase_before'][frame], phase, 'phase before')
        close(z['dwell_before'][frame], dwell, 'dwell before', 1e-6)
        before = phase.copy()
        intervals[active] += 1
        age[active] = intervals[active].astype(np.float32)*dt
        phase_intervals[active] += 1
        phase_age[active] = phase_intervals[active].astype(np.float32)*dt
        physical = active & (z['physical_failed'][frame] | ~z['finite'][frame])
        working = active & ~physical & (before >= 2)
        drift = np.linalg.norm(z['feet'][frame, worlds, opposite, :2]-initial_stance, axis=-1)
        maximum_drift[working] = np.maximum(maximum_drift[working], drift[working])
        gates, quality, values = metrics(frame, before)
        for column, key in enumerate(gate_keys):
            equal(z[key][frame], gates[:, column], key)
        for key, value in values.items():
            close(z[key][frame], value, key, 5e-7)
        close(z['quality_before'][frame], previous_quality, 'potential baseline', 3e-6)
        close(z['quality_measured'][frame], quality, 'independent physical quality', 3e-6)
        potential = np.where(working, c['gamma']*quality-previous_quality, 0.)
        time_cost = -c['active_step_cost']*working
        bonus = np.zeros(16)
        transition = np.zeros(16, bool)
        for column in range(5):
            selected = working & (before == column+2)
            dwell_intervals[selected, column] = np.where(gates[selected, column], dwell_intervals[selected, column]+1, 0)
            dwell[selected, column] = dwell_intervals[selected, column].astype(np.float32)*dt
            success = selected & (dwell_intervals[:, column] >= required[column])
            if column == 4:
                bonus[success & ~latched] = c['success_bonus']
                latched |= success
            else:
                completed[success, column] = True
                bonus[success] = c['transition_bonus']
                transition |= success
                after = (6 if report['level'] == 0 else 3) if column == 0 else ((5 if report['level'] == 1 else 4) if column == 1 else column+3)
                phase[success] = after
        begin = active & ~physical & (before == 1) & (age >= np.float32(report['onset_seconds']))
        phase[begin] = 2
        phase_intervals[transition | begin] = 0
        phase_age[transition | begin] = 0.
        after_gates, after_quality, _ = metrics(frame, phase)
        old_quality = previous_quality.copy()
        previous_quality[active] = after_quality[active]
        settled = latched & (phase == 6) & after_gates[:, 4] & (dwell_intervals[:, 4] >= required[4])
        deadline = active & (intervals >= int(np.ceil(c['episode_seconds']/report['dt'])))
        timeout = deadline & (not report['quiet']) & ~settled & ~physical
        physical_latch |= physical
        timeout_latch |= timeout
        terminal |= physical | deadline
        success = settled & (not report['quiet']) & ~physical_latch & ~timeout_latch
        equal(z['phase_after'][frame], phase, 'measured phase transitions')
        close(z['dwell'][frame], dwell, 'consecutive phase dwell', 1e-6)
        close(z['age'][frame], age, 'task clock', 1e-6)
        close(z['phase_age'][frame], phase_age, 'phase clock', 1e-6)
        equal(z['intervals'][frame], intervals, 'active interval clock')
        equal(z['phase_intervals'][frame], phase_intervals, 'phase interval clock')
        equal(z['dwell_intervals'][frame], dwell_intervals, 'consecutive dwell intervals')
        close(z['maximum_stance_drift'][frame], maximum_drift, 'latched stance displacement', 5e-7)
        for column, key in enumerate(('transfer_completed', 'lift_completed', 'place_completed', 'land_completed')):
            equal(z[key][frame], completed[:, column], key)
        equal(z['success_latched'][frame], latched, 'latched success')
        equal(z['task_success'][frame], success, 'current task success')
        equal(z['task_terminal'][frame], terminal, 'curriculum terminal')
        equal(z['task_timeout'][frame], timeout_latch, 'task deadline failure')
        equal(z['task_failed'][frame], physical_latch | timeout_latch, 'latched task failures')
        equal(z['failed'][frame], (physical_latch | timeout_latch | z['physical_failed'][frame]) & z['alive'][frame], 'combined failure')
        equal(z['terminal'][frame], ((z['native_done'][frame] != 0) | z['physical_failed'][frame] | terminal) & z['alive'][frame], 'native and task terminal')
        equal(z['bootstrap'][frame], (~terminal).astype(np.float32), 'terminal bootstrap')
        failure = np.where(physical, c['physical_failure_reward'], np.where(timeout, c['timeout_reward'], 0.))
        reward = np.where(physical | timeout, failure, potential+time_cost+bonus)
        for key, value in (('reward_potential', potential), ('reward_time', time_cost),
                           ('reward_completion', bonus), ('reward_failure', failure)):
            close(z[key][frame], value, key, 5e-6)
        close(z['reward_before_action_cost'][frame], reward, 'curriculum reward', 5e-6)
        expected_reward = np.where(z['alive'][frame], reward-z['action_cost'][frame], 0.)
        close(z['reward'][frame], expected_reward, 'final reward', 5e-6)
        maximum_reward_error = max(maximum_reward_error, float(np.abs(z['reward'][frame]-expected_reward).max()))
        observation = z['command_after'][frame]
        equal(observation[:, :7], np.eye(7)[phase], 'observable phase')
        equal(observation[:, 7:10], np.eye(3)[np.full(16, report['level'])], 'observable level')
        equal(observation[:, 10:12], np.eye(2)[side], 'observable side')
        equal(observation[:, 12], np.full(16, not report['quiet']), 'observable enable')
        equal(observation[:, 13], (not report['quiet']) & (phase >= 2), 'observable authority')
        close(observation[:, 14], np.full(16, distance/c['step_m']), 'observable distance', 1e-7)
        lifted_phase = (phase == 3) | (phase == 4)
        swing_target = np.where((phase >= 4)[:, None], end, initial_swing)
        com_target = np.where((phase == 2)[:, None], target, np.where(lifted_phase[:, None], initial_stance, settle_com))
        close(z['com_target'][frame], com_target, 'phase COM target', 1e-7)
        equal(observation[:, 15], lifted_phase, 'observable clearance goal')
        close(observation[:, 16], (lifted_phase*c['lift_m']-z['clearance'][frame, worlds, side])/c['lift_m'], 'observable clearance error', 3e-6)
        close(observation[:, 17:19], (swing_target-z['feet'][frame, worlds, side, :2])/c['step_m'], 'observable swing error', 3e-6)
        close(observation[:, 19:21], (initial_stance-z['feet'][frame, worlds, opposite, :2])/c['stance_drift_m'], 'observable stance error', 3e-6)
        close(observation[:, 21:23], (com_target-z['com'][frame, :, :2])/c['transfer_m'], 'observable COM error', 3e-6)
        close(observation[:, 23], np.where(phase == 2, c['transfer_stance_share'], np.where(lifted_phase, 1., .5)), 'observable load-share target', 1e-7)
        close(observation[:, 24], age/c['episode_seconds'], 'observable age', 1e-7)
        close(observation[:, 25], np.maximum(np.float32(report['onset_seconds'])-age, 0)/c['episode_seconds'], 'observable onset countdown', 1e-7)
        close(observation[:, 26], np.maximum(c['episode_seconds']-age, 0)/c['episode_seconds'], 'observable deadline countdown', 1e-7)
        close(observation[:, 27], phase_age/c['episode_seconds'], 'observable phase age', 1e-7)
        columns = np.clip(phase-2, 0, 4)
        close(observation[:, 28], np.where(phase >= 2, np.clip(dwell[worlds, columns]/required_seconds[columns], 0, 1), 0), 'observable dwell', 2e-7)
        close(observation[:, 29], np.where(active, old_quality, z['command_before'][frame, :, 29]), 'previous quality observation', 3e-6)
        close(observation[:, 30], previous_quality, 'next quality observation', 3e-6)
        equal(observation[:, 31], latched, 'observable success latch')
        equal(observation[:, 32:36], completed, 'observable phase completion')
        close(observation[:, 36], maximum_drift/c['stance_drift_m'], 'observable latched slip', 2e-6)
    equal(z['command_before'][1:], z['command_after'][:-1], 'command chronology')
    equal(z['goalfeatures_before'], z['command_before'], 'FSM input observation')
    equal(z['goalfeatures_after'], z['command_after'], 'FSM output observation')
    return dict(intervals=n, maximum_reward_error=maximum_reward_error,
                phase_criteria_independent=True, one_phase_per_interval=True,
                current_successes=int(z['task_success'][-1].sum()))


def check_provenance(out):
    inputs, completed = read(out/'inputs.json'), read(out/'completed.json')
    for path, sha in inputs.items():
        assert digest(Path(path)) == sha, ('changed input', path)
        if Path(path).suffix == '.py' or Path(path).name == 'STEPPING_SKILL_PLAN.md':
            assert digest(out/'sources'/Path(path).name) == sha, ('source archive', path)
    for path, sha in completed['outputs'].items():
        assert digest(out/path) == sha, ('changed output', path)
    assert completed['sources_unchanged'] is True
    from .run_learned_support import PARENT
    assert str(PARENT) in inputs and str(PARENT.with_suffix('.amp.pt')) in inputs
    for row in read(FOUNDATION/'protocol.json')['demonstrations']:
        assert inputs[str(Path(row['path']))] == row['sha256']
    return inputs, completed


def audit(out=OUT):
    inputs, completed = check_provenance(out)
    contract = read(out/'contract.json')
    assert contract['schema'] == 'commanded_stepping_pilot_v1'
    assert contract['input_width'] == 528 and contract['action_outputs'] == 33
    assert contract['seeds'] == [941009, 951009] and contract['initial_seed'] == 940009
    assert contract['gamma'] == .995 and contract['gae_lambda'] == .98
    assert contract['actor_lr'] == 1e-4 and contract['critic_lr'] == 3e-4
    assert contract['epochs'] == 3 and contract['minibatch'] == 256 and contract['kl_limit'] == .01
    assert contract['collections_per_level'] == contract['horizon'] == 64
    assert contract['episodic_terminal_bootstrap'] is False
    equal(contract['train_worlds'], TRAIN_WORLDS, 'declared training worlds')
    assert contract['task']['source_actor_frozen'] and not contract['task']['autonomous_hit_policy']
    assert not contract['task']['external_force'] and contract['task']['terminal_bootstrap'] == 0
    assert contract['task']['goal_width'] == 37 and contract['task']['observation_width'] == 528
    control = contract['task']['controller']
    assert control['motor_count'] == 33 and control['residual_radians'] == .25
    assert control['pd_gain_per_torque_limit'] == 4. and control['pd_damping_time'] == .02
    model = mujoco.MjModel.from_xml_path(str(FOUNDATION/'dummy.xml'))
    assert not model.jnt_stiffness.any(), 'passive springs changed'
    bare, zero = arrays(out/'parity-bare/trace.npz'), arrays(out/'parity-zero/trace.npz')
    parity = read(out/'parity.json')
    assert parity['exact'] and parity['command_enabled'] and not parity['source_targets_rescaled']
    assert parity['cases'] == 16 and parity['seconds'] == 12.
    for key in ('qpos', 'qvel', 'issued', 'active_target', 'addition', 'nominal', 'force'):
        equal(bare[key], zero[key], 'zero-head native parity '+key)
    equal(bare['failed'], zero['physical_failed'], 'zero-head physical failure parity')
    assert not zero['addition'].any() and not zero['sampled_latent'].any()
    reports, checks = {}, {}
    for path in sorted(out.glob('*/trace.npz')):
        if path.parent.name == 'parity-bare':
            continue
        z, report = arrays(path), read(path.with_name('report.json'))
        assert all(np.isfinite(value).all() for value in z.values()), path
        assert report['dt'] == contract['task']['dt']
        reports[path.parent.name] = report
        controls = check_controls(z, contract)
        xy, clearance = check_geometry(z, model)
        curriculum = check_curriculum(z, report, contract['curriculum'])
        result = check_report(z, report, xy, clearance)
        checks[path.parent.name] = dict(controls=controls, curriculum=curriculum, physical=result)
    fits, summary = check_stages(out, contract, reports)
    assert completed['total_transitions'] == summary['total_transitions']
    assert completed['status'] == summary['status']
    initial = [state_dict(out/f'fit-{seed}'/'level-0/initial.pt') for seed in contract['seeds']]
    assert all(torch.equal(initial[0][key], initial[1][key]) for key in initial[0]), 'fresh-seed initial equality'
    expected_models = {f'fit-{seed}/level-{level}/{name}' for seed in contract['seeds']
        for level in range(len(summary['levels'])) for name in ('initial.pt', 'final.pt', 'critic-final.pt')}
    assert {str(path.relative_to(out)).replace('\\', '/') for path in out.rglob('*.pt')} == expected_models
    return dict(passed=True, schema='independent_commanded_stepping_audit_v1',
                inputs=len(inputs), sealed_outputs=len(completed['outputs']),
                traces=len(checks)+1, cases=16*(len(checks)+1), physical=checks, training=fits,
                zero_initial_native_parity=True, identical_fresh_seed_initial_states=True,
                progression_verified_from_training_cases_only=True,
                active_training_transitions=summary['total_transitions'], status=summary['status'],
                physics_integration_steps=0, optimizer_steps=0,
                scope='Saved native trajectories and first/last training collections; no optimizer replay, '
                      'new rollout, or inference of autonomous ball recovery from commanded skills.')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out
    assert not (out/'audit.json').exists(), 'Audit output already exists'
    harness = out/'audit-harness'
    harness.mkdir(exist_ok=False)
    source = Path(__file__)
    shutil.copy2(source, harness/source.name)
    source_hash = digest(source)
    try:
        report = audit(out)
        assert digest(source) == source_hash
        report['audit_source_sha256'] = source_hash
        report['audit_source'] = str(source)
    except Exception as error:
        save(out/'audit.json', dict(passed=False, error_type=type(error).__name__, error=str(error),
                                    audit_source_sha256=source_hash))
        raise
    save(out/'audit.json', report)
    print(json.dumps({key: report[key] for key in ('passed', 'traces', 'cases', 'active_training_transitions', 'status')}), flush=True)


if __name__ == '__main__':
    main()
