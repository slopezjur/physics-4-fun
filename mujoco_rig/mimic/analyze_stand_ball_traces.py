"""Read sealed ball traces with NumPy; never simulate, infer policy or train."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def window_metrics(t, contract, case, indices):
    action = t['action'][indices, case]
    reference = t['reference_motor_q'][indices, case]
    low, high = (np.asarray(contract[k]) for k in ('target_lower_radians', 'target_upper_radians'))
    raw_target = reference + contract['residual_radians']*np.clip(action, -1., 1.)
    target_clamped = (raw_target < low-1e-6) | (raw_target > high+1e-6)
    torque = np.abs(t['substep_torque'][indices, :, case])/np.asarray(contract['torque_scale_nm'])
    joint = t['substep_q'][indices, :, case]
    outside = (joint < low-1e-4) | (joint > high+1e-4)
    peak_joint = int(np.unravel_index(torque.argmax(), torque.shape)[-1])
    return dict(frames=len(indices), action_peak=float(np.abs(action).max()),
        action_clipped_samples=int((np.abs(action) > 1.).sum()),
        target_clamped_samples=int(target_clamped.sum()),
        clamped_target_motors=[contract['actuator_names'][i] for i in np.flatnonzero(target_clamped.any(0))],
        torque_peak_fraction=float(torque.max()), torque_saturated_samples=int((torque >= .99999).sum()),
        peak_torque_motor=contract['actuator_names'][peak_joint],
        joint_outside_limit_samples=int(outside.sum()),
        joint_outside_limit_motors=[contract['actuator_names'][i] for i in np.flatnonzero(outside.any((0, 1)))],
        minimum_joint_limit_margin_rad=float(np.minimum(joint-low, high-joint).min()),
        horizontal_speed_max=float(t['horizontal_speed_m_s'][indices, case].max()),
        chest_tilt_max_degrees=float(t['upper_body_tilt_degrees'][indices, case, 1].max()))


def analyze(folder, metadata):
    contract, report = read(folder/'contract.json'), read(folder/'report.json')
    keys = ['time_s', 'normal_impulse_ns', 'contact_impulse_ns', 'alive_before',
        'balance', 'posture', 'pre_balance', 'pre_posture', 'action', 'reference_motor_q',
        'issued_entry', 'active_targets', 'substep_torque', 'substep_q', 'substep_v',
        'horizontal_speed_m_s', 'upper_body_tilt_degrees', 'pre_upper_body_tilt_degrees',
        'ball_qvel_before', 'qpos_before', 'observation']
    with np.load(folder/'trace.npz', allow_pickle=False) as archive:
        keys += [k for k in archive.files if k.startswith(('balance_gate_', 'posture_gate_',
            'pre_balance_gate_', 'pre_posture_gate_'))]
        t = {key: archive[key] for key in keys}
    dt = contract['control_timestep']
    rows = []
    for case, result in enumerate(report['cases']):
        contacts = np.flatnonzero(t['normal_impulse_ns'][:, case] > 0.)
        if not len(contacts):
            raise ValueError(f'No recorded contact: {folder}, {case}')
        first = int(contacts[0])
        live = t['alive_before'][:, case]
        valid = np.logical_and.reduce([t[k][:, case] for k in ('balance', 'posture', 'pre_balance', 'pre_posture')])
        departures = np.flatnonzero(live & ~valid)
        departure = int(departures[0]) if len(departures) else None
        if departure is not None and departure < first:
            raise ValueError('Pre-contact departure requires a separate interpretation')
        groups = np.split(contacts, np.flatnonzero(np.diff(contacts) > 1)+1)
        clusters = [dict(start_s=float(t['time_s'][g[0]]), end_s=float(t['time_s'][g[-1]]),
            normal_impulse_ns=float(t['normal_impulse_ns'][g, case].sum()),
            vector_impulse_ns=t['contact_impulse_ns'][g, case].sum(0).tolist()) for g in groups]
        post = np.flatnonzero(live & (np.arange(len(live)) >= first))
        before_loss = post if departure is None else post[post <= departure]
        first_second = post[t['time_s'][post] <= t['time_s'][first]+1.]
        gate_departures = {}
        for key in keys:
            if '_gate_' in key:
                bad = np.flatnonzero(live & ~t[key][:, case])
                if len(bad):
                    gate_departures[key] = float(t['time_s'][bad[0]])
        action_baseline = t['action'][max(0, first-int(np.ceil(.5/dt))):first, case].mean(0)
        action_change = np.abs(t['action'][first_second, case]-action_baseline)
        motor_change = action_change.max(0)
        top = np.argsort(motor_change)[-5:][::-1]
        issued = t['issued_entry'][:, case, :33]
        targets = np.clip(t['reference_motor_q'][:, case] + contract['residual_radians']*
            np.clip(t['action'][:, case], -1., 1.),
            contract['target_lower_radians'], contract['target_upper_radians'])
        latency = int(contract['latency_steps'])
        delay_error = np.abs(t['active_targets'][latency:, case]-t['issued_entry'][:-latency, case])
        active = t['active_targets'][:, case]
        expected_torque = (np.clip(4.*(active[:, None, :33]-t['substep_q'][:, :, case]) +
            .08*(active[:, None, 33:66]-t['substep_v'][:, :, case]), -1., 1.) *
            np.asarray(contract['torque_scale_nm'])[None, None, :] * active[:, None, 66:67])
        rows.append(dict(**metadata, case=case, recovered=result['recovery_pass'],
            survived=result['survived'], survival_seconds=result['survival_seconds'],
            first_contact_seconds=float(t['time_s'][first]),
            first_departure_seconds=None if departure is None else float(t['time_s'][departure]),
            first_departure_after_contact_seconds=None if departure is None else float((departure-first)*dt),
            total_normal_impulse_ns=float(t['normal_impulse_ns'][:, case].sum()),
            vector_impulse_ns=t['contact_impulse_ns'][:, case].sum(0).tolist(),
            contact_clusters=clusters,
            later_cluster_normal_impulse_ns=sum(c['normal_impulse_ns'] for c in clusters[1:]),
            ball_velocity_at_first_contact=t['ball_qvel_before'][first, case, :3].tolist(),
            chest_tilt_before_first_contact=float(t['pre_upper_body_tilt_degrees'][first, case, 1]),
            root_position_before_first_contact=t['qpos_before'][first, case, :3].tolist(),
            root_velocity_before_first_contact=t['observation'][first, case, 9:12].tolist(),
            gate_departures=dict(sorted(gate_departures.items(), key=lambda p: p[1])),
            windows=dict(through_first_departure=window_metrics(t, contract, case, before_loss),
                first_second=window_metrics(t, contract, case, first_second),
                all_post_contact=window_metrics(t, contract, case, post)),
            largest_first_second_action_changes=[dict(motor=contract['actuator_names'][i],
                absolute_change=float(motor_change[i]), target_change_rad=float(.25*motor_change[i])) for i in top],
            maximum_issued_target_reconstruction_error_rad=float(np.abs(issued[live]-targets[live]).max()),
            maximum_delay_error=float(delay_error[live[latency:]].max()),
            maximum_torque_formula_error_nm=float(np.abs(expected_torque[live]-t['substep_torque'][live, :, case]).max())))
    return rows


def run(args):
    if args.out.exists():
        raise ValueError('Output already exists')
    pilot = Path('logs/mimickit-training/stand-ball-pilot-20261006-01')
    curve = Path('logs/mimickit-training/stand-ball-checkpoints-20261006-01')
    protocol, seal = read(curve/'protocol.json'), read(curve/'final-verification.json')
    if not seal['verified'] or not seal['inputs_and_sources_unchanged']:
        raise ValueError('Verified checkpoint assessment required')
    manifest = {str(Path(k).resolve()): v for k, v in {**protocol['inputs'], **seal['output_hashes']}.items()}
    jobs, inputs = [], {}
    for seed in (301027, 301028):
        for update in (0, 100, 200, 300):
            for arm in (('source',) if update == 0 else ('quiet', 'single_ball')):
                root = (pilot/str(seed)/'source' if update == 0 else
                    pilot/str(seed)/arm/'evaluation' if update == 300 else
                    curve/str(update)/str(seed)/arm/'selected')
                for sign in (1, -1):
                    for backend in ('native', 'gpu'):
                        folder = root/f'5kg-1mps-{sign:+d}-{backend}'
                        for name in ('contract.json', 'report.json', 'trace.npz'):
                            path = folder/name
                            value = digest(path)
                            if manifest.get(str(path.resolve())) != value:
                                raise ValueError(f'Evidence differs from sealed hash: {path}')
                            inputs[str(path)] = value
                        jobs.append((folder, dict(seed=seed, update=update, arm=arm, sign=sign, backend=backend)))
    args.out.mkdir(parents=True)
    for path in (curve/'protocol.json', curve/'final-verification.json', Path(__file__)):
        inputs[str(path)] = digest(path)
    write(args.out/'protocol.json', dict(schema='stand_ball_trace_audit_v1', inputs=inputs,
        traces=len(jobs), existing_cases=8*len(jobs), new_physical_cases=0, new_training_updates=0,
        windows='First contact through first pre/post gate departure inclusive; if no departure, all remaining live frames. Also first second and all live post-contact.',
        clusters='Contiguous control intervals with nonzero ball-character normal impulse; not individual substep contacts.',
        limit_tolerances=dict(target_clamp_rad=1e-6, joint_limit_exceedance_rad=1e-4, torque_saturation_fraction=.99999),
        scope='Observational description of sealed traces; does not isolate causal effects'))
    rows = []
    for folder, metadata in jobs:
        rows.extend(analyze(folder, metadata))
    summaries = []
    for sign in (1, -1):
        for recovered in (False, True):
            group = [r for r in rows if r['sign'] == sign and r['recovered'] == recovered]
            if not group:
                continue
            w = [r['windows']['through_first_departure'] for r in group]
            summaries.append(dict(sign=sign, recovered=recovered, cases=len(group),
                cases_with_action_clipping=sum(x['action_clipped_samples'] > 0 for x in w),
                cases_with_target_clamping=sum(x['target_clamped_samples'] > 0 for x in w),
                cases_with_torque_saturation=sum(x['torque_saturated_samples'] > 0 for x in w),
                cases_with_joint_limit_exceedance=sum(x['joint_outside_limit_samples'] > 0 for x in w),
                maximum_action=max(x['action_peak'] for x in w),
                maximum_torque_fraction=max(x['torque_peak_fraction'] for x in w),
                normal_impulse_range_ns=[min(r['total_normal_impulse_ns'] for r in group), max(r['total_normal_impulse_ns'] for r in group)]))
    result = dict(cases=rows, summaries=summaries, existing_cases=len(rows),
        maximum_delay_error=max(r['maximum_delay_error'] for r in rows),
        maximum_torque_formula_error_nm=max(r['maximum_torque_formula_error_nm'] for r in rows),
        maximum_issued_target_reconstruction_error_rad=max(r['maximum_issued_target_reconstruction_error_rad'] for r in rows))
    write(args.out/'report.json', result)
    for name, expected in inputs.items():
        if digest(Path(name)) != expected:
            raise ValueError(f'Input changed during analysis: {name}')
    write(args.out/'verification.json', dict(verified=True, inputs_unchanged=True,
        checked_inputs=len(inputs), existing_cases=len(rows), new_physical_cases=0,
        new_training_updates=0, outputs={str(p): digest(p) for p in args.out.iterdir() if p.is_file()}))
    print(json.dumps(dict(existing_cases=len(rows), summaries=summaries,
        maximum_delay_error=result['maximum_delay_error'],
        maximum_torque_formula_error_nm=result['maximum_torque_formula_error_nm'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
