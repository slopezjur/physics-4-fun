"""Independent frozen-command inference and archived-trace arithmetic audit."""
import json
from pathlib import Path

import numpy as np
import torch

from .run_support_skill_study import digest, save

OUT = Path('logs/mimickit-training/recovery-command-generalization-20261009-01')
HIGH = Path('logs/mimickit-training/recovery-retracted-actor-rate-20261009-01')
LOW = Path('logs/mimickit-training/recovery-retracted-learning-20261009-02')
DATA = Path('logs/mimickit-training/recovery-local-actor-updates-20261009-01')
SIGMA = .08717797887081347
SIGN_BAND = 1e-7
NUMPY_FORWARD_ATOL = 5e-7
NUMPY_FORWARD_RTOL = 2e-6


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def close(actual, expected, label, atol=1e-10, rtol=2e-7):
    assert np.allclose(actual, expected, atol=atol, rtol=rtol), label


def compare(actual, expected, label):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), label
        for key, value in expected.items():
            compare(actual[key], value, label+'.'+key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), label
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, label+'.'+str(index))
    elif isinstance(expected, (bool, int, str)) or expected is None:
        assert actual == expected, (label, actual, expected)
    else:
        assert np.isclose(actual, expected, atol=1e-10, rtol=2e-7), (label, actual, expected)


def numpy_forward(state, features):
    """Float32 NumPy evaluation, independent of the actor's PyTorch forward method."""
    assert features.dtype == np.float32 and features.ndim == 2 and features.shape[1] == 427
    value = features.copy()
    expected_keys = {f'net.{layer}.{field}' for layer in (0, 2, 4) for field in ('weight', 'bias')}
    assert set(state) == expected_keys
    for layer, shape in ((0, (128, 427)), (2, (128, 128)), (4, (1, 128))):
        weight = state[f'net.{layer}.weight'].detach().cpu().numpy()
        bias = state[f'net.{layer}.bias'].detach().cpu().numpy()
        assert weight.dtype == bias.dtype == np.float32 and weight.shape == shape and bias.shape == (shape[0],)
        assert np.isfinite(weight).all() and np.isfinite(bias).all()
        value = value@weight.T+bias
        if layer != 4:
            negative = value <= 0
            value[negative] = np.expm1(value[negative])
    return np.float32(.9)*features[:, -1:]+np.float32(.1)*value


def verify_probe_identity(source):
    x = source['probe_x']
    assert x.shape == (36, 427) and x.dtype == np.float32 and np.isfinite(x).all()
    assert not x[:, -2:].any()
    branch, onset, pose, quiet_index = (source['probe_'+key] for key in ('branch', 'onset', 'pose', 'quiet_index'))
    expected = [(time, sign, p) for time in (44, 74) for sign in (0, 1, 2) for p in range(6)]
    assert list(zip(onset.tolist(), branch.tolist(), pose.tolist())) == expected
    assert quiet_index.shape == (36,) and np.issubdtype(quiet_index.dtype, np.integer)
    for index in range(36):
        paired = int(quiet_index[index])
        assert branch[paired] == 0 and onset[paired] == onset[index] and pose[paired] == pose[index]
        assert paired == (0 if onset[index] == 44 else 18)+pose[index]
    assert np.all(source['probe_preferred'][branch == 1] == -1)
    assert np.all(source['probe_preferred'][branch == 2] == 1)
    return branch, onset, pose, quiet_index


def probe_moments(value):
    value = np.asarray(value, dtype=np.float64)
    assert value.ndim == 1 and value.size and np.isfinite(value).all()
    return dict(count=int(value.size), mean=float(np.mean(value)),
                rms=float(np.sqrt(np.mean(np.square(value)))),
                maximum_absolute=float(np.max(np.abs(value))))


def classifications(value, reference):
    directed = np.asarray(value, dtype=np.float64)*reference
    result = np.zeros(directed.shape, dtype=np.int64)
    result[directed > SIGN_BAND] = 1
    result[directed < -SIGN_BAND] = -1
    result[np.asarray(reference) == 0] = 0
    return result


def probe_group(evidence, mask):
    selected_push = mask & (evidence['branch'] != 0)
    def counts(key):
        values = evidence[key][selected_push]
        return dict(cases=int(values.size), aligned=int(np.count_nonzero(values == 1)),
                    opposed=int(np.count_nonzero(values == -1)), within_band=int(np.count_nonzero(values == 0)))
    keys = ('mean', 'memory', 'innovation', 'paired_mean', 'paired_innovation',
            'mean_over_sigma', 'innovation_over_sigma', 'paired_mean_over_sigma',
            'desired_addition', 'requested_addition', 'paired_requested_addition')
    return dict(cases=int(mask.sum()), **{key: probe_moments(evidence[key][mask]) for key in keys},
                raw_reference=counts('raw_alignment'), paired_reference=counts('paired_alignment'),
                slew_limited=int(evidence['slew_limited'][mask].sum()))


def verify_probes(folder, root, seed, source):
    branch, onset, pose, quiet_index = verify_probe_identity(source)
    evidence, report = arrays(folder/'probes.npz'), read(folder/'probes.json')
    checkpoint = root/f'fit-{seed}'/'final.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    independent = numpy_forward(state, source['probe_x'])[:, 0]
    mean = evidence['mean']
    assert mean.shape == (36,) and mean.dtype == np.float32 and np.isfinite(mean).all()
    close(mean, independent, 'independent NumPy actor', NUMPY_FORWARD_ATOL, NUMPY_FORWARD_RTOL)
    # Labels use sealed Torch predictions, not the looser independent BLAS tolerance.
    previous_latent = source['probe_x'][:, -1]
    memory = np.float32(.9)*previous_latent
    previous_addition = np.float32(.1)*source['probe_x'][:, -2]
    desired = np.float32(.1)*np.tanh(mean)
    requested = previous_addition+np.clip(desired-previous_addition, np.float32(-.02), np.float32(.02))
    reference = np.select((branch == 1, branch == 2), (-1, 1), default=0).astype(np.int64)
    expected = dict(branch=branch, onset=onset, pose=pose, quiet_index=quiet_index,
                    reference_sign=reference, mean=mean, previous_latent=previous_latent,
                    memory=memory, innovation=mean-memory, previous_addition=previous_addition,
                    desired_addition=desired, requested_addition=requested,
                    slew_limited=np.abs(desired-previous_addition) > np.float32(.02))
    for key, value in expected.items():
        assert evidence[key].shape == (36,) and evidence[key].dtype == value.dtype, key
        if key in ('desired_addition', 'requested_addition'):
            close(evidence[key], value, key, atol=1e-8, rtol=2e-7)
        else:
            assert np.array_equal(evidence[key], value), key
    # Derived summaries must exactly use the persisted prediction/mapping values.
    for key in ('desired_addition', 'requested_addition'):
        expected[key] = evidence[key]
    for source_key, target in (('mean', 'paired_mean'), ('innovation', 'paired_innovation'),
                               ('requested_addition', 'paired_requested_addition')):
        value = expected[source_key].astype(np.float64)
        expected[target] = value-value[quiet_index]
    for key in ('mean', 'innovation', 'paired_mean'):
        expected[key+'_over_sigma'] = expected[key].astype(np.float64)/SIGMA
    expected['raw_alignment'] = classifications(mean, reference)
    expected['paired_alignment'] = classifications(expected['paired_mean'], reference)
    assert evidence.keys() == expected.keys()
    for key, value in expected.items():
        assert evidence[key].shape == value.shape and evidence[key].dtype == value.dtype, key
        close(evidence[key], value, key)
    labels = {-1: 'opposed', 0: 'within_band', 1: 'aligned'}
    rows = []
    for index in range(36):
        pushed = branch[index] != 0
        row = dict(index=index, onset_frame=int(onset[index]), pose=int(pose[index]),
                   branch=('quiet', 'positive_x', 'negative_x')[branch[index]],
                   quiet_probe_index=int(quiet_index[index]),
                   reference_sign=int(reference[index]) if pushed else None,
                   raw_reference=labels[int(expected['raw_alignment'][index])] if pushed else None,
                   paired_reference=labels[int(expected['paired_alignment'][index])] if pushed else None,
                   slew_limited=bool(expected['slew_limited'][index]))
        for key in ('mean', 'previous_latent', 'memory', 'innovation', 'paired_mean', 'paired_innovation',
                    'mean_over_sigma', 'innovation_over_sigma', 'paired_mean_over_sigma',
                    'previous_addition', 'desired_addition', 'requested_addition', 'paired_requested_addition'):
            row[key] = float(expected[key][index])
        rows.append(row)
    masks = dict(all=np.ones(36, bool), quiet=branch == 0, push=branch != 0,
                 positive_x=branch == 1, negative_x=branch == 2)
    assert isinstance(report['scope'], str) and report['scope']
    expected_report = dict(probes=36, quiet_probes=12, pushed_probes=24, poses=list(range(6)),
        onsets=[44, 74], zero_action_memory_verified=True, pairing_verified=True,
        sigma=SIGMA, sign_band=SIGN_BAND, rho=.9, addition_radius=.1, addition_slew=.02,
        groups={name: probe_group(expected, mask) for name, mask in masks.items()},
        by_onset={str(value): probe_group(expected, onset == value) for value in (44, 74)},
        per_probe=rows, scope=report['scope'], checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint), frozen_parameters_unchanged=True)
    compare(report, expected_report, 'probes report')
    return report, float(np.max(np.abs(mean.astype(np.float64)-independent.astype(np.float64))))


def trace_moments(values):
    value = np.asarray(values, dtype=np.float64)
    assert value.size and np.isfinite(value).all()
    average = float(value.mean())
    second = float(np.mean(value*value))
    return dict(samples=int(value.size), mean=average, rms=float(np.sqrt(second)),
                second_moment=second, variance=float(np.mean((value-average)**2)),
                maximum_absolute=float(np.max(np.abs(value))))


def trace_decomposition(latent, memory, innovation):
    latent, memory, innovation = (np.asarray(value, dtype=np.float64) for value in (latent, memory, innovation))
    close(latent, memory+innovation, 'memory identity', 1e-14, 1e-12)
    l, m, i = (trace_moments(value) for value in (latent, memory, innovation))
    cross = float(np.mean(memory*innovation))
    covariance = float(np.mean((memory-memory.mean())*(innovation-innovation.mean())))
    energy_error = l['second_moment']-m['second_moment']-i['second_moment']-2*cross
    variance_error = l['variance']-m['variance']-i['variance']-2*covariance
    assert abs(energy_error) <= 1e-14 and abs(variance_error) <= 1e-14
    return dict(latent=l, memory=m, innovation=i, mean_memory_times_innovation=cross,
                twice_cross_second_moment=2*cross, covariance=covariance, twice_covariance=2*covariance,
                second_moment_identity_error=float(energy_error), variance_identity_error=float(variance_error))


def trace_reference(value, reference):
    directed = np.asarray(value, dtype=np.float64)*reference
    return dict(samples=int(directed.size), agreement=int(np.count_nonzero(directed > SIGN_BAND)),
                opposition=int(np.count_nonzero(directed < -SIGN_BAND)),
                tied=int(np.count_nonzero(np.abs(directed) <= SIGN_BAND)),
                signed_mean=float(directed.mean()), signed_minimum=float(directed.min()),
                signed_maximum=float(directed.max()), tolerance_latent=SIGN_BAND)


def trace_selection(evidence, selected):
    indices = np.asarray(selected, dtype=np.int64)
    branches = {}
    for branch, name in enumerate(('push', 'quiet')):
        branches[name] = dict(
            decomposition=trace_decomposition(*(evidence[key][branch][:, indices, 0]
                                                for key in ('latent', 'memory', 'innovation'))),
            previous_addition_rad=trace_moments(evidence['previous_addition'][branch][:, indices, 0]),
            addition_rad=trace_moments(evidence['addition'][branch][:, indices, 0]),
            active_intervals=int(evidence['alive'][branch][:, indices].sum()),
            inactive_intervals=int((~evidence['alive'][branch][:, indices]).sum()),
            failed_intervals=int(evidence['failed'][branch][:, indices].sum()))
    value = evidence['difference_latent'][:, indices, 0]
    reference = np.where(indices < 8, -1., 1.)
    paired = dict(decomposition=trace_decomposition(*(evidence['difference_'+key][:, indices, 0]
                                                     for key in ('latent', 'memory', 'innovation'))),
                  addition_rad=trace_moments(evidence['difference_addition'][:, indices, 0]),
                  window_mean_reference=trace_reference(value.mean(axis=0), reference),
                  per_interval_reference=trace_reference(value, reference[None]),
                  paired_active_intervals=int(evidence['paired_alive'][:, indices].sum()),
                  paired_inactive_intervals=int((~evidence['paired_alive'][:, indices]).sum()))
    return dict(cases=indices.tolist(), branches=branches, pushed_minus_quiet=paired)


def check_trace(trace, dt):
    count = len(trace['time'])
    assert count > 133 and trace['time'].shape == (count,)
    assert trace['time'][0] == dt
    close(trace['time'], (np.arange(count)+1)*dt, 'trace timestamps', 1e-10, 0)
    for key in ('latent', 'previous_latent'):
        assert trace[key].shape == (count, 16, 1) and np.isfinite(trace[key]).all()
    for key in ('previous', 'addition'):
        assert trace[key].shape == (count, 16, 33) and np.isfinite(trace[key]).all()
        assert np.array_equal(trace[key][:, :, 21], trace[key][:, :, 29])
        other = np.delete(trace[key], (21, 29), axis=2)
        assert not other.any()
    for key in ('alive', 'failed'):
        assert trace[key].shape == (count, 16) and trace[key].dtype == np.bool_
    assert trace['alive'][0].all() and not trace['previous_latent'][0].any()
    assert not trace['previous'][0].any()
    assert np.array_equal(trace['alive'][1:], trace['alive'][:-1] & ~trace['failed'][:-1])
    previous_latent = trace['previous_latent'][:-1].copy()
    live = trace['alive'][:-1]
    previous_latent[live] = trace['latent'][:-1][live]
    assert np.array_equal(trace['previous_latent'][1:], previous_latent)
    assert np.array_equal(trace['previous'][1:], trace['addition'][:-1])
    previous = trace['previous'][:, :, 21].astype(np.float64)
    desired = .1*np.tanh(trace['latent'][:, :, 0].astype(np.float64))
    requested = previous+np.minimum(.02, np.maximum(-.02, desired-previous))
    requested[~trace['alive']] = previous[~trace['alive']]
    error = float(np.max(np.abs(requested-trace['addition'][:, :, 21])))
    assert error <= 1e-7
    return dict(intervals=count, dt=dt, memory_chronology_exact=True,
                addition_chronology_exact=True, inactive_memory_freezes=True,
                only_tied_ankles_modified=True, action_mapping_maximum_error_rad=error)


def report_paths(root, seed, mode, timing):
    push_name = f'{"matched-" if mode == "matched" else ""}{seed}-t{timing}-g0'
    quiet_name = f'matched-{seed}-t{timing}-quiet' if mode == 'matched' else f'{seed}-quiet'
    return root/push_name, root/quiet_name


def check_case_reports(push, quiet, timing, mode, dt):
    for report, unpushed in ((push, False), (quiet, True)):
        assert report['sampled'] is False and report['quiet'] is unpushed and report['group'] == 0
        assert report['dt'] == dt and len(report['cases']) == 16
        assert report['seconds'] == (60. if unpushed and mode == 'continuous' else 6.)
        for index, row in enumerate(report['cases']):
            assert row['case'] == index and row['pose'] == index % 8
            assert row['heldout'] == (index % 8 >= 6)
            force = [0., 0., 0.] if unpushed else ([150., 0., 0.] if index < 8 else [-150., 0., 0.])
            assert row['force_n'] == force
            if unpushed:
                assert row['hit_seconds'] is None
            else:
                assert abs(row['hit_seconds']-(60, 102)[timing]*dt) < 2e-7


def verify_window(folder, root, seed, rate, mode, timing, dt):
    push_dir, quiet_dir = report_paths(root, seed, mode, timing)
    check_case_reports(read(push_dir/'report.json'), read(quiet_dir/'report.json'), timing, mode, dt)
    traces = [arrays(path/'trace.npz') for path in (push_dir, quiet_dir)]
    checks = {name: check_trace(trace, dt) for name, trace in zip(('push', 'quiet'), traces)}
    issue = (61, 103)[timing]
    window = slice(issue, issue+30)
    expected = dict(frame=np.arange(issue, issue+30, dtype=np.int64), time=traces[0]['time'][window])
    assert np.array_equal(expected['time'], traces[1]['time'][window])
    for key in ('alive', 'failed', 'latent', 'previous_latent'):
        expected[key] = np.stack([trace[key][window] for trace in traces])
    for source, target in (('previous', 'previous_addition'), ('addition', 'addition')):
        expected[target] = np.stack([np.take(trace[source][window], [21, 29], axis=2) for trace in traces])
    expected['memory'] = .9*expected['previous_latent'].astype(np.float64)
    expected['innovation'] = expected['latent'].astype(np.float64)-expected['memory']
    for key in ('latent', 'memory', 'innovation', 'addition'):
        expected['difference_'+key] = expected[key][0].astype(np.float64)-expected[key][1].astype(np.float64)
    expected['paired_alive'] = expected['alive'][0] & expected['alive'][1]
    name = f'{mode}-t{timing}'
    evidence, report = arrays(folder/f'{name}.npz'), read(folder/f'{name}.json')
    assert evidence.keys() == expected.keys()
    for key, value in expected.items():
        assert evidence[key].dtype == value.dtype and np.array_equal(evidence[key], value), (name, key)
    ids = np.arange(16)
    masks = dict(all=np.ones(16, bool), seen_pose=ids % 8 < 6, reserved_pose=ids % 8 >= 6,
                 positive_x=ids < 8, negative_x=ids >= 8,
                 seen_positive_x=(ids % 8 < 6) & (ids < 8),
                 seen_negative_x=(ids % 8 < 6) & (ids >= 8),
                 reserved_positive_x=(ids % 8 >= 6) & (ids < 8),
                 reserved_negative_x=(ids % 8 >= 6) & (ids >= 8))
    scope = report['scope']
    assert scope['fixed_windows'] is True and scope['survival_filtering'] is False
    assert scope['statistics_include_all_selected_intervals'] is True
    assert set(scope) == {'fixed_windows', 'survival_filtering', 'statistics_include_all_selected_intervals',
                          'latent_decomposition', 'energy', 'comparison', 'signs', 'sign_tolerance', 'quiet_copies'}
    assert all(isinstance(value, str) and value for key, value in scope.items()
               if key not in ('fixed_windows', 'survival_filtering', 'statistics_include_all_selected_intervals'))
    expected_report = dict(seed=seed, rate=rate, mode=mode, timing=timing, issue_frame=issue,
        intervals=30, cases=16, dt=dt, duration_seconds=30*dt, branch_order=['push', 'quiet'],
        ankles=[21, 29], checks=checks, all_selected_intervals_active=bool(expected['alive'].all()),
        selected_failed_intervals=int(expected['failed'].sum()),
        grouped={group: trace_selection(expected, ids[mask]) for group, mask in masks.items()},
        per_case=[dict(case=index, pose=index % 8, reserved_pose=index % 8 >= 6,
                       direction='positive_x' if index < 8 else 'negative_x',
                       reference_polarity=-1 if index < 8 else 1,
                       **trace_selection(expected, [index])) for index in range(16)],
        scope=scope, pushed_source=str(push_dir/'trace.npz'), quiet_source=str(quiet_dir/'trace.npz'))
    compare(report, expected_report, name)
    return report


def verify_provenance():
    inputs = read(OUT/'inputs.json')
    completed = read(OUT/'completed.json')
    assert inputs and all(digest(Path(path)) == sha for path, sha in inputs.items())
    for relative, sha in completed['outputs'].items():
        path = OUT/relative
        assert path.resolve().is_relative_to(OUT.resolve())
        assert digest(path) == sha, str(path)
    required = {str(path) for path in Path(__file__).parent.glob('*.py')}
    required.add(str(Path(__file__).with_name('RECOVERY_COMMAND_GENERALIZATION_PLAN.md')))
    consumed_counts = {}
    for root in (HIGH, LOW, DATA):
        for name in ('completed.json', 'audit.json'):
            required.add(str(root/name))
        assert read(root/'audit.json')['passed'] is True
        old_outputs = read(root/'completed.json')['outputs']
        artifacts = {'inputs.json'}
        if root == DATA:
            artifacts.update(str(Path(str(seed))/'dataset.npz') for seed in (921009, 931009))
        else:
            artifacts.add('contract.json')
            for seed in (921009, 931009):
                artifacts.add(str(Path(f'fit-{seed}')/'final.pt'))
                for mode in ('continuous', 'matched'):
                    for timing in (0, 1):
                        for path in report_paths(root, seed, mode, timing):
                            artifacts.update(str((path/name).relative_to(root)) for name in ('report.json', 'trace.npz'))
        for relative in artifacts:
            path = root/relative
            required.add(str(path))
            assert old_outputs[relative] == digest(path), str(path)
        consumed_counts[str(root)] = len(artifacts)
    assert set(inputs) == required, (set(inputs)-required, required-set(inputs))
    for path in Path(__file__).parent.glob('*.py'):
        assert digest(OUT/'sources'/path.name) == inputs[str(path)], path.name
    plan = Path(__file__).with_name('RECOVERY_COMMAND_GENERALIZATION_PLAN.md')
    assert digest(OUT/'sources'/plan.name) == inputs[str(plan)]
    historical = {}
    for root in (HIGH, LOW):
        previous = read(root/'inputs.json')
        shared = sorted(set(previous) & set(inputs))
        assert shared and all(previous[path] == inputs[path] for path in shared)
        historical[str(root)] = dict(passed=True, count=len(shared), shared_paths=shared)
    compare(read(OUT/'historical-inputs.json'), historical, 'historical shared sources')
    expected_outputs = {'contract.json', 'inputs.json', 'historical-inputs.json', 'results.json'}
    for rate in ('lower_rate', 'higher_rate'):
        for seed in (921009, 931009):
            for name in ('probes', 'continuous-t0', 'continuous-t1', 'matched-t0', 'matched-t1'):
                expected_outputs.update(str(Path(f'{rate}-{seed}')/(name+suffix)) for suffix in ('.npz', '.json'))
    assert set(completed['outputs']) == expected_outputs
    actual_outputs = {str(path.relative_to(OUT)) for path in OUT.rglob('*')
                      if path.is_file() and 'sources' not in path.parts
                      and path.name not in ('completed.json', 'audit.json')}
    assert actual_outputs == expected_outputs
    assert completed['wall_seconds'] >= 0 and np.isfinite(completed['wall_seconds'])
    expected_completion = dict(wall_seconds=completed['wall_seconds'], inputs_unchanged=True,
        frozen_parameters_unchanged=True, common_probe_features_exact=True,
        physical_rollouts=0, optimizer_updates=0, critic_updates=0, new_checkpoints=0,
        actors=4, common_probe_states=36, same_input_actor_evaluations=144,
        unique_trace_sources=28, window_comparisons=16, paired_sample_rows=7680,
        historical_shared_inputs={root: row['count'] for root, row in historical.items()},
        outputs=completed['outputs'])
    compare(completed, expected_completion, 'completion')
    return inputs, dict(input_files=len(inputs), output_files=len(expected_outputs),
                        consumed_historical_outputs=consumed_counts,
                        historical_shared_inputs=expected_completion['historical_shared_inputs'])


def verify_contract():
    lower, higher = read(LOW/'contract.json'), read(HIGH/'contract.json')
    assert higher['actor_learning_rate'] == 1e-3
    assert higher['control_actor_learning_rate'] == 1e-4 and higher['critic_learning_rate'] == 3e-4
    for key in ('sigma', 'dt', 'ankles', 'control', 'training_seeds', 'initial_actor_seed'):
        assert lower[key] == higher[key], 'source contract '+key
    assert higher['sigma'] == SIGMA and higher['ankles'] == [21, 29]
    assert higher['training_seeds'] == [921009, 931009] and higher['initial_actor_seed'] == 920009
    assert abs(higher['dt']-.016668) < 1e-12
    contract = read(OUT/'contract.json')
    prose = ('reference_polarity', 'trajectory_interpretation', 'requested_addition_interpretation')
    assert all(isinstance(contract[key], str) and contract[key] for key in prose)
    expected = dict(actor_rates={'lower_rate': 1e-4, 'higher_rate': 1e-3}, seeds=[921009, 931009],
        source_roots={'lower_rate': str(LOW), 'higher_rate': str(HIGH)}, dataset_root=str(DATA),
        common_probe_keys=['probe_x', 'probe_branch', 'probe_onset', 'probe_pose', 'probe_quiet_index'],
        same_inputs_across_seeds=True, probe_count=36, quiet_probes=12, pushed_probes=24,
        probe_onsets=[44, 74], probe_poses=list(range(6)), probe_memory_zero=True,
        latent_sign_band=SIGN_BAND, sigma=SIGMA, dt=higher['dt'], ankles=[21, 29], rho=.9,
        action_radius=.1, action_slew=.02, modes=['continuous', 'matched'], onset_frames=[60, 102],
        issue_frames=[61, 103], window_intervals=30, case_slots=16, trace_poses=list(range(8)),
        reserved_poses=[6, 7], window_comparisons=16, paired_case_windows=256,
        paired_sample_rows=7680, unique_trace_sources=28, same_input_actor_evaluations=144,
        physical_rollouts=0, optimizer_updates=0, critic_updates=0, new_checkpoints=0,
        reserved_cases_remain_diagnostic=True, **{key: contract[key] for key in prose})
    compare(contract, expected, 'contract')
    return contract


def audit():
    inputs, provenance = verify_provenance()
    contract = verify_contract()
    datasets = {seed: arrays(DATA/str(seed)/'dataset.npz') for seed in (921009, 931009)}
    for source in datasets.values():
        verify_probe_identity(source)
    for key in contract['common_probe_keys']:
        assert np.array_equal(datasets[921009][key], datasets[931009][key]), key
    results = read(OUT/'results.json')
    assert set(results) == {'lower_rate', 'higher_rate'}
    maximum_forward_error, actors, windows = {}, 0, 0
    unique_trace_paths = set()
    for rate, root in (('lower_rate', LOW), ('higher_rate', HIGH)):
        assert set(results[rate]) == {'921009', '931009'}
        for seed in (921009, 931009):
            folder = OUT/f'{rate}-{seed}'
            probes, error = verify_probes(folder, root, seed, datasets[seed])
            maximum_forward_error[f'{rate}-{seed}'] = error
            saved_windows = {}
            for mode in ('continuous', 'matched'):
                for timing in (0, 1):
                    summary = verify_window(folder, root, seed, rate, mode, timing, contract['dt'])
                    saved_windows[f'{mode}-t{timing}'] = summary
                    unique_trace_paths.update(summary[key] for key in ('pushed_source', 'quiet_source'))
                    windows += 1
            compare(results[rate][str(seed)], dict(probes=probes, windows=saved_windows), f'{rate}-{seed} results')
            actors += 1
    assert actors == 4 and windows == 16 and len(unique_trace_paths) == 28
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    return dict(passed=True, **provenance, actors=actors, independent_forward_evaluations=144,
        maximum_numpy_forward_error=maximum_forward_error,
        independent_forward_atol=NUMPY_FORWARD_ATOL, independent_forward_rtol=NUMPY_FORWARD_RTOL,
        probe_mapping_atol_rad=1e-8, trace_mapping_atol_rad=1e-7,
        summary_atol=1e-10, summary_rtol=2e-7, reference_sign_band_latent=SIGN_BAND,
        classification_source='Persisted float32 Torch predictions independently verified by NumPy; no sign-band widening.',
        common_inputs_exact=True, probe_pairing_and_polarity_exact=True, output_arrays_reconstructed=True,
        archived_memory_chronology_exact=True, all_groups_and_cases_recomputed=True,
        window_comparisons=windows, unique_trace_sources=len(unique_trace_paths),
        paired_case_windows=256, paired_sample_rows=7680,
        survival_filtering=False, inputs_unchanged=True, physical_rollouts=0,
        optimizer_updates=0, critic_updates=0, new_checkpoints=0,
        scope='Frozen inference and archived arithmetic only. Reference polarity is historical; no physical or causal recovery claim.')


def main():
    destination = OUT/'audit.json'
    assert not destination.exists(), 'Preserve prior audit; use a successor for any repair.'
    try:
        result = audit()
    except Exception as error:
        save(destination, dict(passed=False, error_type=type(error).__name__, error=str(error)))
        raise
    save(destination, result)
    print('COMMAND GENERALIZATION AUDIT PASSED', flush=True)


if __name__ == '__main__':
    main()
