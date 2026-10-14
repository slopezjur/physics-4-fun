"""Frozen command/memory decomposition over fixed archived evaluation windows."""
import numpy as np


RHO = .9
RADIUS = .1
SLEW = .02
LATENT_TOLERANCE = 1e-7
ANKLES = (21, 29)
BRANCHES = ('push', 'quiet')


def moments(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.size and np.isfinite(values).all()
    mean = float(values.mean())
    second = float(np.mean(values**2))
    return dict(samples=int(values.size), mean=mean, rms=float(np.sqrt(second)),
        second_moment=second, variance=float(np.mean((values-mean)**2)),
        maximum_absolute=float(np.abs(values).max()))


def decomposition(latent, memory, innovation):
    latent, memory, innovation = (np.asarray(value, dtype=np.float64) for value in (latent, memory, innovation))
    assert latent.shape == memory.shape == innovation.shape and latent.size
    assert np.allclose(latent, memory+innovation, atol=1e-14, rtol=1e-12)
    total, carried, new = (moments(value) for value in (latent, memory, innovation))
    cross = float(np.mean(memory*innovation))
    covariance = float(np.mean((memory-memory.mean())*(innovation-innovation.mean())))
    energy_error = total['second_moment']-carried['second_moment']-new['second_moment']-2*cross
    variance_error = total['variance']-carried['variance']-new['variance']-2*covariance
    assert np.isclose(energy_error, 0., atol=1e-14, rtol=0)
    assert np.isclose(variance_error, 0., atol=1e-14, rtol=0)
    return dict(latent=total, memory=carried, innovation=new,
        mean_memory_times_innovation=cross, twice_cross_second_moment=2*cross,
        covariance=covariance, twice_covariance=2*covariance,
        second_moment_identity_error=float(energy_error), variance_identity_error=float(variance_error))


def reference_counts(values, polarity):
    """Count agreement with a declared reference sign; this is not a return ranking."""
    signed = np.asarray(values, dtype=np.float64)*np.asarray(polarity, dtype=np.float64)
    assert signed.size and np.isfinite(signed).all()
    agreement = int((signed > LATENT_TOLERANCE).sum())
    opposition = int((signed < -LATENT_TOLERANCE).sum())
    tied = int((np.abs(signed) <= LATENT_TOLERANCE).sum())
    assert agreement+opposition+tied == signed.size
    return dict(samples=int(signed.size), agreement=agreement, opposition=opposition, tied=tied,
        signed_mean=float(signed.mean()), signed_minimum=float(signed.min()),
        signed_maximum=float(signed.max()), tolerance_latent=LATENT_TOLERANCE)


def trace_checks(trace):
    time = np.asarray(trace['time'])
    count = len(time)
    assert count > 133 and time.shape == (count,) and np.isfinite(time).all()
    dt = float(time[0])
    assert dt > 0 and np.allclose(time, (np.arange(count)+1)*dt, atol=1e-10, rtol=0)
    assert trace['latent'].shape == trace['previous_latent'].shape == (count, 16, 1)
    assert trace['previous'].shape == trace['addition'].shape == (count, 16, 33)
    assert trace['alive'].shape == trace['failed'].shape == (count, 16)
    assert trace['alive'].dtype == trace['failed'].dtype == np.bool_
    for key in ('latent', 'previous_latent', 'previous', 'addition'):
        assert np.isfinite(trace[key]).all(), key
    assert trace['alive'][0].all() and not trace['previous_latent'][0].any()
    assert np.array_equal(trace['alive'][1:], trace['alive'][:-1] & ~trace['failed'][:-1])
    expected_memory = np.where(trace['alive'][:-1, :, None], trace['latent'][:-1], trace['previous_latent'][:-1])
    assert np.array_equal(trace['previous_latent'][1:], expected_memory)
    assert not trace['previous'][0].any()
    assert np.array_equal(trace['previous'][1:], trace['addition'][:-1])
    for key in ('previous', 'addition'):
        assert np.array_equal(trace[key][:, :, ANKLES[0]], trace[key][:, :, ANKLES[1]])
        assert not trace[key][:, :, [index for index in range(33) if index not in ANKLES]].any()
    previous = trace['previous'][:, :, ANKLES[0]].astype(np.float64)
    desired = RADIUS*np.tanh(trace['latent'][:, :, 0].astype(np.float64))
    expected_addition = previous+np.clip(desired-previous, -SLEW, SLEW)
    expected_addition[~trace['alive']] = previous[~trace['alive']]
    error = float(np.abs(trace['addition'][:, :, ANKLES[0]]-expected_addition).max())
    assert error <= 1e-7
    return dict(intervals=count, dt=dt, memory_chronology_exact=True,
        addition_chronology_exact=True, inactive_memory_freezes=True,
        only_tied_ankles_modified=True, action_mapping_maximum_error_rad=error)


def selection_summary(arrays, ids):
    ids = np.asarray(ids, dtype=int)
    assert ids.size
    result = dict(cases=ids.tolist(), branches={})
    for branch, name in enumerate(BRANCHES):
        result['branches'][name] = dict(
            decomposition=decomposition(arrays['latent'][branch][:, ids, 0],
                arrays['memory'][branch][:, ids, 0], arrays['innovation'][branch][:, ids, 0]),
            previous_addition_rad=moments(arrays['previous_addition'][branch][:, ids, 0]),
            addition_rad=moments(arrays['addition'][branch][:, ids, 0]),
            active_intervals=int(arrays['alive'][branch][:, ids].sum()),
            inactive_intervals=int((~arrays['alive'][branch][:, ids]).sum()),
            failed_intervals=int(arrays['failed'][branch][:, ids].sum()))
    latent = arrays['difference_latent'][:, ids, 0]
    reference = np.where(ids < 8, -1., 1.)
    result['pushed_minus_quiet'] = dict(
        decomposition=decomposition(latent, arrays['difference_memory'][:, ids, 0],
                                     arrays['difference_innovation'][:, ids, 0]),
        addition_rad=moments(arrays['difference_addition'][:, ids, 0]),
        window_mean_reference=reference_counts(latent.mean(axis=0), reference),
        per_interval_reference=reference_counts(latent, reference[None]),
        paired_active_intervals=int(arrays['paired_alive'][:, ids].sum()),
        paired_inactive_intervals=int((~arrays['paired_alive'][:, ids]).sum()))
    return result


def analyze(push, quiet, timing, mode, seed, rate):
    """Extract and describe one fixed window; perform no policy or physics execution."""
    assert timing in (0, 1) and mode in ('continuous', 'matched')
    assert seed in (921009, 931009) and rate in ('lower_rate', 'higher_rate')
    checks = {name: trace_checks(trace) for name, trace in zip(BRANCHES, (push, quiet), strict=True)}
    assert checks['push']['dt'] == checks['quiet']['dt']
    issue = (61, 103)[timing]
    window = slice(issue, issue+30)
    assert np.array_equal(push['time'][window], quiet['time'][window])
    arrays = dict(frame=np.arange(issue, issue+30, dtype=np.int64),
        time=np.array(push['time'][window], copy=True))
    for key in ('alive', 'failed', 'latent', 'previous_latent'):
        arrays[key] = np.stack([trace[key][window] for trace in (push, quiet)])
    arrays['previous_addition'] = np.stack([trace['previous'][window][:, :, ANKLES] for trace in (push, quiet)])
    arrays['addition'] = np.stack([trace['addition'][window][:, :, ANKLES] for trace in (push, quiet)])
    arrays['memory'] = RHO*arrays['previous_latent'].astype(np.float64)
    arrays['innovation'] = arrays['latent'].astype(np.float64)-arrays['memory']
    for key in ('latent', 'memory', 'innovation', 'addition'):
        arrays['difference_'+key] = arrays[key][0].astype(np.float64)-arrays[key][1].astype(np.float64)
    arrays['paired_alive'] = arrays['alive'][0] & arrays['alive'][1]
    ids = np.arange(16)
    masks = dict(all=np.ones(16, dtype=bool), seen_pose=ids % 8 < 6, reserved_pose=ids % 8 >= 6,
        positive_x=ids < 8, negative_x=ids >= 8,
        seen_positive_x=(ids % 8 < 6) & (ids < 8), seen_negative_x=(ids % 8 < 6) & (ids >= 8),
        reserved_positive_x=(ids % 8 >= 6) & (ids < 8), reserved_negative_x=(ids % 8 >= 6) & (ids >= 8))
    cases = [dict(case=index, pose=index % 8, reserved_pose=index % 8 >= 6,
        direction='positive_x' if index < 8 else 'negative_x', reference_polarity=-1 if index < 8 else 1,
        **selection_summary(arrays, [index])) for index in range(16)]
    summary = dict(seed=seed, rate=rate, mode=mode, timing=timing, issue_frame=issue,
        intervals=30, cases=16, dt=checks['push']['dt'],
        duration_seconds=30*checks['push']['dt'], branch_order=list(BRANCHES), ankles=list(ANKLES),
        checks=checks, all_selected_intervals_active=bool(arrays['alive'].all()),
        selected_failed_intervals=int(arrays['failed'].sum()),
        grouped={name: selection_summary(arrays, ids[mask]) for name, mask in masks.items()}, per_case=cases,
        scope=dict(fixed_windows=True, survival_filtering=False, statistics_include_all_selected_intervals=True,
            latent_decomposition='Float64 remainder after subtracting 0.9 times the recorded previous latent; includes floating-point residuals.',
            energy='Second moment equals memory second moment plus innovation second moment plus twice their mean product; these are correlated components, not independent causal fractions.',
            comparison='Pushed-minus-quiet compares separate archived trajectories. It is not same-input actor sensitivity or a causal memory ablation.',
            signs='Agreement/opposition refers only to declared -X-response for +X-force and +X-response for -X-force. It is not demonstrated return optimality.',
            sign_tolerance='1e-7 latent units only; no ankle-angle sign threshold is applied.',
            quiet_copies='Quiet worlds contain two copies of each pose; counts are cases, not independent poses.'))
    return arrays, summary
