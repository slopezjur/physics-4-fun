"""Prespecified paired action-credit analysis of audited frozen continuations."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .run_support_skill_study import digest, save

OUT = Path('logs/mimickit-training/recovery-local-action-credit-20261009-01')
ACTORS = (921009, 931009)
ONSETS = (44, 74)
NOISE_SEEDS = (951011, 951012, 951013, 951014)
METRICS = ('reward_return', 'gae0', 'lambda_reward0', 'value_contribution0',
           'return64', 'bootstrapped_return64')
OUTCOMES = ('survived', 'settled', 'whole_stand')
RANK_TOLERANCE = 1e-3
DECOMPOSITION_TOLERANCE = 2e-4
COMPARISONS = {
    'gae_vs_full_return': ('gae0', 'reward_return'),
    'gae_vs_observed_return64': ('gae0', 'return64'),
    'gae_vs_bootstrapped_return64': ('gae0', 'bootstrapped_return64'),
    'gae_vs_lambda_reward': ('gae0', 'lambda_reward0'),
    'lambda_reward_vs_full_return': ('lambda_reward0', 'reward_return'),
    'observed_return64_vs_full_return': ('return64', 'reward_return'),
    'bootstrap_vs_observed_return64': ('bootstrapped_return64', 'return64'),
}
SCOPE = (
    'Positive-minus-negative first-innovation contrasts under common future noise. '
    'Raw and conservative-tolerance signs are descriptive, not significance tests. '
    'Four noise draws are repeated continuations of the same start, not four '
    'independent physical start states. GAE, lambda-weighted rewards, observed '
    'returns and bootstrapped returns have different horizons or targets. A '
    'ranking reversal does not by itself establish critic miscalibration. '
    'No optimizer, new physical simulation or policy checkpoint is used.'
)


def sign(value, tolerance=RANK_TOLERANCE):
    return 1 if value > tolerance else -1 if value < -tolerance else 0


def sign_counts(values, tolerance=RANK_TOLERANCE):
    counts = Counter(sign(float(value), tolerance) for value in values)
    return dict(positive=counts[1], negative=counts[-1], tied=counts[0])


def relation(left, right, tolerance):
    a, b = sign(left, tolerance), sign(right, tolerance)
    if a == 0 or b == 0:
        return 'both_tied' if a == b else 'left_tied' if a == 0 else 'right_tied'
    return 'agree' if a == b else 'oppose'


def comparisons(deltas):
    return {name: dict(raw=relation(deltas[a], deltas[b], 0.),
                       robust=relation(deltas[a], deltas[b], RANK_TOLERANCE))
            for name, (a, b) in COMPARISONS.items()}


def summarize(rows, unit):
    def numeric(values):
        values = np.asarray(values, dtype=float)
        return dict(mean=float(values.mean()), median=float(np.median(values)),
                    minimum=float(values.min()), maximum=float(values.max()))

    return dict(unit=unit, count=len(rows),
        metrics={metric: dict(**numeric([r['deltas'][metric] for r in rows]),
            raw_signs=sign_counts([r['deltas'][metric] for r in rows], 0.),
            robust_signs=sign_counts([r['deltas'][metric] for r in rows]))
            for metric in METRICS},
        comparisons={name: {kind: dict(Counter(r['comparisons'][name][kind] for r in rows))
            for kind in ('raw', 'robust')} for name in COMPARISONS},
        outcome_delta_signs={key: sign_counts([r['outcome_deltas'][key] for r in rows], 0.)
            for key in OUTCOMES},
        positive_outcome_mean={key: float(np.mean([r['positive'][key] for r in rows]))
            for key in OUTCOMES},
        negative_outcome_mean={key: float(np.mean([r['negative'][key] for r in rows]))
            for key in OUTCOMES},
        mean_step_count={arm: float(np.mean([r[arm]['step_count'] for r in rows]))
            for arm in ('positive', 'negative')})


def grouped_summary(rows, unit):
    output = {}
    for actor in ACTORS:
        selected = [r for r in rows if r['actor'] == actor]
        output[str(actor)] = {branch: summarize(
            [r for r in selected if branch == 'all' or r['branch'] == branch
             or (branch == 'push' and r['branch'] != 'quiet')], unit)
            for branch in ('all', 'quiet', 'positive_x', 'negative_x', 'push')}
    return output


def main():
    source = Path(__file__).resolve()
    audit_path, completion_path = OUT/'audit.json', OUT/'completed.json'
    audit = json.loads(audit_path.read_text())
    assert audit['passed']
    completion = json.loads(completion_path.read_text())
    sealed = completion['outputs']
    original_inputs_path = OUT/'inputs.json'
    original_inputs = json.loads(original_inputs_path.read_text())
    source_entries = [sha for path, sha in original_inputs.items()
                      if Path(path).resolve() == source]
    assert source_entries == [digest(source)], 'Analysis must match prespecified run source'
    assert digest(OUT/'sources'/source.name) == digest(source)
    inputs = {str(path): digest(path) for path in
              (source, audit_path, completion_path, original_inputs_path)}
    assert inputs[str(original_inputs_path)] == sealed[str(original_inputs_path.relative_to(OUT))]
    reports = {}
    paths = sorted(OUT.glob('*/report.json'))
    assert len(paths) == 64
    max_decomposition_error = 0.
    for path in paths:
        sha = digest(path)
        assert sha == sealed[str(path.relative_to(OUT))]
        inputs[str(path)] = sha
        report = json.loads(path.read_text())
        actor, onset, noise = report['actor'], report['onset_frame'], report['noise_seed']
        quiet, first_sign = report['quiet'], report['first_sign']
        assert actor in ACTORS and onset in ONSETS and noise in NOISE_SEEDS
        assert type(quiet) is bool and first_sign in (-1, 1)
        expected_worlds = list(range(6)) if quiet else [*range(6), *range(8, 14)]
        assert report['worlds'] == expected_worlds
        cases = {c['world']: c for c in report['cases']}
        assert len(cases) == len(report['cases']) and sorted(cases) == expected_worlds
        for world, case in cases.items():
            assert case['pose'] == world % 8 and case['pose'] < 6
            assert all(np.isfinite(case[key]) for key in METRICS)
            error = abs(case['gae0']-case['lambda_reward0']-case['value_contribution0'])
            assert error <= DECOMPOSITION_TOLERANCE
            max_decomposition_error = max(max_decomposition_error, error)
        key = (actor, onset, quiet, noise, first_sign)
        assert key not in reports
        reports[key] = (path.parent.name, cases)

    pairs = []
    for actor in ACTORS:
        for onset in ONSETS:
            for quiet in (True, False):
                for noise in NOISE_SEEDS:
                    negative_name, negative = reports[actor, onset, quiet, noise, -1]
                    positive_name, positive = reports[actor, onset, quiet, noise, 1]
                    assert negative.keys() == positive.keys()
                    for world in negative:
                        n, p = negative[world], positive[world]
                        arm_values = {arm: {**{key: float(case[key]) for key in METRICS},
                            **{key: bool(case[key]) for key in OUTCOMES},
                            'step_count': len(case['steps'])}
                            for arm, case in (('negative', n), ('positive', p))}
                        deltas = {key: float(p[key]-n[key]) for key in METRICS}
                        decomposition_error = abs(deltas['gae0']-deltas['lambda_reward0']-
                                                  deltas['value_contribution0'])
                        assert decomposition_error <= 2*DECOMPOSITION_TOLERANCE
                        branch = 'quiet' if quiet else 'positive_x' if world < 8 else 'negative_x'
                        pairs.append(dict(actor=actor, onset_frame=onset, branch=branch,
                            world=world, pose=world % 8, noise_seed=noise,
                            negative_folder=negative_name, positive_folder=positive_name,
                            **arm_values, deltas=deltas, comparisons=comparisons(deltas),
                            outcome_deltas={key: int(p[key])-int(n[key]) for key in OUTCOMES},
                            gae_decomposition_delta_error=decomposition_error))
    assert len(pairs) == 288

    by_start = defaultdict(list)
    for row in pairs:
        key = (row['actor'], row['onset_frame'], row['branch'], row['world'])
        by_start[key].append(row)
    means = []
    for (actor, onset, branch, world), rows in sorted(by_start.items()):
        assert sorted(r['noise_seed'] for r in rows) == list(NOISE_SEEDS)
        deltas = {metric: float(np.mean([r['deltas'][metric] for r in rows]))
                  for metric in METRICS}
        arms = {arm: {key: float(np.mean([r[arm][key] for r in rows]))
                      for key in (*METRICS, *OUTCOMES, 'step_count')}
                for arm in ('positive', 'negative')}
        spread = {}
        for metric in METRICS:
            values = np.asarray([r['deltas'][metric] for r in rows], dtype=float)
            spread[metric] = dict(sample_standard_deviation=float(values.std(ddof=1)),
                minimum=float(values.min()), maximum=float(values.max()),
                raw_signs=sign_counts(values, 0.), robust_signs=sign_counts(values),
                all_four_strictly_positive=bool((values > RANK_TOLERANCE).all()),
                all_four_strictly_negative=bool((values < -RANK_TOLERANCE).all()))
        means.append(dict(actor=actor, onset_frame=onset, branch=branch, world=world,
            pose=world % 8, noise_seeds=list(NOISE_SEEDS), continuations_per_arm=4,
            **arms, deltas=deltas, comparisons=comparisons(deltas), noise_spread=spread,
            outcome_deltas={key: arms['positive'][key]-arms['negative'][key]
                            for key in OUTCOMES}))
    assert len(means) == 72
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    contract = dict(direction='positive minus negative first innovation', gamma=.995,
        gae_lambda=.98, gae_intervals=64, robust_sign_absolute_tolerance=RANK_TOLERANCE,
        tolerance_scope='Conservative numerical sign band, not a significance or practical-effect threshold',
        decomposition_absolute_tolerance_per_trajectory=DECOMPOSITION_TOLERANCE,
        case_count=576, paired_comparisons=288, actor_start_means=72,
        noise_repeats_per_arm=4, physical_starts_per_actor=36,
        scope=SCOPE)
    save(OUT/'paired-credit.json', dict(inputs=inputs, contract=contract, pairs=pairs,
        four_noise_start_means=means, max_trajectory_gae_decomposition_error=max_decomposition_error))
    save(OUT/'results.json', dict(inputs=inputs,
        paired_credit_sha256=digest(OUT/'paired-credit.json'), contract=contract,
        individual_pair_summary=grouped_summary(pairs, 'one common-noise paired continuation'),
        four_noise_start_summary=grouped_summary(means, 'mean of four paired continuations at one actor/start'),
        max_trajectory_gae_decomposition_error=max_decomposition_error))
    print(json.dumps(dict(paired_comparisons=len(pairs), actor_start_means=len(means),
        max_trajectory_gae_decomposition_error=max_decomposition_error,
        paired_credit=str(OUT/'paired-credit.json'), results=str(OUT/'results.json'))), flush=True)


if __name__ == '__main__':
    main()
