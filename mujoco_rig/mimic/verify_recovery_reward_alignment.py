"""Independent termination scan and return recursion for saved reward rankings."""
import json
from pathlib import Path

import numpy as np

from .run_support_skill_study import digest, save

OUT = Path('logs/mimickit-training/reward-alignment-20261009-01')


def read(path):
    return json.loads(path.read_text())


def main():
    inputs, done, result = (read(OUT/name) for name in ('inputs.json', 'completed.json', 'results.json'))
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    assert all(digest(OUT/p) == sha for p, sha in done['outputs'].items())
    cache, rows, max_error = {}, {}, 0.
    for row in result['rows']:
        assert row['pose'] < 6 and row['case'] % 8 == row['pose']
        folder = Path(row['source'])
        if str(folder) not in cache:
            with np.load(folder/'trace.npz', allow_pickle=False) as z:
                cache[str(folder)] = {k: z[k] for k in ('reward', 'alive', 'failed', 'settled')}
        z = cache[str(folder)]
        i, begin = row['case'], row['issue_frame']
        assert begin == (61, 103)[row['timing']]
        end = len(z['reward'])
        failure = None
        for f in range(begin, end):
            assert z['alive'][f, i]
            if z['failed'][f, i]:
                failure, end = f, f+1
                assert z['reward'][f, i] == -2.
                assert not z['alive'][end:, i].any()
                break
        total = 0.
        for f in reversed(range(begin, end)):
            total = float(z['reward'][f, i])+result['gamma']*total
        max_error = max(max_error, abs(total-row['observed_return']))
        assert abs(total-row['observed_return']) < 1e-9
        assert end-begin == row['alive_intervals']
        assert row['survived'] == (failure is None)
        assert row['settled'] == (failure is None and bool(z['settled'][-120:, i].all()))
        expected_failure = 0. if failure is None else -2.*result['gamma']**(failure-begin)
        assert abs(row['components']['failure']-expected_failure) < 1e-10
        assert abs(sum(row['components'].values())-total) < .002
        for seed, bonus in row['bootstrap_bonus'].items():
            assert row['survived'] or bonus == 0.
            assert abs(row['bootstrapped_return'][seed]-total-bonus) < 1e-9
        rows[(row['timing'], row['branch'], row['case'], row['action'])] = row
    for pair in result['pairs']:
        key = (pair['timing'], pair['branch'], pair['case'])
        pulse, zero = rows[(*key, pair['action'])], rows[(*key, 'zero')]
        expected = 'rescue' if pulse['settled'] and not zero['settled'] else 'loss' if zero['settled'] and not pulse['settled'] else 'unchanged'
        assert pair['outcome'] == expected
        assert abs(pair['delta']-(pulse['observed_return']-zero['observed_return'])) < 1e-9
        for seed in pair['bootstrap_delta']:
            assert abs(pair['bootstrap_delta'][seed]-(pulse['bootstrapped_return'][seed]-zero['bootstrapped_return'][seed])) < 1e-9
    for branch, outcomes in result['summary'].items():
        for outcome, summary in outcomes.items():
            pairs = [p for p in result['pairs'] if p['branch'] == branch and p['outcome'] == outcome]
            assert summary['count'] == len(pairs)
            assert summary['positive'] == sum(p['delta'] > 1e-6 for p in pairs)
            assert summary['negative'] == sum(p['delta'] < -1e-6 for p in pairs)
            for seed, count in summary['bootstrapped_positive'].items():
                assert count == sum(p['bootstrap_delta'][seed] > 1e-6 for p in pairs)
    for oracle in result['oracle']:
        candidates = [rows[(oracle['timing'], 'push', oracle['case'], action)] for action in ('zero', 'negative', 'positive')]
        best = max(candidates, key=lambda r: r['observed_return'])
        assert oracle['any_settled'] == any(c['settled'] for c in candidates)
        assert oracle['best_reward_action'] == best['action'] and oracle['best_reward_settled'] == best['settled']
    assert len(rows) == 144 and len(result['pairs']) == 96 and len(result['oracle']) == 24
    assert result['max_reward_error'] < 2e-5 and result['max_start_feature_error'] < 2e-5
    assert done['new_physics_intervals'] == done['optimizer_updates'] == 0
    audit = dict(passed=True, trajectories=144, paired_pulse_comparisons=96, pushed_starts=24,
        max_independent_return_error=max_error, first_failure_included=True, frozen_frames_excluded=True,
        reserved_poses_excluded=True, input_hashes=len(inputs), output_hashes=len(done['outputs']))
    save(OUT/'audit.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
