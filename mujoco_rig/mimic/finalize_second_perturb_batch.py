"""Reconcile the twenty completed experiments without running more physics."""
import argparse
import json
from pathlib import Path
import shutil

from .audit_stand_failures import digest, read, save
from .stand_experiment_evidence import seal


def run(args):
    spec = read(args.protocol)
    assert len(spec['experiments']) == 20
    assert [r['step'] for r in spec['experiments']] == list(range(1, 21))
    inputs = {str(args.protocol): digest(args.protocol), str(Path(__file__)): digest(Path(__file__))}
    seen, rows = set(), []

    def checked_report(root):
        root = Path(root)
        proof = read(root / 'final-verification.json')
        assert proof['verified'] and proof['inputs_and_sources_unchanged']
        inputs[str(root / 'final-verification.json')] = digest(root / 'final-verification.json')
        hashes = {str(Path(p).resolve()): h for p, h in proof['output_hashes'].items()}
        path = root / 'report.json'
        assert hashes[str(path.resolve())] == digest(path)
        inputs[str(path)] = digest(path)
        return read(path)

    for experiment in spec['experiments']:
        cases, updates, transitions, presentations = 0, 0, 0, 0
        for root in experiment['evaluations']:
            assert root not in seen
            seen.add(root)
            report = checked_report(root)
            assert report['training_updates'] == 0
            cases += report['evaluation_cases']
        for root in experiment.get('learning', []):
            assert root not in seen
            seen.add(root)
            report = checked_report(root)
            assert report['evaluation_cases'] == 0
            updates += report['training_updates']
            transitions += report['transitions']
            presentations += report.get('rehearsal_example_presentations', 0)
        assert cases == experiment['new_cases']
        assert updates == experiment.get('updates', 0)
        assert transitions == updates * 4096
        for root in experiment['reviews']:
            checked_report(root)
        rows.append(dict(step=experiment['step'], global_stage=experiment['global_stage'],
                         new_frozen_cases=cases, accepted_updates=updates,
                         accepted_transitions=transitions, offline_presentations=presentations))

    excluded = []
    for root_name in spec['excluded_learning_roots']:
        root = Path(root_name)
        inputs[str(root / 'interrupted.json')] = digest(root / 'interrupted.json')
        completed, transitions = 0, 0
        for path in root.glob('*/*/updates.jsonl'):
            last = json.loads(path.read_text().splitlines()[-1])
            completed += last['updates']
            transitions += last['transitions']
            inputs[str(path)] = digest(path)
        assert transitions == completed * 4096
        excluded.append(dict(root=root_name, recorded_completed_updates=completed,
                             recorded_transitions=transitions, excluded_from_comparisons=True))

    timing = checked_report(spec['timing_review'])
    assert timing['all_admitted']
    assert all(not row['lost'] for row in timing['comparisons'])
    assert sum(len(r['right_passes']) for r in timing['comparisons']) > sum(len(r['left_passes']) for r in timing['comparisons'])
    primary = checked_report(spec['primary_review'])
    primary_gains = [r for r in primary['comparisons'] if r['right'] == 's33_varied'
                     and r['left'] in ('parent_strength', 's33_quiet') and r['condition_key'][-2] == 1
                     and r['condition_key'][4] == 1.6]
    assert len(primary_gains) == 8 and all(len(r['right_passes']) > len(r['left_passes']) for r in primary_gains)
    direction = checked_report(spec['direction_review'])
    directional_rows = [r for r in direction['comparisons'] if r['left'] != 'baseline90']
    assert all(r['admitted'] and not r['lost'] for r in directional_rows)
    assert not direction['all_admitted']  # Arm-only lateral clusters remain rejected.
    repeated = checked_report(spec['repeated_review'])
    assert repeated['all_admitted']
    for row in repeated['records']:
        if row['label'].startswith('candidate_'):
            baseline = next(r for r in repeated['records'] if r['label'] == row['label'].replace('candidate_', 'baseline_')
                            and r['seed'] == row['seed'] and r['sign'] == row['sign'])
            for backend in ('native', 'gpu'):
                old = {c['case'] for c in baseline['cases'][backend] if c['recovery_pass']}
                new = {c['case'] for c in row['cases'][backend] if c['recovery_pass']}
                assert old <= new
    retention = checked_report(spec['retention_root'])
    assert retention['full_interval_stand'] and retention['evaluation_cases'] == 32
    replicate = checked_report(spec['replicate_varied_root'])
    first = next(r for r in replicate['results'] if r['seed'] == 301027
                 and r['sign'] == 1 and r['condition']['speed_m_s'] == 1.6)
    assert all(a['recovery_passes'] == 3 for a in first['assessments'].values())

    summary = dict(completed_scientific_steps=20, experiments=rows,
        new_frozen_cases=sum(r['new_frozen_cases'] for r in rows),
        accepted_training_updates=sum(r['accepted_updates'] for r in rows),
        accepted_training_transitions=sum(r['accepted_transitions'] for r in rows),
        offline_example_presentations=sum(r['offline_presentations'] for r in rows),
        excluded_learning=excluded, matched_starting_parent_count=2,
        interpretation='Counts include both simulator backends and repeated eight-pose cases; they are not independent trained humanoids.',
        first_candidate_retained_120_second_stand=True,
        first_candidate_better_on_admitted_primary_timing_direction_and_repeated_comparisons=True,
        lateral_condition_excluded_from_chest_comparison=True,
        optimization_replication_gate_passed=False, promoted=False,
        arena_modified_by_this_batch=False, further_training_running=False,
        unit_test_suites_run=0, csharp_builds_run=0,
        next_work='Improve optimization consistency and address the observed late collapse after the second stronger hit; lateral arm-first impacts need their own prospective target-body protocol.')
    assert summary['new_frozen_cases'] == 6784
    assert summary['accepted_training_updates'] == 4200
    assert summary['accepted_training_transitions'] == 17203200
    assert sum(r['recorded_completed_updates'] for r in excluded) == 256
    args.out.mkdir(exist_ok=False)
    shutil.copy2(args.protocol, args.out / 'protocol.json')
    shutil.copy2(__file__, args.out / 'review-source.py')
    save(args.out / 'report.json', summary)
    seal(args.out, inputs, [], verified=True, completed_scientific_steps=20,
         additional_simulations=0, additional_training_updates=0, promoted=False)
    print({k: summary[k] for k in ('completed_scientific_steps', 'new_frozen_cases',
          'accepted_training_updates', 'accepted_training_transitions', 'promoted')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
