"""Run four sequential continuation arms and apply the predeclared replication gate."""
import argparse
from pathlib import Path
import subprocess
import sys
import traceback

from .audit_stand_failures import read, save
from .stand_experiment_evidence import protect, seal
from .train_perturb_continuation import BACKENDS, read_settings


def recovery_scores(assessment, backend):
    return {force: sum(assessment['pushes'][f'{force}N-{direction}'][backend]['recovery_passes']
                       for direction in ('xp', 'xm')) for force in (30, 60, 90)}


def compare_pair(quiet, perturb):
    assert quiet['seed'] == perturb['seed']
    assert quiet['initial']['full_tensor_hash'] == perturb['initial']['full_tensor_hash']
    assert quiet['updates'] == perturb['updates'] == 300
    assert quiet['transitions'] == perturb['transitions'] == 1228800
    assert quiet['forced_world_control_intervals'] == 0 < perturb['forced_world_control_intervals']
    comparison = {}
    for backend in BACKENDS:
        baseline = recovery_scores(quiet['initial'], backend)
        repeated = recovery_scores(perturb['initial'], backend)
        control = recovery_scores(quiet['final'], backend)
        candidate = recovery_scores(perturb['final'], backend)
        # An unstable baseline cannot support a clean improvement claim.
        baseline_agrees = all(quiet['initial']['pushes'][f'{force}N-{direction}'][backend]['recovery_cases']
            == perturb['initial']['pushes'][f'{force}N-{direction}'][backend]['recovery_cases']
            for force in (30, 60, 90) for direction in ('xp', 'xm'))
        retention = all(report['final']['quiet'][backend]['standing_passes'] == 8 for report in (quiet, perturb))
        no_regression = all(candidate[f] >= max(baseline[f], control[f]) for f in (30, 60, 90))
        gain = sum(candidate[f] for f in (60, 90)) > max(
            sum(baseline[f] for f in (60, 90)), sum(control[f] for f in (60, 90)))
        comparison[backend] = dict(baseline=baseline, repeated_baseline=repeated, quiet_control=control,
            perturb=candidate, repeated_baseline_agrees=baseline_agrees, quiet_retained=retention,
            no_force_level_regression=no_regression, moderate_recovery_improved=gain,
            passes=bool(baseline_agrees and retention and no_regression and gain))
    return comparison


def run(args):
    if not __debug__:
        raise RuntimeError('Evidence checks require Python assertions enabled')
    settings = read_settings(args.protocol)
    inputs, sources = protect(Path(settings['prior']), args.out, [Path(__file__),
        Path(__file__).with_name('train_perturb_continuation.py')], extra_inputs=[args.protocol])
    prior_protocol = read(Path(settings['prior'])/'protocol.json')
    save(args.out/'protocol.json', dict(inputs=inputs, sources=sources, settings=settings,
        config=prior_protocol['config'], demonstrations=prior_protocol['demonstrations'],
        model_sha256=prior_protocol['model_sha256'], reference_sha256=prior_protocol['reference_sha256'],
        arm_order={'301027': ['quiet', 'perturb'], '301028': ['perturb', 'quiet']},
        maximum_training_updates=1200, maximum_cases=896,
        promotion_rule='Both seeds and backends: stable repeated baseline, both arms retain 8/8 quiet '
            '120-second Stand, no loss at any assessed force level versus source or quiet continuation, '
            'and strictly higher combined 60/90 N recovery than both comparators. No automatic promotion.'))
    prior, reports, roots = Path(settings['prior']), {}, {}
    for seed in settings['seeds']:
        reports[str(seed)], roots[str(seed)] = {}, {}
        order = ('quiet', 'perturb') if seed == settings['seeds'][0] else ('perturb', 'quiet')
        for arm in order:
            root = args.out/f'{seed}-{arm}'
            command = [sys.executable, '-B', '-u', '-m', 'mujoco_rig.mimic.train_perturb_continuation',
                '--prior', str(prior), '--out', str(root), '--protocol', str(args.protocol),
                '--seed', str(seed), '--arm', arm, '--mimickit', str(args.mimickit)]
            with (args.out/f'{seed}-{arm}.console.log').open('x', encoding='utf-8') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            assert read(root/'final-verification.json')['verified']
            reports[str(seed)][arm] = read(root/'report.json')
            roots[str(seed)][arm] = str(root)
            prior = root
            save(args.out/'progress.json', dict(completed_arms=roots))
        quiet_root, perturb_root = (Path(roots[str(seed)][a]) for a in ('quiet', 'perturb'))
        assert read(quiet_root/'protocol.json')['config'] == read(perturb_root/'protocol.json')['config']
        assert read(quiet_root/'training-start.json') == read(perturb_root/'training-start.json')
    comparisons = {seed: compare_pair(pair['quiet'], pair['perturb']) for seed, pair in reports.items()}
    eligible = all(row['passes'] for pair in comparisons.values() for row in pair.values())
    result = dict(status='complete', comparisons=comparisons, arm_roots=roots,
        qualifies_for_next_stage=eligible, training_updates=1200, transitions=4915200, cases=896,
        training_seconds=sum(r['training_seconds'] for pair in reports.values() for r in pair.values()),
        arena_changed=False, promoted=False,
        decision='Prepare a separately bounded next stage' if eligible else 'Retain Stand sources; diagnose saved failures')
    # Seal the comparison as a successor of the final arm, keeping the evolving
    # orchestrator progress/log files outside its protected input manifest.
    comparison_root = args.out/'comparison'
    comparison_inputs, comparison_sources = protect(prior, comparison_root, [Path(__file__)],
        extra_inputs=[args.out/'protocol.json'])
    save(comparison_root/'protocol.json', dict(inputs=comparison_inputs, sources=comparison_sources,
        config=prior_protocol['config'], demonstrations=prior_protocol['demonstrations'],
        model_sha256=prior_protocol['model_sha256'], reference_sha256=prior_protocol['reference_sha256'],
        settings=settings, arm_roots=roots))
    save(comparison_root/'report.json', result)
    seal(comparison_root, comparison_inputs, comparison_sources, verified=True, cases=896,
        training_updates=1200, matched_warm_starts=True, qualifies_for_next_stage=eligible)
    save(args.out/'report.json', result)
    seal(args.out, inputs, sources, verified=True, cases=896, training_updates=1200,
        qualifies_for_next_stage=eligible)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, required=True)
    arguments = parser.parse_args()
    existed = arguments.out.exists()
    try:
        run(arguments)
    except Exception:
        if not existed and arguments.out.is_dir():
            save(arguments.out/'failure.json', dict(error=traceback.format_exc()))
        raise
