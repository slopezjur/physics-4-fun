"""Finish the authorized arms while retaining the failed initial-state gate."""
import argparse
from pathlib import Path
import subprocess
import sys
import traceback

from .audit_stand_failures import read, save
from .run_perturb_pilot import compare_pair
from .stand_experiment_evidence import protect, seal
from .train_perturb_continuation import read_settings


def run(args):
    if not __debug__:
        raise RuntimeError('Evidence checks require Python assertions enabled')
    settings = read_settings(args.protocol)
    diagnostic = args.pilot/'reset-diagnosis'
    assert read(diagnostic/'final-verification.json')['verified']
    failure = read(args.pilot/'failure.json')['error']
    assert "training-start.json" in failure and 'AssertionError' in failure
    inputs, sources = protect(diagnostic, args.out, [Path(__file__)],
        extra_inputs=[args.protocol, args.pilot/'protocol.json', args.pilot/'failure.json'])
    parent = read(diagnostic/'protocol.json')
    roots = {'301027': {arm: str(args.pilot/f'301027-{arm}') for arm in ('quiet', 'perturb')}}
    save(args.out/'protocol.json', dict(inputs=inputs, sources=sources, settings=settings,
        config=parent['config'], demonstrations=parent['demonstrations'],
        model_sha256=parent['model_sha256'], reference_sha256=parent['reference_sha256'],
        existing_arms=roots, remaining_order=['301028-perturb', '301028-quiet'],
        maximum_new_updates=600, maximum_new_cases=448,
        guard_handling='Record the failed bitwise environment gate; never waive it or promote a brain. '
            'Complete only the two remaining originally authorized arms, using the unchanged sealed trainer. '
            'New reset diagnostics cannot reconstruct unsaved original training-start components.'))
    reports = {'301027': {}}
    for arm, root in roots['301027'].items():
        assert read(Path(root)/'final-verification.json')['verified']
        reports['301027'][arm] = read(Path(root)/'report.json')
    prior = diagnostic
    roots['301028'], reports['301028'] = {}, {}
    for arm in ('perturb', 'quiet'):
        root = args.out/f'301028-{arm}'
        command = [sys.executable, '-B', '-u', '-m', 'mujoco_rig.mimic.train_perturb_continuation',
            '--prior', str(prior), '--out', str(root), '--protocol', str(args.protocol),
            '--seed', '301028', '--arm', arm, '--mimickit', str(args.mimickit)]
        with (args.out/f'301028-{arm}.console.log').open('x', encoding='utf-8') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert read(root/'final-verification.json')['verified']
        roots['301028'][arm] = str(root)
        reports['301028'][arm] = read(root/'report.json')
        prior = root
        save(args.out/'progress.json', dict(completed_arms=roots))
    comparisons, starts = {}, {}
    for seed, pair in reports.items():
        quiet_root, perturb_root = (Path(roots[seed][arm]) for arm in ('quiet', 'perturb'))
        assert read(quiet_root/'protocol.json')['config'] == read(perturb_root/'protocol.json')['config']
        quiet, perturb = (read(root/'training-start.json') for root in (quiet_root, perturb_root))
        for key in ('learner', 'optimizers', 'training_mode'):
            assert quiet['fingerprint'][key] == perturb['fingerprint'][key]
        assert quiet['rng'] == perturb['rng']
        starts[seed] = dict(learner_optimizer_rng_match=True,
            environment_bitwise_equal=quiet['fingerprint']['environment'] == perturb['fingerprint']['environment'],
            raw_start_components_saved=False,
            limitation='The reset diagnostic explains a reproducible mechanism, not the exact historical difference')
        comparisons[seed] = compare_pair(pair['quiet'], pair['perturb'])
    result = dict(status='completed_with_initial_state_guard_failure', arm_roots=roots,
        comparisons=comparisons, training_start_checks=starts,
        performance_rule_passed=all(row['passes'] for pair in comparisons.values() for row in pair.values()),
        original_pair_admission_passed=False, qualifies_for_next_stage=False, promoted=False,
        training_updates=1200, transitions=4915200, cases=896,
        new_updates_in_completion=600, new_cases_in_completion=448,
        training_seconds=sum(r['training_seconds'] for pair in reports.values() for r in pair.values()),
        arena_changed=False, decision='Preserve Stand selection; retain the failed guard and all per-arm outcomes')
    comparison = args.out/'comparison'
    protected, copied_sources = protect(prior, comparison, [Path(__file__)],
        extra_inputs=[args.out/'protocol.json'])
    save(comparison/'protocol.json', dict(inputs=protected, sources=copied_sources,
        config=parent['config'], demonstrations=parent['demonstrations'],
        model_sha256=parent['model_sha256'], reference_sha256=parent['reference_sha256'], settings=settings,
        guard_handling='The original comparison gate remains failed; results are descriptive, not promoted'))
    save(comparison/'report.json', result)
    seal(comparison, protected, copied_sources, verified=True, training_updates=1200, cases=896,
        original_pair_admission_passed=False, qualifies_for_next_stage=False)
    save(args.out/'report.json', result)
    seal(args.out, inputs, sources, verified=True, new_training_updates=600, new_cases=448,
        original_pair_admission_passed=False, qualifies_for_next_stage=False)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', type=Path, required=True)
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
