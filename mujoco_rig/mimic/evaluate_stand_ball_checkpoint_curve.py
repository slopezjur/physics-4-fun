"""Frozen intermediate-checkpoint comparison; no learning or automatic promotion."""
import argparse
import gc
from pathlib import Path
import shutil
import traceback

import torch

from .audit_stand_failures import digest, read, save
from .evaluate_stand_ball_boundary import evaluate as native_ball
from .evaluate_stand_ball_gpu import evaluate as gpu_ball
from .evaluate_stand_sustained import evaluate_duration, verify_duration
from .replicate_stand_representations import tensor_hash
from .rig import Rig
from .runtime import activate
from .stand_experiment_evidence import load_actor, protect, seal


BACKENDS = ('native', 'gpu')
EVALUATORS = dict(native=native_ball, gpu=gpu_ball)


def settings_from(path):
    settings = read(path)
    fixed = dict(schema='stand_ball_checkpoint_curve_v1', seeds=[301027, 301028],
        arms=['quiet', 'single_ball'], checkpoints=[100, 200], seconds=12.,
        launch_seconds=5., launch_distance_m=.65, quiet_retention_seconds=120.,
        screen_cases=256, maximum_conditional_cases=384,
        maximum_evaluation_cases=640, training_updates=0)
    if any(settings.get(key) != value for key, value in fixed.items()):
        raise ValueError('Changed scientific settings require a successor protocol')
    return settings


def assess_impacts(actor, rig, reference, world, out, settings, mass, speed):
    out.mkdir()
    expected = tensor_hash(actor.state_dict())
    impacts = {}
    for sign in (1, -1):
        name = f'{mass}kg-{speed:g}mps-{sign:+d}'
        impacts[name] = {}
        for backend, evaluator in EVALUATORS.items():
            result = evaluator(rig, reference, actor, world, out/f'{name}-{backend}',
                settings, speed, sign)
            assert result['actor_hash'] == expected
            impacts[name][backend] = result
        save(out/'progress.json', impacts)
    assert tensor_hash(actor.state_dict()) == expected
    result = dict(actor_hash=expected, impacts=impacts, cases=32)
    save(out/'report.json', result)
    return result


def compare_selected(source, endpoints):
    rows = {}
    names = ('5kg-1mps-+1', '5kg-1mps--1')
    for backend in BACKENDS:
        actors = dict(source=source, **endpoints)
        counts = {label: {name: actor['impacts'][name][backend]['recovery_passes']
                         for name in names} for label, actor in actors.items()}
        totals = {label: sum(values.values()) for label, values in counts.items()}
        hits = all(actor['impacts'][name][backend]['hit_cases'] == 8
                   for actor in actors.values() for name in names)
        no_regression = all(counts['single_ball'][name] >=
            max(counts['source'][name], counts['quiet'][name]) for name in names)
        gain = totals['single_ball'] > max(totals['source'], totals['quiet'])
        rows[backend] = dict(counts=counts, totals=totals, all_shots_hit=hits,
            no_direction_regression=no_regression, strict_gain=gain,
            passed=bool(hits and no_regression and gain))
    return dict(passed=all(row['passed'] for row in rows.values()), backends=rows)


def run(args):
    if not __debug__:
        raise RuntimeError('Scientific checks require Python assertions enabled')
    settings = settings_from(args.protocol)
    activate(args.mimickit)
    from util import mp_util
    mp_util.init(0, 1, 'cuda:0', None)
    torch.set_num_threads(1)
    prior = Path(settings['prior'])
    verified = read(prior/'final-verification.json')
    if not verified.get('verified') or not verified.get('inputs_and_sources_unchanged'):
        raise ValueError('Verified completed pilot required')
    parent = read(prior/'protocol.json')
    package = Path(__file__).parent
    inputs, sources = protect(prior, args.out, [Path(__file__),
        package/'scripts/evaluate-stand-ball-checkpoints.ps1'], extra_inputs=[args.protocol])
    canonical = {str(Path(path).resolve()): value for path, value in inputs.items()}
    historical = read(prior/'report.json')
    assert historical['updates'] == 1200 and historical['evaluation_cases'] == 480
    save(args.out/'protocol.json', dict(inputs=inputs, sources=sources, settings=settings,
        config=parent['config'], demonstrations=parent['demonstrations'],
        interpretation='Exploratory curve, never a replacement for the sealed 300-update result',
        conditional_rule='Same checkpoint: strict selected-level gain above source and matched quiet on both seeds/backends; no direction regression; every shot hits',
        conditional_assessments='All four actors at each qualifying checkpoint: 120-second Stand and 2kg/2mps on both backends',
        selection='Report every checkpoint; no automatic selection or promotion',
        reused_assessments='Sealed source and update-300 results; no new cases counted for reuse',
        automatic_arena_selection=False))
    rig = Rig.load(args.out/'dummy.xml', policy_exclude=())
    reference = args.out/'stationary_reference.npz'
    worlds = {}
    for mass in (2, 5):
        worlds[mass] = args.out/f'ball-{mass}kg.xml'
        shutil.copy2(prior/worlds[mass].name, worlds[mass])

    def actor_at(seed, arm, update):
        checkpoint = prior/str(seed)/arm/f'updates-{update}'/'model.pt'
        for path in (checkpoint, checkpoint.with_suffix('.amp.pt')):
            if canonical.get(str(path.resolve())) != digest(path):
                raise ValueError(f'Checkpoint differs from sealed pilot: {path}')
        return load_actor(rig, reference, parent, checkpoint)

    assessments, comparisons, cases = {}, {}, 0
    for update in settings['checkpoints']:
        key = str(update)
        assessments[key], comparisons[key] = {}, {}
        for seed in settings['seeds']:
            seed_key = str(seed)
            endpoints = {}
            for arm in settings['arms']:
                folder = args.out/key/seed_key/arm
                folder.mkdir(parents=True)
                actor = actor_at(seed, arm, update)
                endpoints[arm] = assess_impacts(actor, rig, reference, worlds[5],
                    folder/'selected', settings, 5, 1.)
                cases += 32
                del actor
                gc.collect()
                torch.cuda.empty_cache()
            assessments[key][seed_key] = endpoints
            comparisons[key][seed_key] = compare_selected(
                historical['results'][seed_key]['initial'], endpoints)
            save(args.out/'progress.json', dict(evaluation_cases=cases,
                assessments=assessments, comparisons=comparisons))
    assert cases == settings['screen_cases']
    comparisons['300'] = {seed: compare_selected(row['initial'], row['final'])
                          for seed, row in historical['results'].items()}
    qualifying = [update for update in settings['checkpoints']
                  if all(row['passed'] for row in comparisons[str(update)].values())]
    retention = {}
    for update in qualifying:
        key = str(update)
        retention[key] = {}
        for seed in settings['seeds']:
            retention[key][str(seed)] = {}
            for arm in settings['arms']:
                folder = args.out/key/str(seed)/arm
                actor = actor_at(seed, arm, update)
                expected = tensor_hash(actor.state_dict())
                assert expected == assessments[key][str(seed)][arm]['actor_hash']
                quiet = {}
                for backend in BACKENDS:
                    quiet[backend] = evaluate_duration(rig, reference, 'target_pd',
                        backend, actor, folder/f'quiet-{backend}',
                        seconds=settings['quiet_retention_seconds'])
                    verify_duration(folder/f'quiet-{backend}', settings['quiet_retention_seconds'])
                    assert quiet[backend]['actor_hash'] == expected
                    cases += 8
                easy = assess_impacts(actor, rig, reference, worlds[2],
                    folder/'easy', settings, 2, 2.)
                cases += 32
                passed = all(r['standing_passes'] == 8 for r in quiet.values()) and all(
                    r['hit_cases'] == 8 and r['recovery_passes'] == 8
                    for levels in easy['impacts'].values() for r in levels.values())
                retention[key][str(seed)][arm] = dict(quiet=quiet, easy=easy, passed=passed)
                save(args.out/'retention-progress.json', dict(evaluation_cases=cases, retention=retention))
                del actor
                gc.collect()
                torch.cuda.empty_cache()
    exploratory_candidates = [int(key) for key, seeds in retention.items()
        if all(row['passed'] for arms in seeds.values() for row in arms.values())]
    assert cases == 256 + 192*len(qualifying) <= settings['maximum_evaluation_cases']
    result = dict(assessments=assessments, comparisons=comparisons, retention=retention,
        qualifying_checkpoints=qualifying, exploratory_candidates=exploratory_candidates,
        evaluation_cases=cases, new_training_updates=0, promoted=False, arena_changed=False,
        historical_pilot_replicated_improvement=historical['replicated_improvement'])
    save(args.out/'report.json', result)
    seal(args.out, inputs, sources, verified=True, evaluation_cases=cases,
        training_updates=0, exploratory_candidates=exploratory_candidates, promoted=False)
    print(dict(completed=True, evaluation_cases=cases,
        exploratory_candidates=exploratory_candidates, promoted=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, required=True)
    args = parser.parse_args()
    existed = args.out.exists()
    try:
        run(args)
    except Exception:
        if not existed and args.out.is_dir():
            save(args.out/'failure.json', dict(error=traceback.format_exc(), promoted=False))
        raise
