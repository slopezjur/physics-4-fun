"""Read-only learning-curve assessment with conditional long-Stand qualification."""
import argparse
import gc
from pathlib import Path
import traceback

import torch

from .audit_stand_failures import digest, read, save
from .ball_study_support import FOUNDATION, protect_used
from .evaluate_stand_sustained import evaluate_duration, verify_duration
from .rig import Rig
from .runtime import activate
from .run_chest_ball_pilot import source_assessment, compare as compare_seed
from .run_chest_learning_screen import impacts, recipe_gain
from .stand_experiment_evidence import load_actor, seal


def run(args):
    if not __debug__:
        raise RuntimeError('Assertions required')
    settings = read(args.protocol)
    expected = dict(schema='chest_checkpoint_screen_v1', seeds=[301027, 301028],
        actors=['source'], arms=['quiet', 'single_ball'], checkpoints=[100, 200],
        seconds=12, launch_seconds=5, launch_distance_m=.25,
        assessment_levels=[[5, 1], [5, 1.4]], training_speed_m_s=1.4,
        quiet_retention_seconds=120, maximum_evaluation_cases=640, training_updates=0)
    if any(settings.get(k) != v for k, v in expected.items()):
        raise ValueError('Unsupported bounded checkpoint screen')
    root = Path(settings['checkpoint_root'])
    verification = read(root / 'final-verification.json')
    if not verification['verified'] or not verification['inputs_and_sources_unchanged']:
        raise ValueError('Unverified checkpoint source')
    hashes = {str(Path(p).resolve()): h for p, h in verification['output_hashes'].items()}
    checkpoints, extra = {}, [str(root / 'final-verification.json')]
    for update in settings['checkpoints']:
        for seed in settings['seeds']:
            for arm in settings['arms']:
                model = root / str(seed) / arm / f'updates-{update}' / 'model.pt'
                for path in (model, model.with_suffix('.amp.pt')):
                    if hashes.get(str(path.resolve())) != digest(path):
                        raise ValueError(f'Checkpoint no longer matches its source seal: {path}')
                    extra.append(str(path))
                checkpoints[update, seed, arm] = model
    activate(args.mimickit)
    from util import mp_util
    from .stand_ball_native import write_world
    mp_util.init(0, 1, 'cuda:0', None)
    torch.set_num_threads(1)
    used = dict(settings, evidence=[*settings['evidence'], *extra])
    dependencies = [Path(__file__).with_name(p) for p in ('run_chest_learning_screen.py',
        'run_chest_ball_pilot.py', 'chest_reward_variant.py', 'chest_ball_curriculum.py')]
    parent, inputs, sources = protect_used(args.out, [Path(__file__), *dependencies], args.protocol, used)
    save(args.out / 'protocol.json', dict(settings=settings, inputs=inputs, sources=sources,
        config=parent['config'], demonstrations=parent['demonstrations'], training_updates=0,
        automatic_promotion=False, conditional_stand='Only replicated absolute impact gains receive fresh 120-second Stand; otherwise null'))
    rig = Rig.load(args.out / 'dummy.xml', policy_exclude=())
    reference, world = args.out / 'stationary_reference.npz', args.out / 'ball-5kg.xml'
    write_world(args.out / 'dummy.xml', FOUNDATION / 'ball-5kg.xml', world, 5.)
    results, cases = {}, 0
    for update in settings['checkpoints']:
        folder = args.out / str(update)
        folder.mkdir()
        seeds = {}
        for seed in settings['seeds']:
            seed_folder = folder / str(seed)
            seed_folder.mkdir()
            initial = source_assessment(seed, settings)
            finals = {}
            for arm in settings['arms']:
                actor = load_actor(rig, reference, parent, checkpoints[update, seed, arm])
                finals[arm] = impacts(actor, rig, reference, world, seed_folder / arm, settings)
                cases += 64
                save(args.out / 'case-progress.json', dict(evaluation_cases=cases, maximum_cases=640,
                    update=update, seed=seed, arm=arm))
                del actor
                gc.collect()
                torch.cuda.empty_cache()
            seeds[str(seed)] = dict(initial=initial, final=finals,
                recipe={arm: recipe_gain(initial, final, 1.4) for arm, final in finals.items()})
            save(folder / 'progress.json', seeds)
        eligible = {arm: all(r['recipe'][arm]['eligible'] for r in seeds.values()) for arm in settings['arms']}
        for arm, yes in eligible.items():
            if not yes:
                continue
            for seed in settings['seeds']:
                actor = load_actor(rig, reference, parent, checkpoints[update, seed, arm])
                final = seeds[str(seed)]['final'][arm]
                for backend in ('native', 'gpu'):
                    path = folder / str(seed) / arm / f'quiet-{backend}'
                    final['quiet'][backend] = evaluate_duration(rig, reference, 'target_pd', backend, actor, path, seconds=120.)
                    verify_duration(path, 120.)
                    assert final['quiet'][backend]['actor_hash'] == final['actor_hash']
                    cases += 8
                final.update(quiet_assessed=True, full_interval_stand=all(r['standing_passes'] == 8
                    for r in final['quiet'].values()), cases=80)
                save(folder / str(seed) / arm / 'report.json', final)
                del actor
                gc.collect()
                torch.cuda.empty_cache()
        for result in seeds.values():
            comparison = compare_seed(result['initial'], result['final'], 1.4)
            comparison['all_arms_stand_retained'] = comparison.pop('stand_retained')
            comparison['source_and_ball_stand_retained'] = (result['initial']['full_interval_stand']
                and result['final']['single_ball']['full_interval_stand'])
            for row in comparison['backends'].values():
                row['passed'] = bool(comparison['source_and_ball_stand_retained'] and row['all_shots_hit']
                    and row['no_direction_or_level_regressions'] and row['strict_selected_level_gain'])
            comparison['passed'] = all(r['passed'] for r in comparison['backends'].values()) and all(
                f['admitted'] and f['chest_first'] for f in result['final'].values())
            result['comparison'] = comparison
        gains = {arm: bool(yes and all(r['final'][arm]['full_interval_stand'] for r in seeds.values()))
                 for arm, yes in eligible.items()}
        results[str(update)] = dict(results=seeds, impact_eligible_arms=eligible,
            replicated_recipe_gain=gains, replicated_impact_practice_gain=all(r['comparison']['passed'] for r in seeds.values()))
        save(folder / 'report.json', results[str(update)])
        save(args.out / 'progress.json', dict(results=results, evaluation_cases=cases))
    assert 512 <= cases <= 640
    qualified = [int(u) for u, r in results.items() if r['replicated_impact_practice_gain']]
    report = dict(results=results, evaluation_cases=cases, training_updates=0,
                  qualified_impact_checkpoints=qualified, promoted=False)
    save(args.out / 'report.json', report)
    seal(args.out, inputs, sources, verified=True, evaluation_cases=cases, training_updates=0,
         qualified_impact_checkpoints=qualified, promoted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    existed = args.out.exists()
    try:
        run(args)
    except Exception:
        if not existed and args.out.is_dir():
            save(args.out / 'failure.json', dict(error=traceback.format_exc()))
        raise
