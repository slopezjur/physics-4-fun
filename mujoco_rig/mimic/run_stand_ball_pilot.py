"""Bounded, matched ball-versus-quiet continuation with isolated endpoint assessments."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from .audit_stand_failures import digest, read, save
from .evaluate_stand_ball_boundary import evaluate as native_ball
from .evaluate_stand_ball_gpu import evaluate as gpu_ball
from .evaluate_stand_sustained import evaluate_duration, verify_duration
from .measure_stand_learning_curve import rng_state, restore_rng, training_fingerprint, tree_hash
from .policy import StandPolicy
from .prepare_stand_ball_pair import prepare_pair, extra_state, identical
from .replicate_stand_representations import tensor_hash
from .rig import Rig
from .runtime import activate
from .stand_ball_training_evidence import TrainingEvidence
from .stand_experiment_evidence import protect, seal, load_actor
from .stand_start_guard import capture, compare


BACKENDS = ('native', 'gpu')


def read_settings(path):
    settings = read(path)
    fixed = dict(schema='stand_ball_matched_pilot_v1', seeds=[301027, 301028],
        arms=['quiet', 'single_ball'], envs=64, rollout_steps=64, updates_per_arm=300,
        maximum_training_seconds_per_arm=600., training_mass_kg=5., training_speed_m_s=1.,
        quiet_fraction=.5, seconds=12., launch_seconds=5., launch_distance_m=.65,
        quiet_retention_seconds=120., assessment_levels=[[2, 2.], [5, 1.]],
        checkpoint_interval=100, maximum_updates=1200, maximum_transitions=4915200,
        maximum_evaluation_cases=480)
    if any(settings.get(key) != value for key, value in fixed.items()):
        raise ValueError('Changed scientific settings require a successor protocol')
    return settings


def pair_fingerprint(pair):
    return {arm: dict(common=training_fingerprint(agent, task),
        extra=tree_hash({key: torch.from_numpy(value) for key, value in extra_state(task).items()}),
        rng=tree_hash(rng_state(task))) for arm, (agent, task, _) in pair.items()}


def assess(actor, rig, reference, worlds, out, settings):
    out.mkdir()
    quiet, impacts = {}, {}
    actor.eval().requires_grad_(False)
    expected_hash = tensor_hash(actor.state_dict())
    for backend in BACKENDS:
        folder = out/f'quiet-{backend}'
        quiet[backend] = evaluate_duration(rig, reference, 'target_pd', backend, actor,
            folder, seconds=settings['quiet_retention_seconds'])
        verify_duration(folder, settings['quiet_retention_seconds'])
        assert quiet[backend]['actor_hash'] == expected_hash
    for mass, speed in settings['assessment_levels']:
        for sign in (1, -1):
            name = f'{mass}kg-{speed:g}mps-{sign:+d}'
            impacts[name] = {}
            for backend, evaluator in (('native', native_ball), ('gpu', gpu_ball)):
                result = evaluator(rig, reference, actor, worlds[mass], out/f'{name}-{backend}',
                    settings, speed, sign)
                assert result['actor_hash'] == expected_hash
                impacts[name][backend] = result
            save(out/'impact-progress.json', impacts)
    result = dict(actor_hash=expected_hash, quiet=quiet, impacts=impacts, cases=80,
        full_interval_stand=all(r['standing_passes'] == 8 for r in quiet.values()))
    save(out/'report.json', result)
    return result


def save_checkpoint(agent, out):
    from .amp_recovery import validate_sidecar
    out.mkdir()
    agent.save(str(out/'model.pt'))
    full = torch.load(out/'model.amp.pt', map_location='cpu', weights_only=True)
    validate_sidecar(torch.load(out/'model.pt', map_location='cpu', weights_only=True), full)
    if tensor_hash(full) != tensor_hash(agent.state_dict()):
        raise RuntimeError('Saved full learner differs from the live learner')


def train_arm(pair, arm, seed, starts, out, settings):
    agent, task, initial_rng = pair[arm]
    restore_rng(task, initial_rng)
    out.mkdir()
    capture(out/'actual-start', agent, task)
    start = compare(starts/arm, out/'actual-start')
    with np.load(starts/arm/'extra-state.npz', allow_pickle=False) as saved:
        current = extra_state(task)
        extras_equal = all(identical(saved[key], current[key]) for key in saved.files)
    save(out/'start-admission.json', dict(common=start, extras_equal=extras_equal))
    if not start['passed'] or not extras_equal:
        raise RuntimeError('Actual training start differs from the admitted live pair')
    save(out/'contract.json', task.observation_contract())
    save_checkpoint(agent, out/'initial')
    records = out/'rollouts'
    records.mkdir()
    elapsed, launches, contacts = 0., 0, 0
    with TrainingEvidence(task) as evidence:
        while agent._iter < settings['updates_per_arm']:
            if elapsed >= settings['maximum_training_seconds_per_arm']:
                raise RuntimeError('Training time cap reached; incomplete arms cannot be compared')
            task.launch_records.clear()
            torch.cuda.synchronize()
            began = time.perf_counter()
            info = agent._train_iter()
            torch.cuda.synchronize()
            elapsed += time.perf_counter()-began
            agent._sample_count = agent._update_sample_count()
            agent._iter += 1
            counts = evidence.flush(records/f'{agent._iter:04d}.npz')
            launches += counts['launches']
            contacts += counts['contact_world_intervals']
            row = dict(seed=seed, arm=arm, updates=agent._iter, transitions=agent._sample_count,
                training_seconds=elapsed, launches=launches, contact_world_intervals=contacts,
                actor_optimizer_steps=agent._actor_optimizer.get_steps(),
                critic_optimizer_steps=agent._critic_optimizer.get_steps(),
                **{key: float(value) for key, value in info.items()})
            assert all(np.isfinite(value) for key, value in row.items() if key != 'arm')
            assert row['transitions'] == 4096*agent._iter
            assert row['actor_optimizer_steps'] == 48*agent._iter
            assert row['critic_optimizer_steps'] == 64*agent._iter
            assert all(torch.isfinite(value).all() for value in agent.state_dict().values())
            with (out/'updates.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            save(out/'progress.json', row)
            if agent._iter % settings['checkpoint_interval'] == 0:
                save_checkpoint(agent, out/f'updates-{agent._iter}')
                print(row, flush=True)
    if elapsed > settings['maximum_training_seconds_per_arm']:
        raise RuntimeError('Last update exceeded training cap; no successful comparison')
    if arm == 'single_ball' and (launches == 0 or contacts == 0):
        raise RuntimeError('Ball arm received no physical impact practice')
    assert arm != 'quiet' or (launches == 0 and contacts == 0)
    save_checkpoint(agent, out/'final')
    result = dict(updates=agent._iter, transitions=agent._sample_count, training_seconds=elapsed,
        launches=launches, contact_world_intervals=contacts)
    save(out/'training-report.json', result)
    return result


def compare_seed(initial, finals):
    quiet, ball = finals['quiet'], finals['single_ball']
    rows = {}
    retained = initial['full_interval_stand'] and quiet['full_interval_stand'] and ball['full_interval_stand']
    for backend in BACKENDS:
        counts = {label: {name: result['impacts'][name][backend]['recovery_passes']
                          for name in initial['impacts']}
                  for label, result in (('source', initial), ('quiet', quiet), ('single_ball', ball))}
        all_hits = all(result['impacts'][name][backend]['hit_cases'] == 8
                       for result in (initial, quiet, ball) for name in initial['impacts'])
        no_regressions = all(counts['single_ball'][name] >= max(counts['source'][name], counts['quiet'][name])
                             for name in initial['impacts'])
        selected = {label: sum(value for name, value in level.items() if name.startswith('5kg-'))
                    for label, level in counts.items()}
        improvement = selected['single_ball'] > max(selected['source'], selected['quiet'])
        rows[backend] = dict(passed=bool(retained and all_hits and no_regressions and improvement),
            counts=counts, selected_level_totals=selected, all_shots_hit=all_hits,
            no_direction_or_level_regressions=no_regressions, strict_selected_level_gain=improvement)
    return dict(passed=all(r['passed'] for r in rows.values()), stand_retained=retained, backends=rows)


def run(args):
    if not __debug__:
        raise RuntimeError('Scientific checks require Python assertions enabled')
    settings = read_settings(args.protocol)
    activate(args.mimickit)
    from util import mp_util
    mp_util.init(0, 1, 'cuda:0', None)
    torch.set_num_threads(1)
    prior = Path(settings['prior'])
    verified = read(prior/'final-verification.json')
    if not all(verified.get(k) for k in ('verified', 'gpu_ball_admitted', 'matched_start_probe_passed')):
        raise ValueError('Verified GPU admission and matched starts are required')
    parent = read(prior/'protocol.json')
    package = Path(__file__).parent
    inputs, sources = protect(prior, args.out, [Path(__file__), package/'stand_ball_training_evidence.py',
        package/'scripts/train-stand-ball-pilot.ps1'], extra_inputs=[args.protocol])
    save(args.out/'protocol.json', dict(inputs=inputs, sources=sources, settings=settings,
        config=parent['config'], demonstrations=parent['demonstrations'],
        model_sha256=parent['model_sha256'], reference_sha256=parent['reference_sha256'],
        shared_initial_evaluation='One frozen source evaluation per seed before preparing either live arm',
        optimizer_state='Fresh; full weights and learned normalizers warm-started in both arms',
        time_budget='CUDA-synchronized PPO plus evidence capture; excludes saved evidence and evaluations',
        promotion_rule='Both seeds and backends: retain Stand, all throws hit, no direction/level regression, strictly more selected-level recovery than source and quiet continuation',
        automatic_arena_selection=False))
    rig, reference = Rig.load(args.out/'dummy.xml', policy_exclude=()), args.out/'stationary_reference.npz'
    worlds = {}
    for mass in (2, 5):
        worlds[mass] = args.out/f'ball-{mass}kg.xml'
        shutil.copy2(prior/worlds[mass].name, worlds[mass])
    results = {}
    canonical = {str(Path(p).resolve()): h for p, h in inputs.items()}
    for seed in settings['seeds']:
        root = Path(parent['settings']['sources'][str(seed)])
        source = read(root/'protocol.json')
        checkpoint = root/'final/model.pt'
        for path in (root/'protocol.json', checkpoint, checkpoint.with_suffix('.amp.pt')):
            if canonical.get(str(path.resolve())) != digest(path):
                raise ValueError('Source lineage differs from GPU admission')
        cfg = source['config']
        assert cfg['steps_per_iter'] == 64 and cfg['actor_epochs'] == 3 and cfg['critic_epochs'] == 2
        assert cfg['actor_optimizer']['learning_rate'] == 3e-4
        assert cfg['actor_batch_size'] == 4 and cfg['critic_batch_size'] == 2
        folder = args.out/str(seed)
        folder.mkdir()
        initial = assess(load_actor(rig, reference, source, checkpoint), rig, reference, worlds,
            folder/'source', settings)
        if not initial['full_interval_stand']:
            raise RuntimeError('Source failed repeated 120-second Stand; no training for this seed')
        pair = prepare_pair(rig, reference, worlds[5], source, checkpoint, seed, folder/'starts')
        assert all(tensor_hash(StandPolicy(agent).state_dict()) == initial['actor_hash']
                   for agent, _, _ in pair.values())
        finals, training = {}, {}
        order = ('quiet', 'single_ball') if seed == 301027 else ('single_ball', 'quiet')
        for arm in order:
            training[arm] = train_arm(pair, arm, seed, folder/'starts', folder/arm, settings)
            agent, task, _ = pair[arm]
            state, before = rng_state(task), pair_fingerprint(pair)
            try:
                finals[arm] = assess(StandPolicy(agent), rig, reference, worlds,
                    folder/arm/'evaluation', settings)
            finally:
                restore_rng(task, state)
            after = pair_fingerprint(pair)
            rng_restored = tree_hash(state) == tree_hash(rng_state(task))
            save(folder/arm/'evaluation-isolation.json', dict(before=before, after=after,
                rng_restored=rng_restored))
            if before != after or not rng_restored:
                raise RuntimeError('Evaluation mutated a live training arm')
        result = dict(initial=initial, final=finals, training=training,
            comparison=compare_seed(initial, finals))
        save(folder/'report.json', result)
        results[str(seed)] = result
        save(args.out/'progress.json', dict(completed_seeds=list(results), results=results))
        del pair, agent, task
        gc.collect()
        torch.cuda.empty_cache()
    improved = all(r['comparison']['passed'] for r in results.values())
    report = dict(results=results, replicated_improvement=improved, promoted=False, arena_changed=False,
        updates=sum(a['updates'] for r in results.values() for a in r['training'].values()),
        transitions=sum(a['transitions'] for r in results.values() for a in r['training'].values()),
        training_seconds=sum(a['training_seconds'] for r in results.values() for a in r['training'].values()),
        evaluation_cases=480)
    assert report['updates'] == settings['maximum_updates'] and report['transitions'] == settings['maximum_transitions']
    save(args.out/'report.json', report)
    seal(args.out, inputs, sources, verified=True, training_updates=report['updates'],
        evaluation_cases=480, replicated_improvement=improved, promoted=False)


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
