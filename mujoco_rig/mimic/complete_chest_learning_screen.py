"""Complete Stage 14 after a native crash, preserving and auditing completed arms."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from .audit_stand_failures import read, save
from .ball_study_support import FOUNDATION, checkpoint, protect_used
from .chest_ball_curriculum import curriculum
from .chest_reward_variant import reward_variant
from .evaluate_stand_ball_boundary import summarize
from .evaluate_stand_ball_gpu import compare as backend_compare
from .measure_stand_learning_curve import rng_state, restore_rng, tree_hash
from .policy import StandPolicy
from .prepare_stand_ball_pair import prepare_pair, identical
from .replicate_stand_representations import check_protected, tensor_hash, verify_control_trace
from .rig import Rig
from .runtime import activate
from .run_chest_ball_pilot import source_assessment, compare as compare_seed
from .run_chest_learning_screen import impacts, recipe_gain
from . import run_stand_ball_pilot as pilot
from .stand_experiment_evidence import load_actor, seal
from .stand_start_guard import compare as compare_start
from .verify_stand_learning_curve import verify_gates


def audit_arm(root, seed, arm, rig, reference, source, settings):
    from .amp_recovery import validate_sidecar
    folder = root / str(seed) / arm
    assert compare_start(root / str(seed) / 'starts' / arm, folder / 'actual-start')['passed']
    admission = read(folder / 'start-admission.json')
    assert admission['common']['passed'] and admission['extras_equal']
    isolation = read(folder / 'evaluation-isolation.json')
    assert isolation['before'] == isolation['after'] and isolation['rng_restored']
    training, evaluation = read(folder / 'training-report.json'), read(folder / 'evaluation/report.json')
    records = [json.loads(line) for line in (folder / 'updates.jsonl').read_text().splitlines()]
    assert len(records) == training['updates'] == settings['updates_per_arm'] == 300
    launches = contacts = 0
    for index, row in enumerate(records, 1):
        assert row['updates'] == index and row['transitions'] == index * 4096
        assert row['actor_optimizer_steps'] == 48 * index and row['critic_optimizer_steps'] == 64 * index
        with np.load(folder / 'rollouts' / f'{index:04d}.npz', allow_pickle=False) as data:
            values = {k: data[k] for k in data.files}
        assert all(np.isfinite(v).all() for v in values.values())
        assert values['steps'].shape == (64, 64)
        expected = (values['steps'] == 300) & (np.arange(64)[None] < 32)
        if arm == 'quiet':
            expected[:] = False
        assert np.array_equal(values['launched'], expected)
        assert (values['position_error'] <= 2e-6).all() and (values['velocity_error'] <= 2e-6).all()
        quiet = np.arange(64) >= (0 if arm == 'quiet' else 32)
        assert not values['normal_impulse'][:, quiet].any() and not values['contact_impulse'][:, quiet].any()
        assert (values['normal_impulse'] >= 0).all()
        assert ((values['steps'][1:] == values['steps'][:-1] + 1) | (values['steps'][1:] == 0)).all()
        launches += int(values['launched'].sum())
        contacts += int(np.count_nonzero(values['normal_impulse'] > 0))
    assert launches == training['launches'] and contacts == training['contact_world_intervals']
    assert training['transitions'] == 300 * 4096
    assert records[-1]['training_seconds'] == training['training_seconds'] <= settings['maximum_training_seconds_per_arm']
    for name in ('initial', 'updates-100', 'updates-200', 'updates-300', 'final'):
        path = folder / name / 'model.pt'
        portable = torch.load(path, map_location='cpu', weights_only=True)
        full = torch.load(path.with_suffix('.amp.pt'), map_location='cpu', weights_only=True)
        validate_sidecar(portable, full)
        assert all(torch.isfinite(v).all() for v in full.values())
    actor = load_actor(rig, reference, source, folder / 'final/model.pt')
    assert tensor_hash(actor.state_dict()) == evaluation['actor_hash']
    maximum_action_error = 0.
    dt = rig.model.opt.timestep * rig.decimation
    for name, backends in evaluation['impacts'].items():
        for backend, report in backends.items():
            path = folder / 'evaluation' / f'{name}-{backend}'
            with np.load(path / 'trace.npz', allow_pickle=False) as data:
                trace = {k: data[k] for k in data.files}
            assert all(np.isfinite(v).all() for v in trace.values())
            verify_gates(trace, '')
            verify_gates(trace, 'pre_')
            assert verify_control_trace(trace, read(path / 'contract.json')) == report['max_control_formula_error_nm']
            assert all(report[k] == v for k, v in summarize(trace, dt, 12., 5., False).items())
            with torch.no_grad():
                predicted = actor(torch.from_numpy(trace['observation'].reshape(-1, 374))).numpy().reshape(trace['action'].shape)
            error = float(np.abs(predicted - trace['action'])[trace['alive_before']].max(initial=0.))
            maximum_action_error = max(maximum_action_error, error)
            assert error <= 2e-6
            if backend == 'native':
                with np.load(path / 'body-impulses.npz', allow_pickle=False) as body:
                    assert np.abs(body['normal'].sum(-1) - trace['normal_impulse_ns']).max() <= 1e-10
        assert backend_compare(backends['native'], backends['gpu'], dt) == evaluation['backend_parity'][name]
    return dict(passed=True, seed=seed, arm=arm, updates=300, evaluation_cases=64,
                maximum_actor_replay_error=maximum_action_error), training, evaluation


def run(args):
    settings = read(args.protocol)
    prior = read(args.prior / 'protocol.json')
    assert settings == prior['settings'] and settings['updates_per_arm'] == 300
    check_protected(prior['inputs'], prior['sources'])
    activate(args.mimickit)
    from util import mp_util
    from .stand_ball_native import write_world
    mp_util.init(0, 1, 'cuda:0', None)
    torch.set_num_threads(1)
    used = dict(settings, evidence=[*settings['evidence'], *[str(p) for p in args.prior.rglob('*') if p.is_file()]])
    dependencies = [Path(s['path']) for s in prior['sources']]
    _, inputs, sources = protect_used(args.out, [Path(__file__), *dependencies], args.protocol, used)
    source = {k: prior[k] for k in ('config', 'demonstrations')}
    save(args.out / 'protocol.json', dict(inputs=inputs, sources=sources, settings=settings, **source,
        partial_source=str(args.prior), completion_scope='Audit/copy three finished arms; restart only seed 301028 quiet from admitted original start', automatic_promotion=False))
    rig, reference = Rig.load(args.out / 'dummy.xml', policy_exclude=()), args.out / 'stationary_reference.npz'
    world = args.out / 'ball-5kg.xml'
    write_world(args.out / 'dummy.xml', FOUNDATION / 'ball-5kg.xml', world, 5.)
    results, audits = {}, []
    for seed in settings['seeds']:
        folder = args.out / str(seed)
        folder.mkdir()
        initial = source_assessment(seed, settings)
        results[str(seed)] = dict(initial=initial, final={}, training={})
        for arm in settings['arms']:
            if (seed, arm) == (301028, 'quiet'):
                continue
            audit, training, evaluation = audit_arm(args.prior, seed, arm, rig, reference, source, settings)
            audits.append(audit)
            shutil.copytree(args.prior / str(seed) / arm, folder / arm)
            results[str(seed)]['training'][arm] = training
            results[str(seed)]['final'][arm] = evaluation
            save(args.out / 'reuse-audit.json', dict(arms=audits))
            print(dict(audited_seed=seed, arm=arm, **{k: audit[k] for k in ('passed', 'maximum_actor_replay_error')}), flush=True)
    folder = args.out / '301028'
    with curriculum(settings), reward_variant(settings):
        pair = prepare_pair(rig, reference, world, source, checkpoint(301028, 'source'), 301028, folder / 'starts')
        cross = {}
        for arm in settings['arms']:
            cross[arm] = compare_start(args.prior / '301028/starts' / arm, folder / 'starts' / arm)
            with np.load(args.prior / '301028/starts' / arm / 'extra-state.npz', allow_pickle=False) as before:
                with np.load(folder / 'starts' / arm / 'extra-state.npz', allow_pickle=False) as after:
                    cross[arm]['extras_equal'] = all(identical(before[k], after[k]) for k in before.files)
        save(args.out / 'cross-process-start-admission.json', cross)
        assert all(c['passed'] and c['extras_equal'] for c in cross.values())
        training = pilot.train_arm(pair, 'quiet', 301028, folder / 'starts', folder / 'quiet', settings)
        agent, task, _ = pair['quiet']
        state, before = rng_state(task), pilot.pair_fingerprint(pair)
        try:
            evaluation = impacts(StandPolicy(agent), rig, reference, world, folder / 'quiet/evaluation', settings)
        finally:
            restore_rng(task, state)
        after = pilot.pair_fingerprint(pair)
        restored = tree_hash(state) == tree_hash(rng_state(task))
        save(folder / 'quiet/evaluation-isolation.json', dict(before=before, after=after, rng_restored=restored))
        assert before == after and restored
        results['301028']['training']['quiet'] = training
        results['301028']['final']['quiet'] = evaluation
        del pair, agent, task
        gc.collect()
        torch.cuda.empty_cache()
    for seed, row in results.items():
        row['recipe'] = {arm: recipe_gain(row['initial'], final, 1.4) for arm, final in row['final'].items()}
        row['comparison'] = compare_seed(row['initial'], row['final'], 1.4)
        row['comparison']['all_arms_stand_retained'] = row['comparison'].pop('stand_retained')
        row['comparison']['source_and_ball_stand_retained'] = None
        save(args.out / seed / 'report.json', row)
    eligible = {arm: all(r['recipe'][arm]['eligible'] for r in results.values()) for arm in settings['arms']}
    # Completed seed 301027 arms already fail absolute gain, so no conditional Stand is due.
    assert not any(eligible.values())
    report = dict(results=results, impact_eligible_arms=eligible, replicated_recipe_gain=dict(quiet=False, single_ball=False),
        replicated_impact_practice_gain=False, evaluation_cases=256, reused_evaluation_cases=192, new_evaluation_cases=64,
        updates=sum(a['updates'] for r in results.values() for a in r['training'].values()),
        transitions=sum(a['transitions'] for r in results.values() for a in r['training'].values()),
        training_seconds=sum(a['training_seconds'] for r in results.values() for a in r['training'].values()),
        new_updates=300, discarded_crashed_arm_updates=126, promoted=False)
    assert report['updates'] == 1200 and report['transitions'] == 4915200
    save(args.out / 'report.json', report)
    seal(args.out, inputs, sources, verified=True, evaluation_cases=256, new_evaluation_cases=64,
         training_updates=1200, new_training_updates=300, promoted=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = p.parse_args()
    existed = args.out.exists()
    try:
        run(args)
    except Exception:
        if not existed and args.out.is_dir():
            save(args.out / 'failure.json', dict(error=traceback.format_exc()))
        raise
