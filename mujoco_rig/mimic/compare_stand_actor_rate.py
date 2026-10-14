"""Two fresh, matched-update comparisons of actor SGD learning rate."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
import yaml

from . import measure_stand_learning_curve as curve
from . import replicate_stand_representations as screen
from .audit_stand_failures import digest, read, save
from .rig import Rig
from .runtime import activate
from .train_stand_imitation import tensor_hash
from .train_torque_stand import build_agent

PRIOR = Path('logs/mimickit-training/stand-feedback-inputs-20261005-01')
DIAGNOSIS = Path('logs/mimickit-training/stand-duration-diagnosis-20261003-01')
SEEDS = (301020, 301021)
UPDATES = 100
MAX_SECONDS = 300.
REWARD_COLUMNS = ('base_mean', 'shaped_mean', 'factor_mean', 'speed_mean_m_s', 'speed_max_m_s', 'formula_max_error')


def build_task_type():
    from .representation_task import AMPRepresentationTask

    class ReferenceTraceTask(AMPRepresentationTask):
        def __init__(self, *args, **kwargs):
            self.reward_records = []
            super().__init__(*args, **kwargs)

        def reward(self):
            base = super().reward()
            speed_squared = self.engine.get_root_vel(0)[:, :2].square().sum(-1)
            factor = torch.ones_like(base)
            shaped = base*factor
            speed = torch.sqrt(speed_squared)
            independently_computed = base
            self.reward_records.append(torch.stack((base.mean(), shaped.mean(), factor.mean(), speed.mean(),
                speed.max(), (shaped-independently_computed).abs().max())).detach())
            return shaped

    return ReferenceTraceTask


def protect(out):
    prior, verification = read(PRIOR/'protocol.json'), read(PRIOR/'final-verification.json')
    protected = {**prior['inputs'], **verification['output_hashes']}
    for path in (PRIOR/'final-verification.json', DIAGNOSIS/'report.json', DIAGNOSIS/'verification.json'):
        protected[str(path)] = digest(path)
    paths = [Path(__file__), Path(__file__).with_name('verify_stand_actor_rate.py'),
        Path(__file__).with_name('audit_stand_duration_failures.py')]
    for source in prior['sources']:
        paths.append(Path(source['path']))
        for key in ('path', 'snapshot'):
            protected[source[key]] = source['sha256']
    paths = list({str(path.resolve()): path for path in paths}.values())
    for path in paths:
        expected = digest(path)
        assert str(path) not in protected or protected[str(path)] == expected
        protected[str(path)] = expected
    screen.check_protected(protected, [])
    out.mkdir(parents=True, exist_ok=False)
    (out/'sources').mkdir()
    sources = []
    for index, path in enumerate(paths):
        snapshot = out/'sources'/f'{index:02d}-{path.name}'
        shutil.copy2(path, snapshot)
        sources.append(dict(path=str(path), snapshot=str(snapshot), sha256=digest(path)))
    return protected, sources


def compare(candidate, baseline, dt):
    result = curve.continuation(candidate, baseline, dt)
    for backend in curve.BACKENDS:
        paired = screen.comparison(candidate['results'][backend], baseline['results'][backend], dt)
        row = result['backends'][backend]
        row['paired_survival'] = paired
        row['candidate_wins'] = bool(row['continue_training'] and
            (row['stand_pass_gain'] > 0 or paired['improved_cases'] > paired['worsened_cases']))
    result['candidate_wins'] = all(v['candidate_wins'] for v in result['backends'].values())
    return result


def train_arm(rig, reference, clips, config, seed, mode, out, expected_hash):
    task_type = build_task_type()
    config = copy.deepcopy(config)
    config['actor_optimizer']['learning_rate'] = 3e-4 if mode == 'actor_lr_3e4' else 1e-4
    task = task_type(rig, reference, 64, 'cuda:0', representation='target_pd',
        reward_mode='reference_v9', episode_seconds=60., seed=seed, demonstrations=clips, demonstration_rig=rig)
    task.mirror_demos = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    agent = build_agent(config, task, 'cuda:0')
    initial_hash = tensor_hash(agent.state_dict())
    assert expected_hash is None or expected_hash == initial_hash
    assert int(agent._obs_norm._count) == int(agent._a_norm._count) == 0
    assert task.get_obs_space().shape == (374,) and task.get_disc_obs_space().shape == (364,)
    out.mkdir()
    (out/'agent.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    save(out/'contract.json', task.observation_contract())
    save(out/'reward-contract.json', dict(mode=mode,
        formula='reference_v9', actor_learning_rate=config['actor_optimizer']['learning_rate'],
        sampling='post-physics root velocity, same as balance gate', amp_weight=.2, task_weight=.8,
        physical_failures='unchanged; failed transitions zero task reward', evaluation='unchanged 12-second gates'))
    save(out/'initialization.json', dict(seed=seed, sampling_seed=seed+701, initialize_from=None,
        fresh=True, full_tensor_hash=initial_hash))
    initial = curve.assess(agent, task, rig, reference, out/'initial', 0., 0.)
    agent._init_train()
    torch.manual_seed(seed+701)
    np.random.seed(seed+701)
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    elapsed, reward_arrays = 0., []
    while agent._iter < UPDATES and elapsed < MAX_SECONDS:
        task.reward_records.clear()
        torch.cuda.synchronize()
        started = time.perf_counter()
        info = agent._train_iter()
        torch.cuda.synchronize()
        elapsed += time.perf_counter()-started
        agent._sample_count = agent._update_sample_count()
        agent._iter += 1
        records = torch.stack(task.reward_records).cpu().numpy()
        assert records.shape == (64, len(REWARD_COLUMNS)) and np.isfinite(records).all()
        assert records[:, -1].max() < 1e-6
        assert np.all(records[:, 1] <= records[:, 0]+1e-7)
        assert (records[:, 2] >= 0).all() and (records[:, 2] <= 1).all()
        if mode in ('reference', 'actor_lr_3e4'):
            assert np.array_equal(records[:, 0], records[:, 1]) and (records[:, 2] == 1).all()
        reward_arrays.append(records)
        row = dict(iteration=agent._iter, transitions=agent._sample_count, training_seconds=elapsed,
            actor_optimizer_steps=agent._actor_optimizer.get_steps(),
            critic_optimizer_steps=agent._critic_optimizer.get_steps(),
            **{key: float(value) for key, value in info.items()},
            **{'reward_'+key: float(value) for key, value in zip(REWARD_COLUMNS, records.mean(0))})
        assert all(np.isfinite(value) for value in row.values())
        assert all(torch.isfinite(value).all() for value in agent.state_dict().values())
        with (out/'updates.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        save(out/'progress.json', row)
        if agent._iter % 20 == 0:
            print(dict(seed=seed, mode=mode, updates=agent._iter, seconds=elapsed), flush=True)
    np.savez_compressed(out/'reward-records.npz', values=np.stack(reward_arrays), columns=np.asarray(REWARD_COLUMNS))
    final = curve.assess(agent, task, rig, reference, out/'final', elapsed, elapsed)
    report = dict(seed=seed, mode=mode, initial=initial, final=final, training_seconds=elapsed,
        ppo_updates=agent._iter, transitions=agent._sample_count, completed_budget=agent._iter == UPDATES,
        final_checkpoint=final['checkpoint'], initialize_from=None, initial_full_tensor_hash=initial_hash)
    save(out/'report.json', report)
    return report


def run(args):
    activate(args.mimickit)
    from util import mp_util
    mp_util.init(0, 1, 'cuda:0', None)
    torch.set_num_threads(1)
    protected, sources = protect(args.out)
    prior = read(PRIOR/'protocol.json')
    for name in ('dummy.xml', 'stationary_reference.npz', 'stationary_reference.json'):
        shutil.copy2(PRIOR/name, args.out/name)
    rig, reference = Rig.load(args.out/'dummy.xml', policy_exclude=()), args.out/'stationary_reference.npz'
    assert digest(rig.path) == prior['model_sha256'] and digest(reference) == prior['reference_sha256']
    assert len(rig.actuators) == 33 and not np.any(rig.model.jnt_stiffness) and not np.any(rig.model.actuator_biasprm)
    config, records = prior['config'], prior['demonstrations']
    for record in records:
        assert digest(Path(record['path'])) == record['sha256']
    (args.out/'agent.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    save(args.out/'protocol.json', dict(inputs=protected, sources=sources, seeds=SEEDS,
        orders=[['reference', 'actor_lr_3e4'], ['actor_lr_3e4', 'reference']], config=config,
        model_sha256=digest(rig.path), reference_sha256=digest(reference), demonstrations=records,
        updates_per_arm=UPDATES, maximum_seconds_per_arm=MAX_SECONDS, transitions_per_update=4096,
        evaluation_seconds=12., environments=64, training_episode_seconds=60.,
        maximum_assessment_cases=128, only_intervention='Actor SGD learning rate 1e-4 versus 3e-4; unchanged reward',
        selection='Final update, never best checkpoint', replication_required=True,
        win_rule='Both backends: more Stand passes, or equal passes and minimum/mean survival and mean longest valid interval gain >= one control interval with more paired survival improvements than regressions',
        no_budget_extension=True, arena_changed=False, perturbations=False))
    arms, decisions, hashes = {}, {}, []
    for seed, order in zip(SEEDS, (('reference', 'actor_lr_3e4'), ('actor_lr_3e4', 'reference'))):
        seed_dir = args.out/f'seed-{seed}'
        seed_dir.mkdir()
        pair, common_hash = {}, None
        for mode in order:
            print(dict(starting_seed=seed, mode=mode), flush=True)
            pair[mode] = train_arm(rig, reference, [Path(r['path']) for r in records], config, seed,
                mode, seed_dir/mode, common_hash)
            common_hash = pair[mode]['initial_full_tensor_hash']
            if not pair[mode]['completed_budget']:
                raise RuntimeError('Training safety cap reached before matched update budget; comparison incomplete')
        assert common_hash not in hashes
        hashes.append(common_hash)
        arms[str(seed)] = pair
        decisions[str(seed)] = compare(pair['actor_lr_3e4']['final'], pair['reference']['final'],
            rig.model.opt.timestep*rig.decimation)
        save(seed_dir/'decision.json', decisions[str(seed)])
        from .verify_stand_actor_rate import verify_pair
        save(seed_dir/'verification.json', verify_pair(args.out, seed, pair, decisions[str(seed)]))
        print(dict(seed=seed, candidate_wins=decisions[str(seed)]['candidate_wins']), flush=True)
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), arms=arms, decisions=decisions,
        replicated_gain=all(d['candidate_wins'] for d in decisions.values()),
        full_learned_stand=all(pair['actor_lr_3e4']['final']['results'][b]['standing_passes'] == 8
            for pair in arms.values() for b in curve.BACKENDS),
        total_assessment_cases=128, perturbations_ready=False, arena_changed=False)
    screen.check_protected(protected, sources)
    save(args.out/'report.json', report)
    save(args.out/'completion-verification.json', dict(protected_inputs=len(protected), sources=len(sources),
        output_hashes={str(p): digest(p) for p in sorted(args.out.rglob('*')) if p.is_file()}))
    print(dict(completed=True, replicated_gain=report['replicated_gain']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            save(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise
