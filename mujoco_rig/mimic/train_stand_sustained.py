"""Independent fixed-recipe learning with isolated 12/60-second assessments."""
import argparse
import copy
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import yaml
from .audit_stand_failures import read, save
from .compare_stand_actor_rate import build_task_type, REWARD_COLUMNS
from .evaluate_stand_sustained import evaluate_duration
from .measure_stand_learning_curve import assess, rng_state, restore_rng, training_fingerprint, tree_hash
from .policy import StandPolicy
from .runtime import activate
from .rig import Rig
from .stand_experiment_evidence import protect, seal
from .train_stand_imitation import tensor_hash
from .train_torque_stand import build_agent


def assess_sustained(agent, task, rig, reference, out, checkpoint):
    out.mkdir()
    state, before = rng_state(task), training_fingerprint(agent, task)
    try:
        results = {b: evaluate_duration(rig, reference, 'target_pd', b,
            StandPolicy(agent), out/b, seconds=60.) for b in ('native', 'gpu')}
    finally:
        restore_rng(task, state)
    after, restored = training_fingerprint(agent, task), rng_state(task)
    assert before == after and tree_hash(state) == tree_hash(restored)
    torch.save(dict(before=state, restored=restored), out/'rng.pt')
    save(out/'isolation.json', dict(before=before, after=after, rng_before=tree_hash(state), rng_after=tree_hash(restored)))
    result = dict(results=results, checkpoint=checkpoint, ppo_updates=agent._iter,
        full_interval_stand=all(r['standing_passes'] == 8 for r in results.values()))
    save(out/'report.json', result)
    return result


def run(args):
    assert args.seed in (301025, 301026)
    activate(args.mimickit)
    from util import mp_util
    mp_util.init(0, 1, 'cuda:0', None)
    torch.set_num_threads(1)
    inputs, sources = protect(args.prior, args.out, [Path(__file__),
        Path(__file__).with_name('verify_stand_sustained_training.py'),
        Path(__file__).with_name('evaluate_stand_sustained.py')])
    prior = read(args.prior/'protocol.json')
    config = copy.deepcopy(prior['config'])
    assert config['actor_optimizer']['learning_rate'] == 3e-4
    (args.out/'agent.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    rig, reference = Rig.load(args.out/'dummy.xml', policy_exclude=()), args.out/'stationary_reference.npz'
    save(args.out/'protocol.json', dict(inputs=inputs, sources=sources, config=config, seed=args.seed,
        sampling_seed=args.seed+701, prior_root=str(args.prior), demonstrations=prior['demonstrations'],
        model_sha256=prior['model_sha256'], reference_sha256=prior['reference_sha256'],
        checkpoint_updates=[0,100,300,600,1200], maximum_training_seconds=2400., maximum_cases=144,
        transitions_per_update=4096, initialize_from=None, stop_on_sustained_16_passes=True,
        scope='Fresh fixed-recipe replication and bounded duration extension; no perturbations'))
    task = build_task_type()(rig, reference, 64, 'cuda:0', representation='target_pd',
        reward_mode='reference_v9', episode_seconds=60., seed=args.seed,
        demonstrations=[Path(r['path']) for r in prior['demonstrations']], demonstration_rig=rig)
    task.mirror_demos = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    agent = build_agent(config, task, 'cuda:0')
    assert int(agent._obs_norm._count) == int(agent._a_norm._count) == 0
    assert tensor_hash(agent.state_dict()) != read(args.prior/'report.json')['assessments']['initial']['full_tensor_hash']
    save(args.out/'contract.json', task.observation_contract())
    assessments = {'initial': assess(agent, task, rig, reference, args.out/'initial', 0., 0.)}
    elapsed, records, sustained, reason = 0., [], {}, None
    agent._init_train()
    torch.manual_seed(args.seed+701)
    np.random.seed(args.seed+701)
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    for target in (100,300,600,1200):
        while agent._iter < target and elapsed < 2400.:
            task.reward_records.clear()
            torch.cuda.synchronize()
            started = time.perf_counter()
            info = agent._train_iter()
            torch.cuda.synchronize()
            elapsed += time.perf_counter()-started
            agent._sample_count = agent._update_sample_count()
            agent._iter += 1
            reward = torch.stack(task.reward_records).cpu().numpy()
            assert reward.shape == (64,6) and np.isfinite(reward).all()
            assert np.array_equal(reward[:,0], reward[:,1]) and (reward[:,2] == 1).all()
            records.append(reward)
            row = dict(iteration=agent._iter, transitions=agent._sample_count, training_seconds=elapsed,
                actor_optimizer_steps=agent._actor_optimizer.get_steps(), critic_optimizer_steps=agent._critic_optimizer.get_steps(),
                **{key: float(value) for key,value in info.items()})
            assert all(np.isfinite(v) for v in row.values())
            assert all(torch.isfinite(v).all() for v in agent.state_dict().values())
            with (args.out/'updates.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            save(args.out/'progress.json', row)
            if agent._iter % 25 == 0:
                print(dict(seed=args.seed, updates=agent._iter, training_seconds=elapsed), flush=True)
        if agent._iter != target:
            raise RuntimeError('Synchronized training cap reached; incomplete protocol')
        stage = f'updates-{target}'
        assessments[stage] = assess(agent, task, rig, reference, args.out/stage, elapsed, elapsed)
        save(args.out/'assessments.json', assessments)
        if target >= 600 or all(assessments[stage]['results'][b]['standing_passes'] == 8 for b in ('native','gpu')):
            sustained[f'sustained-{target}'] = assess_sustained(agent, task, rig, reference,
                args.out/f'sustained-{target}', assessments[stage]['checkpoint'])
            save(args.out/'sustained-assessments.json', sustained)
            if sustained[f'sustained-{target}']['full_interval_stand']:
                reason = 'sustained_stand_16_of_16'
                break
    np.savez_compressed(args.out/'reward-records.npz', values=np.stack(records), columns=np.asarray(REWARD_COLUMNS))
    report = dict(assessments=assessments, sustained_assessments=sustained,
        stop_reason=reason or 'maximum_updates_reached', ppo_updates=agent._iter,
        transitions=agent._sample_count, training_seconds=elapsed,
        final_checkpoint=list(assessments.values())[-1]['checkpoint'],
        sustained_stand_verified=reason == 'sustained_stand_16_of_16', perturbations_ready=False, arena_changed=False)
    save(args.out/'report.json', report)
    from .verify_stand_sustained_training import verify
    verification = verify(args.out)
    seal(args.out, inputs, sources, **verification)
    print(dict(completed=True, seed=args.seed, updates=agent._iter, stop_reason=report['stop_reason']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            save(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise
