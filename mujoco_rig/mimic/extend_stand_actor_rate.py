"""Conditional fresh third-seed duration check after replicated actor-rate gain."""
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
from .measure_stand_learning_curve import assess
from .runtime import activate
from .rig import Rig
from .stand_experiment_evidence import protect, seal
from .train_stand_imitation import tensor_hash
from .train_torque_stand import build_agent

PRIOR = Path('logs/mimickit-training/stand-actor-rate-20261005-01')
SEED = 301022


def run(args):
    assert read(PRIOR/'report.json')['replicated_gain'], 'Duration branch requires replicated gain'
    activate(args.mimickit)
    from util import mp_util
    mp_util.init(0, 1, 'cuda:0', None)
    torch.set_num_threads(1)
    inputs, sources = protect(PRIOR, args.out,
        [Path(__file__), Path(__file__).with_name('verify_stand_actor_extension.py')])
    prior = read(PRIOR/'protocol.json')
    config = copy.deepcopy(prior['config'])
    config['actor_optimizer']['learning_rate'] = 3e-4
    (args.out/'agent.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    rig, reference = Rig.load(args.out/'dummy.xml', policy_exclude=()), args.out/'stationary_reference.npz'
    save(args.out/'protocol.json', dict(inputs=inputs, sources=sources, config=config, seed=SEED,
        sampling_seed=SEED+701, demonstrations=prior['demonstrations'],
        model_sha256=prior['model_sha256'], reference_sha256=prior['reference_sha256'],
        checkpoint_updates=[0, 100, 300], maximum_training_seconds=900., maximum_cases=48,
        transitions_per_update=4096, initialize_from=None, stop_on_16_passes=True,
        scope='Fresh third-seed duration check of replicated actor rate; no additional coefficient change'))
    task = build_task_type()(rig, reference, 64, 'cuda:0', representation='target_pd',
        reward_mode='reference_v9', episode_seconds=60., seed=SEED,
        demonstrations=[Path(r['path']) for r in prior['demonstrations']], demonstration_rig=rig)
    task.mirror_demos = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    agent = build_agent(config, task, 'cuda:0')
    assert int(agent._obs_norm._count) == int(agent._a_norm._count) == 0
    initial_hash = tensor_hash(agent.state_dict())
    assert all(initial_hash != pair['reference']['initial_full_tensor_hash']
        for pair in read(PRIOR/'report.json')['arms'].values())
    save(args.out/'contract.json', task.observation_contract())
    assessments = {'initial': assess(agent, task, rig, reference, args.out/'initial', 0., 0.)}
    elapsed, records, reason = 0., [], None
    agent._init_train()
    torch.manual_seed(SEED+701)
    np.random.seed(SEED+701)
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    for target in (100, 300):
        while agent._iter < target and elapsed < 900.:
            task.reward_records.clear()
            torch.cuda.synchronize()
            started = time.perf_counter()
            info = agent._train_iter()
            torch.cuda.synchronize()
            elapsed += time.perf_counter()-started
            agent._sample_count = agent._update_sample_count()
            agent._iter += 1
            reward = torch.stack(task.reward_records).cpu().numpy()
            assert reward.shape == (64, 6) and np.isfinite(reward).all()
            assert np.array_equal(reward[:, 0], reward[:, 1]) and (reward[:, 2] == 1).all()
            records.append(reward)
            row = dict(iteration=agent._iter, transitions=agent._sample_count, training_seconds=elapsed,
                actor_optimizer_steps=agent._actor_optimizer.get_steps(),
                critic_optimizer_steps=agent._critic_optimizer.get_steps(),
                **{key: float(value) for key, value in info.items()})
            assert all(np.isfinite(v) for v in row.values())
            assert all(torch.isfinite(v).all() for v in agent.state_dict().values())
            with (args.out/'updates.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            save(args.out/'progress.json', row)
            if agent._iter % 25 == 0:
                print(dict(updates=agent._iter, training_seconds=elapsed, transitions=agent._sample_count), flush=True)
        if agent._iter != target:
            raise RuntimeError('Duration safety cap reached; incomplete protocol')
        stage = f'updates-{target}'
        assessments[stage] = assess(agent, task, rig, reference, args.out/stage, elapsed, elapsed)
        save(args.out/'assessments.json', assessments)
        if all(assessments[stage]['results'][b]['standing_passes'] == 8 for b in ('native', 'gpu')):
            reason = 'stand_16_of_16'
            break
    np.savez_compressed(args.out/'reward-records.npz', values=np.stack(records), columns=np.asarray(REWARD_COLUMNS))
    report = dict(assessments=assessments, stop_reason=reason or 'maximum_updates_reached',
        ppo_updates=agent._iter, transitions=agent._sample_count, training_seconds=elapsed,
        final_checkpoint=list(assessments.values())[-1]['checkpoint'], full_learned_stand=reason == 'stand_16_of_16',
        sustained_stand_verified=False, perturbations_ready=False, arena_changed=False)
    save(args.out/'report.json', report)
    from .verify_stand_actor_extension import verify
    verification = verify(args.out)
    seal(args.out, inputs, sources, **verification)
    print(dict(completed=True, stop_reason=report['stop_reason'], updates=agent._iter), flush=True)


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
