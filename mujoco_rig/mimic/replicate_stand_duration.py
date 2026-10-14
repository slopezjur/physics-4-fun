"""Fresh duration replication and conditional extension with fixed settings."""
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
from .measure_stand_learning_curve import assess, continuation
from .runtime import activate
from .rig import Rig
from .stand_experiment_evidence import protect, seal
from .train_stand_imitation import tensor_hash
from .train_torque_stand import build_agent

ORIGINAL = Path('logs/mimickit-training/stand-actor-rate-duration-20261005-01')
REPLICATION = Path('logs/mimickit-training/stand-duration-replication-20261005-01')


def run(args):
    extending = args.mode == 'extension'
    prior_root = REPLICATION if extending else ORIGINAL
    seed, targets, cap = (301024, (100, 300, 600), 1500.) if extending else (301023, (100, 300), 900.)
    if extending:
        for root in (ORIGINAL, REPLICATION):
            results = read(root/'report.json')['assessments']
            assert 'updates-300' in results, 'Both duration runs must reach 300'
            assert continuation(results['updates-300'], results['updates-100'],
                read(root/'contract.json')['control_timestep'])['continue_training']
    activate(args.mimickit)
    from util import mp_util
    mp_util.init(0, 1, 'cuda:0', None)
    torch.set_num_threads(1)
    inputs, sources = protect(prior_root, args.out,
        [Path(__file__), Path(__file__).with_name('verify_stand_duration_replication.py')])
    prior = read(prior_root/'protocol.json')
    config = copy.deepcopy(prior['config'])
    assert config['actor_optimizer']['learning_rate'] == 3e-4
    (args.out/'agent.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    rig, reference = Rig.load(args.out/'dummy.xml', policy_exclude=()), args.out/'stationary_reference.npz'
    save(args.out/'protocol.json', dict(inputs=inputs, sources=sources, config=config, seed=seed,
        sampling_seed=seed+701, mode=args.mode, prior_root=str(prior_root), demonstrations=prior['demonstrations'],
        model_sha256=prior['model_sha256'], reference_sha256=prior['reference_sha256'],
        checkpoint_updates=[0, *targets], maximum_training_seconds=cap, maximum_cases=16*(len(targets)+1),
        transitions_per_update=4096, initialize_from=None, stop_on_16_passes=True,
        scope='Fresh duration replication or conditional extension; unchanged actor rate and reward'))
    task = build_task_type()(rig, reference, 64, 'cuda:0', representation='target_pd',
        reward_mode='reference_v9', episode_seconds=60., seed=seed,
        demonstrations=[Path(r['path']) for r in prior['demonstrations']], demonstration_rig=rig)
    task.mirror_demos = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    agent = build_agent(config, task, 'cuda:0')
    assert int(agent._obs_norm._count) == int(agent._a_norm._count) == 0
    initial_hash = tensor_hash(agent.state_dict())
    assert initial_hash != read(prior_root/'report.json')['assessments']['initial']['full_tensor_hash']
    save(args.out/'contract.json', task.observation_contract())
    assessments = {'initial': assess(agent, task, rig, reference, args.out/'initial', 0., 0.)}
    elapsed, records, reason = 0., [], None
    agent._init_train()
    torch.manual_seed(seed+701)
    np.random.seed(seed+701)
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    for target in targets:
        while agent._iter < target and elapsed < cap:
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
        if extending and target == 300 and not continuation(assessments[stage], assessments['updates-100'],
                task.observation_contract()['control_timestep'])['continue_training']:
            reason = 'duration_gain_not_repeated'
            break
    np.savez_compressed(args.out/'reward-records.npz', values=np.stack(records), columns=np.asarray(REWARD_COLUMNS))
    report = dict(assessments=assessments, stop_reason=reason or 'maximum_updates_reached',
        ppo_updates=agent._iter, transitions=agent._sample_count, training_seconds=elapsed,
        final_checkpoint=list(assessments.values())[-1]['checkpoint'], full_learned_stand=reason == 'stand_16_of_16',
        sustained_stand_verified=False, perturbations_ready=False, arena_changed=False)
    save(args.out/'report.json', report)
    from .verify_stand_duration_replication import verify
    verification = verify(args.out)
    seal(args.out, inputs, sources, **verification)
    print(dict(completed=True, stop_reason=report['stop_reason'], updates=agent._iter), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('replication', 'extension'), required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            save(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise
