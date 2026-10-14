"""Approved fixed-budget angular-reward pilots, preflight and frozen audits."""
import argparse
from pathlib import Path
import traceback

import numpy as np
import torch

from .angular_speed_reward import augment_trace, contract, cost, task_factory
from .audit_stand_failures import read
from .baseline import sha256
from .diagnose_stand_foundations import write_json


def sources():
    return [Path(__file__), Path('mujoco_rig/mimic/angular_speed_reward.py'),
        Path('mujoco_rig/mimic/compare_stand_exploration.py'), Path('mujoco_rig/mimic/audit_stand_policy.py')]


def preflight(args):
    historical = Path('logs/mimickit-training/stand-angular-reward-audit-20261002-01')
    protocol, report = read(historical/'protocol.json'), read(historical/'report.json')
    protected = dict(protocol['inputs'])
    for path in sources()+[historical/'protocol.json', historical/'report.json']:
        protected[str(path)] = sha256(path)
    results = {}
    for label, record in report['traces'].items():
        directory = Path(record['path'])
        with np.load(directory/'trace.npz', allow_pickle=False) as archive:
            original = {key: archive[key] for key in archive.files}
        score_path = historical/label/'scores.npz'
        protected[str(score_path)] = sha256(score_path)
        with np.load(score_path, allow_pickle=False) as scores:
            values = {}
            for device in ('cpu', 'cuda:0'):
                velocity = torch.tensor(original['qvel'][..., 3:6], dtype=torch.float32, device=device)
                speed, measured_cost = cost(velocity)
                measured_cost = measured_cost.cpu().numpy()
                assert np.allclose(measured_cost, scores['angular_cost'], rtol=0, atol=1e-6)
                valid = original['done'] != protocol['fail_value']
                reward = (scores['reference_reward'].astype(np.float32)-.25*measured_cost)*valid
                assert np.allclose(reward, scores['rescored_task_reward'], rtol=0, atol=1e-6)
                assert not reward[~valid].any()
                values[device] = dict(max_cost_error=float(np.abs(measured_cost-scores['angular_cost']).max()),
                    max_reward_error=float(np.abs(reward-scores['rescored_task_reward']).max()),
                    mean_live_cost=float(measured_cost[original['alive_before']].mean()),
                    max_cost=float(measured_cost.max()))
            results[label] = values
    assert all(values['max_cost'] == 0 for values in results['teacher'].values())
    assert all(sha256(Path(path)) == expected for path, expected in protected.items()), 'Historical input/source changed'
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/'preflight.json', dict(contract=contract(.25), inputs=protected, results=results,
        cpu_gpu_formula_verified=True, teacher_zero_cost_verified=True, failed_reward_zero_verified=True,
        inputs_unchanged=True, physics_rollouts=0, training_updates=0))
    print(results, flush=True)


def run(args):
    if args.mode == 'preflight':
        return preflight(args)
    variant = contract(args.weight)
    if args.mode == 'train':
        from .compare_stand_exploration import run as train
        args.action_std, args.actor_learning_rate, args.critic_learning_rate = .1, 1e-4, 1e-3
        args.seconds, args.seed = 180., 301010
        if args.paired_with:
            paired = read(args.paired_with/'protocol.json')['reward_variant']
            paired['weight'] = args.weight
            assert paired == variant, 'Non-weight reward contract mismatch'
        factory = lambda *a, **kw: task_factory(args.weight)(*a, **kw)
        return train(args, task_factory=factory, source_paths=sources(), reward_variant=variant)
    from .audit_stand_policy import run as audit
    assert read(args.run/'protocol.json')['reward_variant'] == variant
    factory = lambda *a, **kw: task_factory(args.weight, record_components=True)(*a, **kw)
    return audit(args, task_factory=factory, source_paths=sources()[:2], reward_variant=variant,
        trace_augmenter=augment_trace)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('preflight', 'train', 'audit'), required=True)
    parser.add_argument('--weight', type=float, choices=(0., .25), default=0.)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--paired-with', type=Path)
    parser.add_argument('--paired-audit', type=Path, default=Path('logs/mimickit-training/stand-limit-reward-20261002-01/audit-weight-0'))
    parser.add_argument('--dataset', type=Path, default=Path('logs/mimickit-training/stand-teacher-20261001-01'))
    parser.add_argument('--corrections', type=Path, default=Path('logs/mimickit-training/stand-corrections-20261001-01'))
    parser.add_argument('--config', type=Path, default=Path('logs/mimickit-training/stand-correction-pilot-20261001-01/agent.yaml'))
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.out.is_dir():
            write_json(args.out/'failure.json', dict(error=traceback.format_exc()))
        raise
