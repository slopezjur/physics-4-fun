"""Bounded paired reward pilots and frozen behavior audits."""
import argparse
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import torch

from .baseline import sha256
from .diagnose_stand_foundations import write_json
from .outward_limit_reward import augment_trace, contract, cost, task_factory


def read(path):
    import json
    return json.loads(path.read_text())


def sources():
    return [Path(__file__), Path('mujoco_rig/mimic/outward_limit_reward.py'),
        Path('mujoco_rig/mimic/compare_stand_exploration.py'), Path('mujoco_rig/mimic/audit_stand_policy.py')]


def preflight(args):
    historical = Path('logs/mimickit-training/stand-failure-traces-20261002-02')
    recorded = read(historical/'report.json')
    manifest = read(historical/'protocol.json')
    model_run = Path('logs/mimickit-training/stand-ppo-critic-lr1em3-30min-20261002-01')
    control = read(model_run/'contract.json')
    xml = ET.parse(model_run/'dummy.xml').getroot()
    hinges = list(xml.find('worldbody').iter('joint'))
    assert [node.get('name') for node in hinges] == control['joint_names']
    indices = np.array([control['joint_names'].index(name) for name in control['actuator_names']])
    ranges = np.array([[float(x) for x in hinges[index].get('range').split()] for index in indices], dtype=np.float32)
    scale = np.asarray(control['torque_scale_nm'], dtype=np.float32)
    protected = {str(path):sha256(path) for path in sources()}
    for filename in ('contract.json', 'dummy.xml'):
        path = model_run/filename
        protected[str(path)] = sha256(path)
    results = {}
    for label, entry in recorded['traces'].items():
        directory = Path(entry['path'])
        for filename in ('trace.npz', 'report.json'):
            path = directory/filename
            assert sha256(path) == manifest['inputs'][str(path)]
            protected[str(path)] = sha256(path)
        with np.load(directory/'trace.npz', allow_pickle=False) as archive:
            a = {key:archive[key] for key in archive.files}
        expected = np.zeros_like(a['action'])
        expected[2:] = a['action'][:-2].clip(-1, 1)*scale
        if 'held_torque_nm' in a:
            assert np.array_equal(expected, a['held_torque_nm'])
        position = a['qpos'][:, :, 7+indices].astype(np.float32)
        u = expected/scale
        values = cost(torch.from_numpy(position), torch.from_numpy(u),
            torch.from_numpy(ranges[:, 0]), torch.from_numpy(ranges[:, 1])).numpy()
        margin = np.float32(np.deg2rad(5.))
        lower = np.clip((ranges[:, 0]+margin-position)/margin, 0, 1)
        upper = np.clip((position-ranges[:, 1]+margin)/margin, 0, 1)
        independent = np.maximum(lower*np.maximum(-u, 0), upper*np.maximum(u, 0)).mean(-1)
        assert np.allclose(values, independent, rtol=0, atol=1e-6)
        assert not values[:2].any() and np.isfinite(values).all()
        live = a['alive_before']
        results[label] = dict(mean_live_cost=float(values[live].mean()), max_live_cost=float(values[live].max()),
            per_episode_mean_cost=[float(values[live[:, case], case].mean()) for case in range(8)],
            recorded_torque_verified='held_torque_nm' in a, independent_formula_verified=True,
            historical_trace_hash_verified=True)
    assert results['teacher']['max_live_cost'] == 0
    assert all(sha256(Path(path)) == digest for path,digest in protected.items())
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/'preflight.json', dict(contract=contract(1), results=results,
        inputs=protected, inputs_unchanged=True, teacher_zero_cost_verified=True,
        physics_rollouts=0, training_updates=0))
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
    return audit(args, task_factory=factory, source_paths=[sources()[0], sources()[1]],
        reward_variant=variant, trace_augmenter=augment_trace)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('preflight', 'train', 'audit'), required=True)
    parser.add_argument('--weight', type=float, choices=(0., 1.), default=0.)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--paired-with', type=Path)
    parser.add_argument('--paired-audit', type=Path, default=Path('logs/mimickit-training/stand-critic-audit-20261002-01'))
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
