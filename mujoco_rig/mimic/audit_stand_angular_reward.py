"""Rescore protected Stand traces with angular-speed feedback; no simulation."""
import argparse
import ast
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np

from .audit_stand_failures import balance_gate, digest, read, save


THRESHOLD = 1.
SCALE = 1.
WEIGHT = .25
DISCOUNT = .99


def angular_cost(qvel):
    speed = np.linalg.norm(qvel[..., 3:6], axis=-1)
    excess = np.maximum(speed-THRESHOLD, 0)/SCALE
    return speed, -np.expm1(-excess**2)


def done_flags(path):
    definition = next(node for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == 'DoneFlags')
    return {node.targets[0].id: ast.literal_eval(node.value) for node in definition.body
        if isinstance(node, ast.Assign)}


def discounted(values):
    return float(DISCOUNT**np.arange(len(values)) @ values)


def score_case(arrays, case, count):
    return {key: discounted(arrays[key][:count, case]) for key in
        ('reference_reward', 'angular_cost', 'applied_angular_penalty', 'rescored_task_reward')}


def run(args):
    comparison_path = args.root/'comparison-inputs.json'
    protected = {str(Path(path).resolve()): expected for path, expected in read(comparison_path)['inputs'].items()}
    flags_source = args.mimickit/'mimickit/envs/base_env.py'
    sources = [Path(__file__), Path('mujoco_rig/mimic/audit_stand_failures.py'),
        Path('mujoco_rig/mimic/torque_stand_task.py'), flags_source]
    for path in sources+[comparison_path, args.root/'comparison.json']:
        protected[str(path.resolve())] = digest(path)
    flags = done_flags(flags_source)
    assert flags == dict(NULL=0, FAIL=1, SUCC=2, TIME=3)
    assert balance_gate(sources[2])['max_angular_speed_rad_s'] == THRESHOLD
    for weight in (0, 1):
        run_path = args.root/f'weight-{weight}'
        config = read(run_path/'protocol.json')['config']
        assert config['discount'] == DISCOUNT
        for name in ('contract.json', 'initialization.json'):
            path = run_path/name
            protected[str(path.resolve())] = digest(path)
    entries = [('teacher', Path('logs/mimickit-training/stand-reward-audit-20261002-01/teacher'))]
    for weight in (0, 1):
        audit = args.root/f'audit-weight-{weight}'
        entries.append((f'weight-{weight}-deterministic', audit/'deterministic-native'))
        entries.extend((f'weight-{weight}-sampled-{seed}', audit/f'trained_actor-{seed}')
            for seed in (301010, 301011, 301012))
    traces = {}
    for label, directory in entries:
        for name in ('trace.npz', 'report.json'):
            path = directory/name
            assert str(path.resolve()) in protected, f'Historical manifest missing {path}'
        with np.load(directory/'trace.npz', allow_pickle=False) as archive:
            original = {key: archive[key] for key in archive.files}
        report = read(directory/'report.json')
        live = original['alive_before']
        assert live.shape[1] == 8 and np.isfinite(original['qvel']).all()
        reference = original.get('reference_reward', original['task_reward']).astype(np.float64)
        if 'reference_reward' in original:
            assert np.allclose(original['task_reward'], reference-original['reward_penalty'], rtol=0, atol=1e-7)
        speed, cost = angular_cost(original['qvel'])
        independent = 1-np.exp(-(np.maximum(speed-THRESHOLD, 0)/SCALE)**2)
        assert np.allclose(cost, independent, rtol=0, atol=1e-15)
        assert np.isfinite(cost).all() and (cost >= 0).all() and (cost <= 1).all()
        failed = original['done'] == flags['FAIL']
        assert not reference[failed & live].any()
        penalty = WEIGHT*cost*(~failed)
        reward = (reference-penalty)*(~failed)
        assert not reward[failed].any()
        arrays = dict(alive_before=live, time_s=original['time_s'], done=original['done'],
            angular_speed_rad_s=speed, angular_cost=cost, reference_reward=reference,
            applied_angular_penalty=penalty, rescored_task_reward=reward)
        cases = []
        for case, recorded in enumerate(report['cases']):
            n = int(live[:, case].sum())
            assert n == recorded['frames'] and live[:n, case].all() and not live[n:, case].any()
            assert int(original['done'][n-1, case]) == flags[recorded['terminal_flag']]
            loss = np.flatnonzero(~original['balance'][:n, case])
            index = int(loss[0]) if len(loss) else None
            first_loss = None if index is None else dict(frame=index, seconds=float(original['time_s'][index]),
                angular_speed_rad_s=float(speed[index, case]), cost=float(cost[index, case]),
                applied_penalty=float(penalty[index, case]), reference_reward=float(reference[index, case]),
                rescored_task_reward=float(reward[index, case]),
                previous_limit_cost=float(original['outward_limit_cost'][index, case]) if 'outward_limit_cost' in original else None)
            cases.append(dict(case=case, frames=n, survival_seconds=recorded['survival_seconds'],
                terminal_flag=recorded['terminal_flag'], standing_pass=recorded['standing_pass'],
                full_live=score_case(arrays, case, n), first_balance_loss=first_loss,
                cost_positive_live_fraction=float((cost[:n, case] > 0).mean()),
                cost_ge_099_live_fraction=float((cost[:n, case] >= .99).mean()),
                negative_rescored_live_fraction=float((reward[:n, case] < 0).mean())))
        traces[label] = dict(path=str(directory), original=original, arrays=arrays, cases=cases)
    teacher = traces['teacher']
    assert not teacher['arrays']['angular_cost'].any(), 'Teacher must remain unpenalized'
    common_counts = [min(trace['cases'][case]['frames'] for trace in traces.values()) for case in range(8)]
    results = {}
    for label, trace in traces.items():
        assert all(np.array_equal(trace['original'][key], teacher['original'][key])
            for key in ('initial_qpos', 'initial_qvel'))
        for case, row in enumerate(trace['cases']):
            count = common_counts[case]
            row['common_prefix_frames'] = count
            row['common_prefix_seconds'] = float(trace['arrays']['time_s'][count-1])
            row['common_prefix'] = score_case(trace['arrays'], case, count)
            teacher_scores = score_case(teacher['arrays'], case, row['frames'])
            row['teacher_same_duration'] = teacher_scores
        loss_rows = [row['first_balance_loss'] for row in trace['cases'] if row['first_balance_loss'] is not None]
        keys = ('reference_reward', 'angular_cost', 'applied_angular_penalty', 'rescored_task_reward')
        summary = dict(episodes=8, standing_passes=sum(row['standing_pass'] for row in trace['cases']),
            full_live={key: float(np.mean([row['full_live'][key] for row in trace['cases']])) for key in keys},
            common_prefix={key: float(np.mean([row['common_prefix'][key] for row in trace['cases']])) for key in keys},
            balance_loss_cases=len(loss_rows), first_loss_positive_cost_cases=sum(row['cost'] > 0 for row in loss_rows),
            first_loss_mean_penalty=float(np.mean([row['applied_penalty'] for row in loss_rows])) if loss_rows else None,
            first_loss_mean_reference_reward=float(np.mean([row['reference_reward'] for row in loss_rows])) if loss_rows else None,
            first_loss_mean_rescored_reward=float(np.mean([row['rescored_task_reward'] for row in loss_rows])) if loss_rows else None,
            cost_positive_live_fraction=float(np.mean([row['cost_positive_live_fraction'] for row in trace['cases']])),
            cost_ge_099_live_fraction=float(np.mean([row['cost_ge_099_live_fraction'] for row in trace['cases']])),
            negative_rescored_live_fraction=float(np.mean([row['negative_rescored_live_fraction'] for row in trace['cases']])) )
        if label != 'teacher':
            summary['teacher_greater_full_return_cases'] = sum(teacher['cases'][case]['full_live']['rescored_task_reward'] > row['full_live']['rescored_task_reward'] for case, row in enumerate(trace['cases']))
            summary['teacher_greater_same_duration_return_cases'] = sum(row['teacher_same_duration']['rescored_task_reward'] > row['full_live']['rescored_task_reward'] for row in trace['cases'])
        results[label] = dict(path=trace['path'], cases=trace['cases'], summary=summary)
    assert all(digest(Path(path)) == expected for path, expected in protected.items()), 'Protected input/source mismatch'
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'sources').mkdir()
    snapshots = []
    for path in sources:
        snapshot = args.out/'sources'/path.name
        shutil.copy2(path, snapshot)
        snapshots.append(dict(path=str(path), snapshot=str(snapshot), sha256=digest(path)))
    for label, trace in traces.items():
        directory = args.out/label
        directory.mkdir()
        np.savez_compressed(directory/'scores.npz', **trace['arrays'])
        save(directory/'report.json', results[label])
    learners = [row for label, result in results.items() if label != 'teacher' for row in result['cases']]
    losses = [row['first_balance_loss'] for row in learners]
    assert all(row is not None for row in losses)
    summary = dict(learner_episodes=len(learners), first_loss_positive_cost_cases=sum(row['cost'] > 0 for row in losses),
        first_loss_mean_penalty=float(np.mean([row['applied_penalty'] for row in losses])),
        first_loss_mean_reference_reward=float(np.mean([row['reference_reward'] for row in losses])),
        first_loss_mean_rescored_reward=float(np.mean([row['rescored_task_reward'] for row in losses])),
        first_loss_zero_previous_limit_cost_cases=sum(row['previous_limit_cost'] == 0 for row in losses),
        teacher_max_speed_rad_s=float(teacher['arrays']['angular_speed_rad_s'].max()),
        teacher_zero_cost_verified=True, teacher_reward_unchanged=bool(np.array_equal(teacher['arrays']['reference_reward'], teacher['arrays']['rescored_task_reward'])))
    assert all(digest(Path(path)) == expected for path, expected in protected.items())
    assert all(digest(Path(row['snapshot'])) == row['sha256'] for row in snapshots)
    save(args.out/'protocol.json', dict(inputs=protected, sources=snapshots, discount=DISCOUNT,
        threshold_rad_s=THRESHOLD, scale_rad_s=SCALE, weight=WEIGHT, fail_value=flags['FAIL'],
        composition='reference_reward - 0.25*(1-exp(-(max(norm(qvel[3:6])-1,0)/1)**2)); failed task reward zero',
        common_prefix_frames=common_counts, policy_traces_fixed=True,
        scoring='Original reference task reward only; the previous limit cost is removed, not stacked. No AMP, critic or timeout-bootstrap rescoring.',
        aggregation='Case means have equal weight. Full-live teacher values are twelve-second truncated prefixes.',
        training_updates=0, physics_rollouts=0, simulator_imported=False, arena_changed=False))
    save(args.out/'report.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(), traces=results,
        summary=summary, episodes_analyzed=72, inputs_unchanged=True, source_snapshots_verified=True,
        reset_states_identical=True, independent_formula_verified=True, failed_task_reward_zero_verified=True,
        training_updates=0, physics_rollouts=0, simulator_imported=False, arena_changed=False))
    print(summary, flush=True)
    print({label: result['summary'] for label, result in results.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('logs/mimickit-training/stand-limit-reward-20261002-01'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    run(parser.parse_args())
