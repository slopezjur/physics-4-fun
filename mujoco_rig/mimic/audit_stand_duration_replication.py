"""Compare sealed duration results and diagnose their saved live trajectories."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np

from .audit_stand_failures import read, save
from .measure_stand_learning_curve import continuation
from .stand_experiment_evidence import protect, seal, verify_rollout


def diagnose(root):
    protocol, report, contract = (read(root/name) for name in ('protocol.json', 'report.json', 'contract.json'))
    scale = np.asarray(contract['torque_scale_nm'])
    lower, upper = np.asarray(contract['target_lower_radians']), np.asarray(contract['target_upper_radians'])
    dt = contract['control_timestep']
    stages, cases = report['assessments'], []
    for stage in stages:
        for backend in ('native', 'gpu'):
            trace, verified = verify_rollout(root/stage/backend)
            assert verified == stages[stage]['results'][backend]
            gates = [key for key in trace if key.startswith(('balance_gate_', 'posture_gate_'))]
            for case in range(8):
                live = trace['alive_before'][:, case]
                failed = np.flatnonzero(live & ~(trace['balance'][:, case] & trace['posture'][:, case]))
                terminal = int(np.flatnonzero(live)[-1])
                departure = int(failed[0]) if len(failed) else None
                endpoint = departure if departure is not None else terminal
                start = max(0, endpoint-30)
                window = slice(start, endpoint+1)
                q = trace['substep_q'][window, :, case]
                fraction = (np.minimum(q-lower, upper-q) < np.deg2rad(5.)).mean(axis=(0, 1))
                cases.append(dict(stage=stage, backend=backend, case=case,
                    first_failed_gates=[key for key in gates if not trace[key][departure, case]] if departure is not None else [],
                    first_departure_seconds=float(trace['time_s'][departure]) if departure is not None else None,
                    survival_seconds=float(trace['time_s'][terminal]), terminal_done=int(trace['done'][terminal, case]),
                    terminal_pelvis_below_065=bool(trace['pelvis_height_m'][terminal, case] < .65),
                    terminal_nonfoot_contact=bool(trace['non_foot_ground_contacts_n'][terminal, case].max() > 1.),
                    terminal_nonfinite=not bool(trace['finite_state'][terminal, case]),
                    preceding_window_seconds=float((endpoint-start+1)*dt),
                    action_max_abs=float(np.abs(trace['action'][window, case]).max()),
                    saturated_motor_fraction=float((np.abs(trace['substep_torque'][window, :, case])/scale >= .99).mean()),
                    near_limit_motor_fractions={name: float(value) for name, value in zip(contract['actuator_names'], fraction) if value > 0},
                    departure_root_velocity_xy=trace['qvel_after'][departure, case, :2].tolist() if departure is not None else None))
    counts = {stage: dict(Counter(gate for row in cases if row['stage'] == stage for gate in row['first_failed_gates'])) for stage in stages}
    rows = [json.loads(line) for line in (root/'updates.jsonl').read_text().splitlines()]
    with np.load(root/'reward-records.npz', allow_pickle=False) as archive:
        rewards, reward_columns = archive['values'], archive['columns'].tolist()
    assert rewards.shape == (len(rows), 64, len(reward_columns))
    statistics = {}
    cumulative = {'iteration', 'transitions', 'training_seconds', 'actor_optimizer_steps', 'critic_optimizer_steps', 'num_eps'}
    for label, lo, hi in [('first_20', 0, 20)]+[(f'ending_at_{n}', n-20, n) for n in (100, 300, 600) if len(rows) >= n]:
        statistics[label] = dict(first_update=lo+1, last_update=hi,
            means={key: float(np.mean([row[key] for row in rows[lo:hi]])) for key in rows[0] if key not in cumulative},
            task_reward_means={key: float(rewards[lo:hi, :, index].mean()) for index, key in enumerate(reward_columns)},
            training_seconds_per_update=(rows[hi-1]['training_seconds']-(rows[lo-1]['training_seconds'] if lo else 0.))/(hi-lo))
    comparisons = {}
    for before, after in ((100, 300), (300, 600)):
        if f'updates-{after}' not in stages:
            continue
        previous, current = stages[f'updates-{before}'], stages[f'updates-{after}']
        pairs = {}
        for backend in ('native', 'gpu'):
            a, b = previous['results'][backend]['cases'], current['results'][backend]['cases']
            assert [c['case'] for c in a] == [c['case'] for c in b] == list(range(8))
            deltas = {key: [y[key]-x[key] for x, y in zip(a, b)] for key in ('survival_seconds', 'longest_valid_seconds')}
            pairs[backend] = dict(changes= deltas,
                survival_improved=sum(v >= dt-1e-12 for v in deltas['survival_seconds']),
                survival_regressed=sum(v <= -dt+1e-12 for v in deltas['survival_seconds']))
        comparisons[f'{before}_to_{after}'] = dict(decision=continuation(current, previous, dt), pairs=pairs)
    return dict(seed=protocol['seed'], root=str(root), cases=cases, first_failed_gate_counts=counts,
        ppo_windows=statistics, comparisons=comparisons,
        no_predeparture_saturation=all(c['saturated_motor_fraction'] == 0 for c in cases),
        ppo_updates=report['ppo_updates'], transitions=report['transitions'], training_seconds=report['training_seconds'],
        final={b: list(stages.values())[-1]['results'][b] for b in ('native', 'gpu')})


def run(args):
    inputs, sources = protect(args.roots[-1], args.out, [Path(__file__)])
    # Verify each explicit root belongs to the protected evidence chain.
    protected = {str(Path(p).resolve()) for p in inputs}
    for root in args.roots:
        assert str((root/'final-verification.json').resolve()) in protected
    results = [diagnose(root) for root in args.roots]
    durations = [r['comparisons'].get('100_to_300') for r in results]
    replicated = all(d is not None and d['decision']['continue_training'] for d in durations)
    report = dict(results=results, duration_gain_repeated_in_all_listed_seeds=replicated,
        causal_conclusion=False, new_physics_steps=0, new_forward_passes=0, new_training_updates=0)
    save(args.out/'report.json', report)
    seal(args.out, inputs, sources, verified=True, analyzed_cases=sum(len(r['cases']) for r in results),
        saved_gates_and_controls_recomputed=True, new_physics_steps=0, new_forward_passes=0, new_training_updates=0)
    print(dict(seeds=[r['seed'] for r in results], duration_gain_repeated=replicated,
        first_failed_gate_counts={r['seed']: r['first_failed_gate_counts'] for r in results}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--roots', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
