"""Saved-evidence diagnosis of the target-PD duration screen."""
from collections import Counter
from pathlib import Path
import json
import numpy as np

from .audit_stand_failures import digest, read, save
from .replicate_stand_representations import check_protected

PRIOR = Path('logs/mimickit-training/stand-target-pd-curve-20261003-01')
OUT = Path('logs/mimickit-training/stand-duration-diagnosis-20261003-01')


def run():
    verification = read(PRIOR/'final-verification.json')
    protocol = read(PRIOR/'protocol.json')
    inputs = {**protocol['inputs'], **verification['output_hashes'],
        str(PRIOR/'final-verification.json'): digest(PRIOR/'final-verification.json'),
        str(Path(__file__)): digest(Path(__file__))}
    check_protected(inputs, protocol['sources'])
    OUT.mkdir(parents=True, exist_ok=False)
    contract = read(PRIOR/'contract.json')
    scale = np.asarray(contract['torque_scale_nm'])
    lower, upper = np.asarray(contract['target_lower_radians']), np.asarray(contract['target_upper_radians'])
    names = contract['actuator_names']
    cases, counts = [], Counter()
    for stage in ('initial', 'seconds-180', 'seconds-360'):
        for backend in ('native', 'gpu'):
            with np.load(PRIOR/stage/backend/'trace.npz', allow_pickle=False) as data:
                trace = {key: data[key] for key in data.files}
            gates = [k for k in trace if k.startswith(('balance_gate_', 'posture_gate_'))]
            for case in range(8):
                live = trace['alive_before'][:, case]
                departure = int(np.flatnonzero(live & ~(trace['balance'][:, case] & trace['posture'][:, case]))[0])
                terminal = int(np.flatnonzero(live)[-1])
                start = max(0, departure-30)
                window = slice(start, departure+1)
                first = [key for key in gates if not trace[key][departure, case]]
                counts.update(first)
                q = trace['substep_q'][window, :, case]
                near = np.minimum(q-lower, upper-q) < np.deg2rad(5.)
                fraction = near.mean(axis=(0, 1))
                terminal_flags = dict(pelvis_below_065=bool(trace['pelvis_height_m'][terminal, case] < .65),
                    nonfoot_contact=bool(trace['non_foot_ground_contacts_n'][terminal, case].max() > 1.),
                    nonfinite=not bool(trace['finite_state'][terminal, case]))
                cases.append(dict(stage=stage, backend=backend, case=case, first_failed_gates=first,
                    first_departure_seconds=float(trace['time_s'][departure]),
                    physical_failure_seconds=float(trace['time_s'][terminal]), physical_failure=terminal_flags,
                    preceding_window_seconds=float((departure-start+1)*contract['control_timestep']),
                    action_max_abs=float(np.abs(trace['action'][window, case]).max()),
                    saturated_motor_fraction=float((np.abs(trace['substep_torque'][window, :, case])/scale >= .99).mean()),
                    near_limit_motor_fractions={name: float(f) for name, f in zip(names, fraction) if f > 0},
                    departure_root_velocity_xy=trace['qvel_after'][departure, case, :2].tolist()))
    rows = [json.loads(line) for line in (PRIOR/'updates.jsonl').read_text().splitlines()]
    statistics = {}
    for name, lo, hi in (('first_20', 0, 20), ('ending_at_180', 89, 109), ('ending_at_360', 203, 223)):
        statistics[name] = dict(first_update=lo+1, last_update=hi,
            means={key: float(np.mean([row[key] for row in rows[lo:hi]]))
                for key in rows[0] if key not in ('iteration', 'transitions', 'training_seconds')})
    pairs = []
    for backend in ('native', 'gpu'):
        before = read(PRIOR/'seconds-180'/backend/'report.json')['cases']
        after = read(PRIOR/'seconds-360'/backend/'report.json')['cases']
        delta = [b['survival_seconds']-a['survival_seconds'] for a, b in zip(before, after)]
        pairs.append(dict(backend=backend, survival_changes_seconds=delta,
            improved=sum(d > 0 for d in delta), worsened=sum(d < 0 for d in delta)))
    report = dict(inputs=inputs, cases=cases, first_failed_gate_counts=dict(counts), ppo_windows=statistics,
        three_to_six_minute_pairs=pairs, no_predeparture_saturation=all(c['saturated_motor_fraction'] == 0 for c in cases),
        hypothesis='Strengthen quiet horizontal-speed preference while retaining reference imitation and AMP',
        current_task_reward_loss_at_speed_02_other_errors_zero=float(.1*(1-np.exp(-.2**2))),
        candidate_task_factor_at_speed_02=float(np.exp(-1.)),
        causal_conclusion=False, new_physics_steps=0, new_forward_passes=0, new_training_updates=0)
    check_protected(inputs, protocol['sources'])
    save(OUT/'report.json', report)
    save(OUT/'verification.json', dict(inputs_unchanged=True, inputs=len(inputs), cases=len(cases),
        report_sha256=digest(OUT/'report.json'), source_sha256=digest(Path(__file__))))
    print(json.dumps(dict(first_failed_gates=dict(counts), predeparture_saturation=False, pairs=pairs)))


if __name__ == '__main__':
    run()
