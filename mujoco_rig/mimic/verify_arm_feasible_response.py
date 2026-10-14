"""Audit delayed command transmission and paired finite-horizon responses."""
import json
from pathlib import Path

import numpy as np

from .run_support_skill_study import digest, save, FOUNDATION
from .probe_arm_feasible_response import OUT, PHASES, BRANCHES, fingerprint


def first_changed(values, threshold=1e-7):
    indices = np.flatnonzero(abs(values) > threshold)
    return int(indices[0]) if len(indices) else None


def main():
    from .rig import Rig
    out = Path('logs/mimickit-training/arm-feasible-response-review-20261008-01')
    out.mkdir(exist_ok=False)
    p = json.loads((OUT / 'protocol.json').read_text())
    report = json.loads((OUT / 'report.json').read_text())
    assert all(digest(Path(path)) == h for path, h in p['inputs'].items())
    assert all(digest(Path(path)) == h for path, h in report['output_hashes'].items())
    rig = Rig.load(FOUNDATION / 'dummy.xml', policy_exclude=())
    motor, qi, dt = p['motor'], p['qpos_index'], p['dt']
    joints = rig.model.actuator_trnid[rig.actuators, 0]
    lower, upper = rig.model.jnt_range[joints].astype(np.float32).T
    cases = []
    broad_underflow = 0
    max_pd_error = 0.
    for phase in PHASES:
        state = dict(np.load(OUT / f'{phase}-state.npz'))
        hashes = {r['start_sha256'] for r in report['results'] if r['phase'] == phase}
        assert hashes == {fingerprint(state)}
        base = np.load(OUT / f'{phase}-zero.npz')
        old_root = Path('logs/mimickit-training/arm-action-response-20261008-01')
        old_state = dict(np.load(old_root / f'{phase}-state.npz'))
        assert all(np.array_equal(v, old_state[k]) for k, v in state.items())
        old_base = np.load(old_root / f'{phase}-zero.npz')
        assert all(np.array_equal(base[k], old_base[k]) for k in old_base.files if k != 'broad_reward')
        for name, magnitude, duration in BRANCHES:
            z = np.load(OUT / f'{phase}-{name}.npz')
            assert np.array_equal(z['applied'][2:], z['issued'][:-2])
            for offset in range(2):
                slot = (int(state['cursor']) + offset) % 2
                assert np.array_equal(z['applied'][offset], state['control_pending'][slot, :8, :33])
            assert np.allclose(z['issued'], np.clip(z['source'] + z['addition'], lower, upper), atol=2e-6, rtol=0)
            assert np.array_equal(z['injected_action'], z['model_action'])
            injected = z['correction']
            expected = np.zeros_like(injected)
            expected[:duration, :, motor] = magnitude * z['alive'][:duration]
            assert np.allclose(injected, expected, atol=1e-7, rtol=0)
            nominal = z['previous_addition'] + np.clip(z['desired_addition'] - z['previous_addition'], -.02, .02)
            assert np.array_equal(nominal, z['nominal_addition'])
            proposed = nominal + injected * z['authority'][:, :, None] * z['fade'][:, :, None]
            projected = np.clip(z['previous_addition'] + np.clip(proposed - z['previous_addition'], -.02, .02), -1, 1)
            expected_addition = np.where(injected != 0, projected, nominal)
            expected_addition[z['quiet'] | ~z['alive']] = 0
            assert np.array_equal(expected_addition, z['addition'])
            live_command = z['alive'] & ~z['quiet']
            assert abs(z['addition'][live_command] - z['previous_addition'][live_command]).max() <= .0200001
            assert abs(z['addition'][z['alive']]).max() <= 1.0000001
            assert (z['issued'] >= lower - 1e-6).all() and (z['issued'] <= upper + 1e-6).all()

            normalized = 4 * (z['applied'][:, None] - z['substep_q']) + .08 * (z['applied_velocity'][:, None] - z['substep_v'])
            expected_torque = np.clip(normalized, -1, 1) * np.asarray(rig.action_scale, dtype=np.float32)
            pd_error = float(abs(expected_torque - z['substep_torque']).max())
            max_pd_error = max(max_pd_error, pd_error)
            assert pd_error < 2e-5
            assert np.allclose(z['components'].sum(-1), z['reward'], atol=1e-6, rtol=0)
            common = z['alive'] & base['alive'] & ~z['failed'] & ~base['failed']
            broad_underflow += int(((z['components'][:, :, :2] == 0) & common[:, :, None]).sum())
            if name == 'zero':
                continue
            assert np.array_equal(z['qpos'][:2], base['qpos'][:2])
            for world in np.flatnonzero(z['alive'][0]):
                valid = common[:, world]
                last = np.flatnonzero(valid)[-1]
                dq = z['qpos'][:, world, qi] - base['qpos'][:, world, qi]
                da = z['applied'][:, world, motor] - base['applied'][:, world, motor]
                narrow = (z['reward'][:, world] - base['reward'][:, world])[valid]
                broad = (z['broad_reward'][:, world] - base['broad_reward'][:, world])[valid]
                offsets = np.flatnonzero(valid)
                checkpoints = {}
                for offset in (5, 11, 29, 59):
                    if valid[offset]:
                        goal = z['goal'][offset, world, motor]
                        checkpoints[str(offset)] = dict(q_difference_rad=float(dq[offset]),
                            absolute_goal_error_difference_rad=float(abs(z['qpos'][offset, world, qi] - goal) - abs(base['qpos'][offset, world, qi] - goal)))
                initial_goal = float(z['goal'][0, world, motor])
                cases.append(dict(phase=phase, branch=name, pose=int(world), heldout=bool(world >= 6),
                    frames_common=int(valid.sum()), final_common_frame=int(last),
                    issued_initial_delta_rad=float(z['issued'][0, world, motor] - base['issued'][0, world, motor]),
                    issued_initial_transmission_fraction=float((z['issued'][0, world, motor] - base['issued'][0, world, motor]) / magnitude),
                    initial_authority=float(z['authority'][0, world]), initial_fade=float(z['fade'][0, world]),
                    initial_nominal_slew_direction=float(np.sign(z['nominal_addition'][0, world, motor] - z['previous_addition'][0, world, motor])),
                    initial_inward=bool(magnitude * (z['nominal_addition'][0, world, motor] - z['previous_addition'][0, world, motor]) < 0),
                    initial_slew_limited=bool(abs(z['desired_addition'][0, world, motor] - z['previous_addition'][0, world, motor]) > .02),
                    initial_raw_minus_source_rad=float(z['raw_target'][0, world, motor] - z['source'][0, world, motor]),
                    first_applied_difference_frame=first_changed(da), first_q_difference_frame=first_changed(dq),
                    peak_abs_q_difference_rad=float(abs(dq[valid]).max()), checkpoints=checkpoints,
                    initial_actual_minus_goal_rad=float(state['observations'][world, qi + 8] - initial_goal),
                    narrow_discounted_reward_difference=float(np.sum(narrow * .995**offsets)),
                    broad_discounted_reward_difference=float(np.sum(broad * .995**offsets)),
                    narrow_reward_only_gae_weighted_difference=float(np.sum(narrow * (.995 * .98)**offsets)),
                    broad_reward_only_gae_weighted_difference=float(np.sum(broad * (.995 * .98)**offsets)),
                    changed_supported_frames=int((z['supported'][valid, world] != base['supported'][valid, world]).sum()),
                    branch_failed=bool((z['alive'][:, world] & z['failed'][:, world]).any()),
                    baseline_failed=bool((base['alive'][:, world] & base['failed'][:, world]).any())))
    summary = []
    for phase in PHASES:
        for name, _, _ in BRANCHES[1:]:
            rows = [r for r in cases if r['phase'] == phase and r['branch'] == name]
            summary.append(dict(phase=phase, branch=name, cases=len(rows),
                transmission_range=[min(r['issued_initial_transmission_fraction'] for r in rows), max(r['issued_initial_transmission_fraction'] for r in rows)],
                applied_onsets=sorted({r['first_applied_difference_frame'] for r in rows if r['first_applied_difference_frame'] is not None}),
                q_onsets=sorted({r['first_q_difference_frame'] for r in rows if r['first_q_difference_frame'] is not None}),
                peak_q_change_range=[min(r['peak_abs_q_difference_rad'] for r in rows), max(r['peak_abs_q_difference_rad'] for r in rows)],
                changed_at_first_issue=sum(abs(r['issued_initial_delta_rad']) > 1e-7 for r in rows),
                inward_cases=sum(r['initial_inward'] for r in rows),
                positive_narrow_returns=sum(r['narrow_discounted_reward_difference'] > 0 for r in rows),
                positive_broad_returns=sum(r['broad_discounted_reward_difference'] > 0 for r in rows)))
    result = dict(inputs_and_outputs_unchanged=True, matched_states_exact=True, command_delay_exact=True,
        maximum_substep_pd_reconstruction_error_nm=max_pd_error, narrow_reward_zero_components_common_alive=broad_underflow,
        dt=dt, reward_only_gae_half_life_seconds=float(np.log(.5) / np.log(.995 * .98) * dt),
        reward_only_gae_weight_at_one_second=float((.995 * .98)**(1 / dt)),
        bounds_verified=True, old_states_and_baselines_exact=True, cases=cases, summary=summary, saved_bytes=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()))
    save(out / 'report.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
