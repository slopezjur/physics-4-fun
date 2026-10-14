"""Independent saved-trace checks for the frozen quiet-ankle intervention."""
import json
from pathlib import Path

import mujoco
import numpy as np

from .corrective_step_audit import foot_geometry, measure_steps
from .run_support_skill_study import FOUNDATION, digest, save

OUT = Path('logs/mimickit-training/ankle-response-causal-20261009-01')


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def main():
    inputs, done, contract = (read(OUT/name) for name in ('inputs.json', 'completed.json', 'contract.json'))
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    assert all(digest(OUT/p) == sha for p, sha in done['outputs'].items())
    assert done['new_cases'] == 64 and done['training_transitions'] == 0
    pilot = Path(contract['pilot'])
    dt, ankles = contract['dt'], contract['ankles']
    lower = np.array(contract['control']['target_lower_radians'])
    upper = np.array(contract['control']['target_upper_radians'])
    model = mujoco.MjModel.from_xml_path(str(FOUNDATION/'dummy.xml'))
    summaries, comparisons, measured_steps = {}, [], 0
    for seed in contract['seeds']:
        quiet = arrays(pilot/f'{seed}-quiet'/'trace.npz')
        rows = []
        for timing, onset in enumerate((1., 1.7)):
            name = f'{seed}-t{timing}'
            z = arrays(OUT/name/'trace.npz')
            normal = arrays(pilot/f'{name}-g0'/'trace.npz')
            report = read(OUT/name/'report.json')
            old_report = read(pilot/f'{name}-g0'/'report.json')
            hit, frames = int(np.ceil(onset/dt)), len(z['time'])
            switch = hit+1
            assert frames == int(np.ceil(6/dt))
            assert all(np.isfinite(value).all() for value in z.values())
            assert not z['warning'].any()
            assert np.allclose(z['time'], (np.arange(frames)+1)*dt, atol=1e-10, rtol=0)
            for key in z:
                assert np.array_equal(z[key][:switch], normal[key][:switch]), (name, key, 'prefix')
            for key in ('qpos', 'qvel', 'active_target'):
                assert np.array_equal(z[key][:switch+2], normal[key][:switch+2]), (name, key, 'delay')
            assert np.array_equal(z['latent'][switch:], quiet['latent'][switch:frames])
            mask = z['alive'][switch:]
            assert np.allclose(z['addition'][switch:][mask], quiet['addition'][switch:frames][mask], atol=1e-7, rtol=0)
            for key in ('qpos', 'qvel', 'addition', 'latent', 'history_before'):
                assert np.array_equal(normal[key][:hit], quiet[key][:hit]), (name, key, 'quiet prefix')
            history = np.concatenate((z['balance_features'][:-1, :, None], z['history_before'][:-1, :, :2]), axis=2)
            history[~z['alive'][:-1]] = z['history_before'][:-1][~z['alive'][:-1]]
            assert np.array_equal(history, z['history_before'][1:])
            memory = np.where(z['alive'][:-1, :, None], z['latent'][:-1], z['previous_latent'][:-1])
            assert np.array_equal(memory, z['previous_latent'][1:])
            expected = z['previous'][:, :, ankles[:1]] + np.clip(
                .1*np.tanh(z['latent'])-z['previous'][:, :, ankles[:1]], -.02, .02)
            expected[~z['alive']] = z['previous'][:, :, ankles[:1]][~z['alive']]
            assert np.allclose(z['addition'][:, :, ankles[0]], expected[:, :, 0], atol=1e-7, rtol=0)
            assert np.array_equal(z['addition'][:, :, ankles[0]], z['addition'][:, :, ankles[1]])
            assert not z['addition'][:, :, [i for i in range(33) if i not in ankles]].any()
            assert np.allclose(z['issued'], np.clip(z['nominal']+z['addition'], lower, upper), atol=1e-7, rtol=0)
            assert not z['active_target'][:2, :, -1].any()
            assert np.array_equal(z['active_target'][2:, :, :33], z['issued'][:-2])
            force = np.zeros_like(z['force'])
            force[hit:hit+6, :8, 0] = 150.
            force[hit:hit+6, 8:, 0] = -150.
            force[~z['alive']] = 0
            assert np.array_equal(force, z['force'])
            xy, clearance = foot_geometry(model, z['qpos'])
            assert np.allclose(xy, z['xy'], atol=2e-6, rtol=0)
            assert np.allclose(clearance, z['clearance'], atol=2e-6, rtol=0)
            alive = z['alive'] & ~z['failed']
            assert z['alive'][0].all()
            assert np.array_equal(z['alive'][1:], alive[:-1])
            for c, old in zip(report['cases'], old_report['cases'], strict=True):
                i = c['case']
                assert i == old['case'] and c['pose'] == i % 8 and c['heldout'] == (i % 8 >= 6)
                assert abs(c['hit_seconds']-hit*dt) < 1e-6
                assert c['survived'] == bool(alive[:, i].all())
                assert c['settled'] == bool(alive[:, i].all() and z['settled'][-int(np.ceil(2/dt)):, i].all())
                assert c['whole_stand'] == bool(alive[:, i].all() and z['stand'][:, i].all())
                events = measure_steps(z['time'], xy[:, i], clearance[:, i], z['loads'][:, i], alive[:, i],
                                       z['settled'][:, i], hit_time=c['hit_seconds'])
                assert events == c['steps']
                assert c['recovered_with_step'] == (c['settled'] and any(e['settled_window_start'] is not None for e in events))
                measured_steps += len(events)
                rows.append(dict(timing=timing, case=i, pose=i % 8, heldout=c['heldout'],
                    sign='+X' if i < 8 else '-X', normal=old['settled'], replay=c['settled'],
                    normal_survived=old['survived'], replay_survived=c['survived']))
            for key in ('survived', 'settled', 'whole_stand', 'recovered_with_step'):
                assert report[key] == sum(c[key] for c in report['cases'])
            common = z['alive'][switch:] & normal['alive'][switch:]
            delta = z['addition'][switch:, :, ankles[0]]-normal['addition'][switch:, :, ankles[0]]
            comparisons.append(dict(name=name, switch_frame=switch, exact_prefix_frames=switch,
                exact_physical_prefix_frames=switch+2, addition_difference_rms=float(np.sqrt(np.mean(delta[common]**2))),
                addition_difference_max=float(np.max(np.abs(delta[common]))),
                live_source_nominal_difference_max=float(np.max(np.abs(z['nominal']-normal['nominal'])))))
        summaries[str(seed)] = dict(normal_settled=sum(r['normal'] for r in rows),
            replay_settled=sum(r['replay'] for r in rows),
            normal_survived=sum(r['normal_survived'] for r in rows),
            replay_survived=sum(r['replay_survived'] for r in rows),
            normal_only=[r for r in rows if r['normal'] and not r['replay']],
            replay_only=[r for r in rows if r['replay'] and not r['normal']],
            heldout_normal=sum(r['normal'] and r['heldout'] for r in rows),
            heldout_replay=sum(r['replay'] and r['heldout'] for r in rows), cases=rows)
    result = dict(passed=True, traces=4, cases=64, measured_steps=measured_steps,
        input_hashes=len(inputs), output_hashes=len(done['outputs']),
        exact_prefixes=True, two_interval_delay=True, quiet_replay_verified=True,
        geometry_and_outcomes_verified=True, comparisons=comparisons, seeds=summaries)
    save(OUT/'audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'seeds'}), flush=True)
    print(json.dumps({s: {k: v for k, v in r.items() if k != 'cases'} for s, r in summaries.items()}), flush=True)


if __name__ == '__main__':
    main()
