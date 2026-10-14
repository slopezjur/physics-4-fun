"""Independent cached saved-trace audit for recovery-start learning."""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .corrective_step_audit import foot_geometry, measure_steps
from .run_support_skill_study import FOUNDATION, digest, save

OUT = Path('logs/mimickit-training/recovery-start-learning-20261009-01')
PILOT = Path('logs/mimickit-training/learned-ankle-20261009-01')


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def main():
    inputs, done, contract = (read(OUT/p) for p in ('inputs.json', 'completed.json', 'contract.json'))
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    assert all(digest(OUT/path) == sha for path, sha in done['outputs'].items())
    assert read(OUT/'checks.json')['passed']
    model = mujoco.MjModel.from_xml_path(str(FOUNDATION/'dummy.xml'))
    data, jac, rotation = mujoco.MjData(model), np.zeros((3, model.nv)), np.zeros(9)
    feet = [model.body(f'Foot_{side}').id for side in ('L', 'R')]
    reference = arrays(FOUNDATION/'stationary_reference.npz')['qpos'].astype(np.float32)
    mass = float(model.body_mass.sum())
    dt, ankles = contract['dt'], contract['ankles']
    lower, upper = (np.array(contract['control'][f'target_{side}_radians']) for side in ('lower', 'upper'))
    reports, event_count, max_feature_error = {}, 0, 0.
    for path in sorted(OUT.glob('*/trace.npz')):
        z, r = arrays(path), read(path.with_name('report.json'))
        name = path.parent.name
        reports[name] = r
        assert all(np.isfinite(value).all() for value in z.values()) and not z['warning'].any()
        assert np.allclose(z['time'], (np.arange(len(z['time']))+1)*dt, atol=1e-10, rtol=0)
        assert len(z['time']) == int(np.ceil(r['seconds']/dt))
        assert not z['previous_latent'][0].any()
        assert np.array_equal(z['history_before'][0, :, 0], z['history_before'][0, :, 2])
        history = np.concatenate((z['balance_features'][:-1, :, None], z['history_before'][:-1, :, :2]), axis=2)
        history[~z['alive'][:-1]] = z['history_before'][:-1][~z['alive'][:-1]]
        assert np.array_equal(z['history_before'][1:], history)
        memory = np.where(z['alive'][:-1, :, None], z['latent'][:-1], z['previous_latent'][:-1])
        assert np.array_equal(z['previous_latent'][1:], memory)
        addition = z['previous'][:, :, ankles[:1]] + np.clip(.1*np.tanh(z['latent'])-z['previous'][:, :, ankles[:1]], -.02, .02)
        addition[~z['alive']] = z['previous'][:, :, ankles[:1]][~z['alive']]
        assert np.allclose(z['addition'][:, :, ankles[0]], addition[:, :, 0], atol=1e-7, rtol=0)
        assert np.array_equal(z['addition'][:, :, ankles[0]], z['addition'][:, :, ankles[1]])
        assert not z['addition'][:, :, [i for i in range(33) if i not in ankles]].any()
        assert np.allclose(z['issued'], np.clip(z['nominal']+z['addition'], lower, upper), atol=1e-7, rtol=0)
        assert not z['active_target'][:2, :, -1].any()
        assert np.array_equal(z['active_target'][2:, :, :33], z['issued'][:-2])
        xy, clearance = foot_geometry(model, z['qpos'])
        assert np.allclose(xy, z['xy'], atol=2e-6, rtol=0)
        assert np.allclose(clearance, z['clearance'], atol=2e-6, rtol=0)
        for frame in sorted(set([0, len(z['time'])-1, *range(0, len(z['time']), 30)])):
            for i in range(16):
                data.qpos[:], data.qvel[:] = z['qpos'][frame, i], z['qvel'][frame, i]
                mujoco.mj_forward(model, data)
                mujoco.mj_jacSubtreeCom(model, data, jac, 1)
                velocity = jac@data.qvel
                mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
                angular = rotation.reshape(3, 3)@data.qvel[3:6]
                com, fp = z['com'][frame, i], data.xpos[feet]
                center = fp[:, :2].mean(0)
                capture = com[:2]+velocity[:2]*np.sqrt(max(.05, com[2]-fp[:, 2].min())/9.81)
                expected = np.r_[(com[:2]-center)/.1, velocity[:2]/.5, angular/2, (capture-center)/.1,
                    z['loads'][frame, i]/(mass*9.81), (data.qpos[2]-reference[i % 8, 2])/.15,
                    1-2*(data.qpos[4]**2+data.qpos[5]**2), ((fp[:, :2]-com[:2])/.2).ravel()]
                max_feature_error = max(max_feature_error, float(np.abs(expected-z['balance_features'][frame, i]).max()))
        alive = z['alive'] & ~z['failed']
        assert z['alive'][0].all() and np.array_equal(z['alive'][1:], alive[:-1])
        for c in r['cases']:
            i = c['case']
            assert c['pose'] == i % 8 and c['heldout'] == (i % 8 >= 6)
            assert c['survived'] == bool(alive[:, i].all())
            assert c['settled'] == bool(alive[:, i].all() and z['settled'][-int(np.ceil(2/dt)):, i].all())
            assert c['whole_stand'] == bool(alive[:, i].all() and z['stand'][:, i].all())
            force = np.zeros_like(z['force'][:, i])
            if not r['quiet']:
                hit = int(round(c['hit_seconds']/dt))
                force[hit:hit+6] = c['force_n']
                force[~z['alive'][:, i]] = 0
            assert np.array_equal(force, z['force'][:, i])
            events = measure_steps(z['time'], xy[:, i], clearance[:, i], z['loads'][:, i], alive[:, i],
                z['settled'][:, i], hit_time=c['hit_seconds'])
            assert events == c['steps']
            assert c['recovered_with_step'] == (c['settled'] and any(e['settled_window_start'] is not None for e in events))
            event_count += len(events)
        for key in ('survived', 'settled', 'whole_stand', 'recovered_with_step'):
            assert r[key] == sum(c[key] for c in r['cases'])
        if name.startswith('matched-'):
            desc = read(path.with_name('descriptor.json'))
            baseline = arrays(Path(desc['baseline'])/'trace.npz')
            issue = desc['issue_frame']
            end = len(z['time']) if desc['entire_trace_exact'] else issue
            assert all(np.array_equal(z[key][:end], baseline[key][:end]) for key in z)
            assert all(np.array_equal(z[key][:issue+2], baseline[key][:issue+2]) for key in ('qpos', 'qvel'))
    initial = torch.load(PILOT/'fit-921009'/'initial.pt', weights_only=True, map_location='cpu')
    fits, results = {}, {}
    for seed in contract['training_seeds']:
        folder = OUT/f'fit-{seed}'
        resets, history, report, account = (read(folder/name) for name in ('resets.json', 'history.json', 'report.json', 'start-accounting.json'))
        assert report['transitions'] == account['collected'] == 24576 and len(history) == 32
        assert all(r['pose'] < 6 and r['world'] % 8 == r['pose'] and r['onset_frame'] in contract['training_onset_frames']
                   and r['start_frame'] == r['onset_frame']+1 for r in resets)
        assert all(r['force_n'][1:] == [0., 0.] and r['force_n'][0] in (-150., 0., 150.) for r in resets)
        assert account['optimizer_prefix_transitions'] == 0
        assert account['reset_records'] == account['exact_transfers'] == len(resets)
        assert account['source_prefix_intervals'] == sum(r['start_frame'] for r in resets)
        assert not ({60, 102} & set(contract['training_onset_frames']))
        assert all(h['transitions'] == (i+1)*768 for i, h in enumerate(history))
        assert all(0 <= kl <= .01 for h in history for kl in h['accepted_kl'])
        saved = torch.load(folder/'initial.pt', weights_only=True, map_location='cpu')
        final = torch.load(folder/'final.pt', weights_only=True, map_location='cpu')
        assert all(torch.equal(value, saved[key]) for key, value in initial.items())
        assert any(not torch.equal(value, final[key]) for key, value in saved.items())
        fits[str(seed)] = dict(transitions=account['collected'], prefix_intervals=account['source_prefix_intervals'],
            resets=len(resets), quiet_resets=sum(r['force_n'][0] == 0 for r in resets),
            updates=sum(h['actor_updates'] for h in history), rollbacks=sum(h['rolled_back'] for h in history),
            wall_seconds=report['wall_seconds'])
        result = {}
        for mode in ('continuous', 'matched'):
            groups = {}
            for group in range(4):
                names = [f'{seed}-t{t}-g{group}' if mode == 'continuous' else f'matched-{seed}-t{t}-g{group}' for t in (0, 1)]
                cases = [c for name in names for c in reports[name]['cases']]
                groups[str(group)] = dict(settled=sum(c['settled'] for c in cases),
                    heldout_settled=sum(c['settled'] and c['heldout'] for c in cases),
                    by_sign=[sum(c['settled'] for c in cases if (c['case'] < 8) == (s == 0)) for s in (0, 1)])
                recorded = done['results'][str(seed)][mode]['groups'][str(group)] if mode == 'continuous' else done['results'][str(seed)][mode][str(group)]
                assert groups[str(group)]['settled'] == recorded['settled']
                assert groups[str(group)]['heldout_settled'] == recorded['heldout_settled']
            quiet = reports[f'{seed}-quiet']['whole_stand'] if mode == 'continuous' else sum(reports[f'matched-{seed}-t{t}-quiet']['whole_stand'] for t in (0, 1))
            result[mode] = dict(groups=groups, total_settled=sum(g['settled'] for g in groups.values()), quiet=quiet)
        result['sampled_settled'] = sum(reports[f'{seed}-sampled-t{t}']['settled'] for t in (0, 1))
        results[str(seed)] = result
    assert len(reports) == 45 and len(reports)*16 == done['cases'] == 720
    assert max_feature_error < 2e-5 and done['total_transitions'] == 49152
    audit = dict(passed=True, traces=45, cases=720, measured_steps=event_count,
        input_hashes=len(inputs), output_hashes=len(done['outputs']),
        max_independent_balance_feature_error=max_feature_error, fits=fits, results=results)
    save(OUT/'audit.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
