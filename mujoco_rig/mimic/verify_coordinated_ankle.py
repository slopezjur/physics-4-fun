"""Audit scalar action history, live balance observations and frozen evaluations."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .corrective_step_audit import foot_geometry, measure_steps
from .run_support_skill_study import FOUNDATION, digest, save


def read(path):
    return json.loads(path.read_text())


def verify(out):
    done, inputs, contract = (read(out/n) for n in ('completed.json', 'inputs.json', 'contract.json'))
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    assert all(digest(out/p) == sha for p, sha in done['outputs'].items())
    assert read(out/'checks.json')['passed']
    model = mujoco.MjModel.from_xml_path(str(FOUNDATION/'dummy.xml'))
    data = mujoco.MjData(model)
    jac, rotation = np.zeros((3, model.nv)), np.zeros(9)
    feet = [model.body(f'Foot_{s}').id for s in ('L', 'R')]
    reference = np.load(FOUNDATION/'stationary_reference.npz')['qpos'].astype(np.float32)
    mass = float(model.body_mass.sum())
    ankles = contract['ankles']
    lower = np.array(contract['control']['target_lower_radians'])
    upper = np.array(contract['control']['target_upper_radians'])
    traces = cases = steps = 0
    feature_error = 0.
    reports = {}
    for path in sorted(out.glob('*/trace.npz')):
        report = read(path.with_name('report.json'))
        reports[path.parent.name] = report
        with np.load(path, allow_pickle=False) as z:
            assert all(np.isfinite(z[k]).all() for k in z.files)
            assert not z['warning'].any()
            assert np.allclose(z['time'], (np.arange(len(z['time']))+1)*contract['dt'], rtol=0, atol=1e-10)
            assert not z['previous_latent'][0].any()
            assert np.array_equal(z['history_before'][0, :, 0], z['history_before'][0, :, 2])
            expected_history = np.concatenate((z['balance_features'][:-1, :, None], z['history_before'][:-1, :, :2]), axis=2)
            expected_history[~z['alive'][:-1]] = z['history_before'][:-1][~z['alive'][:-1]]
            assert np.array_equal(expected_history, z['history_before'][1:])
            expected_memory = np.where(z['alive'][:-1, :, None], z['latent'][:-1], z['previous_latent'][:-1])
            assert np.array_equal(expected_memory, z['previous_latent'][1:])
            desired = .1*np.tanh(z['latent'])
            expected_addition = z['previous'][:, :, ankles[:1]] + np.clip(desired-z['previous'][:, :, ankles[:1]], -.02, .02)
            expected_addition[~z['alive']] = z['previous'][:, :, ankles[:1]][~z['alive']]
            assert np.allclose(z['addition'][:, :, ankles[0]], expected_addition[:, :, 0], rtol=0, atol=1e-7)
            assert np.array_equal(z['addition'][:, :, ankles[0]], z['addition'][:, :, ankles[1]])
            assert not z['addition'][:, :, [i for i in range(33) if i not in ankles]].any()
            assert np.allclose(z['issued'], np.clip(z['nominal']+z['addition'], lower, upper), rtol=0, atol=1e-7)
            assert not z['active_target'][:2, :, -1].any()
            assert np.array_equal(z['active_target'][2:, :, :33], z['issued'][:-2])
            xy, clearance = foot_geometry(model, z['qpos'])
            assert np.allclose(xy, z['xy'], atol=2e-6, rtol=0)
            assert np.allclose(clearance, z['clearance'], atol=2e-6, rtol=0)
            # Independently reconstruct features at sampled physical frames, including endpoints.
            frames = sorted(set([0, len(z['time'])-1, *range(0, len(z['time']), 30)]))
            for f in frames:
                for i in range(16):
                    data.qpos[:], data.qvel[:] = z['qpos'][f, i], z['qvel'][f, i]
                    mujoco.mj_forward(model, data)
                    mujoco.mj_jacSubtreeCom(model, data, jac, 1)
                    velocity = jac@data.qvel
                    mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
                    angular = rotation.reshape(3, 3)@data.qvel[3:6]
                    com, fp = z['com'][f, i], data.xpos[feet]
                    center = fp[:, :2].mean(0)
                    capture = com[:2]+velocity[:2]*np.sqrt(max(.05, com[2]-fp[:, 2].min())/9.81)
                    expected = np.r_[(com[:2]-center)/.1, velocity[:2]/.5, angular/2,
                        (capture-center)/.1, z['loads'][f, i]/(mass*9.81),
                        (data.qpos[2]-reference[i % 8, 2])/.15,
                        1-2*(data.qpos[4]**2+data.qpos[5]**2), ((fp[:, :2]-com[:2])/.2).ravel()]
                    feature_error = max(feature_error, float(np.abs(expected-z['balance_features'][f, i]).max()))
            alive = z['alive'] & ~z['failed']
            for c in report['cases']:
                i = c['case']
                assert c['survived'] == bool(alive[:, i].all())
                assert c['settled'] == bool(alive[:, i].all() and z['settled'][-int(np.ceil(2/contract['dt'])):, i].all())
                expected_force = np.zeros_like(z['force'][:, i])
                if c['hit_seconds'] is not None:
                    frame = int(round(c['hit_seconds']/contract['dt']))
                    expected_force[frame:frame+6] = c['force_n']
                    expected_force[~z['alive'][:, i]] = 0
                assert np.array_equal(expected_force, z['force'][:, i])
                events = measure_steps(z['time'], xy[:, i], clearance[:, i], z['loads'][:, i], alive[:, i],
                                       z['settled'][:, i], hit_time=c['hit_seconds'])
                assert events == c['steps']
                steps += len(events)
            for key in ('survived', 'settled', 'whole_stand', 'recovered_with_step'):
                assert report[key] == sum(c[key] for c in report['cases'])
        traces += 1
        cases += 16
    assert feature_error < 2e-5, feature_error
    initials = []
    for seed in (921009, 931009):
        folder = out/f'fit-{seed}'
        resets, history, report = (read(folder/n) for n in ('resets.json', 'history.json', 'report.json'))
        assert report['transitions'] == 24576 and len(history) == 32
        assert {i % 8 for i in report['train_world_ids']} == set(range(6))
        assert all(r['pose'] < 6 and r['onset_frame'] in contract['training_onset_frames'] for r in resets)
        assert all(r['force_n'][1:] == [0., 0.] and r['force_n'][0] in (-150., 0., 150.) for r in resets)
        assert not ({60, 102} & set(contract['training_onset_frames']))
        assert all(h['transitions'] == (i+1)*768 for i, h in enumerate(history))
        assert all(0 <= kl <= .01 for h in history for kl in h['accepted_kl'])
        initials.append(torch.load(folder/'initial.pt', weights_only=True, map_location='cpu'))
        final = torch.load(folder/'final.pt', weights_only=True, map_location='cpu')
        assert any(not torch.equal(initials[-1][k], final[k]) for k in final)
    assert all(torch.equal(initials[0][k], initials[1][k]) for k in initials[0])
    for name, summary in done['results'].items():
        for group in range(4):
            rows = [c for t in (0, 1) for c in reports[f'{name}-t{t}-g{group}']['cases']]
            assert summary['groups'][str(group)]['settled'] == sum(c['settled'] for c in rows)
            assert summary['groups'][str(group)]['heldout_settled'] == sum(c['heldout'] and c['settled'] for c in rows)
        assert summary['quiet'] == reports[f'{name}-quiet']['whole_stand']
    assert traces == 33 and cases == 528
    result = dict(passed=True, traces=traces, cases=cases, measured_steps=steps,
        input_hashes=len(inputs), output_hashes=len(done['outputs']), transitions=49152,
        max_independent_balance_feature_error=feature_error, history_and_action_memory_verified=True,
        training_poses_and_timing_excluded=True, identical_initial_heads=True)
    save(out/'audit.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    verify(parser.parse_args().out)


if __name__ == '__main__':
    main()
