"""Independent cached-trace audit of matched native recovery starts."""
import json
from pathlib import Path

import mujoco
import numpy as np

from .corrective_step_audit import foot_geometry, measure_steps
from .run_support_skill_study import FOUNDATION, digest, save

OUT = Path('logs/mimickit-training/recovery-starts-20261009-01')


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def main():
    done, inputs, contract = (read(OUT/name) for name in ('completed.json', 'inputs.json', 'contract.json'))
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    assert all(digest(OUT/path) == sha for path, sha in done['outputs'].items())
    assert len(read(OUT/'checks.json')) == 2 and all(c['passed'] for c in read(OUT/'checks.json'))
    model = mujoco.MjModel.from_xml_path(str(FOUNDATION/'dummy.xml'))
    dt, ankles = contract['dt'], contract['ankles']
    lower, upper = (np.array(contract['control'][f'target_{side}_radians']) for side in ('lower', 'upper'))
    records, steps, parity_traces = {}, 0, 0
    for timing in (0, 1):
        common = arrays(OUT/f't{timing}-common.npz')
        quiet = arrays(OUT/f't{timing}-quiet-start.npz')
        pushed = arrays(OUT/f't{timing}-push-start.npz')
        hit = int(common['task_steps'][0])
        assert (common['task_steps'] == hit).all()
        assert np.array_equal(quiet['pending_oldest_first'], pushed['pending_oldest_first'])
        assert np.array_equal(quiet['control_active'], pushed['control_active'])
        assert not common['env_previous'].any() and not common['env_last_latent'].any()
        assert not np.array_equal(quiet['native_integration'], pushed['native_integration'])
        assert not np.array_equal(quiet['features'], pushed['features'])
        for branch, start in (('quiet', quiet), ('push', pushed)):
            zero = arrays(OUT/f't{timing}-{branch}-zero'/'trace.npz')
            for label in ('zero', 'negative', 'positive'):
                folder = OUT/f't{timing}-{branch}-{label}'
                report, z = read(folder/'report.json'), arrays(folder/'trace.npz')
                records[folder.name] = report
                assert all(np.isfinite(value).all() for value in z.values()) and not z['warning'].any()
                assert len(z['time']) == int(np.ceil(6/dt))
                assert np.allclose(z['time'], (np.arange(len(z['time']))+1)*dt, atol=1e-10, rtol=0)
                assert all(np.array_equal(z[key][:hit+1], zero[key][:hit+1]) for key in z)
                assert all(np.array_equal(z[key][:hit+3], zero[key][:hit+3]) for key in ('qpos', 'qvel'))
                assert np.array_equal(z['history_before'][hit+1], start['env_history'])
                assert np.array_equal(z['previous_latent'][hit+1], start['env_last_latent'])
                assert np.array_equal(z['previous'][hit+1], start['env_previous'])
                assert np.array_equal(z['active_target'][hit+1:hit+3], start['pending_oldest_first'])
                if report['reference']:
                    archived = arrays(Path(report['reference'])/'trace.npz')
                    fields = sorted(set(archived) & set(z))
                    assert fields == report['exact_archived_fields']
                    assert all(np.array_equal(z[key], archived[key][:len(z['time'])]) for key in fields)
                    parity_traces += 1
                expected = np.zeros_like(z['addition'])
                expected[hit+1:hit+13, :, ankles] = report['amplitude']
                expected = z['previous'] + np.clip(expected-z['previous'], -.02, .02)
                expected[~z['alive']] = z['previous'][~z['alive']]
                assert np.allclose(z['addition'], expected, atol=1e-7, rtol=0)
                assert np.allclose(z['issued'], np.clip(z['nominal']+z['addition'], lower, upper), atol=1e-7, rtol=0)
                assert np.array_equal(z['active_target'][2:, :, :33], z['issued'][:-2])
                assert not z['active_target'][:2, :, -1].any()
                force = np.zeros_like(z['force'])
                if branch == 'push':
                    force[hit:hit+6, :8, 0] = 150.
                    force[hit:hit+6, 8:, 0] = -150.
                force[~z['alive']] = 0
                assert np.array_equal(force, z['force'])
                xy, clearance = foot_geometry(model, z['qpos'])
                assert np.allclose(xy, z['xy'], atol=2e-6, rtol=0)
                assert np.allclose(clearance, z['clearance'], atol=2e-6, rtol=0)
                alive = z['alive'] & ~z['failed']
                assert z['alive'][0].all() and np.array_equal(z['alive'][1:], alive[:-1])
                for c in report['cases']:
                    i = c['case']
                    assert c['pose'] == i % 8 and c['heldout'] == (i % 8 >= 6)
                    assert c['survived'] == bool(alive[:, i].all())
                    assert c['settled'] == bool(alive[:, i].all() and z['settled'][-int(np.ceil(2/dt)):, i].all())
                    events = measure_steps(z['time'], xy[:, i], clearance[:, i], z['loads'][:, i], alive[:, i],
                        z['settled'][:, i], hit_time=None if branch == 'quiet' else hit*dt)
                    assert c['steps'] == events
                    assert c['recovered_with_step'] == (c['settled'] and any(e['settled_window_start'] is not None for e in events))
                    steps += len(events)
                for key in ('survived', 'settled', 'whole_stand'):
                    assert report[key] == sum(c[key] for c in report['cases'])
    summary = {}
    for label in ('zero', 'negative', 'positive'):
        cases = [c for t in (0, 1) for c in records[f't{t}-push-{label}']['cases']]
        baseline = [c for t in (0, 1) for c in records[f't{t}-push-zero']['cases']]
        summary[label] = dict(settled=sum(c['settled'] for c in cases),
            heldout_settled=sum(c['settled'] and c['heldout'] for c in cases),
            quiet_settled=sum(records[f't{t}-quiet-{label}']['settled'] for t in (0, 1)),
            quiet_whole_stand=sum(records[f't{t}-quiet-{label}']['whole_stand'] for t in (0, 1)),
            by_sign=[sum(c['settled'] for c in cases if (c['case'] < 8) == (s == 0)) for s in (0, 1)],
            gains=[dict(timing=i//16, case=c['case']) for i, (c, b) in enumerate(zip(cases, baseline)) if c['settled'] and not b['settled']],
            losses=[dict(timing=i//16, case=c['case']) for i, (c, b) in enumerate(zip(cases, baseline)) if b['settled'] and not c['settled']])
    assert len(records) == 12 and parity_traces == 8 and done['cases'] == 192
    result = dict(passed=True, cases=192, traces=12, exact_archived_traces=parity_traces,
        measured_steps=steps, input_hashes=len(inputs), output_hashes=len(done['outputs']),
        matching_starts=True, preserved_queue_and_history=True, summary=summary)
    save(OUT/'audit.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
