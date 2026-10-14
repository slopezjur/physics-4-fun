"""Read-only independent checks of the bounded direct-recovery evidence."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .corrective_step_audit import foot_geometry, measure_steps
from .run_support_skill_study import FOUNDATION, digest, save


def verify(out):
    completed = json.loads((out/'completed.json').read_text())
    inputs = json.loads((out/'inputs.json').read_text())
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    assert all(digest(out/p) == sha for p, sha in completed['outputs'].items())
    contract = json.loads((out/'contract.json').read_text())
    lo, hi = [np.array(contract['control'][k]) for k in ('target_lower_radians', 'target_upper_radians')]
    model = mujoco.MjModel.from_xml_path(str(FOUNDATION/'dummy.xml'))
    traces = cases = events = 0
    for path in sorted(out.glob('*/trace.npz')):
        report = json.loads(path.with_name('report.json').read_text())
        with np.load(path, allow_pickle=False) as z:
            alive = z['alive'] & ~z['failed']
            assert not z['warning'].any()
            assert all(np.isfinite(z[k]).all() for k in z.files)
            assert np.allclose(np.diff(z['time']), report['dt'], rtol=0, atol=1e-10)
            assert np.all(np.abs(z['addition']) <= .1000001)
            assert np.all(np.abs(z['addition']-z['previous']) <= .0200001)
            assert np.allclose(z['issued'], np.clip(z['nominal']+z['addition'], lo, hi), rtol=0, atol=1e-7)
            assert not z['active_target'][:2, :, -1].any()
            assert (z['active_target'][2:, :, -1] == 1).all()
            assert np.array_equal(z['active_target'][2:, :, :33], z['issued'][:-2])
            xy, clearance = foot_geometry(model, z['qpos'])
            assert np.allclose(xy, z['xy'], rtol=0, atol=2e-6)
            assert np.allclose(clearance, z['clearance'], rtol=0, atol=2e-6)
            for c in report['cases']:
                i = c['case']
                t = z['time']-report['dt']
                expected = np.zeros_like(z['force'][:, i])
                if c['hit_seconds'] is not None:
                    window = (t >= c['hit_seconds']-1e-8) & (t < c['hit_seconds']+6*report['dt']-1e-8)
                    expected[window & z['alive'][:, i]] = c['force_n']
                assert np.array_equal(expected, z['force'][:, i])
                assert np.allclose(z['force'][:, i].sum(0)*report['dt'], c['impulse_ns'])
                assert c['survived'] == bool(alive[:, i].all())
                assert c['settled'] == bool(alive[:, i].all() and z['settled'][-int(np.ceil(2/report['dt'])):, i].all())
                measured = measure_steps(z['time'], xy[:, i], clearance[:, i], z['loads'][:, i],
                                         alive[:, i], z['settled'][:, i], hit_time=c['hit_seconds'])
                assert measured == c['steps']
                assert c['recovered_with_step'] == (c['settled'] and any(e['settled_window_start'] is not None for e in measured))
                events += len(measured)
            for key in ('survived', 'settled', 'whole_stand', 'recovered_with_step'):
                assert report[key] == sum(c[key] for c in report['cases'])
            assert report['measured_steps'] == sum(len(c['steps']) for c in report['cases'])
        traces += 1
        cases += len(report['cases'])
    for tag in ('quiet', 'push'):
        with np.load(out/f'parity-bare-{tag}'/'trace.npz') as a, np.load(out/f'parity-zero-{tag}'/'trace.npz') as b:
            assert all(np.array_equal(a[k], b[k]) for k in a.files)
    initial = []
    for seed in (901009, 911009):
        fit = out/f'fit-{seed}'
        history = json.loads((fit/'history.json').read_text())
        report = json.loads((fit/'report.json').read_text())
        assert len(history) == 32 and report['transitions'] == 24576
        assert {i % 8 for i in report['train_world_ids']} == set(range(6))
        assert all(h['transitions'] == (i+1)*768 for i, h in enumerate(history))
        assert all(0 <= kl <= .01 for h in history for kl in h['accepted_kl'])
        initial.append(torch.load(fit/'initial.pt', map_location='cpu', weights_only=True))
        final = torch.load(fit/'final.pt', map_location='cpu', weights_only=True)
        assert any(not torch.equal(initial[-1][k], final[k]) for k in final)
    assert all(torch.equal(initial[0][k], initial[1][k]) for k in initial[0])
    for name, summary in completed['results'].items():
        records = [json.loads((out/f'{name}-push-{i}'/'report.json').read_text()) for i in range(4)]
        for key in ('survived', 'settled', 'measured_steps', 'recovered_with_step'):
            assert summary[key] == sum(r[key] for r in records)
        assert summary['heldout_settled'] == sum(c['heldout'] and c['settled'] for r in records for c in r['cases'])
    result = dict(passed=True, traces=traces, evaluated_cases=cases, audited_steps=events,
                  input_hashes=len(inputs), output_hashes=len(completed['outputs']),
                  transitions=49152, exact_zero_parity=True, identical_initial_heads=True,
                  heldout_poses_excluded=True, force_window_and_delay_verified=True)
    save(out/'audit.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    verify(parser.parse_args().out)


if __name__ == '__main__':
    main()
