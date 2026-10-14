"""Independently verify force timing, interventions and recovery classifications."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .corrective_step_audit import foot_geometry, measure_steps
from .run_support_skill_study import FOUNDATION, digest, save


def read(path):
    return json.loads(path.read_text())


def trace(folder):
    with np.load(folder/'trace.npz', allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def verify(out):
    completed, inputs, contract = (read(out/name) for name in ('completed.json', 'inputs.json', 'contract.json'))
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    assert all(digest(out/p) == sha for p, sha in completed['outputs'].items())
    model = mujoco.MjModel.from_xml_path(str(FOUNDATION/'dummy.xml'))
    lo, hi = np.array(contract['lower']), np.array(contract['upper'])
    legs = contract['leg_indices']
    all_reports, all_descriptors = {}, {}
    event_count = recovered_steps = 0
    for path in sorted(out.glob('*/trace.npz')):
        folder = path.parent
        report, descriptor, z = read(folder/'report.json'), read(folder/'descriptor.json'), trace(folder)
        all_reports[folder.name], all_descriptors[folder.name] = report, descriptor
        assert not z['warning'].any()
        assert all(np.isfinite(v).all() for v in z.values())
        assert np.allclose(z['time'], (np.arange(len(z['time']))+1)*contract['dt'], rtol=0, atol=1e-10)
        alive = z['alive'] & ~z['failed']
        assert (np.abs(z['addition']) <= .1000001).all()
        assert (np.abs(z['addition']-z['previous']) <= .0200001).all()
        assert np.allclose(z['issued'], np.clip(z['nominal']+z['addition'], lo, hi), atol=1e-7, rtol=0)
        assert not z['active_target'][:2, :, -1].any()
        assert (z['active_target'][2:, :, -1] == 1).all()
        assert np.array_equal(z['active_target'][2:, :, :33], z['issued'][:-2])
        desired = np.zeros_like(z['addition'])
        if descriptor['profile']:
            begin = descriptor['issue_frame']
            desired[begin:begin+12] = descriptor['amplitude'] * np.array(contract['combinations'][descriptor['profile']])
        expected = z['previous'] + np.clip(desired-z['previous'], -.02, .02)
        expected[~z['alive']] = z['previous'][~z['alive']]
        assert np.allclose(expected, z['addition'], rtol=0, atol=1e-7)
        assert not z['addition'][:, :, [i for i in range(33) if i not in legs]].any()
        xy, clearance = foot_geometry(model, z['qpos'])
        assert np.allclose(xy, z['xy'], atol=2e-6, rtol=0)
        assert np.allclose(clearance, z['clearance'], atol=2e-6, rtol=0)
        for c in report['cases']:
            i = c['case']
            force = np.zeros_like(z['force'][:, i])
            if not descriptor['quiet']:
                first = int(np.ceil(descriptor['onset']/contract['dt']))
                force[first:first+6] = c['force_n']
                force[~z['alive'][:, i]] = 0
            assert np.array_equal(force, z['force'][:, i])
            assert c['survived'] == bool(alive[:, i].all())
            assert c['settled'] == bool(alive[:, i].all() and z['settled'][-int(np.ceil(2/contract['dt'])):, i].all())
            events = measure_steps(z['time'], xy[:, i], clearance[:, i], z['loads'][:, i],
                                   alive[:, i], z['settled'][:, i], hit_time=c['hit_seconds'])
            assert events == c['steps']
            assert c['recovered_with_step'] == (c['settled'] and any(e['settled_window_start'] is not None for e in events))
            event_count += len(events)
            recovered_steps += c['recovered_with_step']
        for key in ('survived', 'settled', 'whole_stand', 'recovered_with_step'):
            assert report[key] == sum(c[key] for c in report['cases'])
        if descriptor['baseline_folder']:
            base_folder = Path(descriptor['baseline_folder'])
            base, br = trace(base_folder), read(base_folder/'report.json')
            start = descriptor['issue_frame']
            assert all(np.array_equal(z[k][:start], base[k][:start])
                       for k in ('qpos', 'qvel', 'issued', 'active_target', 'force', 'alive'))
            assert all(np.array_equal(z[k][:start+2], base[k][:start+2]) for k in ('qpos', 'qvel'))
            gains = [i for i in range(16) if report['cases'][i]['settled'] and not br['cases'][i]['settled']]
            losses = [i for i in range(16) if br['cases'][i]['settled'] and not report['cases'][i]['settled']]
            assert gains == descriptor['paired']['gained_cases'] and losses == descriptor['paired']['lost_cases']
            if descriptor['replay_folder']:
                begin, end = descriptor['replay_window']
                assert np.array_equal(z['source_action'][begin:end][:, :, legs], base['source_action'][begin:end][:, :, legs])
                if not descriptor['profile']:
                    assert all(np.array_equal(z[k], base[k]) for k in base)
    candidates = read(out/'candidates.json')
    assert len(candidates) == 24
    for c in candidates:
        gains = losses = heldout_gain = heldout_loss = 0
        for timing in (0, 1):
            name = f"probe-{c['profile']}-a{c['amplitude_index']}-t{timing}"
            d = all_descriptors[name]['paired']
            gains += sum(i % 8 < 6 for i in d['gained_cases'])
            losses += sum(i % 8 < 6 for i in d['lost_cases'])
            heldout_gain += sum(i % 8 >= 6 for i in d['gained_cases'])
            heldout_loss += sum(i % 8 >= 6 for i in d['lost_cases'])
        assert (gains, losses, heldout_gain, heldout_loss) == (c['train_gain'], c['train_loss'], c['heldout_gain'], c['heldout_loss'])
    selected = max(candidates, key=lambda c: (c['train_gain']-c['train_loss'], c['train_gain'],
        c['training_survival_seconds'], -abs(c['amplitude'])))
    assert selected == read(out/'selected.json') == completed['selected']
    assert len(all_reports) == 63
    result = dict(passed=True, traces=63, evaluated_cases=sum(len(r['cases']) for r in all_reports.values()),
        input_hashes=len(inputs), output_hashes=len(completed['outputs']), measured_steps=event_count,
        recovered_with_step=recovered_steps, exact_prefix_and_delay=True, zero_replay_exact=True,
        selected_without_heldout_outcomes=True, optimizer_updates=0)
    save(out/'audit.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    verify(parser.parse_args().out)


if __name__ == '__main__':
    main()
