"""Independent saved-trace audit of the frozen weighted actor's quiet replay."""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from util.torch_util import quat_to_tan_norm
from .corrective_step_audit import foot_geometry, measure_steps
from .rig import Rig
from .run_learned_support import PARENT
from .run_support_skill_study import FOUNDATION, digest, save
from .stand_experiment_evidence import load_actor

OUT = Path('logs/mimickit-training/recovery-weighted-quiet-replay-20261009-01')
CANDIDATE = Path('logs/mimickit-training/recovery-age-weighted-learning-20261009-01')
ZERO = Path('logs/mimickit-training/learned-ankle-20261009-01')


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


@torch.no_grad()
def source_targets(trace, anchor, reference, control):
    """Rebuild source observations using the previously audited stationary layout."""
    future = torch.cat((reference[:, :3], quat_to_tan_norm(reference[:, [4, 5, 6, 3]]),
                        reference[:, 7:]), -1).repeat(1, 3)
    lower = torch.tensor(control['target_lower_radians'], dtype=torch.float32)
    upper = torch.tensor(control['target_upper_radians'], dtype=torch.float32)
    q_indices = control['target_qpos_indices']
    max_error, checked = 0., 0
    for frame in range(2, len(trace['time'])):
        q, v = (torch.from_numpy(trace[key][frame-1]) for key in ('qpos', 'qvel'))
        angular = 2*torch.from_numpy(trace['balance_features'][frame-1, :, 4:7])
        pending = [torch.cat((torch.from_numpy(trace['issued'][i]), torch.zeros(16, 33),
                             torch.ones(16, 1)), -1) for i in (frame-2, frame-1)]
        raw = torch.cat((q[:, :3], quat_to_tan_norm(q[:, [4, 5, 6, 3]]), v[:, :3], angular,
                         q[:, 7:], .1*v[:, 6:], .001*torch.from_numpy(trace['loads'][frame-1]),
                         *pending, future, torch.zeros(16, 1)), -1)
        assert raw.shape == (16, 374)
        prediction = (reference[:, q_indices]+control['residual_radians']*anchor(raw)).clamp(lower, upper).numpy()
        active = trace['alive'][frame]
        if active.any():
            error = float(np.abs(prediction[active]-trace['nominal'][frame, active]).max())
            max_error = max(max_error, error)
            checked += int(active.sum())
    assert max_error < 2e-5, ('live source target reconstruction', max_error)
    return dict(active_targets_checked=checked, max_absolute_error_rad=max_error,
                first_checked_frame=2, physics_integrations=0)


def physical_audit(z, report, contract, model):
    dt, ankles, control = contract['dt'], contract['ankles'], contract['control']
    lower, upper = (np.asarray(control[f'target_{side}_radians']) for side in ('lower', 'upper'))
    frames = int(np.ceil(6/dt))
    assert len(z['time']) == frames and report['seconds'] == 6.
    assert report['dt'] == dt and report['sampled'] is False and report['group'] == 0
    assert all(np.isfinite(value).all() for value in z.values()) and not z['warning'].any()
    assert np.allclose(z['time'], (np.arange(frames)+1)*dt, atol=1e-10, rtol=0)
    assert not z['previous_latent'][0].any() and not z['previous'][0].any()
    assert np.array_equal(z['history_before'][0, :, 0], z['history_before'][0, :, 2])
    history = np.concatenate((z['balance_features'][:-1, :, None], z['history_before'][:-1, :, :2]), axis=2)
    history[~z['alive'][:-1]] = z['history_before'][:-1][~z['alive'][:-1]]
    assert np.array_equal(history, z['history_before'][1:])
    memory = np.where(z['alive'][:-1, :, None], z['latent'][:-1], z['previous_latent'][:-1])
    assert np.array_equal(memory, z['previous_latent'][1:])
    assert np.array_equal(z['previous'][1:], z['addition'][:-1])
    expected = z['previous'][:, :, ankles[:1]]+np.clip(
        .1*np.tanh(z['latent'])-z['previous'][:, :, ankles[:1]], -.02, .02)
    expected[~z['alive']] = z['previous'][:, :, ankles[:1]][~z['alive']]
    assert np.allclose(z['addition'][:, :, ankles[0]], expected[:, :, 0], atol=1e-7, rtol=0)
    assert np.array_equal(z['addition'][:, :, ankles[0]], z['addition'][:, :, ankles[1]])
    assert not z['addition'][:, :, [i for i in range(33) if i not in ankles]].any()
    assert np.abs(z['addition']).max() <= .1+1e-7
    assert np.abs(z['addition']-z['previous']).max() <= .02+1e-7
    assert np.allclose(z['issued'], np.clip(z['nominal']+z['addition'], lower, upper), atol=1e-7, rtol=0)
    assert not z['active_target'][:2, :, -1].any()
    assert np.array_equal(z['active_target'][2:, :, :33], z['issued'][:-2])
    assert not z['active_target'][:, :, 33:66].any() and (z['active_target'][2:, :, -1] == 1).all()
    alive = z['alive'] & ~z['failed']
    assert z['alive'][0].all() and np.array_equal(z['alive'][1:], alive[:-1])
    xy, clearance = foot_geometry(model, z['qpos'])
    assert np.allclose(xy, z['xy'], atol=2e-6, rtol=0)
    assert np.allclose(clearance, z['clearance'], atol=2e-6, rtol=0)
    count = 0
    assert len(report['cases']) == 16
    for i, case in enumerate(report['cases']):
        assert case['case'] == i and case['pose'] == i % 8 and case['heldout'] == (i % 8 >= 6)
        assert case['survived'] == bool(alive[:, i].all())
        assert case['settled'] == bool(alive[:, i].all() and z['settled'][-int(np.ceil(2/dt)):, i].all())
        assert case['whole_stand'] == bool(alive[:, i].all() and z['stand'][:, i].all())
        expected_force = np.zeros_like(z['force'][:, i])
        if report['quiet']:
            assert case['hit_seconds'] is None and case['force_n'] == [0., 0., 0.]
        else:
            hit = int(round(case['hit_seconds']/dt))
            assert abs(case['hit_seconds']-hit*dt) < 1e-10
            assert case['force_n'] == [150. if i < 8 else -150., 0., 0.]
            expected_force[hit:hit+6] = case['force_n']
            expected_force[~z['alive'][:, i]] = 0
        assert np.array_equal(expected_force, z['force'][:, i])
        assert np.allclose(case['impulse_ns'], z['force'][:, i].sum(0)*dt, atol=1e-6, rtol=1e-7)
        events = measure_steps(z['time'], xy[:, i], clearance[:, i], z['loads'][:, i], alive[:, i],
                               z['settled'][:, i], hit_time=case['hit_seconds'])
        assert events == case['steps']
        assert case['recovered_with_step'] == (case['settled'] and any(e['settled_window_start'] is not None for e in events))
        post = alive[:, i] & (z['time'] >= (case['hit_seconds'] or 0))
        unilateral = (z['loads'][:, i].min(-1) < 5) & (z['loads'][:, i].max(-1) >= 30)
        lifted = ((clearance[:, i] >= .015) & (z['loads'][:, i] < 5)
                  & (z['loads'][:, i, ::-1] >= 30)).any(-1)
        assert case['unilateral_frames'] == int((post & unilateral).sum())
        assert case['lifted_frames'] == int((post & lifted).sum())
        count += len(events)
    for key in ('survived', 'settled', 'whole_stand', 'recovered_with_step'):
        assert report[key] == sum(c[key] for c in report['cases'])
    assert report['measured_steps'] == count
    return count


def main():
    torch.set_num_threads(1)
    inputs, done, contract = (read(OUT/name) for name in ('inputs.json', 'completed.json', 'contract.json'))
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    assert all(digest(OUT/path) == sha for path, sha in done['outputs'].items())
    assert read(CANDIDATE/'audit.json')['passed'] and read(ZERO/'audit.json')['passed']
    assert contract['seed'] == 931009 and Path(contract['candidate']) == CANDIDATE and Path(contract['zero']) == ZERO
    assert contract['issue_frames'] == [61, 103] and contract['training_transitions'] == 0
    assert (done['traces'], done['cases'], done['push_cases'], done['quiet_control_cases']) == (3, 48, 32, 16)
    assert all(done[key] is True for key in ('frozen_source_unchanged', 'frozen_actor_unchanged', 'inputs_unchanged'))
    assert not list(OUT.rglob('*.pt'))
    assert len(list(OUT.glob('*/trace.npz'))) == 3
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    anchor = load_actor(rig, FOUNDATION/'stationary_reference.npz', read(FOUNDATION/'protocol.json'), PARENT)
    reference = torch.from_numpy(arrays(FOUNDATION/'stationary_reference.npz')['qpos'].astype(np.float32))[torch.arange(16) % 8]
    quiet = arrays(CANDIDATE/'931009-quiet'/'trace.npz')
    comparisons, rows, target_checks, measured_steps = [], [], {}, 0
    summary = {row['name']: row for row in read(OUT/'summary.json')}
    for name in ('quiet-parity-t0', 'replay-t0', 'replay-t1'):
        folder = OUT/name
        z, report, desc = arrays(folder/'trace.npz'), read(folder/'report.json'), read(folder/'descriptor.json')
        measured_steps += physical_audit(z, report, contract, rig.model)
        target_checks[name] = source_targets(z, anchor, reference, contract['control'])
        issue, frames = desc['issue_frame'], len(z['time'])
        assert desc['calls'] == frames and issue == contract['issue_frames'][desc['timing']]
        assert Path(desc['reference_quiet']) == CANDIDATE/'931009-quiet'
        assert summary[name]['replay_settled'] == report['settled']
        assert summary[name]['issue_frame'] == issue and summary[name]['quiet'] == report['quiet']
        if name == 'quiet-parity-t0':
            assert desc['mode'] == 'quiet-parity' and report['quiet'] is True
            assert Path(desc['reference_live']) == CANDIDATE/'931009-quiet'
            assert all(np.array_equal(z[key], quiet[key][:frames]) for key in z)
            assert report['survived'] == report['settled'] == report['whole_stand'] == 16
            continue
        timing = desc['timing']
        assert desc['mode'] == 'replay' and report['quiet'] is False
        roots = dict(live=CANDIDATE/f'931009-t{timing}-g0',
                     matched=CANDIDATE/f'matched-931009-t{timing}-g0',
                     zero=ZERO/f'baseline-t{timing}-g0')
        for key, path in roots.items():
            assert Path(desc[f'reference_{key}']) == path
        live = arrays(roots['live']/'trace.npz')
        old_reports = {key: read(path/'report.json') for key, path in roots.items()}
        assert summary[name]['live_settled'] == old_reports['live']['settled']
        assert all(np.array_equal(z[key][:issue], live[key][:issue]) for key in z)
        assert all(np.array_equal(z[key][:issue+2], live[key][:issue+2]) for key in ('qpos', 'qvel', 'active_target'))
        assert np.array_equal(z['latent'][:issue], live['latent'][:issue])
        assert np.array_equal(z['latent'][issue:], quiet['latent'][issue:frames])
        assert all(np.array_equal(live[key][:issue-1], quiet[key][:issue-1])
                   for key in ('qpos', 'qvel', 'addition', 'latent', 'history_before'))
        active = z['alive'][issue:]
        assert np.allclose(z['addition'][issue:][active], quiet['addition'][issue:frames][active], atol=1e-7, rtol=0)
        for case in report['cases']:
            i = case['case']
            assert int(round(case['hit_seconds']/contract['dt'])) == issue-1
            row = dict(timing=timing, case=i, pose=case['pose'], heldout=case['heldout'],
                       sign='+X' if i < 8 else '-X', replay=case['settled'], replay_survived=case['survived'])
            for key, previous in old_reports.items():
                old = previous['cases'][i]
                assert old['case'] == i and old['heldout'] == case['heldout']
                row[key], row[key+'_survived'] = old['settled'], old['survived']
                row[key+'_comparison'] = ('gain' if case['settled'] and not old['settled']
                                          else 'loss' if old['settled'] and not case['settled'] else 'unchanged')
            rows.append(row)
        common = z['alive'][issue:] & live['alive'][issue:]
        difference = z['addition'][issue:, :, contract['ankles'][0]]-live['addition'][issue:, :, contract['ankles'][0]]
        comparisons.append(dict(timing=timing, issue_frame=issue, exact_prefix_frames=issue,
                                exact_physical_prefix_frames=issue+2,
                                addition_difference_rms_rad=float(np.sqrt(np.mean(difference[common]**2))),
                                addition_difference_max_rad=float(np.abs(difference[common]).max()),
                                nominal_difference_from_quiet_max_rad=float(np.abs(z['nominal'][issue:]-quiet['nominal'][issue:frames]).max())))
    assert len(rows) == 32
    outcomes = {key: dict(settled=sum(row[key] for row in rows),
                         survived=sum(row[key+'_survived'] for row in rows),
                         heldout_settled=sum(row[key] and row['heldout'] for row in rows))
                for key in ('replay', 'live', 'matched', 'zero')}
    pairs = dict(cases=rows, totals=outcomes,
                 timing1_cases3_and4=[row for row in rows if row['timing'] == 1 and row['case'] in (3, 4)])
    audit = dict(passed=True, traces=3, cases=48, push_cases=32, quiet_control_cases=16,
                 training_transitions=0, new_checkpoints=0, measured_steps=measured_steps,
                 input_hashes=len(inputs), output_hashes=len(done['outputs']),
                 exact_quiet_control=True, exact_live_prefixes=True,
                 two_interval_physical_delay=True, quiet_latent_suffix_verified=True,
                 geometry_and_outcomes_verified=True, source_targets=target_checks,
                 comparisons=comparisons, outcomes=outcomes,
                 scope='Only the learned ankle correction is replaced after the first force interval. '
                 'The frozen Stand source retains state feedback. Matched-start and zero outcomes '
                 'are descriptive comparators with different prefixes, not the intervention control.')
    save(OUT/'paired-outcomes.json', pairs)
    save(OUT/'audit.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
