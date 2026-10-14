"""Audit saved PPO outcomes without rerunning learning or changing admission gates."""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_handoff import PRIOR
from .train_arm_hold_ppo_reward import INITIAL, OUT, TRAIN_IDS


def main():
    from .rig import Rig
    root = OUT
    out = Path('logs/mimickit-training/arm-hold-ppo-reward-review-20261008-01')
    out.mkdir(exist_ok=False)
    protocol = json.loads((root / 'protocol.json').read_text())
    report = json.loads((root / 'report.json').read_text())
    assert all(digest(Path(p)) == h for p, h in protocol['inputs'].items())
    assert all(digest(Path(p)) == h for p, h in report['output_hashes'].items())
    assert set(i % 8 for i in TRAIN_IDS.tolist()) == set(range(6))
    initial = torch.load(INITIAL, weights_only=True)
    rig = Rig.load(FOUNDATION / 'dummy.xml', policy_exclude=())
    model = rig.model
    dt = model.opt.timestep * rig.decimation
    qi = model.jnt_qposadr[model.actuator_trnid[rig.actuators, 0]]
    poses = np.load(PRIOR / 'goals.npz')['poses']
    data = mujoco.MjData(model)
    cases = []
    folders = [('baseline', INITIAL.parent.parent / 'nominal')]
    names = [f'{label}-{seed}' for seed in (501028, 511028) for label in ('control', 'broad')]
    folders += [(name, root / f'{name}-nominal') for name in names]
    folders += [(p.name, p) for p in root.glob('*-varied')]
    for label, folder in folders:
        r = json.loads((folder / 'report.json').read_text())
        z = np.load(folder / 'trace.npz')
        hold = (z['time'] > r['hold_end'] - 2) & (z['time'] <= r['hold_end'])
        assert np.allclose(z['time'], (np.arange(len(z['time'])) + 1) * dt, atol=1e-12, rtol=0)
        for i, c in enumerate(r['cases']):
            data.qpos[:] = poses[i, 0]
            mujoco.mj_forward(model, data)
            xy = data.xpos[[model.body('Foot_L').id, model.body('Foot_R').id], :2].copy()
            data.qpos[:] = poses[i, 1]
            mujoco.mj_forward(model, data)
            com = data.subtree_com[1].copy()
            error = np.sqrt(np.mean((z['qpos'][:, i, qi] - poses[i, 1, qi]) ** 2, axis=-1))
            assert np.allclose(error, z['tracking_rmse'][:, i], atol=1e-8, rtol=0)
            stable = bool(hold.sum() >= int(2 / dt) - 1 and z['alive'][hold, i].all() and z['support'][hold, i].all())
            com_error = np.linalg.norm(z['com'][hold, i, :2] - com[:2], axis=-1)
            foot_error = np.linalg.norm(z['xy'][hold, i] - xy, axis=-1)
            tracked = bool(stable and (error[hold] <= .05).all() and (com_error <= .02).all() and (foot_error <= .02).all())
            survived = len(z['time']) == int(np.ceil(r['seconds'] / dt)) and z['alive'][-1, i] and z['done'][-1, i] != 1
            returned = bool(survived and z['stand'][-int(np.ceil(2 / dt)):, i].all())
            assert (stable, tracked, returned, tracked and returned) == (c['stable_hold'], c['tracked_hold'], c['returned_stand'], c['complete'])
            cases.append(dict(condition=label, pose=i, stable=stable, returned=returned, complete=tracked and returned,
                peak_motor_rmse_rad=float(error[hold].max()), mean_motor_rmse_rad=float(error[hold].mean()),
                peak_com_error_m=float(com_error.max()), peak_foot_error_m=float(foot_error.max())))
    quiet_baseline = np.load('logs/mimickit-training/learned-support-review-20261008-01/zero-addition-quiet30/trace.npz')
    training = []
    for seed in names:
        folder = root / str(seed)
        start = torch.load(folder / 'initial.pt', weights_only=True)
        final = torch.load(folder / 'final.pt', weights_only=True)
        assert start.keys() == initial.keys() == final.keys()
        assert all(torch.equal(v, start[k]) for k, v in initial.items())
        assert all(torch.isfinite(v).all() for v in final.values())
        for key in ('mean', 'scale', 'output_scale'):
            assert torch.equal(initial[key], final[key])
        assert any(not torch.equal(v, final[k]) for k, v in initial.items())
        quiet = np.load(root / f'quiet-{seed}/trace.npz')
        assert all(np.array_equal(quiet[k], quiet_baseline[k]) for k in ('qpos', 'issued'))
        history = json.loads((folder / 'history.json').read_text())
        run = json.loads((folder / 'report.json').read_text())
        assert len(history) == 32 and run['transitions'] == 32 * 64 * 12
        assert all(h['transitions'] == (i + 1) * 768 for i, h in enumerate(history))
        assert sum(h['actor_steps'] for h in history) == run['actor_gradient_steps']
        assert all(np.isfinite(h['critic_loss']) and np.isfinite(h['max_checked_kl']) for h in history)
        post_kl = [v for h in history for v in h['post_update_full_batch_kl']]
        assert len(post_kl) == run['actor_gradient_steps']
        assert all(np.isfinite(v) and v >= 0 for v in post_kl)
        phases = {}
        for h in history:
            assert sum(v['count'] for v in h['reward_parts'].values()) == 768
            total_reward = sum(sum(v['sums'].values()) for v in h['reward_parts'].values()) / 768
            assert abs(total_reward - h['mean_reward']) < 1e-6
            assert len(h['post_update_full_batch_kl']) == h['actor_steps']
            for phase, values in h['reward_parts'].items():
                entry = phases.setdefault(phase, dict(count=0, sums={k: 0. for k in values['sums']}, tracking_below_one_percent=0, com_below_one_percent=0))
                entry['count'] += values['count']
                for key, value in values['sums'].items():
                    entry['sums'][key] += value
                for key in ('tracking_below_one_percent', 'com_below_one_percent'):
                    entry[key] += values[key]
        for entry in phases.values():
            entry['means'] = {k: v / entry['count'] for k, v in entry['sums'].items()}
            for key in ('tracking_below_one_percent', 'com_below_one_percent'):
                entry[key + '_fraction'] = entry[key] / entry['count']
        control_identical = None
        if seed.startswith('control-'):
            old = Path('logs/mimickit-training/arm-hold-ppo-rate-20261008-01')
            old_weights = torch.load(old / f"lower-{run['seed']}" / 'final.pt', weights_only=True)
            control_identical = all(torch.equal(v, old_weights[k]) for k, v in final.items())
            old_trace = np.load(old / f"lower-{run['seed']}-nominal/trace.npz")
            new_trace = np.load(root / f'{seed}-nominal/trace.npz')
            assert control_identical
            assert all(np.array_equal(new_trace[k], old_trace[k]) for k in old_trace.files)
        training.append(dict(**run, condition=seed, early_stopped_cycles=sum(h['actor_early_stop'] for h in history),
            peak_checked_kl=max(h['max_checked_kl'] for h in history), full_batch_kl_max=max(post_kl),
            full_batch_kl_mean=float(np.mean(post_kl)), full_batch_kl_p95=float(np.percentile(post_kl,95)),
            full_batch_updates_above_threshold=sum(v > .01 for v in post_kl),
            control_weights_and_trace_identical_to_previous_lower_rate=control_identical, phases=phases))
    result = dict(inputs_and_outputs_unchanged=True, initial_weights_identical=True,
        quiet_traces_exact=True, holdouts_excluded_from_both_losses=True,
        cases=cases, training=training,
        saved_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))
    save(out / 'report.json', result)
    print(json.dumps(dict(checks='passed', training=training), indent=2))


if __name__ == '__main__':
    main()
