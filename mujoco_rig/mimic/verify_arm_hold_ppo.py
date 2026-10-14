"""Audit saved PPO outcomes without rerunning learning or changing admission gates."""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_handoff import PRIOR
from .train_arm_hold_ppo import INITIAL, OUT, TRAIN_IDS


def main():
    from .rig import Rig
    root = OUT
    out = Path('logs/mimickit-training/arm-hold-ppo-review-20261008-01')
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
    folders += [(str(seed), root / f'{seed}-nominal') for seed in (501028, 511028)]
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
    for seed in (501028, 511028):
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
        training.append(dict(**run, early_stopped_cycles=sum(h['actor_early_stop'] for h in history),
            peak_checked_kl=max(h['max_checked_kl'] for h in history)))
    # The frozen collector uses the look-ahead goal for its training success test;
    # the evaluator uses the fixed hold goal. Record the boundary discrepancy.
    from .predictive_target_support import blend_pose
    frames = np.arange(int(np.ceil(9 / dt)))
    times = (frames + 1) * dt
    mask = (times > 3.5) & (times <= 5.5)
    boundary = mask & (frames * dt + 2 * dt >= 5.5)
    max_boundary_error = 0.
    for frame in frames[boundary]:
        for pose in poses:
            goal = blend_pose(pose[1], pose[0], frame * dt + 2 * dt - 5.5, 1.5)
            max_boundary_error = max(max_boundary_error, float(abs(goal[qi] - pose[1, qi]).max()))
    reward_gate_caveat = dict(lookahead_hold_boundary_frames=int(boundary.sum()),
        maximum_goal_difference_rad=max_boundary_error,
        bonus_awarded_in_either_training_run=any(r['completed_training_episodes'] for r in training),
        frozen_evaluation_uses_correct_fixed_hold_target=True,
        correction_required_before_success_bonus_is_used=True)
    result = dict(inputs_and_outputs_unchanged=True, initial_weights_identical=True,
        quiet_traces_exact=True, holdouts_excluded_from_both_losses=True,
        cases=cases, training=training, reward_gate_caveat=reward_gate_caveat,
        saved_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))
    save(out / 'report.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
