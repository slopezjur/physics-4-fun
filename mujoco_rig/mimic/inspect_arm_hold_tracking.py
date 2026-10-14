"""Locate joint tracking errors in existing frozen traces; no simulation or fitting."""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .rig import Rig
from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_handoff import PRIOR
from .train_arm_hold_ppo_reward import OUT
from .train_arm_hold_ppo import INITIAL
from .ablate_arm_hold_queue import QueueBlindLearner


def main():
    rig = Rig.load(FOUNDATION / 'dummy.xml', policy_exclude=())
    model = rig.model
    joints = model.actuator_trnid[rig.actuators, 0]
    qi = model.jnt_qposadr[joints]
    lower, upper = model.jnt_range[joints].T
    poses = np.load(PRIOR / 'goals.npz')['poses']
    folders = [Path('logs/mimickit-training/arm-hold-queue-ablation-20261008-01/nominal'),
        Path('logs/mimickit-training/arm-hold-learning-20261008-01/teacher-nominal')]
    folders += sorted(OUT.glob('*-nominal'))
    results = []
    for folder in folders:
        trace = np.load(folder / 'trace.npz')
        report = json.loads((folder / 'report.json').read_text())
        hold = (trace['time'] > 3.5) & (trace['time'] <= 5.5)
        stable = np.array([c['stable_hold'] for c in report['cases']])
        if not stable.any():
            continue
        goal = poses[:8, 1, qi][stable]
        actual = trace['qpos'][hold][:, stable][:, :, qi]
        issued = trace['issued'][hold][:, stable]
        error = actual - goal
        rms = np.sqrt(np.mean(error**2, axis=(0, 1)))
        motor_rows = []
        for index in np.argsort(rms)[::-1]:
            motor_rows.append(dict(joint=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joints[index])),
                qpos_index=int(qi[index]), rms_actual_error_rad=float(rms[index]),
                mean_actual_minus_goal_rad=float(error[:, :, index].mean()),
                mean_issued_minus_goal_rad=float((issued - goal)[:, :, index].mean()),
                rms_issued_minus_goal_rad=float(np.sqrt(np.mean((issued - goal)[:, :, index]**2))),
                issued_near_joint_limit_fraction=float(((abs(issued[:, :, index] - lower[index]) < 1e-4) |
                    (abs(issued[:, :, index] - upper[index]) < 1e-4)).mean())))
        results.append(dict(folder=str(folder), stable_poses=np.flatnonzero(stable).tolist(),
            trace_sha256=digest(folder / 'trace.npz'), motors= motor_rows))
    out = Path('logs/mimickit-training/arm-hold-ppo-reward-review-20261008-01')
    # Compare outputs on identical archived teacher inputs, not different live states.
    torch.set_num_threads(2)
    samples_path = folders[1] / 'samples.npz'
    samples = np.load(samples_path)
    times = (samples['frame'] + 1) * model.opt.timestep * rig.decimation
    hold = (times > 3.5) & (times <= 5.5)
    x = torch.tensor(samples['x'][hold])
    initial = torch.load(INITIAL, weights_only=True)
    net = QueueBlindLearner(initial['mean'], initial['scale'], initial['output_scale'])
    net.load_state_dict(initial)
    net.eval()
    policy_changes = []
    with torch.no_grad():
        before = net(x)
        for folder in sorted(OUT.glob('*-nominal')):
            name = folder.name.removesuffix('-nominal')
            net.load_state_dict(torch.load(OUT / name / 'final.pt', weights_only=True))
            delta = (net(x) - before).numpy()
            index = int(np.flatnonzero(qi == model.jnt_qposadr[model.joint('UpperArm_L_ry').id])[0])
            policy_changes.append(dict(condition=name, examples=len(x), rms_all_motor_output_change_rad=float(np.sqrt(np.mean(delta**2))),
                maximum_abs_output_change_rad=float(abs(delta).max()),
                left_upper_arm_ry_rms_output_change_rad=float(np.sqrt(np.mean(delta[:, index]**2)))))
    save(out / 'joint-tracking.json', dict(
        interpretation='Descriptive per-condition supported-hold cohorts; commands may legitimately compensate dynamics. Not a causal responsiveness test.',
        results=results, fixed_teacher_input_policy_changes=policy_changes, teacher_samples_sha256=digest(samples_path)))
    for result in results:
        print(result['folder'], result['stable_poses'], result['motors'][:2])
    print('FIXED_INPUT_POLICY_CHANGES', policy_changes)


if __name__ == '__main__':
    main()
