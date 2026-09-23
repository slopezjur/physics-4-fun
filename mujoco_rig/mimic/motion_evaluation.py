"""Native tracking gates shared by motion probes and checkpoint selection."""
import mujoco
import numpy as np
import torch
from .motion_protocol import tracking_checks


def tracking_metrics(task, metrics, traces):
    rig, model = task.rig, task.rig.model
    feet = [model.body(n).id for n in ("Foot_L", "Foot_R")]
    joints = model.jnt_qposadr[model.actuator_trnid[rig.actuators, 0]]
    actual_data, desired_data = mujoco.MjData(model), mujoco.MjData(model)
    rows = []
    for episode in range(metrics["episodes"]):
        count = round(metrics["survival_seconds"][episode] / task.dt)
        phase = episode / max(1, metrics["episodes"] - 1) * (task.reference.duration - task.episode_seconds)
        desired, _ = task.reference.sample(torch.arange(count) * task.dt + phase)
        actual = traces["qpos"][:count, episode]
        errors, positions = [], []
        for q, target in zip(actual, desired.numpy()):
            actual_data.qpos[:], desired_data.qpos[:] = q, target
            mujoco.mj_kinematics(model, actual_data)
            mujoco.mj_kinematics(model, desired_data)
            errors.append(np.linalg.norm(actual_data.xpos[feet] - desired_data.xpos[feet], axis=-1))
            positions.append(actual_data.xpos[feet].copy())
        rows.append(dict(phase=phase, survival_seconds=metrics["survival_seconds"][episode],
                         mean_foot_error_m=float(np.mean(errors)),
                         joint_rmse_rad=float(np.sqrt(np.mean((actual[:, joints] - desired.numpy()[:, joints])**2))),
                         foot_lift_range_m=np.ptp(np.array(positions)[..., 2], axis=0).tolist()))
    checks = tracking_checks(metrics, rows)
    return dict(passed=all(checks.values()), checks=checks, cases=rows)
