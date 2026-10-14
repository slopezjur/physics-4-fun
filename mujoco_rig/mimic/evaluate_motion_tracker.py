"""Score one native episode per phase, including a held terminal stance."""
import numpy as np
import torch

GATES = dict(seconds=30., root_error_m=.03, foot_error_m=.03, joint_rmse_rad=.15,
    displacement_rmse_m=.015, relative_displacement_error=.5, displacement_noise_floor_m=.002,
    final_stable_seconds=2.)


@torch.no_grad()
def evaluate(task, actor):
    from envs.base_env import EnvMode, DoneFlags
    task.set_mode(EnvMode.TEST)
    task.episode_seconds = GATES['seconds']
    task.reset()
    count = task.get_num_envs()
    active = torch.ones(count, dtype=torch.bool)
    survived = torch.zeros(count, dtype=torch.bool)
    lengths = torch.zeros(count, dtype=torch.long)
    stable = torch.zeros(count, dtype=torch.long)
    roots, feet, joints, displacement, target_displacement = [torch.zeros(count) for _ in range(5)]
    motor_q = task.engine.policy_control.q_indices
    actual_start = task.engine.native_qpos()[:, :2].clone()
    target_start = task.sample_reference(task.offset)[0][:, :2].clone()
    trace = {key: [] for key in ('qpos', 'qvel', 'target_qpos', 'action', 'active', 'foot_loads')}
    for _ in range(int(np.ceil(GATES['seconds'] / task.dt))):
        mask = active.clone()
        action = actor(task.observations())
        _, _, done, _ = task.step(action)
        actual = task.engine.native_qpos()
        target, _ = task.sample_reference(task.offset + task.steps * task.dt)
        desired_pos, _ = task.mapper.from_native_qpos(target)
        actual_feet = task.engine.get_body_pos(0)[:, task.feet]
        error = torch.linalg.vector_norm(actual[:, :3] - target[:, :3], dim=-1)
        foot_error = torch.linalg.vector_norm(actual_feet - desired_pos[:, task.feet], dim=-1).mean(-1)
        joint_error = (actual[:, motor_q] - target[:, motor_q]).square().mean(-1)
        moved = actual[:, :2] - actual_start
        requested = target[:, :2] - target_start
        roots += error * mask
        feet += foot_error * mask
        joints += joint_error * mask
        displacement += (moved - requested).square().sum(-1) * mask
        target_displacement += requested.square().sum(-1) * mask
        raw_loads = task.engine.get_ground_contact_forces(0)[:, task.allowed_ground, 2].abs()
        loads = torch.stack((raw_loads[:, 0] + raw_loads[:, 2], raw_loads[:, 1] + raw_loads[:, 3]), -1)
        settled = (actual[:, 2] >= .75) & (1 - 2 * (actual[:, 4].square() + actual[:, 5].square()) >= np.cos(np.deg2rad(15.)))
        settled &= torch.linalg.vector_norm(task.engine.get_root_vel(0)[:, :2], dim=-1) <= .2
        settled &= torch.linalg.vector_norm(task.engine.get_root_ang_vel(0), dim=-1) <= 1.
        settled &= (loads >= 30.).all(-1) & (foot_error <= .03) & (done != DoneFlags.FAIL.value)
        stable = torch.where(mask, torch.where(settled, stable + 1, 0), stable)
        lengths += mask
        survived |= mask & (done == DoneFlags.TIME.value)
        active &= done == DoneFlags.NULL.value
        for key, value in zip(trace, (actual, task.engine._array('qvel'), target, action, mask, loads)):
            trace[key].append(value.cpu().numpy().copy())
        if not active.any():
            break
        task.reset(torch.nonzero(done != DoneFlags.NULL.value).flatten())
    cases = []
    for i in range(count):
        length = max(1, int(lengths[i]))
        reference_rms = float((target_displacement[i] / length).sqrt())
        limit = min(.015, max(.002, .5 * reference_rms))
        row = dict(case=i, survived=bool(survived[i]), survival_seconds=float(lengths[i] * task.dt),
            mean_root_error_m=float(roots[i] / length), mean_foot_error_m=float(feet[i] / length),
            joint_rmse_rad=float((joints[i] / length).sqrt()),
            displacement_rmse_m=float((displacement[i] / length).sqrt()),
            reference_displacement_rms_m=reference_rms, displacement_limit_m=limit,
            final_stable_seconds=float(stable[i] * task.dt))
        row['passed'] = (row['survived'] and row['survival_seconds'] >= 30.
            and row['mean_root_error_m'] < .03 and row['mean_foot_error_m'] < .03
            and row['joint_rmse_rad'] < .15 and row['displacement_rmse_m'] < limit
            and row['final_stable_seconds'] >= 2.)
        cases.append(row)
    return dict(schema='native_motion_tracker_foundation_v1', gates=GATES, cases=cases,
        passed=all(c['passed'] for c in cases), successes=sum(c['survived'] for c in cases)), {
            key: np.asarray(values) for key, values in trace.items()}
