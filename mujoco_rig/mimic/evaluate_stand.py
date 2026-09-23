"""Fixed-phase deterministic evaluation; success means surviving the full episode."""
import numpy as np
import torch
from .ball_protocol import VALIDATION_SCHEMA
from .perturb import RECOVERY


@torch.no_grad()
def evaluate(task, actor, episodes=None, *, capture_actions=False):
    from envs.base_env import EnvMode, DoneFlags
    ball = hasattr(task, "validation_cases")
    if episodes is None:
        episodes = len(task.validation_cases) if ball else 8
    if ball and episodes != len(task.validation_cases):
        raise ValueError("Ball selection must score the complete validation suite")
    if not 1 <= episodes <= task.get_num_envs():
        raise ValueError("Evaluation requires at least one world per scored case")
    task.set_mode(EnvMode.TEST)
    task.reset()
    # Use identical phases even when the training engine contains more worlds.
    if not ball:
        task.offset[:] = (task.ids % episodes) / max(1, episodes - 1) * (
            task.reference.duration - task.episode_seconds)
        q, v = task.reference.sample(task.offset)
        task.engine.reset_envs(task.ids, q, v)
    active = torch.ones(task.get_num_envs(), dtype=torch.bool, device=task.device)
    returns = torch.zeros_like(task.offset)
    lengths = torch.zeros_like(task.steps)
    successes = torch.zeros_like(active)
    tracking = torch.zeros_like(task.offset)
    trajectories, observations = [], []
    actions, poses_after, ball_positions = [], [], []
    hits = torch.zeros_like(active)
    stable = torch.zeros_like(lengths)
    first_hit = torch.full_like(lengths, -1)
    recovery_time = torch.full_like(task.offset, -1.)
    settle_steps = int(np.ceil(RECOVERY["settle_seconds"] / task.dt))
    for _ in range(int(np.ceil(task.episode_seconds / task.dt))):
        obs = task.observations()
        observations.append(obs[:episodes].cpu().numpy())
        trajectories.append(task.engine.native_qpos()[:episodes].cpu().numpy())
        action = actor(obs)
        obs, reward, done, _ = task.step(action)
        if capture_actions:
            actions.append(action[:episodes].cpu().numpy())
            poses_after.append(task.engine.native_qpos()[:episodes].cpu().numpy())
            if ball:
                ball_positions.append(task.engine.ball_position()[:episodes].cpu().numpy())
        if hasattr(task.engine, "hit"):
            hits |= task.engine.hit.to(task.device) & active
            first_hit[(first_hit < 0) & hits] = lengths[(first_hit < 0) & hits] + 1
        if ball:
            q = task.engine.native_qpos()
            loads = task.engine.get_ground_contact_forces(0)[:, task.allowed_ground, 2].abs()
            settled = (q[:, 2] >= RECOVERY["height_m"]) & (
                1 - 2 * (q[:, 4].square() + q[:, 5].square()) >= np.cos(np.deg2rad(RECOVERY["tilt_degrees"])))
            settled &= torch.linalg.vector_norm(task.engine.get_root_vel(0)[:, :2], dim=-1) <= RECOVERY["horizontal_speed_m_s"]
            settled &= torch.linalg.vector_norm(task.engine.get_root_ang_vel(0), dim=-1) <= RECOVERY["angular_speed_rad_s"]
            settled &= (loads[:, 0] + loads[:, 2] > RECOVERY["each_foot_load_n"]) & (loads[:, 1] + loads[:, 3] > RECOVERY["each_foot_load_n"])
            settled &= hits & active & (lengths + 1 > first_hit)
            stable = torch.where(settled, stable + 1, 0)
            recovery_time[~settled] = -1
            reached = stable == settle_steps
            recovery_time[reached] = (lengths[reached] + 1 - first_hit[reached]) * task.dt
        returns += reward * active
        lengths += active
        target, _ = task.reference.sample(task.offset + task.steps * task.dt)
        tracking += torch.linalg.vector_norm(task.engine.get_root_pos(0) - target[:, :3], dim=-1) * active
        successes |= active & (done == DoneFlags.TIME.value)
        active &= done == DoneFlags.NULL.value
        if not active[:episodes].any():
            break
        # Keep finished worlds numerically healthy without counting another episode.
        task.reset(torch.nonzero(done != DoneFlags.NULL.value).flatten())
    count = slice(0, episodes)
    metrics = {"episodes": episodes, "successes": int(successes[count].sum()),
            "mean_survival_seconds": float(lengths[count].float().mean() * task.dt),
            "mean_return": float(returns[count].mean()),
            "mean_root_tracking_error_m": float((tracking[count] / lengths[count].clamp_min(1)).mean()),
            "survival_seconds": (lengths[count] * task.dt).cpu().tolist()}
    if hasattr(task.engine, "hit"):
        metrics.update(confirmed_hits=int(hits[count].sum()), survived_hits=int((hits & successes)[count].sum()))
    if ball:
        recovered = hits & successes & (stable >= settle_steps)
        metrics.update(evaluation_schema=VALIDATION_SCHEMA, recovered_hits=int(recovered[count].sum()),
                       recovery_criteria=RECOVERY,
                       cases=[{**task.validation_cases[i], "hit": bool(hits[i]), "survived": bool(successes[i]),
                               "recovered": bool(recovered[i]), "first_hit_step": int(first_hit[i]),
                               "survival_seconds": float(lengths[i] * task.dt),
                               "recovery_seconds": float(recovery_time[i]) if recovered[i] else None}
                              for i in range(episodes)])
    traces = {"qpos": np.array(trajectories), "observations": np.array(observations)}
    if capture_actions:
        traces.update(actions=np.array(actions), qpos_after=np.array(poses_after), ball_position=np.array(ball_positions))
    return metrics, traces
