"""Fixed-phase deterministic evaluation; success means surviving the full episode."""
import numpy as np
import torch


@torch.no_grad()
def evaluate(task, actor, episodes=8):
    from envs.base_env import EnvMode, DoneFlags
    task.set_mode(EnvMode.TEST)
    task.reset()
    # Use identical phases even when the training engine contains more worlds.
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
    hits = torch.zeros_like(active)
    for _ in range(int(np.ceil(task.episode_seconds / task.dt))):
        obs = task.observations()
        observations.append(obs[:episodes].cpu().numpy())
        trajectories.append(task.engine.native_qpos()[:episodes].cpu().numpy())
        obs, reward, done, _ = task.step(actor(obs))
        if hasattr(task.engine, "hit"):
            hits |= task.engine.hit.to(task.device) & active
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
    return metrics, {
                "qpos": np.array(trajectories), "observations": np.array(observations)}
