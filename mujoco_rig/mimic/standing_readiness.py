"""Physical-time standing objective and simulated episode-age preparation."""
import math

import torch

from .evaluation_cache import actor_fingerprint


def objective(dt, mode):
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Require a positive control timestep')
    if mode != 'retention_30s_v1':
        raise ValueError('Unknown standing objective')
    discount = math.exp(-dt / 30.)
    return dict(schema=mode, discount=discount, reward_scale=-math.expm1(-dt / 30.),
                decay_seconds=30., timeout='continuing_value_bootstrap',
                reward='Unchanged DeepMimic tracking, scaled by (1-discount); failure reward/value zero',
                interpretation='Normalized discounted continuing tracking until failure; 30 seconds is the e-folding time, not a hard horizon')


@torch.no_grad()
def initialize_value_scale(agent, source_discount, source_reward_scale, destination):
    """Transfer the critic's units as a prior, never claim to convert its horizon."""
    if not 0 < source_discount < 1 or not source_reward_scale > 0:
        raise ValueError('Invalid source value units')
    factor = ((1-source_discount) / source_reward_scale
              * destination['reward_scale'] / (1-destination['discount']))
    head = agent._model._critic_out
    head.weight.mul_(factor)
    head.bias.mul_(factor)
    agent._critic_optimizer._optimizer.state.clear()
    return dict(scale=factor, interpretation='Output-unit initialization only; new horizon requires validated critic calibration')


def age_counts(ages, ids):
    """Four physical-age strata per posture; no synthetic clock observations."""
    bins = torch.bucketize(ages, torch.tensor([5., 15., 25.], device=ages.device), right=True)
    return torch.bincount(ids % 8 * 4 + bins, minlength=32).reshape(8, 4)


def require_coverage(counts, retained_cases, minimum=4):
    if counts.shape != (8, 4) or not retained_cases or minimum < 1:
        raise ValueError('Require posture/age coverage and at least one retained posture')
    missing = [i for i in sorted(retained_cases)
               if counts[i, 0] < minimum or counts[i, 2:].sum() < minimum]
    if missing:
        raise ValueError(f'Insufficient early/late live states for retained postures: {missing}')


@torch.no_grad()
def prepare_episodes(agent, task, retained_cases, seconds=30.):
    """Simulate a frozen policy and stagger resets; preserve the final physical state."""
    from envs.base_env import DoneFlags
    from learning.base_agent import AgentMode
    if seconds < 30 or task.get_num_envs() < 64:
        raise ValueError('Temporal preparation requires at least 30 seconds and 64 worlds')
    fingerprint = actor_fingerprint(agent)
    agent.eval()
    agent.set_mode(AgentMode.TRAIN)
    obs, info = task.reset()
    steps = math.ceil(seconds / task.dt)
    # Each posture receives the same evenly distributed scheduled ages.
    replicas = (task.get_num_envs() + 7) // 8
    desired_age = ((task.ids // 8) * (steps-1) // max(1, replicas-1)).long()
    scheduled_reset = steps - desired_age
    for step in range(1, steps+1):
        action, _ = agent._decide_action(obs, info)
        _, _, done, _ = task.step(action)
        reset = (done != DoneFlags.NULL.value) | (scheduled_reset == step)
        obs, info = task.reset(task.ids[reset])
        if step % 300 == 0:
            print(f'Stationary preparation: {step}/{steps} control steps', flush=True)
    counts = age_counts(task.get_env_time(), task.ids)
    require_coverage(counts, retained_cases)
    if actor_fingerprint(agent) != fingerprint:
        raise RuntimeError('Preparation changed actor, critic or normalization')
    report = dict(schema='mimic_standing_readiness_v1', control_steps=steps,
                  counts_by_posture_age=counts.cpu().tolist(), age_edges_seconds=[0,5,15,25,30],
                  retained_cases=sorted(retained_cases), minimum_early_and_late_worlds=4,
                  passed=True, weights_unchanged=True,
                  scope='Simulated live states only. Warmup samples excluded from PPO; no replay or synthetic age assignment.')
    return (obs, info), report
