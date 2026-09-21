"""Collect fixed incumbent targets on CPU, using seeds separate from evaluation.

Only trajectories that survive and settle are retained. The reference therefore
protects demonstrated successes without requiring imitation of failed recoveries.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib

import numpy as np
import torch

from env_config import FALL_FRACTION
from perturb_env import PerturbEnv
from policy_reference import REFERENCE_VERSION
from ppo import ActorCritic
from recovery_reward import CONFIG
from observation_contract import LEGACY, VERSIONS, checkpoint_version


def collect(net, *, ball, num_envs, seconds, seed, capacity=4096, observation_version=LEGACY,
            teacher_width=None):
    env = PerturbEnv(num_envs=num_envs, episode_seconds=seconds, seed=seed,
                     model='dummy_ball.xml' if ball else 'dummy.xml',
                     ball_speed=6.0, ball_every=(1.5, 1.8), observation_version=observation_version)
    teacher_width = env.num_obs if teacher_width is None else teacher_width
    if teacher_width > env.num_obs:
        raise ValueError('Cannot collect a narrower observation than the teacher uses')
    obs = env.reset_all()
    env.auto_reset = False
    env.max_shots_per_episode = 1
    alive = np.ones(num_envs, dtype=bool)
    observations, means = [], []
    for step in range(env.max_episode_length):
        with torch.no_grad():
            action = net.actor(obs[:, :teacher_width])
        # Impact examples include approach, response and settling. Never use
        # evaluation seed 17 to construct training targets.
        # Keep every control phase before uniform subsampling. A fixed temporal
        # stride can alias alternating ankle actions and miss half the behavior.
        if not ball or step * env.dt * env.decimation >= 1.5:
            observations.append(obs.clone())
            means.append(action.clone())
        obs, _, _, _ = env.step(action)
        for i, d in enumerate(env.datas):
            alive[i] &= (d.xpos[env.pelvis, 2] >= FALL_FRACTION * env.rest_pelvis_z
                         and d.xmat[env.pelvis].reshape(3, 3)[2, 2] >= 0.5)
    successful = alive & (env.settle_hold >= CONFIG.settle_seconds)
    if ball:
        successful &= env.shots_fired > 0
    if not successful.any() or (not ball and not successful.all()):
        raise ValueError('Reference policy did not demonstrate the required successful trajectories')
    obs = torch.stack(observations)[:, successful].reshape(-1, env.num_obs)
    mean = torch.stack(means)[:, successful].reshape(-1, env.num_actions)
    indices = np.random.default_rng(seed).permutation(len(obs))[:capacity]
    print(f"[reference] {'impact' if ball else 'quiet'}: {successful.sum()}/{num_envs} "
          f"successful trajectories, {len(indices)} states, seed {seed}", flush=True)
    return {'obs': obs[indices], 'mean': mean[indices]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--observation_version', choices=VERSIONS, default=None)
    args = parser.parse_args()
    path = pathlib.Path(args.checkpoint)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    teacher_version = checkpoint_version(checkpoint)
    version = args.observation_version or teacher_version
    net = ActorCritic(checkpoint['num_obs'], checkpoint['num_actions'])
    net.load_state_dict(checkpoint['model'])
    net.eval()
    collection = dict(observation_version=version, teacher_width=checkpoint['num_obs'])
    groups = {'quiet': collect(net, ball=False, num_envs=16, seconds=20, seed=101, **collection),
              'impact': collect(net, ball=True, num_envs=64, seconds=6, seed=103, **collection)}
    root = pathlib.Path(__file__).resolve().parents[1]
    source = {'checkpoint': str(path.resolve()), 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'quiet_seed': 101, 'impact_seed': 103, 'ball_speed': 6.0,
              'sampling': 'uniform_all_control_steps',
              'observation_version': version, 'teacher_observation_version': teacher_version,
              'model_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in ('dummy.xml', 'dummy_ball.xml')}}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'version': REFERENCE_VERSION, 'source': source, 'groups': groups,
                'scale': net.log_std.detach().exp().clamp_min(net.min_std)}, out)
    print(f'[reference] saved {out}', flush=True)


if __name__ == '__main__':
    main()
