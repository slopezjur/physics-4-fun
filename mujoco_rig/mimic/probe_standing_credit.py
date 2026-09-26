"""Frozen-policy, full-episode audit of stationary PPO credit assignment."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from .baseline import sha256
from .checkpoints import validate_contract
from .continuous_stand import CONTRACT, GATES
from .critic_calibration import discounted_targets
from .evaluation_cache import actor_fingerprint
from .motion_features import install
from .rig import Rig
from .runtime import activate, MIMICKIT_REVISION


def analyze(trace, dt, discount, td_lambda, rollout_steps):
    """Use upstream targets, preserving failure zero and timeout bootstrap semantics."""
    from learning.rl_util import compute_td_lambda_return
    reward, done, active = (torch.as_tensor(trace[k]) for k in ('reward', 'done', 'active'))
    values, following = (torch.as_tensor(trace[k]) for k in ('value', 'next_value'))
    if reward.ndim != 2 or any(x.shape != reward.shape for x in (done, active, values, following)):
        raise ValueError('Credit traces must have matching [time, world] dimensions')
    if active.dtype != torch.bool or not active[0].all() or (active[1:] & ~active[:-1]).any():
        raise ValueError('Require contiguous first-episode masks')
    if not all(torch.isfinite(x).all() for x in (reward, values, following)):
        raise ValueError('Nonfinite credit trace')
    if not 0 < discount < 1 or not 0 <= td_lambda <= 1 or dt <= 0 or rollout_steps < 1:
        raise ValueError('Invalid return parameters')
    lengths = active.sum(0)
    last_done = done[lengths - 1, torch.arange(done.shape[1])]
    if not ((last_done == 1) | (last_done == 3)).all():
        raise ValueError('Every first episode must end in failure or timeout')
    following = following.clone()
    following[done == 1] = 0.
    mc = discounted_targets(reward, done != 0, following, discount, stride=1)
    targets = torch.empty_like(reward)
    for start in range(0, len(reward), rollout_steps):
        stop = start + rollout_steps
        targets[start:stop] = compute_td_lambda_return(
            reward[start:stop], following[start:stop], done[start:stop], discount, td_lambda)
    advantage, mc_advantage = targets - values, mc - values
    age = torch.arange(len(reward))[:, None] * dt
    until_end = (lengths[None, :] - torch.arange(len(reward))[:, None]) * dt

    def stats(mask):
        count = int(mask.sum())
        if not count:
            return dict(samples=0)
        residual = (values - mc)[mask]
        adv, actual_adv = advantage[mask], mc_advantage[mask]
        return dict(samples=count, mean_reward=float(reward[mask].mean()),
                    value_mean=float(values[mask].mean()), return_mean=float(mc[mask].mean()),
                    value_bias=float(residual.mean()), value_rmse=float(residual.square().mean().sqrt()),
                    td_advantage_mean=float(adv.mean()), mc_advantage_mean=float(actual_adv.mean()),
                    positive_td_fraction=float((adv > 0).float().mean()),
                    positive_td_negative_mc_fraction=float(((adv > 0) & (actual_adv < 0)).float().mean()))

    groups = torch.arange(reward.shape[1])[None, :] % 8
    ages = [(0., 8.534016), (8.534016, 15.), (15., 25.), (25., GATES['seconds'] + dt)]
    result = dict(all=stats(active), by_age=[dict(start_seconds=a, end_seconds=b,
        **stats(active & (age >= a) & (age < b))) for a, b in ages],
        before_failure=[dict(window_seconds=s, **stats(active & (last_done[None, :] == 1)
            & (until_end <= s))) for s in (0.5, 2., 5.)],
        postures=[dict(posture=i, **stats(active & (groups == i))) for i in range(8)],
        survival=[dict(posture=i, worlds=int((groups[0] == i).sum()),
            timeouts=int(((last_done == 3) & (groups[0] == i)).sum()),
            mean_seconds=float(lengths[groups[0] == i].float().mean() * dt))
            for i in range(8) if (groups[0] == i).any()])
    return result, dict(mc_return=mc.numpy(), td_target=targets.numpy(), raw_advantage=advantage.numpy())


@torch.no_grad()
def collect(actor, critics, task, seed):
    from envs.base_env import DoneFlags
    from learning.base_agent import AgentMode
    torch.manual_seed(seed)
    actor.eval()
    actor.set_mode(AgentMode.TRAIN)
    obs, _ = task.reset()
    active = torch.ones(task.get_num_envs(), device=task.device, dtype=torch.bool)
    rows = {name:[] for name in ('reward', 'done', 'active')}
    estimates = {name:dict(value=[], next_value=[]) for name in critics}
    for step in range(math.ceil(GATES['seconds'] / task.dt)):
        action, _ = actor._decide_action(obs, {})
        following, reward, done, _ = task.step(action)
        rows['reward'].append((reward * active).cpu().numpy())
        rows['done'].append(done.cpu().numpy())
        rows['active'].append(active.cpu().numpy().copy())
        for name, critic in critics.items():
            for key, observation in (('value', obs), ('next_value', following)):
                value = critic._model.eval_critic(critic._obs_norm.normalize(observation)).flatten()
                estimates[name][key].append(value.cpu().numpy())
        active &= done == DoneFlags.NULL.value
        if (step + 1) % 300 == 0 or not active.any():
            print(json.dumps(dict(control_steps=step+1, seconds=(step+1)*task.dt,
                                  active_first_episodes=int(active.sum()))), flush=True)
        if not active.any():
            break
        obs, _ = task.reset(torch.nonzero(done != DoneFlags.NULL.value).flatten())
    if active.any():
        raise RuntimeError('Probe did not finish all first episodes')
    common = {k:np.asarray(v) for k,v in rows.items()}
    return {name:{**common, **{k:np.asarray(v) for k,v in data.items()}} for name,data in estimates.items()}


def run(args):
    if args.envs < 32 or args.envs % 8:
        raise ValueError('Require at least 32 worlds in multiples of eight')
    activate(args.mimickit)
    from .continuous_stand_task import ContinuousStandTask
    from .guarded_ppo import GuardedPPO
    from util import mp_util

    class FrozenAgent(GuardedPPO):
        def _build_model(self, config):
            super()._build_model(config)
            if 'motion_feature_indices' in config:
                install(self, config['motion_feature_indices'])

        def _update_model(self):
            raise RuntimeError('Credit audit cannot optimize weights')

    torch.set_num_threads(1)
    mp_util.init(0, 1, 'cuda:0', None)
    config = yaml.safe_load((args.run / 'agent.yaml').read_text())
    contract = json.loads((args.run / 'contract.json').read_text())
    if contract['observation_contract'] != CONTRACT:
        raise ValueError('Require an explicit stationary training run')
    rig = Rig.load(Path(__file__).resolve().parents[1] / 'dummy.xml')
    task = ContinuousStandTask(rig, args.reference, args.envs, 'cuda:0', seed=args.seed)
    task.reward_scale = float(config.get('standing_objective', {}).get('reward_scale', 1.))
    if not math.isfinite(task.reward_scale) or task.reward_scale <= 0:
        raise ValueError('Invalid recorded standing reward scale')
    validate_contract(contract, task.observation_contract())
    agents = {}
    files = dict(initial='initial.pt', calibrated='calibrated.pt', candidate='model.pt')
    for label, name in files.items():
        agent = FrozenAgent(config, task, 'cuda:0').eval()
        agent.load(str(args.run / name))
        agents[label] = agent
    from .policy import StandPolicy
    if actor_fingerprint(StandPolicy(agents['initial'])) != actor_fingerprint(StandPolicy(agents['calibrated'])):
        raise ValueError('Calibration must preserve the initial actor')
    fingerprints = {name:actor_fingerprint(a) for name,a in agents.items()}
    viewers = Path(__file__).resolve().parents[2] / 'logs/mimickit-viewer'
    viewer_hashes = {str(p):sha256(p) for p in viewers.glob('*.json')}
    args.out.mkdir(parents=True, exist_ok=False)
    gamma, lam = float(config['discount']), float(config['td_lambda'])
    report = dict(schema='mimic_standing_credit_audit_v1', envs=args.envs, seed=args.seed,
        contract=contract, mimickit_revision=MIMICKIT_REVISION,
        source_hashes={name:sha256(Path(__file__).with_name(name)) for name in
            ('probe_standing_credit.py', 'critic_calibration.py', 'continuous_stand_task.py', 'stand_task.py')},
        input_hashes={str(args.run / n):sha256(args.run / n) for n in (*files.values(), 'agent.yaml', 'contract.json')},
        discount=gamma, td_lambda=lam, rollout_steps=config['steps_per_iter'],
        standing_objective=config.get('standing_objective', dict(schema='tracking_v1', reward_scale=1.)),
        discount_half_life_seconds=task.dt * math.log(.5)/math.log(gamma),
        td_trace_half_life_seconds=task.dt * math.log(.5)/math.log(gamma*lam),
        future_reward_weights={str(s):gamma**(s/task.dt) for s in (1, 5, 10, 20, 30)},
        viewer_hashes=viewer_hashes, results={},
        scope='Frozen stochastic first episodes on GPU. Initial/calibrated critics share identical actor trajectories. '
              'Candidate uses its own actor. Returns bootstrap timeouts from each evaluated critic; '
              'Monte Carlo comparisons are noisy sampled returns, not true advantages. '
              '32-step raw TD-lambda targets use upstream code; no advantage normalization or optimization. '
              'World timeouts are survival only, not the settled native retention gate.')
    for label in ('calibrated', 'candidate'):
        print(json.dumps(dict(collecting=label)), flush=True)
        critics = {k:agents[k] for k in ('initial', 'calibrated')} if label == 'calibrated' else {label:agents[label]}
        traces = collect(agents[label], critics, task, args.seed)
        for name, trace in traces.items():
            metrics, derived = analyze(trace, task.dt, gamma, lam, config['steps_per_iter'])
            path = args.out / f'{name}.npz'
            np.savez_compressed(path, **trace, **derived)
            report['results'][name] = dict(metrics=metrics, trace_sha256=sha256(path))
        (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    assert all(actor_fingerprint(agents[n]) == h for n,h in fingerprints.items())
    assert all(sha256(Path(p)) == h for p,h in {**report['input_hashes'], **viewer_hashes}.items())
    report.update(complete=True, weights_unchanged=True, viewers_unchanged=True)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mimickit', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--envs', type=int, default=256)
    parser.add_argument('--seed', type=int, default=210922)
    run(parser.parse_args())
