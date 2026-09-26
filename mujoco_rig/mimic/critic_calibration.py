"""Bounded value fitting on frozen-policy episodes, split by world for validation."""
import copy
import json
from dataclasses import dataclass
import torch

from .evaluation_cache import actor_fingerprint
from .policy import StandPolicy


def discounted_targets(rewards, ends, bootstraps, discount, stride=4):
    returns = torch.zeros_like(rewards[0])
    targets = []
    for index in reversed(range(len(rewards))):
        returns = rewards[index] + discount * torch.where(ends[index], bootstraps[index], returns)
        if index % stride == 0:
            targets.append(returns.clone())
    return torch.stack(targets[::-1])


@dataclass
class CalibrationProtocol:
    group_ids: torch.Tensor
    group_names: tuple
    validation_worlds: torch.Tensor
    horizon_seconds: float
    reward: dict
    split_description: str
    temporal_validation: bool = False

    def validate(self, worlds):
        if (self.group_ids.shape != (worlds,) or self.validation_worlds.shape != (worlds,)
                or self.group_ids.dtype != torch.long or self.validation_worlds.dtype != torch.bool
                or not self.group_names or len(set(self.group_names)) != len(self.group_names)
                or not 0 < self.horizon_seconds < float('inf')
                or (self.group_ids < 0).any() or (self.group_ids >= len(self.group_names)).any()):
            raise ValueError('Invalid calibration protocol')
        for group in range(len(self.group_names)):
            selected = self.group_ids == group
            if not (selected & self.validation_worlds).any() or not (selected & ~self.validation_worlds).any():
                raise ValueError('Every calibration group needs separate training and validation worlds')


def temporal_slices(ages, remaining, failed):
    return {'age_0_5': ages < 5., 'age_5_15': (ages >= 5.) & (ages < 15.),
            'age_15_plus': ages >= 15.,
            **{f'before_failure_{s:g}s': failed & (remaining <= s) for s in (.5, 2., 5.)}}


def calibration_improves(current, baseline, best, groups):
    def loss(result):
        return sum(result[name]['rmse'] ** 2 for name in groups) / len(groups)
    return (current.keys() == baseline.keys()
            and all(current[k]['rmse'] <= baseline[k]['rmse'] for k in baseline)
            and loss(current) < loss(best))


def calibrate(agent, task, steps, out, *, protocol=None):
    from envs.base_env import DoneFlags
    from learning.base_agent import AgentMode
    if steps < 1 or task.get_num_envs() < 16:
        raise ValueError("Critic calibration needs positive steps and at least sixteen worlds")
    if protocol is None:
        if task.quiet_worlds < 4 or task.get_num_envs() - task.quiet_worlds < 4:
            raise ValueError("Critic calibration needs at least four worlds in each task")
        protocol = CalibrationProtocol((task.ids >= task.quiet_worlds).long(), ('quiet', 'motion'),
            task.ids % 4 == 3, max(task.motion_seconds, task.quiet_seconds), task.reward_contract,
            'world index modulo four equals three')
    protocol.validate(task.get_num_envs())
    fingerprint = actor_fingerprint(StandPolicy(agent))
    original = copy.deepcopy(agent._model.state_dict())
    agent.eval()
    agent.set_mode(AgentMode.TRAIN)
    observations, _ = task.reset()
    alive = torch.ones(task.get_num_envs(), dtype=torch.bool, device=task.device)
    saved_obs, valid, rewards, ends, bootstraps = [], [], [], [], []
    end_step = torch.zeros_like(task.ids)
    failed = torch.zeros_like(alive)
    stride = 4
    with torch.no_grad():
        for index in range(int(protocol.horizon_seconds / task.dt) + 2):
            if index % stride == 0:
                saved_obs.append(observations.clone())
                valid.append(alive.clone())
            action, _ = agent._decide_action(observations, {})
            following, reward, done, _ = task.step(action)
            ended = alive & (done != DoneFlags.NULL.value)
            timeout = ended & (done == DoneFlags.TIME.value)
            end_step[ended] = index+1
            failed |= ended & (done == DoneFlags.FAIL.value)
            bootstrap = torch.zeros_like(reward)
            if timeout.any():
                bootstrap[timeout] = agent._model.eval_critic(agent._obs_norm.normalize(following[timeout])).flatten()
            rewards.append(reward * alive)
            ends.append(ended)
            bootstraps.append(bootstrap)
            alive &= ~ended
            if protocol.temporal_validation and (index+1) % 300 == 0:
                print(f'Temporal calibration: collected {index+1} control steps', flush=True)
            if not alive.any():
                break
            observations, _ = task.reset(torch.nonzero(done != DoneFlags.NULL.value).flatten())
        if alive.any():
            raise RuntimeError("Calibration did not finish every initial episode")
        targets = discounted_targets(rewards, ends, bootstraps, agent._discount, stride)
        mask = torch.stack(valid)
        obs = torch.stack(saved_obs)[mask]
        target = targets[mask]
        worlds = task.ids.expand(len(valid), -1)[mask]
        groups = protocol.group_ids[worlds]
        validation = protocol.validation_worlds[worlds]
        normalized = agent._obs_norm.normalize(obs)
    if not torch.isfinite(normalized).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Nonfinite calibration samples")
    train_ids = (~validation).nonzero().flatten()
    score_masks = {name:groups == i for i,name in enumerate(protocol.group_names)}
    if protocol.temporal_validation:
        ages = (torch.arange(len(valid), device=task.device) * stride * task.dt)[:, None].expand(mask.shape)[mask]
        remaining = end_step[worlds] * task.dt - ages
        for name, selected in temporal_slices(ages, remaining, failed[worlds]).items():
            if (selected & validation).sum() < 32 or (selected & ~validation).sum() < 32:
                raise ValueError(f'Insufficient separate calibration evidence for {name}')
            score_masks[name] = selected

    @torch.no_grad()
    def score():
        residual = torch.cat([agent._model.eval_critic(batch).flatten() for batch in normalized[validation].split(2048)]) - target[validation]
        result = {}
        for name, selected in score_masks.items():
            values = residual[selected[validation]]
            result[name] = dict(rmse=float(values.square().mean().sqrt()), bias=float(values.mean()), samples=len(values))
        return result

    before = score()
    best_score, best = before, copy.deepcopy(original)
    def balanced_mse(result):
        return sum(result[name]["rmse"] ** 2 for name in protocol.group_names) / len(protocol.group_names)
    history = []
    for step in range(steps):
        ids = train_ids[torch.randint(len(train_ids), (min(1024, len(train_ids)),), device=task.device)]
        residual = agent._model.eval_critic(normalized[ids]).flatten() - target[ids]
        # Equal group weight despite different first-episode survival durations.
        losses = [residual[selected[ids]].square().mean() for selected in score_masks.values()
                  if selected[ids].any()]
        agent._critic_optimizer.step(torch.stack(losses).mean())
        if (step + 1) % 20 == 0 or step + 1 == steps:
            current = score()
            if not all(torch.isfinite(torch.tensor(v["rmse"])) for v in current.values()):
                agent._model.load_state_dict(original)
                raise FloatingPointError("Nonfinite calibrated critic")
            history.append(dict(step=step+1, validation=current))
            if calibration_improves(current, before, best_score, protocol.group_names):
                best_score, best = current, copy.deepcopy(agent._model.state_dict())
    passed = balanced_mse(best_score) <= .9 * balanced_mse(before)
    agent._model.load_state_dict(best if passed else original)
    agent._critic_optimizer._optimizer.state.clear()
    if actor_fingerprint(StandPolicy(agent)) != fingerprint:
        raise RuntimeError("Critic calibration changed actor or normalization")
    report = dict(schema="mimic_critic_calibration_v2" if protocol.temporal_validation else "mimic_critic_calibration_v1", passed=passed, actor_unchanged=True,
                  actor_fingerprint=fingerprint, reward=protocol.reward, steps=steps,
                  before=before, best=best_score, history=history,
                  gate="At least 10% balanced posture/task validation MSE reduction; no measured group or temporal slice RMSE may increase",
                  temporal_validation=protocol.temporal_validation,
                  samples=len(obs), validation_world_rule=protocol.split_description,
                  targets="Frozen stochastic actor, first episodes; terminal failure zero, time limits bootstrap from initial critic",
                  scope="Calibration-world validation, not an independent behavior/generalization gate")
    (out / "critic_calibration.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
