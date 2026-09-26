"""Compare support-task critic predictions with native realized returns."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .baseline import sha256
from .policy import StandPolicy
from .probe_steps import validate_reference_substitution
from .rig import Rig
from .runtime import activate
from .support_transition import (SupportTransitionReference, TRACKING_REWARD_SCHEMA,
                                 support_tracking_reward)


def discounted_returns(rewards, discount):
    result = np.zeros_like(rewards, dtype=np.float64)
    future = 0.
    for index in reversed(range(len(rewards))):
        future = rewards[index] + discount * future
        result[index] = future
    return result


def td_lambda_advantages(values, rewards, discount, td_lambda):
    next_values = np.concatenate((values[1:], [0.]))
    residual = rewards + discount * next_values - values
    advantage = np.zeros_like(residual)
    future = 0.
    for index in reversed(range(len(residual))):
        future = residual[index] + discount * td_lambda * future
        advantage[index] = future
    return advantage


def summarize(values, returns, advantages, ages):
    result = {}
    for label, mask in (("age_0_2", ages < 2.), ("age_2_4", (ages >= 2.) & (ages < 4.)),
                        ("age_4_plus", ages >= 4.)):
        if mask.any():
            error = values[mask] - returns[mask]
            result[label] = dict(samples=int(mask.sum()), critic_mean=float(values[mask].mean()),
                                 realized_mean=float(returns[mask].mean()),
                                 bias=float(error.mean()), rmse=float(np.sqrt(np.mean(error ** 2))))
            result[label]["td_lambda_advantage_mean"] = float(advantages[mask].mean())
            result[label]["td_lambda_advantage_positive_fraction"] = float((advantages[mask] > 0).mean())
    return result


def run(args):
    activate(args.mimickit)
    from envs.base_env import DoneFlags, EnvMode
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .native_engine import NativeDummyEngine
    from .stand_task import StandTask

    class SupportAuditTask(StandTask):
        def reward(self):
            base = self.reference_reward()
            if args.reward_mode == "reference":
                return base
            q, v = self.sample_reference(self.offset + self.steps * self.dt)
            return support_tracking_reward(base, self.engine.get_root_pos(0)[:, :2],
                                           q[:, :2], self.engine.get_root_vel(0)[:, :2], v[:, :2])

    if (args.out.exists() or not 0 < args.discount < 1 or args.seconds <= 6
            or args.reward_mode not in ("reference", TRACKING_REWARD_SCHEMA)):
        raise ValueError("Require a new output path, valid discount and long episode")
    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    config = yaml.safe_load(args.agent_config.read_text())
    contract = json.loads(args.source_contract.read_text())
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    cases = []
    for label, checkpoint in args.checkpoint:
        if checkpoint.name == "best.pt":
            best = json.loads(checkpoint.with_name("best.json").read_text())
            if best["checkpoint_sha256"] != sha256(checkpoint):
                raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
            validate_reference_substitution(contract, best["contract"])
        task = SupportAuditTask(rig, args.reference, 1, "cpu", engine_factory=NativeDummyEngine,
                         control_mode="target_pd", contact_mode="support_v2",
                         reference_factory=lambda p, r, d: SupportTransitionReference(
                             p, r, d, mode="transition"))
        validate_reference_substitution(contract, task.observation_contract())
        task.episode_seconds = args.seconds
        task.reference_phase = lambda time: (time / task.reference.duration).clamp(max=1.)
        agent = PPOAgent(config, task, "cpu")
        agent.load(str(checkpoint))
        actor = StandPolicy(agent).eval()
        task.set_mode(EnvMode.TEST)
        obs, _ = task.reset()
        values, rewards, root_errors, done_flags = [], [], [], []
        with torch.no_grad():
            for _ in range(round(args.seconds / task.dt) + 1):
                value = agent._model.eval_critic(agent._obs_norm.normalize(obs)).item()
                obs, reward, done, _ = task.step(actor(obs))
                target, _ = task.reference.sample(task.steps * task.dt)
                error = task.engine.get_root_pos(0)[0] - target[0, :3]
                values.append(value)
                rewards.append(float(reward[0]))
                root_errors.append(error.numpy())
                done_flags.append(int(done[0]))
                if done[0] != DoneFlags.NULL.value:
                    break
        if done_flags[-1] != DoneFlags.FAIL.value:
            raise ValueError("Exact return comparison requires a terminal native fall")
        values, rewards = np.asarray(values), np.asarray(rewards)
        root_errors = np.asarray(root_errors)
        returns = discounted_returns(rewards, args.discount)
        advantages = td_lambda_advantages(values, rewards, args.discount, config["td_lambda"])
        ages = np.arange(len(rewards)) * task.dt
        stamps = []
        for second in (0., 1., 2., 3., 4., 5., 6., 7.):
            index = round(second / task.dt)
            if index >= len(rewards):
                continue
            stamps.append(dict(seconds=float(ages[index]), critic_value=float(values[index]),
                               realized_discounted_return=float(returns[index]),
                               td_lambda_advantage=float(advantages[index]),
                               root_error_x_m=float(root_errors[index, 0]),
                               reward=float(rewards[index])))
        cases.append(dict(label=label, checkpoint_sha256=sha256(checkpoint),
                          survival_seconds=len(rewards) * task.dt,
                          first_root_error_5cm_seconds=float(ages[np.flatnonzero(
                              np.linalg.norm(root_errors, axis=1) > .05)[0]])
                          if (np.linalg.norm(root_errors, axis=1) > .05).any() else None,
                          critic_error=summarize(values, returns, advantages, ages), samples=stamps,
                          trace=dict(ages_seconds=ages.tolist(), values=values.tolist(),
                                     realized_returns=returns.tolist(), rewards=rewards.tolist(),
                                     td_lambda_advantages=advantages.tolist(),
                                     root_error_xyz_m=root_errors.tolist())))
        print(json.dumps(dict(label=label, survival_seconds=cases[-1]["survival_seconds"],
                              critic_error=cases[-1]["critic_error"])), flush=True)
    report = dict(schema="mimic_support_critic_audit_v1",
                  source_contract_sha256=sha256(args.source_contract),
                  model_sha256=sha256(rig.path), discount=args.discount,
                  td_lambda=config["td_lambda"],
                  direct_reward_efold_seconds=float(-task.dt / np.log(args.discount * config["td_lambda"])),
                  seconds=args.seconds, reward_mode=args.reward_mode, cases=cases,
                  return_semantics="Exact discounted reward to a native failure; terminal value zero",
                  scope="One deterministic native trajectory per checkpoint; TD-lambda advantages are diagnostic and are not stochastic PPO training batches or a critic validation set")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path, required=True)
    parser.add_argument("--checkpoint", nargs=2, action="append", metavar=("LABEL", "PATH"),
                        required=True)
    parser.add_argument("--discount", type=float, default=.99)
    parser.add_argument("--reward-mode", choices=("reference", TRACKING_REWARD_SCHEMA),
                        default=TRACKING_REWARD_SCHEMA)
    parser.add_argument("--seconds", type=float, default=12.)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    options.checkpoint = [(label, Path(path)) for label, path in options.checkpoint]
    run(options)
