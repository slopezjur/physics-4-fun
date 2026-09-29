"""Frozen-policy assay: can action exploration produce any forward recovery?"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .baseline import sha256
from .policy import StandPolicy
from .runtime import activate
from .rig import Rig


def run(args):
    activate(args.mimickit)
    from envs.base_env import DoneFlags, EnvMode
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .reactive_ball_task import ReactiveBallTask
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    mp_util.init(0, 1, "cuda:0", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    task = ReactiveBallTask(rig, args.reference, 512, "cuda:0", ball_path=args.ball_model,
                            speed_max=1.5, episode_seconds=8., recovery_mode="capture_v3")
    task.set_mode(EnvMode.TEST)
    task.reset()
    agent = PPOAgent(yaml.safe_load((args.checkpoint.parent / "agent.yaml").read_text()), task, task.device)
    agent.load(str(args.checkpoint))
    # MimicKit calls get_obs_space(), which resets the task during construction.
    task.reset()
    task.directions[:] = torch.tensor([1., 0., 0.], device=task.device)
    task.shot_target_body[:] = task.target_indices[(task.ids % 8) % 3]
    assert torch.all(task.directions[:, 0] == 1) and torch.all(task.directions[:, 1:] == 0)
    actor = StandPolicy(agent).eval()
    levels = torch.tensor([0., .05, .15, .3], device=task.device)
    sigma = levels.repeat_interleave(128)[:, None]
    active = torch.ones(512, dtype=torch.bool, device=task.device)
    hits, survived = torch.zeros_like(active), torch.zeros_like(active)
    lengths = torch.zeros(512, device=task.device)
    stable = torch.zeros_like(lengths)
    total_reward = torch.zeros_like(lengths)
    with torch.no_grad():
        for _ in range(480):
            # Paired standardized noise across levels and posture replicas.
            noise = torch.randn(128, 30, device=task.device).repeat(4, 1)
            action = actor(task.observations()) + sigma * noise
            _, reward, done, _ = task.step(action)
            hits |= task.engine.hit & active
            stable = torch.where(active, task.settled_steps.float(), stable)
            total_reward += reward * active
            lengths += active
            survived |= active & (done == DoneFlags.TIME.value)
            active &= done == DoneFlags.NULL.value
            if not active.any():
                break
            ended = torch.nonzero(done != DoneFlags.NULL.value).flatten()
            if len(ended) and not args.leave_completed:
                task.reset(ended)
                task.shot_enabled[ended] = False
    results = []
    for i, value in enumerate(levels.tolist()):
        section = slice(i * 128, (i + 1) * 128)
        recovered = survived[section] & hits[section] & (stable[section] * task.dt >= 2.)
        results.append(dict(action_std=value, episodes=128, hits=int(hits[section].sum()),
                            survived=int(survived[section].sum()), recovered=int(recovered.sum()),
                            mean_seconds=float(lengths[section].mean() * task.dt),
                            mean_return=float(total_reward[section].mean())))
    report = dict(schema="reactive_exploration_assay_v1", seed=args.seed,
                  reset_completed=not args.leave_completed,
                  checkpoint_sha256=sha256(args.checkpoint), ball_sha256=sha256(args.ball_model),
                  speed=1.5, direction="+X", results=results,
                  scope="frozen actor with independent per-step action noise; no learning or promotion")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mimickit", "checkpoint", "reference", "ball-model", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=270950)
    parser.add_argument("--leave-completed", action="store_true")
    run(parser.parse_args())
