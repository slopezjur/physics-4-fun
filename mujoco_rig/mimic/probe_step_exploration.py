"""Test temporally coherent leg exploration under actual hits, without learning."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .baseline import sha256
from .runtime import activate


def run(args):
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .policy import StandPolicy
    from .reactive_ball_task import ReactiveBallTask
    from .reactive_ball_eval import evaluate, selection_cases
    from .rig import Rig
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    mp_util.init(0, 1, "cuda:0", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    task = ReactiveBallTask(rig, args.reference, 512, "cuda:0", ball_path=args.ball_model,
                            speed_max=2., episode_seconds=12., recovery_mode="capture_v3")
    agent = PPOAgent(yaml.safe_load((args.checkpoint.parent / "agent.yaml").read_text()), task, task.device)
    agent.load(str(args.checkpoint))
    actor = StandPolicy(agent).eval()
    settings = [(std, tau) for tau in (.1, .4) for std in (0., .15, .3, .6)]
    sigma = torch.tensor([s for s,t in settings], device=task.device).repeat_interleave(64)[:, None]
    rho = torch.exp(-task.dt / torch.tensor([t for s,t in settings], device=task.device)).repeat_interleave(64)[:, None]
    mask = torch.tensor([rig.model.actuator(int(a)).name.startswith(("Thigh_", "Shin_", "Foot_", "Toe_"))
                         for a in rig.actuators], device=task.device)
    noise = torch.zeros(512, 30, device=task.device)
    hit_age = torch.zeros(512, device=task.device)
    actions, exploration = [], []

    def policy(obs):
        nonlocal noise
        paired = torch.randn(64, 30, device=task.device).repeat(8, 1)
        noise = rho * noise + (1-rho.square()).sqrt() * paired
        enabled = task.engine.hit & (hit_age < args.noise_seconds)
        hit_age.add_(task.engine.hit * task.dt)
        perturbation = sigma * noise * mask * enabled[:, None]
        result = actor(obs) + perturbation
        actions.append(result.cpu().numpy())
        exploration.append(perturbation.cpu().numpy())
        return result

    cases = [dict(c, name=f"setting-{g}-replica-{j}-{c['name']}") for g in range(8)
             for j in range(2) for c in selection_cases(2., 12., task.dt)]
    report, _ = evaluate(task, policy, cases)
    report.update(checkpoint_sha256=sha256(args.checkpoint), seed=args.seed,
                  noise_seconds=args.noise_seconds,
                  scope="Frozen actor with correlated post-hit leg noise; no policy update or admission")
    report["settings"] = [dict(std=s, correlation_seconds=t,
        recovered=sum(c["recovered"] for c in report["cases"][g*64:(g+1)*64]),
        lifted=sum(any(c["lifted_feet"]) for c in report["cases"][g*64:(g+1)*64]),
        landed_step=sum(sum(c["steps_by_foot"]) > 0 for c in report["cases"][g*64:(g+1)*64]),
        recovered_with_step=sum(c["recovered_with_step"] for c in report["cases"][g*64:(g+1)*64]))
        for g,(s,t) in enumerate(settings)]
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "actions.npz", action=np.asarray(actions), exploration=np.asarray(exploration))
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["settings"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mimickit", "checkpoint", "reference", "ball-model", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=280901)
    parser.add_argument("--noise-seconds", type=float, default=.5)
    run(parser.parse_args())
