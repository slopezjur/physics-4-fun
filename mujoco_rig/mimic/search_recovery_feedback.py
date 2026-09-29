"""Search observable velocity feedback as a recovery discovery probe, not admission."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .baseline import sha256
from .runtime import activate


def run(args):
    if args.population < 8 or args.rounds < 1 or args.out.exists():
        raise ValueError("Require a new output directory and at least eight candidates")
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .policy import StandPolicy
    from .reactive_ball_task import ReactiveBallTask
    from .reactive_ball_eval import evaluate, selection_cases
    from .rig import Rig
    from .gated_recovery_net import recovery_gate
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    mp_util.init(0, 1, "cuda:0", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    task = ReactiveBallTask(rig, args.reference, args.population * 32, "cuda:0",
                            ball_path=args.ball_model, speed_max=2., episode_seconds=6.,
                            recovery_mode="step_v4")
    agent = PPOAgent(yaml.safe_load((args.checkpoint.parent / "agent.yaml").read_text()), task, task.device)
    agent.load(str(args.checkpoint))
    actor = StandPolicy(agent).eval()
    names = [rig.model.actuator(int(a)).name for a in rig.actuators]
    legs = [i for i,n in enumerate(names) if n.startswith(("Thigh_", "Shin_", "Foot_", "Toe_"))]
    mean = torch.zeros(len(legs), 4, device=task.device)
    if args.initial_feedback:
        with np.load(args.initial_feedback) as data:
            for key, path in (("checkpoint_sha256", args.checkpoint), ("reference_sha256", args.reference),
                              ("ball_sha256", args.ball_model), ("model_sha256", rig.path)):
                if data[key].item() != sha256(path):
                    raise ValueError("Initial feedback provenance mismatch: " + key)
            if not np.array_equal(data["actuator_indices"], legs):
                raise ValueError("Initial feedback actuator order mismatch")
            mean.copy_(torch.from_numpy(data["gains"].copy()).to(task.device))
    std = torch.full_like(mean, .15 if args.initial_feedback else .4)
    best = mean.clone()
    best_score = -float("inf")
    args.out.mkdir(parents=True)
    protocol = dict(checkpoint_sha256=sha256(args.checkpoint), reference_sha256=sha256(args.reference),
                    ball_sha256=sha256(args.ball_model), model_sha256=sha256(rig.path),
                    seed=args.seed, population=args.population, rounds=args.rounds,
                    features="root vx, vy / 0.5; root wx, wy / 2; clamped to +/-2",
                    gated=args.gated, normalizer_limited_features=args.gated,
                    initial_feedback_sha256=sha256(args.initial_feedback) if args.initial_feedback else "",
                    source_sha256=sha256(Path(__file__)),
                    actuators=[names[i] for i in legs], correction_limit=.8,
                    scope="CEM discovery of feedback reactions; no hit flag, phase, chosen foot or motion clock")
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (args.out / "runner.py").write_bytes(Path(__file__).read_bytes())
    history = []
    for iteration in range(args.rounds):
        gains = mean + std * torch.randn(args.population, len(legs), 4, device=task.device)
        gains[0] = 0
        gains[1] = best
        expanded = gains.repeat_interleave(32, 0)

        def policy(obs):
            raw = agent._obs_norm.normalize(obs) * agent._obs_norm._std + agent._obs_norm._mean if args.gated else obs
            features = torch.cat((raw[:, 9:11] / .5, raw[:, 12:14] / 2.), -1).clamp(-2, 2)
            result = actor(obs)
            correction = torch.einsum("wij,wj->wi", expanded, features).clamp(-.8, .8)
            if args.gated:
                correction *= recovery_gate(raw)[:, None]
            result[:, legs] += correction
            return result.clamp(-1, 1)

        cases = [dict(c, name=f"candidate-{i}-{c['name']}") for i in range(args.population)
                 for c in selection_cases(2., 6., task.dt)]
        report, _ = evaluate(task, policy, cases)
        scores = []
        summaries = []
        for i in range(args.population):
            rows = report["cases"][i*32:(i+1)*32]
            recovered = sum(c["recovered"] for c in rows)
            stepped = sum(c["recovered_with_step"] for c in rows)
            survival = sum(c["survival_seconds"] / 6. for c in rows)
            settled = sum(min(c["final_settled_seconds"], 2.) / 2. for c in rows)
            scores.append(10*recovered + survival + settled + stepped)
            summaries.append(dict(recovered=recovered, recovered_with_step=stepped,
                                  directional=[sum(c["recovered"] for c in rows[d*8:(d+1)*8]) for d in range(4)]))
        ranking = np.argsort(scores)[::-1].copy()
        np.savez_compressed(args.out / f"population_{iteration:03d}.npz", gains=gains.cpu().numpy(), scores=scores)
        (args.out / f"population_{iteration:03d}.json").write_text(json.dumps(summaries, indent=2))
        elites = gains[torch.as_tensor(ranking[:max(4, args.population//8)], device=task.device)]
        mean = .3*mean + .7*elites.mean(0)
        std = (.3*std + .7*elites.std(0, correction=0)).clamp_min(.05)
        winner = int(ranking[0])
        if scores[winner] > best_score:
            best_score = scores[winner]
            best = gains[winner].clone()
            np.savez_compressed(args.out / "feedback.npz", gains=best.cpu().numpy(),
                                actuator_indices=np.asarray(legs), **protocol)
            (args.out / "selected.json").write_text(json.dumps(dict(iteration=iteration,
                candidate=winner, summary=summaries[winner], cases=report["cases"][winner*32:(winner+1)*32]), indent=2))
        row = dict(iteration=iteration, score=scores[winner], best=summaries[winner], baseline=summaries[0],
                   standard_deviation=float(std.mean()))
        history.append(row)
        (args.out / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mimickit", "checkpoint", "reference", "ball-model", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=280904)
    parser.add_argument("--gated", action="store_true")
    parser.add_argument("--initial-feedback", type=Path)
    run(parser.parse_args())
