"""Native-only discovery probe: activate corrections from measured capture risk."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from .baseline import sha256
from .runtime import activate


def run(args):
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .capture_reward import capture_distance
    from .continuous_stand import evaluate_retention
    from .policy import StandPolicy
    from .reactive_ball_task import ReactiveBallTask
    from .reactive_ball_eval import evaluate, selection_cases
    from .rig import Rig
    if not 0 <= args.gate_start < args.gate_full:
        raise ValueError("Require increasing nonnegative capture gate distances")
    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    task = ReactiveBallTask(rig, args.reference, 8 if args.quiet else 32, "cpu", ball_path=args.ball_model,
                            speed_max=2., episode_seconds=12., recovery_mode="step_v4")
    agent = PPOAgent(yaml.safe_load((args.checkpoint.parent / "agent.yaml").read_text()), task, "cpu")
    agent.load(str(args.checkpoint))
    actor = StandPolicy(agent).eval()
    with np.load(args.feedback) as feedback:
        for key, path in (("checkpoint_sha256", args.checkpoint), ("reference_sha256", args.reference),
                          ("ball_sha256", args.ball_model), ("model_sha256", rig.path)):
            if feedback[key].item() != sha256(path):
                raise ValueError("Feedback provenance mismatch: " + key)
        indices = feedback["actuator_indices"].copy()
        gains = torch.from_numpy(feedback["gains"].copy())
    errors, activations = [], []

    def policy(obs):
        if args.quiet:
            task.shot_enabled[:] = False
        for data in task.engine.datas:
            mujoco.mj_subtreeVel(task.engine.rig.model, data)
        com = torch.from_numpy(np.stack([data.subtree_com[1] for data in task.engine.datas])).float()
        velocity = torch.from_numpy(np.stack([data.subtree_linvel[1] for data in task.engine.datas])).float()
        loads = obs[:, 93:95] * 1000.
        error = capture_distance(com, velocity, task.engine.get_body_pos(0)[:, task.feet], loads)
        gate = ((error - args.gate_start) / (args.gate_full-args.gate_start)).clamp(0, 1)
        errors.append(error.numpy().copy())
        activations.append(gate.numpy().copy())
        features = torch.cat((obs[:, 9:11] / .5, obs[:, 12:14] / 2.), -1).clamp(-2, 2)
        result = actor(obs)
        result[:, indices] += (features @ gains.T).clamp(-.8, .8) * gate[:, None]
        return result.clamp(-1, 1)

    if args.quiet:
        report, traces = evaluate_retention(task, policy)
    else:
        report, traces = evaluate(task, policy, selection_cases(2., 12., task.dt))
    report.update(checkpoint_sha256=sha256(args.checkpoint), feedback_sha256=sha256(args.feedback),
                  reference_sha256=sha256(args.reference), model_sha256=sha256(rig.path),
                  ball_sha256=sha256(args.ball_model), gate_start=args.gate_start, gate_full=args.gate_full,
                  maximum_capture_error_m=float(np.max(errors)), maximum_gate=float(np.max(activations)),
                  source_sha256=sha256(Path(__file__)),
                  scope="Native COM-state discovery probe; not portable policy or deployment evidence")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "runner.py").write_bytes(Path(__file__).read_bytes())
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(args.out / "capture.npz", error=errors, gate=activations)
    print(json.dumps({key:report[key] for key in ("passed", "recovered_hits", "recovered_with_step",
                     "maximum_capture_error_m", "maximum_gate") if key in report}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mimickit", "checkpoint", "reference", "ball-model", "feedback", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gate-start", type=float, default=0.)
    parser.add_argument("--gate-full", type=float, default=.02)
    run(parser.parse_args())
