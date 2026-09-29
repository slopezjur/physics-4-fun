"""Native bilateral exploration around a measured successful recovery signal."""
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
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    probe = json.loads(args.probe.read_text())
    if probe["checkpoint_sha256"] != sha256(args.checkpoint):
        raise ValueError("Exploration checkpoint mismatch")
    good = [i for i,c in enumerate(probe["cases"]) if c["recovered_with_step"]]
    if not good:
        raise ValueError("Probe has no measured recovered step")
    source = max(good, key=lambda i: probe["cases"][i]["final_settled_seconds"])
    case = probe["cases"][source]
    with np.load(args.probe.with_name("actions.npz")) as trace:
        signal = trace["exploration"][:, source].copy()
    names = [rig.model.actuator(int(a)).name for a in rig.actuators]
    mirrored = signal.copy()
    for i, name in enumerate(names):
        if name.startswith(("Thigh_", "Shin_", "Foot_", "Toe_")):
            other = name.replace("_L_", "_R_") if "_L_" in name else name.replace("_R_", "_L_")
            mirrored[:, i] = signal[:, names.index(other)] * (1 if name.endswith("_rx") else -1)
    amplitudes = [.7, .85, 1., 1.15]
    signals = np.stack([sig * amplitude for sig in (signal, mirrored)
                        for amplitude in amplitudes for _ in range(8)], axis=1)
    cases = []
    for side in range(2):
        for amplitude in amplitudes:
            for posture in range(8):
                direction = case["direction"] if side == 0 else (-case["direction"]) % 4
                cases.append(dict(selection_cases(2., 12.)[direction * 8 + posture],
                    name=f"bilateral-{side}-{amplitude}-{posture}", target_body=case["target_body"]))
    task = ReactiveBallTask(rig, args.reference, 64, "cpu", ball_path=args.ball_model, speed_max=2.,
                            episode_seconds=12., recovery_mode="step_v4")
    agent = PPOAgent(yaml.safe_load((args.checkpoint.parent / "agent.yaml").read_text()), task, "cpu")
    agent.load(str(args.checkpoint))
    actor = StandPolicy(agent).eval()
    observations, actions = [], []
    timestep = 0

    def policy(obs):
        nonlocal timestep
        action = actor(obs) + torch.tensor(signals[timestep])
        observations.append(obs.numpy().copy())
        actions.append(action.numpy().copy())
        timestep += 1
        return action

    report, _ = evaluate(task, policy, cases)
    report.update(source_checkpoint_sha256=sha256(args.checkpoint), exploration_sha256=sha256(args.probe),
                  reference_sha256=sha256(args.reference), model_sha256=sha256(rig.path),
                  ball_sha256=sha256(args.ball_model), source_world=source,
                  scope="Native exploratory feedback trajectories, not a deployable policy")
    args.out.mkdir(parents=True, exist_ok=False)
    accepted = np.array([c["recovered_with_step"] for c in report["cases"]])
    np.savez_compressed(args.out / "demonstrations.npz", observation=np.asarray(observations)[:, accepted],
                        action=np.asarray(actions)[:, accepted], case=np.flatnonzero(accepted),
                        model_sha256=report["model_sha256"], reference_sha256=report["reference_sha256"],
                        source_checkpoint_sha256=report["source_checkpoint_sha256"])
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(recovered=report["recovered_hits"], recovered_with_step=report["recovered_with_step"],
                         left=sum(c["recovered_with_step"] and c["steps_by_foot"][0] > 0 for c in report["cases"]),
                         right=sum(c["recovered_with_step"] and c["steps_by_foot"][1] > 0 for c in report["cases"]))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mimickit", "checkpoint", "reference", "ball-model", "probe", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
