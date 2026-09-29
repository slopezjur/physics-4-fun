"""Replay a discovered impact and nearby angles across all eight standing postures."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import numpy as np
import yaml

from .baseline import sha256
from .runtime import activate


def run(args):
    if not math.isfinite(args.seconds) or args.seconds < 12 or args.seconds > 60:
        raise ValueError("Require a 12..60 second diagnostic horizon")
    if not args.offsets or not all(math.isfinite(offset) for offset in args.offsets):
        raise ValueError("Require finite angle offsets")
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .balance_observation import task_classes
    from .checkpoints import validate_contract
    from .policy import StandPolicy
    from .reactive_ball_eval import evaluate
    from .rig import Rig

    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    discovery = json.loads(args.discovery.read_text())
    discovery_protocol = json.loads(args.discovery.with_name("protocol.json").read_text())
    if discovery["model_sha256"] != sha256(args.ball_model) or discovery_protocol["reference_sha256"] != sha256(args.reference):
        raise ValueError("Discovery model/reference mismatch")
    matches = [case for case in discovery["cases"] if case["name"] == args.case]
    if len(matches) != 1 or (not args.allow_nonstep_case and not matches[0]["recovered_with_step"]):
        raise ValueError("Require one stepped recovery, or explicitly allow a nonstep diagnostic case")
    discovered = matches[0]
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    contract = json.loads((args.checkpoint.parent / "contract.json").read_text())
    config = yaml.safe_load((args.checkpoint.parent / "agent.yaml").read_text())
    _, BallTask = task_classes(contract)
    offsets = args.offsets
    task = BallTask(rig, args.reference, len(offsets) * 8, "cpu", ball_path=args.ball_model,
                    speed_min=2., speed_max=2., episode_seconds=30., recovery_mode="step_v5")
    validate_contract(contract, task.observation_contract())
    agent = PPOAgent(config, task, "cpu")
    agent.load(str(args.checkpoint))
    cases = [dict(name=f"offset-{offset:g}-posture-{posture}", posture=posture,
        direction=discovered["direction"], angle_radians=discovered["angle_radians"] + math.radians(offset),
        target_body=discovered["target_body"], speed=discovered["speed"], phase=0.,
        launch_step=discovered["launch_step"], trial_steps=math.ceil(args.seconds / task.dt))
        for offset in offsets for posture in range(8)]
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    protocol = dict(schema="native_recovery_neighborhood_v1", discovery=str(args.discovery),
        discovery_sha256=sha256(args.discovery), discovery_protocol_sha256=sha256(args.discovery.with_name("protocol.json")),
        discovered_case=discovered, checkpoint_sha256=sha256(args.checkpoint),
        ball_model_sha256=sha256(args.ball_model), reference_sha256=sha256(args.reference),
        character_sha256=sha256(rig.path), runner_sha256=hashlib.sha256(source).hexdigest(),
        evaluator_sha256=sha256(Path(__file__).with_name("reactive_ball_eval.py")),
        angle_offsets_degrees=offsets, seconds=args.seconds, capture_trace=args.capture,
        allow_nonstep_case=args.allow_nonstep_case,
        scope="Discovered-case neighborhood diagnostic; not fresh held-out admission")
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (args.out / "runner.py").write_bytes(source)
    actor = StandPolicy(agent).eval()
    observations, actions, masks, rewards = [], [], [], []

    @torch.no_grad()
    def policy(observation):
        action = actor(observation)
        if args.capture:
            observations.append(observation.numpy().copy())
            actions.append(action.numpy().copy())
        return action

    def record(task, step, active, done):
        masks.append(active.numpy().copy())
        rewards.append(task.last_transition_reward.numpy().copy())

    metrics, _ = evaluate(task, policy, cases, record if args.capture else None)
    if sha256(args.checkpoint) != protocol["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed during neighborhood audit")
    metrics.update(protocol=protocol, godot_selected=False)
    if args.capture:
        metrics.update(checkpoint_sha256=protocol["checkpoint_sha256"],
            reference_sha256=protocol["reference_sha256"], ball_sha256=protocol["ball_model_sha256"],
            model_sha256=protocol["character_sha256"])
        np.savez_compressed(args.out / "trace.npz", observation=np.asarray(observations),
                            action=np.asarray(actions), active=np.asarray(masks), reward=np.asarray(rewards))
        metrics["trace_sha256"] = sha256(args.out / "trace.npz")
    (args.out / "report.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({key: metrics[key] for key in ("confirmed_hits", "recovered_hits", "recovered_with_step")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mimickit", "checkpoint", "reference", "ball-model", "discovery", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--seconds", type=float, default=33.)
    parser.add_argument("--offsets", type=float, nargs="+", default=[-5., 0., 5., 10.])
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--allow-nonstep-case", action="store_true",
                        help="Replay a known failure or nonstep recovery as a diagnostic; not admission")
    run(parser.parse_args())
