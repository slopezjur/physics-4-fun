"""Native diagnostics of recovery activation or learned damping amplitude."""
import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from .baseline import sha256
from .runtime import activate


def run(args):
    if not args.quiet_only and not args.replays:
        raise ValueError("Provide replay reports or request a quiet-only diagnostic")
    if args.damping_scale is not None and (not math.isfinite(args.damping_scale)
                                          or not 0 < args.damping_scale <= 16):
        raise ValueError("Damping scale must be finite and within (0, 16]")
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from . import gated_recovery_net
    from .balance_observation import task_classes
    from .checkpoints import validate_contract
    from .native_engine import NativeDummyEngine
    from .policy import StandPolicy
    from .reactive_ball_eval import evaluate as evaluate_hits
    from .rig import Rig
    from .validate_stand_foundation import evaluate as evaluate_quiet

    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    contract = json.loads(args.checkpoint.with_name("contract.json").read_text())
    StandTask, BallTask = task_classes(contract)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    quiet = StandTask(rig, args.reference, 8, engine_factory=NativeDummyEngine)
    validate_contract(contract, quiet.observation_contract())
    agent = PPOAgent(yaml.safe_load(args.checkpoint.with_name("agent.yaml").read_text()), quiet, "cpu")
    agent.load(str(args.checkpoint))
    actor = StandPolicy(agent).eval()
    if not actor.actor.capture_observation or actor.actor.support_activation or actor.actor.motion_activation:
        raise ValueError("Diagnostic requires a capture-only or damping actor")
    if actor.actor.damping_activation != (args.damping_scale is not None):
        raise ValueError("Damping actors require an explicit amplitude diagnostic")
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = dict(checkpoint_sha256=sha256(args.checkpoint), reference_sha256=sha256(args.reference),
                    ball_model_sha256=sha256(args.ball_model), rig_sha256=sha256(rig.path),
                    runner_sha256=sha256(Path(__file__)), seconds=33.,
                    quiet_seconds=args.quiet_seconds, quiet_only=args.quiet_only,
                    damping_scale=args.damping_scale,
                    intervention=(f"Multiply only copied actor damping.weight by {args.damping_scale:g}"
                                  if args.damping_scale is not None else
                                  "max(capture gate, clamp((horizontal root speed - 0.15)/0.15,0,1))"),
                    network_sha256=sha256(Path(__file__).with_name("gated_recovery_net.py")),
                    scope="Process-local intervention, unchanged checkpoint and physical limits; not training or admission")

    def velocity_capture_gate(observation):
        capture = ((observation[..., 362] - .001) / .019).clamp(0, 1)
        speed = torch.linalg.vector_norm(observation[..., 9:11], dim=-1)
        return torch.maximum(capture, ((speed - .15) / .15).clamp(0, 1))

    # Reuse the existing gate dispatch only in this diagnostic process.
    original_gate = gated_recovery_net.support_recovery_gate
    if args.damping_scale is None:
        gated_recovery_net.support_recovery_gate = velocity_capture_gate
        actor.actor.support_activation = True
    else:
        with torch.no_grad():
            actor.actor.damping.weight.mul_(args.damping_scale)
    try:
        quiet_result, _ = evaluate_quiet(quiet, actor, args.quiet_seconds)
        results = []
        if quiet_result["passed"] and not args.quiet_only:
            for replay in args.replays:
                saved = json.loads(replay.read_text())
                saved_protocol = saved["protocol"]
                if saved_protocol["checkpoint_sha256"] != protocol["checkpoint_sha256"] or \
                        saved_protocol["ball_model_sha256"] != protocol["ball_model_sha256"] or \
                        saved_protocol["reference_sha256"] != protocol["reference_sha256"]:
                    raise ValueError("Replay provenance differs from diagnostic checkpoint/model/reference")
                task = BallTask(rig, args.reference, 8, "cpu", ball_path=args.ball_model,
                                speed_min=2., speed_max=2., episode_seconds=30., recovery_mode="step_v5")
                validate_contract(contract, task.observation_contract())
                keys = ("name", "posture", "direction", "angle_radians", "target_body", "speed", "phase", "launch_step", "trial_steps")
                cases = [{key: row[key] for key in keys} for row in saved["cases"]]
                metrics, _ = evaluate_hits(task, actor, cases)
                results.append(dict(replay=str(replay), replay_sha256=sha256(replay), metrics=metrics))
        if sha256(args.checkpoint) != protocol["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint changed during diagnostic")
        (args.out / "runner.py").write_bytes(Path(__file__).read_bytes())
        (args.out / "report.json").write_text(json.dumps(dict(protocol=protocol, quiet=quiet_result,
                                                            results=results, godot_selected=False), indent=2))
        print(json.dumps(dict(quiet=quiet_result["passed"], recovered=[r["metrics"]["recovered_hits"] for r in results])), flush=True)
    finally:
        gated_recovery_net.support_recovery_gate = original_gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mimickit", "checkpoint", "reference", "ball-model", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--replays", type=Path, nargs="+", default=[])
    parser.add_argument("--quiet-seconds", type=float, choices=(30., 600.), default=30.)
    parser.add_argument("--quiet-only", action="store_true")
    parser.add_argument("--damping-scale", type=float,
                        help="Scale only the learned damping layer in a copied policy; preserve all physical limits")
    run(parser.parse_args())
