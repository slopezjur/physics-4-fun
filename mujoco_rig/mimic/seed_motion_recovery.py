"""Version a measured velocity/capture gate without changing learned tensors."""
import argparse
import copy
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
    from . import gated_recovery_net as nets
    from .balance_observation import task_classes
    from .checkpoints import BestCheckpoint, ball_regressions, recovery_step_rank, validate_contract
    from .continuous_stand import evaluate_retention
    from .native_engine import NativeDummyEngine
    from .policy import StandPolicy
    from .reactive_ball_eval import evaluate, selection_cases
    from .rig import Rig
    from .training_quiet_admission import validate_quiet_initialization

    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    validate_quiet_initialization(json.loads(args.quiet_report.read_text()), args.checkpoint,
                                  args.reference, rig.path)
    metadata = json.loads(args.checkpoint.with_name("best.json").read_text())
    source_protocol = json.loads(args.checkpoint.with_name("protocol.json").read_text())
    if metadata["checkpoint_sha256"] != sha256(args.checkpoint) or \
            source_protocol["experiment"]["ball_model_sha256"] != sha256(args.ball_model):
        raise ValueError("Require the selected source at the same projectile mass")
    config = yaml.safe_load(args.checkpoint.with_name("agent.yaml").read_text())
    if config["model"]["actor_net"] != nets.CAPTURE_NAME:
        raise ValueError("Require capture-v2 source for this gate-only conversion")
    contract = json.loads(args.checkpoint.with_name("contract.json").read_text())
    StandTask, BallTask = task_classes(contract)
    quiet = StandTask(rig, args.reference, 8, engine_factory=NativeDummyEngine)
    task = BallTask(rig, args.reference, 32, "cpu", ball_path=args.ball_model,
                    speed_min=2., speed_max=2., episode_seconds=12., recovery_mode="step_v5")
    validate_contract(contract, task.observation_contract())
    source = PPOAgent(config, task, "cpu")
    source.load(str(args.checkpoint))
    config = copy.deepcopy(config)
    architecture = nets.DAMPING_NAME if args.damping else nets.MOTION_NAME
    config["model"]["actor_net"] = architecture
    agent = PPOAgent(config, task, "cpu")
    original = source.state_dict()
    converted = agent.state_dict()
    added = set(converted) - set(original)
    expected = ({"_model._actor_layers.damping_gate_specification", "_model._actor_layers.damping.weight"}
                if args.damping else {"_model._actor_layers.motion_gate_thresholds"})
    if added != expected or set(original) - set(converted):
        raise ValueError("Conversion added unexpected state")
    converted.update(original)
    agent.load_state_dict(converted)
    if any(not torch.equal(value, agent.state_dict()[name]) for name, value in original.items()):
        raise RuntimeError("Gate conversion changed a source tensor")

    trace_report = json.loads(args.parity_trace.with_name("report.json").read_text())
    if trace_report["checkpoint_sha256"] != sha256(args.checkpoint) or \
            trace_report["trace_sha256"] != sha256(args.parity_trace):
        raise ValueError("Probe parity observations have different checkpoint provenance")
    with np.load(args.parity_trace) as data:
        observations = data["observation"][data["active"].astype(bool)]
        observations = torch.from_numpy(observations[np.linspace(0, len(observations)-1, min(1024, len(observations))).astype(int)])
    candidate = StandPolicy(agent).eval()
    probe = StandPolicy(source).eval()
    probe.actor.support_activation = not args.damping
    original_gate = nets.support_recovery_gate
    nets.support_recovery_gate = nets.motion_recovery_gate
    try:
        with torch.no_grad():
            parity_error = float((candidate(observations) - probe(observations)).abs().max())
    finally:
        nets.support_recovery_gate = original_gate
    if not np.isfinite(parity_error) or parity_error > 1e-6:
        raise RuntimeError("Versioned actor differs from the tested process-local gate")

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "agent.yaml").write_text(yaml.safe_dump(config))
    (args.out / "contract.json").write_text(json.dumps(contract, indent=2))
    agent.save(str(args.out / "model.pt"))
    protocol = dict(schema="motion_recovery_gate_seed_v1", source_checkpoint=str(args.checkpoint),
                    source_checkpoint_sha256=sha256(args.checkpoint), source_agent_sha256=sha256(args.checkpoint.with_name("agent.yaml")),
                    source_quiet_report_sha256=sha256(args.quiet_report), architecture=architecture,
                    agent_sha256=sha256(args.out / "agent.yaml"), experiment=task.experiment_contract(),
                    contract=contract, correction_limit=.8, motion_gate_thresholds_m_s=[.15, .30],
                    source_tensors_unchanged=list(original), added_state=sorted(added),
                    parity_target="unchanged source actor" if args.damping else "process-local motion gate",
                    trainable_actor="zero-initialized damping layer only" if args.damping else "existing recovery branch",
                    parity_trace_sha256=sha256(args.parity_trace), probe_action_error=parity_error,
                    scope="Gate-only training initialization; no optimizer steps, Arena admission or inherited quiet evidence")
    (args.out / "source").mkdir()
    protocol["source_hashes"] = {}
    for name in ("seed_motion_recovery.py", "gated_recovery_net.py", "reactive_ball_eval.py"):
        path = Path(__file__).with_name(name)
        protocol["source_hashes"][name] = sha256(path)
        (args.out / "source" / name).write_bytes(path.read_bytes())
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    quiet_metrics, _ = evaluate_retention(quiet, candidate)
    metrics, _ = evaluate(task, candidate, selection_cases(2., 12., task.dt))
    metrics.update(quiet_standing_passed=bool(quiet_metrics["passed"]), quiet_native=quiet_metrics)
    regressions = ball_regressions(metrics, metadata["metrics"])
    eligible = metrics["quiet_standing_passed"] and not regressions
    if eligible:
        best = BestCheckpoint(args.out, contract, rank_function=recovery_step_rank,
                              selection_label="Gate-only seed with native directional retention; further admission required")
        best.consider(agent.save, metrics, iteration=0, samples=0)
    report = dict(protocol=protocol, metrics=metrics, regressions=regressions,
                  eligible_for_long_validation=eligible, long_quiet_validated=False,
                  unseen_validated=False, godot_selected=False)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(eligible=eligible, quiet=metrics["quiet_standing_passed"],
                          recovered=metrics["recovered_hits"], steps=metrics["recovered_with_step"],
                          probe_action_error=parity_error, regressions=regressions)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mimickit", "checkpoint", "reference", "ball-model", "quiet-report", "parity-trace", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--damping", action="store_true",
                        help="Preserve the source actor and add a zero-initialized supported-motion damping layer")
    run(parser.parse_args())
