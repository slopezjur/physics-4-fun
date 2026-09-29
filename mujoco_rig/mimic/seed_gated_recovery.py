"""Migrate validated standing weights into a locally registered MimicKit recovery actor."""
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
    from .checkpoints import BestCheckpoint, validate_contract
    from .continuous_stand import evaluate_retention
    from .continuous_stand_task import ContinuousStandTask
    from .gated_recovery_net import NAME, CAPTURE_NAME
    from .balance_observation import CONTRACT as CAPTURE_CONTRACT, task_classes
    from .native_engine import NativeDummyEngine
    from .policy import StandPolicy
    from .reactive_ball_task import ReactiveBallTask
    from .reactive_ball_eval import evaluate, selection_cases
    from .rig import Rig
    torch.set_num_threads(1)
    torch.manual_seed(280905)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    metadata = json.loads((args.checkpoint.parent / "best.json").read_text())
    if metadata["checkpoint_sha256"] != sha256(args.checkpoint) or not metadata["metrics"]["quiet_standing_passed"]:
        raise ValueError("Require a selected standing-retaining source")
    config = yaml.safe_load((args.checkpoint.parent / "agent.yaml").read_text())
    if config["model"]["actor_net"] != "fc_2layers_1024units":
        raise ValueError("Migration expects the standard standing network")
    contract = json.loads((args.checkpoint.parent / "contract.json").read_text())
    quiet = ContinuousStandTask(rig, args.reference, engine_factory=NativeDummyEngine)
    native = ReactiveBallTask(rig, args.reference, 32, "cpu", ball_path=args.ball_model,
                              speed_max=2., episode_seconds=12., recovery_mode="step_v4")
    validate_contract(contract, native.observation_contract())
    source_agent = PPOAgent(config, native, "cpu")
    source_agent.load(str(args.checkpoint))
    if args.capture_observation:
        ContinuousStandTask, ReactiveBallTask = task_classes(dict(observation_contract=CAPTURE_CONTRACT))
        quiet = ContinuousStandTask(rig, args.reference, engine_factory=NativeDummyEngine)
        native = ReactiveBallTask(rig, args.reference, 32, "cpu", ball_path=args.ball_model,
                                  speed_max=2., episode_seconds=12., recovery_mode="step_v4")
        contract = native.observation_contract()
    config = copy.deepcopy(config)
    config["model"]["actor_net"] = CAPTURE_NAME if args.capture_observation else NAME
    agent = PPOAgent(config, native, "cpu")
    state = agent.state_dict()
    for key, value in source_agent.state_dict().items():
        if key.startswith("_model._actor_layers."):
            target = key.replace("_model._actor_layers.", "_model._actor_layers.base.", 1)
        elif key.startswith("_model._action_dist._mean_net."):
            target = key.replace("_model._action_dist._mean_net.", "_model._actor_layers.base_mean.", 1)
        else:
            target = key
        if args.capture_observation and target in ("_obs_norm._mean", "_obs_norm._std"):
            state[target][:362] = value
            state[target][362] = 1. if target.endswith("_std") else 0.
            continue
        if args.capture_observation and target == "_model._critic_layers.0.weight":
            state[target].zero_()
            state[target][:, :362] = value
            continue
        if target not in state or state[target].shape != value.shape:
            raise ValueError("Unsupported source parameter: " + key)
        state[target] = value.clone()
    agent.load_state_dict(state)
    network = agent._model._actor_layers
    with torch.no_grad():
        network.observation_mean.copy_(agent._obs_norm._mean)
        network.observation_std.copy_(agent._obs_norm._std)
        network.action_std.copy_(agent._a_norm._std)
        legs = [i for i,a in enumerate(rig.actuators) if rig.model.actuator(int(a)).name.startswith(
                ("Thigh_", "Shin_", "Foot_", "Toe_"))]
        network.leg_mask[legs] = 1
        sample = native.observations()
        torch.testing.assert_close(StandPolicy(source_agent)(sample[:, :362]), StandPolicy(agent)(sample), rtol=0, atol=0)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "agent.yaml").write_text(yaml.safe_dump(config))
    (args.out / "contract.json").write_text(json.dumps(contract, indent=2))
    protocol = dict(schema="mimic_gated_recovery_seed_v1", source_checkpoint_sha256=sha256(args.checkpoint),
                    feedback_sha256=sha256(args.feedback), experiment=native.experiment_contract(),
                    contract=contract, base="frozen actor layers, readout and normalization",
                    recovery="learned velocity initialization plus trainable nonlinear leg branch",
                    gate=("instantaneous loaded capture error 0.001..0.020m" if args.capture_observation else
                          "observable horizontal speed 0.2..0.4m/s or horizontal angular speed 0.5..1rad/s"),
                    config=config, source_hashes={name:sha256(Path(__file__).with_name(name)) for name in
                        ("seed_gated_recovery.py", "gated_recovery_net.py", "runtime.py", "balance_observation.py",
                         "native_engine.py", "newton_adapter.py")})
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (args.out / "source").mkdir()
    for name in protocol["source_hashes"]:
        (args.out / "source" / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    best = BestCheckpoint(args.out, contract)
    history = []
    for iteration in (0, 1):
        if iteration:
            with np.load(args.feedback) as feedback:
                for key, path in (("checkpoint_sha256", args.checkpoint), ("reference_sha256", args.reference),
                                  ("ball_sha256", args.ball_model), ("model_sha256", rig.path)):
                    if feedback[key].item() != sha256(path):
                        raise ValueError("Feedback provenance mismatch: " + key)
                if not np.array_equal(feedback["actuator_indices"], legs):
                    raise ValueError("Feedback actuator order mismatch")
                with torch.no_grad():
                    network.velocity.weight[legs] = torch.from_numpy(feedback["gains"].copy())
        actor = StandPolicy(agent).eval()
        quiet_metrics, _ = evaluate_retention(quiet, actor)
        metrics, _ = evaluate(native, actor, selection_cases(2., 12., native.dt))
        metrics.update(quiet_standing_passed=bool(quiet_metrics["passed"]), quiet_native=quiet_metrics)
        accepted = best.consider(agent.save, metrics, iteration=iteration, samples=0)
        row = dict(iteration=iteration, metrics=metrics, accepted=accepted)
        history.append(row)
        (args.out / "history.json").write_text(json.dumps(history, indent=2))
        agent.save(str(args.out / ("initial.pt" if iteration == 0 else "model.pt")))
        print(json.dumps(dict(iteration=iteration, quiet=quiet_metrics["passed"],
                              recovered=metrics["recovered_hits"], accepted=accepted)), flush=True)
    (args.out / "report.json").write_text(json.dumps(dict(schema="mimic_gated_recovery_seed_report_v1",
        selected_checkpoint=best.metadata, history=history, godot_selected=False,
        scope="Training seed; long quiet and unseen/repeated admission remain required"), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mimickit", "checkpoint", "reference", "ball-model", "feedback", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--capture-observation", action="store_true")
    run(parser.parse_args())
