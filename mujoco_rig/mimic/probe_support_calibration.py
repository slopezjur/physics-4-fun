"""Fit only the support critic on held-out quiet and entry worlds."""
import argparse
import json
from pathlib import Path

import torch
import yaml

from .baseline import sha256
from .critic_calibration import CalibrationProtocol, calibrate
from .probe_steps import validate_reference_substitution
from .rig import Rig
from .runtime import activate
from .support_transition import TRACKING_REWARD_SCHEMA


def run(args):
    if (args.out.exists() or args.envs < 128 or args.envs % 32
            or args.steps < 1 or not 0 < args.quiet_fraction < 1):
        raise ValueError("Require a new output, at least 128 worlds and valid calibration settings")
    activate(args.mimickit)
    from util import mp_util
    from .guarded_ppo import GuardedPPO
    from .newton_adapter import DummyNewtonEngine
    from .support_rehearsal_task import SupportRehearsalTask

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    mp_util.init(0, 1, "cuda:0", None)
    config = yaml.safe_load(args.agent_config.read_text())
    source_contract = json.loads(args.source_contract.read_text())
    if args.initialize_from.name == "best.pt":
        best = json.loads(args.initialize_from.with_name("best.json").read_text())
        if best["checkpoint_sha256"] != sha256(args.initialize_from):
            raise ValueError("Source checkpoint hash mismatch")
        validate_reference_substitution(source_contract, best["contract"])
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    task = SupportRehearsalTask(
        rig, args.reference, Path(__file__).parent / "assets/stand_reference.npz",
        args.envs, "cuda:0", quiet_fraction=args.quiet_fraction,
        support_seconds=12., quiet_seconds=3., reward_mode=TRACKING_REWARD_SCHEMA,
        seed=args.seed, engine_factory=DummyNewtonEngine)
    validate_reference_substitution(source_contract, task.observation_contract())
    agent = GuardedPPO(config, task, "cuda:0")
    agent.load(str(args.initialize_from))
    if agent._obs_norm.get_count().item() <= 0:
        raise ValueError("Require source observation normalization")
    protocol = CalibrationProtocol(
        (task.ids >= task.quiet_worlds).long(), ("quiet", "support"),
        task.ids % 4 == 3, 12., task.experiment_contract(),
        "Every fourth world validates each fixed quiet/support group")
    args.out.mkdir(parents=True, exist_ok=False)
    provenance = dict(schema="mimic_support_critic_probe_v1",
                      source_checkpoint_sha256=sha256(args.initialize_from),
                      source_contract_sha256=sha256(args.source_contract),
                      agent_config_sha256=sha256(args.agent_config),
                      model_sha256=sha256(rig.path),
                      envs=args.envs, critic_steps=args.steps, seed=args.seed,
                      task=task.experiment_contract(),
                      scope="Frozen actor and normalization; critic-only calibration with separate validation worlds; no export or viewer selection")
    (args.out / "protocol.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    result = calibrate(agent, task, args.steps, args.out, protocol=protocol)
    if result["passed"]:
        checkpoint = args.out / "calibrated.pt"
        agent.save(str(checkpoint))
        provenance["calibrated_checkpoint_sha256"] = sha256(checkpoint)
        (args.out / "protocol.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(dict(passed=result["passed"], before=result["before"],
                          best=result["best"])), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=250925)
    parser.add_argument("--quiet-fraction", type=float, default=.25)
    run(parser.parse_args())
