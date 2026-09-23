"""Recover a selected motion export from saved weights without rerunning training."""
import argparse
import json
from pathlib import Path

import torch
import yaml

from .baseline import sha256
from .checkpoints import validate_contract
from .evaluate_stand import evaluate
from .export_policy import export_policy
from .motion_evaluation import tracking_metrics
from .native_engine import NativeDummyEngine
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate
from .contact_contract import mode_from_contract


def export(args):
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .stand_task import StandTask
    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    metadata = json.loads((args.run / "best.json").read_text())
    protocol = json.loads((args.run / "protocol.json").read_text())
    if protocol["experiment"]["task"] != "motion" or sha256(args.run / "best.pt") != metadata["checkpoint_sha256"]:
        raise ValueError("Requires an intact selected motion checkpoint")
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    task = StandTask(rig, args.reference, 8, "cpu", engine_factory=NativeDummyEngine,
                     control_mode="target_pd", contact_mode=mode_from_contract(metadata["contract"]))
    validate_contract(metadata["contract"], task.observation_contract())
    agent = PPOAgent(yaml.safe_load((args.run / "agent.yaml").read_text()), task, "cpu")
    agent.load(str(args.run / "best.pt"))
    actor = StandPolicy(agent).eval()
    metrics, traces = evaluate(task, actor)
    metrics["motion_tracking"] = tracking_metrics(task, metrics, traces)
    error = export_policy(actor, traces["observations"], args.reference, task.observation_contract(), args.out)
    (args.out / "experiment.json").write_text(json.dumps(protocol["experiment"], indent=2), encoding="utf-8")
    report = dict(schema="mimic_motion_checkpoint_export_v1", checkpoint_sha256=metadata["checkpoint_sha256"],
                  iteration=metadata["iteration"], samples=metadata["samples"], metrics=metrics,
                  onnx_max_action_error=error, scope="Reevaluated/exported saved weights; no training or viewer changes")
    (args.out / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(iteration=metadata["iteration"], onnx_error=error,
                         survived=metrics["successes"], mean_survival_seconds=metrics["mean_survival_seconds"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    export(parser.parse_args())
