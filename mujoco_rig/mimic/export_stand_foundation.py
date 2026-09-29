"""Export a stationary actor only after matching long-hold and unseen-start gates."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import onnxruntime as ort
import torch
import yaml

from .baseline import sha256
from .checkpoints import validate_contract
from .native_engine import NativeDummyEngine
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate
from .validate_stand_foundation import case_passed


def validate_evidence(report, checkpoint_hash, reference_hash, model_hash, randomized):
    if (report.get("schema") != "mimic_standing_foundation_validation_v1"
            or report.get("checkpoint_sha256") != checkpoint_hash
            or report.get("reference_sha256") != reference_hash
            or report.get("model_sha256") != model_hash or report.get("passed") is not True):
        raise ValueError("Validation evidence does not match this actor, reference and plant")
    cases = report.get("cases", [])
    required_seconds, required_cases = (60., 64) if randomized else (600., 8)
    if (report["seconds"] < required_seconds or len(cases) < required_cases
            or sorted({c["posture"] for c in cases}) != list(range(8))
            or any(not case_passed(c, report["seconds"]) for c in cases)
            or (report["velocity_noise_m_s"] < .04 if randomized else report["velocity_noise_m_s"] != 0)):
        raise ValueError("Incomplete long-hold or unseen-start gate")


def run(args):
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .continuous_stand_task import ContinuousStandTask
    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    digest, reference_hash, model_hash = sha256(args.checkpoint), sha256(args.reference), sha256(rig.path)
    training = json.loads((args.checkpoint.parent / "report.json").read_text())
    if (training.get("full_gate_passed") is not True
            or training.get("selected_checkpoint", {}).get("checkpoint_sha256") != digest):
        raise ValueError("Require the selected checkpoint to pass both native and training-backend standing gates")
    evidence = {}
    for label, path in (("long_hold", args.long_report), ("unseen_starts", args.random_report)):
        report = json.loads(path.read_text())
        validate_evidence(report, digest, reference_hash, model_hash, label == "unseen_starts")
        evidence[label] = dict(path=str(path), sha256=sha256(path), cases=len(report["cases"]), seconds=report["seconds"])
    task = ContinuousStandTask(rig, args.reference, engine_factory=NativeDummyEngine)
    contract = json.loads((args.checkpoint.parent / "contract.json").read_text())
    validate_contract(contract, task.observation_contract())
    agent = PPOAgent(yaml.safe_load((args.checkpoint.parent / "agent.yaml").read_text()), task, "cpu")
    agent.load(str(args.checkpoint))
    actor = StandPolicy(agent).eval()
    with np.load(args.checkpoint.parent / "selected-native.npz") as traces:
        observations = traces["observations"].reshape(-1, contract["num_obs"])
        observations = observations[np.linspace(0, len(observations)-1, min(2048, len(observations)), dtype=int)]
    args.out.mkdir(parents=True, exist_ok=False)
    sample = torch.from_numpy(observations)
    path = args.out / "stand.onnx"
    torch.onnx.export(actor, sample[:1], str(path), input_names=["observation"], output_names=["action"],
                      dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
                      opset_version=17, dynamo=False)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        error = float(np.max(np.abs(session.run(None, {"observation": observations})[0] - actor(sample).numpy())))
    if not np.isfinite(error) or error > 1e-5:
        raise RuntimeError(f"ONNX action parity failed: {error}")
    contract.update(actor_sha256=sha256(path), source_checkpoint_sha256=digest,
                    deployment_status="Validated native stationary Stand foundation; requires stationary runtime; no impact admission")
    for source, name in ((args.reference, "stationary_reference.npz"),
                         (args.reference.with_suffix(".json"), "stationary_reference.json"),
                         (args.reference.parent / "ATTRIBUTION.md", "ATTRIBUTION.md")):
        shutil.copy2(source, args.out / name)
    (args.out / "contract.json").write_text(json.dumps(contract, indent=2))
    result = dict(schema="mimic_stand_foundation_export_v1", checkpoint=str(args.checkpoint),
                  checkpoint_sha256=digest, contract_sha256=sha256(args.out / "contract.json"),
                  evidence=evidence, onnx_max_action_error=error,
                  godot_selected=False, stationary_stand_gate_passed=True,
                  readiness="Stand seed for a matching stationary Perturb task; physical-hit validation still required",
                  impact_validation_passed=False)
    (args.out / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--long-report", type=Path, required=True)
    parser.add_argument("--random-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
