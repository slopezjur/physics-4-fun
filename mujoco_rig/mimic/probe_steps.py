"""Measure motion tracking before attempting Perturb training on new step references."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .baseline import sha256
from .checkpoints import validate_contract
from .evaluate_stand import evaluate
from .native_engine import NativeDummyEngine
from .motion_evaluation import tracking_metrics
from .motion_protocol import GATES
from .rig import Rig
from .runtime import activate
from .contact_contract import mode_from_contract


def validate_reference_substitution(source, target):
    """This evaluation explicitly substitutes motion, while every control field must match."""
    adapted = {**source, **{key: target[key] for key in ("reference_sha256", "reference_duration")}}
    validate_contract(adapted, target)


def run(args):
    activate(args.mimickit)
    from .stand_task import StandTask
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    contract = json.loads((args.bundle / "contract.json").read_text())
    actor_hash = sha256(args.bundle / "stand.onnx")
    if actor_hash != contract["actor_sha256"]:
        raise ValueError("Actor hash mismatch")
    initializers = {v.name: onnx.numpy_helper.to_array(v) for v in onnx.load(args.bundle / "stand.onnx").graph.initializer}
    mean, std = initializers["obs_norm._mean"], initializers["obs_norm._std"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.bundle / "stand.onnx"), options, providers=["CPUExecutionProvider"])
    results = []
    for reference in args.reference:
        task = StandTask(rig, reference, 8, "cpu", engine_factory=NativeDummyEngine,
                         control_mode="target_pd", contact_mode=mode_from_contract(contract))
        validate_reference_substitution(contract, task.observation_contract())
        for mode in ("zero_residual", "standing_actor"):
            actor = (lambda obs: torch.zeros(len(obs), 30)) if mode == "zero_residual" else (
                lambda obs: torch.from_numpy(session.run(None, {"observation": obs.numpy()})[0]))
            metrics, traces = evaluate(task, actor)
            tracking = tracking_metrics(task, metrics, traces)
            observed = np.concatenate([traces["observations"][:round(seconds / task.dt), i]
                                       for i, seconds in enumerate(metrics["survival_seconds"])])
            clipped = np.abs((observed - mean) / std) > 10
            clipping = {group["name"]: float(clipped[:, group["offset"]:group["offset"]+group["width"]].mean())
                        for group in contract["observation_layout"]}
            row = dict(reference=str(reference), reference_sha256=task.reference.sha256, mode=mode,
                       metrics=metrics, tracking=tracking, source_normalizer_clipping_fraction=clipping)
            results.append(row)
            np.savez_compressed(args.out / f"probe-{len(results):02d}.npz", **traces)
            print(json.dumps(dict(reference=str(reference), mode=mode, survived=metrics["successes"],
                                  tracking_passed=tracking["passed"], mean_survival=metrics["mean_survival_seconds"])), flush=True)
    report = dict(schema="mimic_step_tracking_probe_v1", actor_sha256=actor_hash, results=results,
                  gates=GATES,
                  scope="Explicit reference substitution for feasibility diagnosis. No training/export/viewer changes. An old standing actor failing a new motion does not establish physical infeasibility.")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
