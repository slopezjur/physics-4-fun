"""Export one selected checkpoint without changing the viewer's accepted bundle."""
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import onnxruntime as ort

from .godot_bundle import prepare


def export_policy(actor, observations, reference: Path, contract, bundle: Path):
    bundle.mkdir(parents=True, exist_ok=False)
    sample = torch.tensor(observations.reshape(-1, contract["num_obs"]))
    torch.onnx.export(actor, sample[:1], str(bundle / "stand.onnx"),
                      input_names=["observation"], output_names=["action"],
                      dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
                      opset_version=17, dynamo=False)
    session = ort.InferenceSession(str(bundle / "stand.onnx"), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        error = float(np.abs(session.run(None, {"observation": sample.numpy()})[0] - actor(sample).numpy()).max())
    if not np.isfinite(error) or error > 1e-5:
        raise RuntimeError(f"ONNX action mismatch: {error}")
    for source in (reference, reference.with_suffix(".json"), reference.parent / "ATTRIBUTION.md"):
        shutil.copy2(source, bundle / source.name)
    (bundle / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    prepare(bundle)
    return error
