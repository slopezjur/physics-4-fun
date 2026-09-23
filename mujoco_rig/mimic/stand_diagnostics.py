"""Matched quiet-standing checks, including reference-relative foot motion."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .baseline import sha256
from .checkpoints import validate_contract
from .contact_contract import mode_from_contract
from .evaluate_stand import evaluate
from .native_engine import NativeDummyEngine
from .rig import Rig
from .runtime import activate


def run(args):
    activate(args.mimickit)
    from .stand_task import StandTask
    torch.set_num_threads(1)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    reference_hash = None
    for bundle in args.bundle:
        contract = json.loads((bundle / "contract.json").read_text())
        if sha256(bundle / "stand.onnx") != contract["actor_sha256"]:
            raise ValueError("Actor hash mismatch")
        task = StandTask(rig, bundle / "stand_reference.npz", 8, "cpu",
                         engine_factory=NativeDummyEngine, control_mode="target_pd",
                         contact_mode=mode_from_contract(contract))
        validate_contract(contract, task.observation_contract())
        if reference_hash is not None and task.reference.sha256 != reference_hash:
            raise ValueError("Matched standing comparison requires the same reference")
        reference_hash = task.reference.sha256
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(bundle / "stand.onnx"), options, providers=["CPUExecutionProvider"])
        actor = lambda obs: torch.from_numpy(session.run(None, {"observation": obs.numpy()})[0])
        row = dict(bundle=str(bundle), contract_sha256=sha256(bundle / "contract.json"),
                   actor_sha256=contract["actor_sha256"], contact_mode=task.contact_mode, trials=[])
        for seconds in (3., 5.):
            task.episode_seconds = seconds
            metrics, traces = evaluate(task, actor, capture_actions=True)
            cases = []
            for case, duration in enumerate(metrics["survival_seconds"]):
                count = round(duration / task.dt)
                poses = torch.tensor(traces["qpos_after"][:count, case])
                feet = task.mapper.from_native_qpos(poses)[0][:, task.feet].numpy()
                phase = case / 7 * (task.reference.duration - seconds)
                target, _ = task.reference.sample(phase + torch.arange(1, count + 1) * task.dt)
                reference_feet = task.mapper.from_native_qpos(target)[0][:, task.feet].numpy()
                error = feet - reference_feet
                velocity_error = np.diff(error, axis=0) / task.dt
                actions = traces["actions"][:count, case]
                cases.append(dict(phase=phase, survival_seconds=duration,
                    foot_tracking_rmse_m=float(np.sqrt(np.mean(np.sum(error**2, axis=-1)))),
                    foot_velocity_error_rms_m_s=float(np.sqrt(np.mean(np.sum(velocity_error**2, axis=-1)))) if count > 1 else None,
                    mean_absolute_action_change=float(np.abs(np.diff(actions, axis=0)).mean()) if count > 1 else None,
                    foot_travel_m=np.linalg.norm(np.diff(feet, axis=0), axis=-1).sum(axis=0).tolist()))
            row["trials"].append(dict(seconds=seconds, metrics=metrics, cases=cases))
        results.append(row)
        print(json.dumps(row), flush=True)
    report = dict(schema="mimic_stand_diagnostics_v1", model_sha256=sha256(rig.path),
                  reference_sha256=reference_hash, results=results,
                  scope="Eight fixed phases at three and five seconds. Foot velocity error is a movement proxy, not a perceptual realism score; terminated episodes exclude reset frames. No training or viewer changes.")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
