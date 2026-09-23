"""Explicit frozen-actor contact ablation; never a checkpoint migration or promotion."""
import argparse
import json
from pathlib import Path
import shutil

import onnxruntime as ort
import torch

from .baseline import sha256
from .checkpoints import validate_contract, ball_selection_groups
from .contact_contract import mode_from_contract
from .evaluate_stand import evaluate
from .native_engine import NativeDummyEngine
from .rig import Rig
from .runtime import activate


def compare(args):
    activate(args.mimickit)
    from .stand_task import StandTask
    from .ball_task import BallTask
    torch.set_num_threads(1)
    source = json.loads((args.bundle / "contract.json").read_text())
    if mode_from_contract(source) != "legacy" or source["action_mode"] != "reference_relative_target_pd":
        raise ValueError("The comparison requires a legacy target-PD baseline")
    if sha256(args.bundle / "stand.onnx") != source["actor_sha256"]:
        raise ValueError("Actor hash mismatch")
    args.out.mkdir(parents=True, exist_ok=False)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    reference = args.bundle / "stand_reference.npz"
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.bundle / "stand.onnx"), options, providers=["CPUExecutionProvider"])
    actor = lambda obs: torch.from_numpy(session.run(None, {"observation": obs.numpy()})[0])
    rows = []
    for mode in ("legacy", "timing_only", "support_v2"):
        contact_mode = "support_v2" if mode == "support_v2" else "legacy"
        ball = BallTask(rig, reference, 96, "cpu", speed_min=1., speed_max=2.5, contact_mode=contact_mode)
        quiet = StandTask(rig, reference, 8, "cpu", engine_factory=NativeDummyEngine,
                          control_mode="target_pd", contact_mode=contact_mode)
        if mode == "legacy":
            validate_contract(source, ball.observation_contract())
        if mode == "timing_only":
            ball.engine.use_last_solve_contacts()
            quiet.engine.use_last_solve_contacts()
        metrics, _ = evaluate(ball, actor)
        quiet3, _ = evaluate(quiet, actor)
        quiet.episode_seconds = 5.
        quiet5, _ = evaluate(quiet, actor)
        row = dict(mode=mode, ball=metrics, direction_groups=ball_selection_groups(metrics)[1],
                   quiet3=quiet3, quiet5=quiet5)
        rows.append(row)
        if mode != "timing_only":
            # Hash-identical weights AND normalizer, with the semantic intervention
            # made explicit. This diagnostic bundle cannot seed training or selection.
            bundle = args.out / mode / "export"
            shutil.copytree(args.bundle, bundle)
            contract = {**source, **ball.observation_contract(), "diagnostic_only": True,
                        "contact_ablation": dict(source_contract_sha256=sha256(args.bundle / "contract.json"),
                                                 mode=mode, weights_and_normalization="unchanged; zero-shot diagnostic")}
            (bundle / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
        (args.out / f"{mode}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(json.dumps(dict(mode=mode, survived=metrics["survived_hits"], recovered=metrics["recovered_hits"],
                              quiet3=quiet3["successes"], quiet5=quiet5["successes"])), flush=True)
    report = dict(schema="mimic_contact_ablation_v1", actor_sha256=source["actor_sha256"],
                  source_contract_sha256=sha256(args.bundle / "contract.json"), results=rows,
                  scope="Same frozen actor and normalization on the same 96 validation impacts and 8 quiet phases. V2 is an explicit zero-shot input/timing intervention, not a trained v2 policy or evidence of better learning. No viewer selection.")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    compare(parser.parse_args())
