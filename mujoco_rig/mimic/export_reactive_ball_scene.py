"""Export a stationary actor, with explicit experimental preview support."""
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
from .continuous_stand import SEMANTICS
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate
from .validate_stand_foundation import case_passed


def require_evidence(args, checkpoint_hash, reference_hash, character_hash, ball_hash):
    run = args.checkpoint.parent
    selection = json.loads((run / "best.json").read_text())
    training = json.loads((run / "report.json").read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    quiet = json.loads(args.quiet_report.read_text())
    unseen = json.loads(args.unseen_report.read_text())
    metrics = selection["metrics"]
    if (selection["checkpoint_sha256"] != checkpoint_hash
        or metrics["confirmed_hits"] != 32 or metrics["recovered_hits"] != 32
        or metrics["quiet_standing_passed"] is not True):
        raise ValueError("Selected actor lacks complete contact and quiet gates")
    if training["selected_checkpoint"]["checkpoint_sha256"] != checkpoint_hash or (
        training["selected_gpu_quiet"]["passed"] is not True
    ):
        raise ValueError("Training backend quiet-standing gate failed")
    experiment = protocol["experiment"]
    if (experiment["ball_model_sha256"] != ball_hash or experiment["ball_mass_kg"] <= 0
        or experiment["reference_sha256"] != reference_hash
        or protocol["contract"]["model_sha256"] != character_hash):
        raise ValueError("Training protocol does not describe this scene plant")
    if (quiet.get("schema") != "mimic_standing_foundation_validation_v1"
        or quiet.get("checkpoint_sha256") != checkpoint_hash
        or quiet.get("reference_sha256") != reference_hash
        or quiet.get("model_sha256") != character_hash
        or quiet.get("seconds", 0) < 600 or len(quiet.get("cases", [])) != 8
        or not all(case_passed(case, quiet["seconds"]) for case in quiet["cases"])):
        raise ValueError("Selected actor lacks eight uninterrupted 600-second standing passes")
    if (unseen.get("schema") != "mimic_reactive_ball_unseen_validation_v1"
        or unseen.get("checkpoint_sha256") != checkpoint_hash
        or unseen.get("reference_sha256") != reference_hash
        or unseen.get("character_model_sha256") != character_hash
        or unseen.get("ball_model_sha256") != ball_hash
        or unseen.get("seconds", 0) < 30
        or any(not 1. <= case["speed"] <= args.scene_speed + 1e-6
               for case in unseen.get("cases", []))
        or min(unseen.get("confirmed_hits", 0), unseen.get("survived_hits", 0),
               unseen.get("recovered_hits", 0)) < 64
        or unseen.get("passed") is not True):
        raise ValueError("Selected actor lacks 64 unseen long-horizon scene-mass recoveries")
    return selection, training, quiet, unseen, experiment["ball_mass_kg"]


def run(args):
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .balance_observation import task_classes
    contract = json.loads((args.checkpoint.parent / "contract.json").read_text())
    _, ReactiveBallTask = task_classes(contract)

    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    checkpoint_hash = sha256(args.checkpoint)
    reference_hash = sha256(args.reference)
    character_hash = sha256(rig.path)
    ball_hash = sha256(args.ball_model)
    report_path = args.checkpoint.parent / "report.json"
    assessment_path = args.checkpoint.parent / "learning_curve.json"
    pending_validation = False
    if args.experimental:
        if args.checkpoint.name != "model.pt":
            raise ValueError("Experimental preview requires the run's final model.pt")
        protocol = json.loads((args.checkpoint.parent / "protocol.json").read_text())
        if report_path.exists():
            training = json.loads(report_path.read_text())
        elif args.allow_pending_validation:
            assessment = json.loads(assessment_path.read_text())[-1]
            update = json.loads((args.checkpoint.parent / "updates.jsonl").read_text().splitlines()[-1])
            if (assessment["iteration"] != update["iteration"]
                    or update["training_iteration_wall_seconds"] < protocol["seconds"]
                    or assessment["metrics"]["quiet_standing_passed"] is not True
                    or assessment["metrics"]["confirmed_hits"] != 32
                    or args.checkpoint.stat().st_mtime > assessment_path.stat().st_mtime):
                raise ValueError("Pending preview requires completed training and final native assessment")
            training = dict(final=assessment["metrics"])
            pending_validation = True
        else:
            raise ValueError("Experimental preview needs a completed report or explicit pending-validation option")
        experiment = protocol["experiment"]
        if (experiment["ball_model_sha256"] != ball_hash
                or experiment["reference_sha256"] != reference_hash
                or protocol["contract"]["model_sha256"] != character_hash):
            raise ValueError("Preview plant or reference differs from the training protocol")
        selection = {"metrics": training["final"]}
        ball_mass = experiment["ball_mass_kg"]
        unseen = None
    else:
        if args.quiet_report is None or args.unseen_report is None:
            raise ValueError("Admitted export requires quiet and unseen validation reports")
        selection, training, quiet, unseen, ball_mass = require_evidence(
            args, checkpoint_hash, reference_hash, character_hash, ball_hash)
    task = ReactiveBallTask(rig, args.reference, 8, "cpu", ball_path=args.ball_model)
    validate_contract(contract, task.observation_contract())
    if contract["reference_semantics"] != SEMANTICS:
        raise ValueError("Scene requires the stationary zero-phase contract")
    agent = PPOAgent(yaml.safe_load((args.checkpoint.parent / "agent.yaml").read_text()), task, "cpu")
    agent.load(str(args.checkpoint))
    actor = StandPolicy(agent).eval()
    sample = task.observations().detach().cpu()
    sample = torch.cat((sample, sample.repeat(32, 1) + .001 * torch.randn(256, sample.shape[1])), 0)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.onnx.export(actor, sample[:1], str(args.out / "stand.onnx"),
                      input_names=["observation"], output_names=["action"],
                      dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
                      opset_version=17, dynamo=False)
    session = ort.InferenceSession(str(args.out / "stand.onnx"), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        error = float(np.abs(session.run(None, {"observation": sample.numpy()})[0]
                             - actor(sample).numpy()).max())
    if not np.isfinite(error) or error > 1e-5:
        raise RuntimeError(f"ONNX actor parity failed: {error}")
    with np.load(args.reference, allow_pickle=False) as reference:
        pose = reference["qpos"][args.posture].astype(np.float32).tolist()
        velocity = reference["qvel"][args.posture].astype(np.float32).tolist()
    if any(velocity):
        raise ValueError("Stationary reference velocity must be zero")
    frames = dict(dt=60., reference_sha256=reference_hash,
                  qpos=[pose, pose], qvel=[velocity, velocity])
    (args.out / "reference_frames.json").write_text(json.dumps(frames))
    for source, name in ((args.reference, "stand_reference.npz"),
                         (args.reference.with_suffix(".json"), "stand_reference.json"),
                         (args.reference.parent / "ATTRIBUTION.md", "ATTRIBUTION.md"),
                         (rig.path, "dummy.xml"),
                         (args.ball_model, "dummy_ball.xml")):
        shutil.copy2(source, args.out / name)
    contract.update(actor_sha256=sha256(args.out / "stand.onnx"),
                    reference_frames_sha256=sha256(args.out / "reference_frames.json"),
                    reference_duration=60., scene_posture=args.posture,
                    source_checkpoint_sha256=checkpoint_hash,
                    deployment_status=("Experimental final actor preview; recovery admission incomplete"
                        if args.experimental else
                        f"Native validated stationary {ball_mass:g} kg recovery to {args.scene_speed:g} m/s"))
    (args.out / "contract.json").write_text(json.dumps(contract, indent=2))
    evidence = dict(schema="mimic_reactive_ball_scene_export_v1",
                    checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
                    ball_model_sha256=ball_hash, ball_mass_kg=ball_mass,
                    reference_sha256=reference_hash,
                    character_model_sha256=character_hash,
                    selection_speed_m_s=selection["metrics"]["cases"][0]["speed"],
                    selection_recovered=None if args.experimental else selection["metrics"]["recovered_hits"],
                    experimental_preview=args.experimental,
                    candidate_recovered=selection["metrics"]["recovered_hits"],
                    scene_ball_speed_m_s=args.scene_speed,
                    quiet_600_report_sha256=None if args.experimental else sha256(args.quiet_report),
                    unseen_scene_mass_report_sha256=None if args.experimental else sha256(args.unseen_report),
                    unseen_recovered=None if unseen is None else unseen["recovered_hits"],
                    onnx_max_action_error=error, scene_posture=args.posture,
                    contract_sha256=sha256(args.out / "contract.json"),
                    pending_native_validation=pending_validation,
                    training_assessment_sha256=sha256(assessment_path) if pending_validation else None,
                    training_report_sha256=None if pending_validation else sha256(report_path))
    (args.out / "validation.json").write_text(json.dumps(evidence, indent=2))
    if checkpoint_hash != sha256(args.checkpoint):
        raise RuntimeError("Checkpoint changed during export")
    print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--ball-model", type=Path, required=True)
    parser.add_argument("--quiet-report", type=Path)
    parser.add_argument("--unseen-report", type=Path)
    parser.add_argument("--experimental", action="store_true",
                        help="Export final model.pt for manual Arena inspection without claiming admission")
    parser.add_argument("--allow-pending-validation", action="store_true",
                        help="Experimental preview after completed training/final native assessment while long checks finish")
    parser.add_argument("--posture", type=int, choices=range(8), default=0)
    parser.add_argument("--scene-speed", type=float, default=1.)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
