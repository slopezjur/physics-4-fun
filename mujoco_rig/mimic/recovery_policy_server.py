"""Local line-protocol adapter for the exact native recovery teacher plus residual."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from .baseline import sha256
from .distill_recovery_step import StepStudent
from .observation_step_teacher import ObservationStepTeacher
from .rig import Rig
from .train_recovery_residual import ResidualController


def load(bundle):
    manifest = json.loads((bundle / "recovery.json").read_text(encoding="utf-8"))
    if manifest["schema"] != "mimic_recovery_scene_bundle_v1":
        raise ValueError("Unknown recovery scene bundle")
    for name, key in (("training_report.json", "training_report_sha256"),
                      ("recovery.pt", "checkpoint_sha256"), ("motion.npz", "motion_sha256"),
                      ("source.json", "source_sha256"), ("search.json", "search_sha256"),
                      ("contract.json", "contract_sha256"), ("dummy.xml", "character_sha256"),
                      ("dummy_ball.xml", "ball_world_sha256")):
        if sha256(bundle / name) != manifest[key]:
            raise ValueError(f"Recovery bundle hash mismatch: {name}")
    report = json.loads((bundle / "training_report.json").read_text(encoding="utf-8"))
    if (not report["passed"] or report["checkpoint_sha256"] != manifest["checkpoint_sha256"]
            or manifest.get("trained_ball_mass_kg", .25) != .25
            or not .1 <= manifest["ball_mass_kg"] <= 8.):
        raise ValueError("Recovery bundle has no accepted selected checkpoint")
    rig = Rig.load(bundle / "dummy.xml")
    contract = json.loads((bundle / "contract.json").read_text(encoding="utf-8"))
    source = json.loads((bundle / "source.json").read_text(encoding="utf-8"))
    search = json.loads((bundle / "search.json").read_text(encoding="utf-8"))
    with np.load(bundle / "motion.npz", allow_pickle=False) as arrays:
        motion = {key: arrays[key].copy() for key in arrays.files}
    cached_support = None
    if "support_schedule_sha256" in manifest:
        schedule_path = bundle / "support_schedule.npz"
        if sha256(schedule_path) != manifest["support_schedule_sha256"]:
            raise ValueError("Recovery support schedule hash mismatch")
        with np.load(schedule_path, allow_pickle=False) as arrays:
            cached_support = {key: arrays[key].copy() for key in ("times", "torques", "centers")}
    if (source["model_sha256"] != manifest["character_sha256"]
            or source["reference_sha256"] != manifest["motion_sha256"]
            or search["source_report_sha256"] != manifest["source_sha256"]
            or search["source_motion_sha256"] != manifest["motion_sha256"]):
        raise ValueError("Recovery teacher sources differ from the selected controller")
    settings = manifest["controller_settings"]
    teacher = ObservationStepTeacher(
        rig, contract, motion, source, search, rig.model.opt.timestep * rig.decimation,
        cached_support=cached_support, **settings)
    actor = StepStudent(bundle / "recovery.pt").eval()
    if len(actor.obs_mean) != 400 or len(actor.action_mean) != 30:
        raise ValueError("Recovery residual tensor contract mismatch")
    return ResidualController(teacher, actor, manifest["residual_scale"]), teacher


def serve(bundle):
    torch.set_num_threads(1)
    controller, teacher = load(bundle)
    trace_path = os.environ.get("P4F_RECOVERY_TRACE")
    trace = open(trace_path, "w", encoding="utf-8") if trace_path else None
    print("READY", flush=True)
    try:
        for line in sys.stdin:
            line = line.strip()
            if line == "RESET":
                teacher.reset()
                if trace:
                    trace.write('{"reset":true}\n')
                    trace.flush()
                print("OK", flush=True)
                continue
            row = np.asarray(json.loads(line), dtype=np.float32)
            if row.shape != (362,) or not np.isfinite(row).all():
                raise ValueError("Expected one finite support-v2 observation")
            action = controller(torch.from_numpy(row[None]))[0].numpy()
            if action.shape != (30,) or not np.isfinite(action).all():
                raise ValueError("Invalid recovery action")
            if trace:
                trace.write(json.dumps({"observation": row.tolist(), "action": action.tolist()}) + "\n")
                trace.flush()
            print(json.dumps(action.tolist(), separators=(",", ":")), flush=True)
    finally:
        if trace:
            trace.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    serve(parser.parse_args().bundle)
