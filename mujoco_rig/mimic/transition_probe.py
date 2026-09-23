"""Isolate a short reference transition without admitting a cropped motion to training."""
import argparse
import json
from pathlib import Path

import numpy as np

from .baseline import sha256
from .force_retarget import refine_support, reference_velocities
from .motion_contacts import landing_seed_for_window
from .retarget_steps import validate_motion
from .rig import Rig


def probe(reference, out, start, stop, max_evaluations=20, linear_solver="lsmr", adjust_landings=False):
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    metadata = json.loads(reference.with_suffix(".json").read_text())
    source_hash = sha256(reference)
    if metadata["model_sha256"] != sha256(rig.path) or metadata["reference_sha256"] != source_hash:
        raise ValueError("Reference or rig hash mismatch")
    if metadata["source"]["license"] != "CC-BY-4.0":
        raise ValueError("Expected the vetted CC-BY-4.0 source")
    if out.exists():
        raise FileExistsError(out)
    with np.load(reference, allow_pickle=False) as arrays:
        if not 0 <= start < stop <= len(arrays["qpos"]) or stop - start < 5:
            raise ValueError("Select at least five source frames with an exclusive stop index")
        poses, targets, contact = (arrays[key][start:stop] for key in ("qpos", "foot_targets", "contact"))
        dt = float(arrays["dt"])
        soles = arrays["sole_contact"][start:stop] if "sole_contact" in arrays else None
        seed = landing_seed_for_window(arrays["contact"], metadata.get("force_fit", {}).get("stance_anchor_offsets_xy"), start, stop)
    clearance = metadata.get("force_fit", {}).get("self_clearance_m", 0.)

    def assess(candidate):
        checks, metrics = validate_motion(rig, candidate, reference_velocities(rig.model, candidate, dt),
                                          targets, contact, dt, soles, clearance)
        return dict(checks=checks, metrics=metrics,
                    root_failure_source_frames=[start + i for i in metrics["root_wrench_consistency"]["infeasible_frames"]],
                    pd_failure_source_frames=[start + i for i in metrics["action_window_consistency"]["failing_frames"]["0.25"]])

    before = assess(poses)
    fitted, targets, contact, fit = refine_support(
        rig, poses, targets, contact, dt, max_evaluations=max_evaluations,
        root_seed_drop_m=0. if metadata.get("force_fit", {}).get("stance_method") else .02,
        source_soles=soles, self_clearance_m=clearance, adjust_landings=adjust_landings,
        landing_seed=seed, linear_solver=linear_solver)
    report = dict(schema="mimic_transition_probe_v1", source_reference_sha256=source_hash,
                  model_sha256=sha256(rig.path), source_frame_start=start, source_frame_stop_exclusive=stop,
                  baseline=before, candidate=assess(fitted), fit=fit, admitted_for_training=False,
                  scope="Free-endpoint short-window diagnostic. Derivatives and command history are local to the window; surrounding-trajectory continuity and closed-loop tracking are untested. No training manifest or policy export is produced.")
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "diagnostic_poses.npz", qpos=fitted, dt=dt)
    report["diagnostic_sha256"] = sha256(out / "diagnostic_poses.npz")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(checks=report["candidate"]["checks"],
                          root_failure_source_frames=report["candidate"]["root_failure_source_frames"],
                          pd_failure_source_frames=report["candidate"]["pd_failure_source_frames"],
                          seconds=fit["seconds"], admitted_for_training=False)), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--max-evaluations", type=int, default=20)
    parser.add_argument("--linear-solver", choices=("lsmr", "exact"), default="lsmr")
    parser.add_argument("--adjust-landings", action="store_true")
    args = parser.parse_args()
    probe(args.reference, args.out, args.start, args.stop, args.max_evaluations, args.linear_solver, args.adjust_landings)
