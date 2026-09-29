"""Bind training, quiet-standing, and unseen-hit evidence for the next mass stage."""
import argparse
import json
from pathlib import Path

from .baseline import sha256
from .validate_stand_foundation import case_passed


def finalize(run: Path, quiet_report: Path, unseen_report: Path,
             stress_report: Path, gpu_report: Path):
    report_path = run / "report.json"
    report = json.loads(report_path.read_text())
    quiet = json.loads(quiet_report.read_text())
    unseen = json.loads(unseen_report.read_text())
    stress = json.loads(stress_report.read_text())
    gpu = json.loads(gpu_report.read_text())
    best = json.loads((run / "best.json").read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    digest = sha256(run / "best.pt")
    if (best["checkpoint_sha256"] != digest
            or report["selected_checkpoint"]["checkpoint_sha256"] != digest
            or quiet["checkpoint_sha256"] != digest
            or unseen["checkpoint_sha256"] != digest
            or stress["checkpoint_sha256"] != digest):
        raise ValueError("Evidence must name one selected checkpoint")
    if (gpu.get("checkpoint_sha256") != digest or gpu.get("ball_model_sha256") !=
            protocol["experiment"]["ball_model_sha256"] or gpu.get("confirmed_hits") != 8
            or gpu.get("recovered_hits") != 8):
        raise ValueError("GPU physical-hit parity is incomplete")
    if (not report["quiet_standing_retained"] or not report["selected_gpu_quiet"]["passed"]
            or protocol["experiment"]["ball_mass_kg"] != 2.
            or stress.get("ball_mass_kg") != 5.
            or best["metrics"]["recovered_hits"] != 32
            or not quiet["passed"] or quiet["seconds"] < 600.
            or len(quiet["cases"]) != 8
            or any(not case_passed(case, quiet["seconds"]) for case in quiet["cases"])
            or unseen["schema"] != "mimic_reactive_ball_unseen_validation_v1"
            or not unseen["passed"] or unseen["seconds"] < 33.
            or unseen["episodes"] < 64 or unseen["recovered_hits"] != unseen["episodes"]
            or unseen["ball_model_sha256"] != protocol["experiment"]["ball_model_sha256"]
            or unseen["reference_sha256"] != protocol["experiment"]["reference_sha256"]
            or quiet["reference_sha256"] != unseen["reference_sha256"]):
        raise ValueError("The 2 kg training seed has not passed its measured gates")
    original = report.get("selected_unseen")
    if original and original.get("recovered_hits") == 0 and all(
            case["survival_seconds"] >= 12. for case in original["cases"]):
        report["reported_unseen_before_timeout_fix"] = original
        report["validation_correction"] = dict(
            issue="Float32 final-step time was below a Python-float episode endpoint",
            previous_result="64 confirmed hits, 0 recorded timeouts",
            correction="Timeout placed halfway through the final control interval",
            prior_report=str(run / "report.pre-timeout-fix.json"),
            validation_runner=str(unseen_report.parent / "runner.py"))
    report["selected_unseen"] = unseen
    report["unseen_hits_complete"] = True
    report["quiet_long"] = dict(path=str(quiet_report), sha256=sha256(quiet_report))
    report["unseen_long"] = dict(path=str(unseen_report), sha256=sha256(unseen_report))
    report["ready_as_2kg_perturb_seed"] = True
    report["godot_selected"] = False
    report_path.write_text(json.dumps(report, indent=2))
    (run / "unseen_report.json").write_text(json.dumps(unseen, indent=2))
    manifest = dict(schema="mimic_reactive_ball_seed_manifest_v1",
        source_stand_checkpoint=protocol["source_checkpoint"],
        source_stand_checkpoint_sha256=protocol["source_checkpoint_sha256"],
        selected_checkpoint=str(run / "best.pt"), selected_checkpoint_sha256=digest,
        reference_sha256=unseen["reference_sha256"],
        character_model_sha256=unseen["character_model_sha256"],
        ball_model_sha256=unseen["ball_model_sha256"], ball_mass_kg=2.,
        selection_recovered=best["metrics"]["recovered_hits"],
        quiet_30_native=best["metrics"]["quiet_standing_passed"],
        quiet_30_gpu=report["selected_gpu_quiet"]["passed"],
        quiet_600_native=quiet["passed"], quiet_600_report_sha256=sha256(quiet_report),
        unseen_33_confirmed=unseen["confirmed_hits"],
        unseen_33_recovered=unseen["recovered_hits"],
        unseen_33_report_sha256=sha256(unseen_report),
        five_kg_stress_recovered=stress["recovered_hits"],
        gpu_front_confirmed=gpu["confirmed_hits"], gpu_front_recovered=gpu["recovered_hits"],
        gpu_front_report_sha256=sha256(gpu_report),
        quiet_max_final_minute_foot_motion_m=max(c["tail_foot_motion_m"] for c in quiet["cases"]),
        five_kg_stress_report_sha256=sha256(stress_report),
        ready_for_next_mass_training=True, godot_selected=False,
        remaining="Train on heavier randomized hits; recheck long quiet standing and new unseen hits before Godot")
    (run / "seed_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--quiet-report", type=Path, required=True)
    parser.add_argument("--unseen-report", type=Path, required=True)
    parser.add_argument("--stress-report", type=Path, required=True)
    parser.add_argument("--gpu-report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.run, args.quiet_report, args.unseen_report,
                              args.stress_report, args.gpu_report)), flush=True)
