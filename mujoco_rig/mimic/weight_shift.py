"""Versioned two-foot weight-shift reference and native pilot gates."""
import argparse
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
import torch

from .baseline import sha256
from .force_retarget import reference_velocities
from .rig import Rig

SCHEMA = "mimic_weight_shift_reference_v1"
EVALUATION_SCHEMA = "mimic_weight_shift_tracking_v2"
REHEARSAL_SCHEMA = "mimic_weight_shift_rehearsal_v1"
GATES = dict(episodes=8, seconds=3., max_root_error_m=.03, max_foot_error_m=.03,
             max_joint_rmse_rad=.15, max_horizontal_displacement_rmse_m=.015,
             max_relative_displacement_error=.5, displacement_noise_floor_m=.002)
PHYSICAL_CHECKS = ("finite", "joint_limits", "penetration", "stance_slide", "ground_support",
                   "stance_contact", "foot_tracking", "joint_speed", "sole_contact",
                   "force_consistency", "root_wrench_consistency", "actuation_feasibility")


def validate_reference(metadata):
    if (metadata.get("schema") != SCHEMA or metadata.get("passed") is not True
            or any(metadata.get("checks", {}).get(k) is not True for k in PHYSICAL_CHECKS)
            or metadata.get("weight_shift_admission") != {"both_feet_planted": True, "horizontal_excursion": True}):
        raise ValueError("Weight-shift training requires its own physical and double-support admission")


def prepare(calibration, out):
    from .track_transfer import CalibrationReference
    from .retarget_steps import validate_motion
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    CalibrationReference(calibration, rig)  # Revalidate the parent without admitting it.
    metadata = json.loads((calibration / "report.json").read_text())
    with np.load(calibration / "diagnostic_poses.npz", allow_pickle=False) as arrays:
        dt = float(arrays["dt"])
        frames = round(metadata["phase_seconds"] / dt)
        if not np.isclose(frames * dt, metadata["phase_seconds"], atol=1e-10):
            raise ValueError("Phase boundaries must coincide with reference frames")
        pieces, targets = [], []
        for leg in range(2):
            start, stop = leg * 4 * frames, (leg * 4 + 1) * frames + 1
            if not arrays["contact"][start:stop].all() or not arrays["sole_contact"][start:stop].all():
                raise ValueError("Weight-shift source interval must keep every sole planted")
            for key, destination in (("qpos", pieces), ("foot_targets", targets)):
                segment = arrays[key][start:stop]
                loop = np.concatenate((segment, segment[-2::-1]))
                destination.append(loop if not leg else loop[1:])
        qpos, foot_targets = np.concatenate(pieces), np.concatenate(targets)
    qvel = reference_velocities(rig.model, qpos, dt)
    contact = np.ones((len(qpos), 2), dtype=bool)
    soles = np.ones((len(qpos), 4), dtype=bool)
    checks, metrics = validate_motion(rig, qpos, qvel, foot_targets, contact, dt, soles)
    admission = dict(both_feet_planted=max(metrics["foot_lift_range_m"]) < .005,
                     horizontal_excursion=float(np.linalg.norm(np.ptp(qpos[:, :2], axis=0))) >= .02)
    passed = all(checks[k] for k in PHYSICAL_CHECKS) and all(admission.values())
    out.mkdir(parents=True, exist_ok=False)
    path = out / "weight_shift_reference.npz"
    np.savez_compressed(path, qpos=qpos, qvel=qvel, dt=dt, foot_targets=foot_targets,
                        contact=contact, sole_contact=soles)
    report = dict(schema=SCHEMA, passed=passed, reference_sha256=sha256(path), model_sha256=sha256(rig.path),
                  source=metadata["source"], parent_calibration_sha256=metadata["diagnostic_sha256"],
                  checks=checks, metrics=metrics, weight_shift_admission=admission,
                  changes="Extract both double-support weight shifts and reverse them back to standing. Recompute velocities and all physical gates. Foot lift is inapplicable to this separate task; stepping admission remains unchanged.")
    path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2(calibration / "ATTRIBUTION.md", out / "ATTRIBUTION.md")
    with (out / "ATTRIBUTION.md").open("a", encoding="utf-8") as file:
        file.write("\nFurther adaptation: two-foot weight shifts with reversed returns; no foot-lift phases.\n")
    print(json.dumps(dict(passed=passed, duration=(len(qpos)-1)*dt, checks=checks, admission=admission)), flush=True)
    return report


def displacement_limit(case):
    return min(GATES["max_horizontal_displacement_rmse_m"],
               max(GATES["displacement_noise_floor_m"],
                   GATES["max_relative_displacement_error"] * case["reference_displacement_rms_m"]))


def tracking_checks(metrics, cases):
    return dict(survival=metrics["successes"] == GATES["episodes"],
                root_tracking=all(c["mean_root_error_m"] < GATES["max_root_error_m"] for c in cases),
                foot_tracking=all(c["mean_foot_error_m"] < GATES["max_foot_error_m"] for c in cases),
                joint_tracking=all(c["joint_rmse_rad"] < GATES["max_joint_rmse_rad"] for c in cases),
                weight_shift=all(c["horizontal_displacement_rmse_m"] < displacement_limit(c) for c in cases))


def tracking_metrics(task, metrics, traces):
    model = task.rig.model
    feet = [model.body(n).id for n in ("Foot_L", "Foot_R")]
    joints = model.jnt_qposadr[model.actuator_trnid[task.rig.actuators, 0]]
    actual_data, target_data = mujoco.MjData(model), mujoco.MjData(model)
    cases = []
    for case, seconds in enumerate(metrics["survival_seconds"]):
        count = round(seconds / task.dt)
        phase = case / (metrics["episodes"] - 1) * (task.reference.duration - task.episode_seconds)
        desired = task.reference.sample(torch.arange(count) * task.dt + phase)[0].numpy()
        actual = traces["qpos"][:count, case]
        foot_errors = []
        for q, target in zip(actual, desired):
            actual_data.qpos[:], target_data.qpos[:] = q, target
            mujoco.mj_kinematics(model, actual_data)
            mujoco.mj_kinematics(model, target_data)
            foot_errors.append(np.linalg.norm(actual_data.xpos[feet] - target_data.xpos[feet], axis=-1))
        displacement_error = (actual[:, :2] - actual[0, :2]) - (desired[:, :2] - desired[0, :2])
        cases.append(dict(phase=phase, survival_seconds=seconds,
                          mean_root_error_m=float(np.linalg.norm(actual[:, :3] - desired[:, :3], axis=-1).mean()),
                          mean_foot_error_m=float(np.mean(foot_errors)),
                          joint_rmse_rad=float(np.sqrt(np.mean((actual[:, joints] - desired[:, joints])**2))),
                          reference_displacement_rms_m=float(np.sqrt(np.mean(np.sum((desired[:, :2] - desired[0, :2])**2, axis=-1)))),
                          horizontal_displacement_rmse_m=float(np.sqrt(np.mean(np.sum(displacement_error**2, axis=-1))))))
    checks = tracking_checks(metrics, cases)
    return dict(passed=all(checks.values()), checks=checks, cases=cases)


def validated_tracking(metrics):
    tracking = metrics["weight_shift_tracking"]
    cases = tracking["cases"]
    fields = ("mean_root_error_m", "mean_foot_error_m", "joint_rmse_rad", "horizontal_displacement_rmse_m", "reference_displacement_rms_m")
    if (metrics["episodes"] != GATES["episodes"] or len(cases) != GATES["episodes"]
            or not all(np.isfinite(c[k]) and c[k] >= 0 for c in cases for k in fields)
            or not all(np.isfinite(metrics[k]) and metrics[k] >= 0 for k in
                       ("successes", "mean_survival_seconds", "mean_root_tracking_error_m"))):
        raise ValueError("Weight-shift selection requires eight finite native tracking cases")
    passed = all(tracking_checks(metrics, cases).values())
    if type(tracking["passed"]) is not bool or passed != tracking["passed"]:
        raise ValueError("Inconsistent weight-shift tracking gate")
    return passed


def checkpoint_rank(metrics):
    from .checkpoints import standing_passed
    schema = metrics.get("evaluation_schema")
    if schema not in (EVALUATION_SCHEMA, REHEARSAL_SCHEMA):
        raise ValueError("Weight-shift selection requires the v2 or rehearsal relative-displacement evaluation")
    passed = validated_tracking(metrics)
    cases = metrics["weight_shift_tracking"]["cases"]
    if type(metrics.get("quiet_standing_passed")) is not bool:
        raise ValueError("Inconsistent weight-shift or standing retention gate")
    quiet_passed = all(standing_passed(metrics[k]) for k in ("quiet_native", "quiet_native_five_seconds"))
    if quiet_passed != metrics["quiet_standing_passed"]:
        raise ValueError("Weight-shift selection must retain both three- and five-second standing")
    if schema == REHEARSAL_SCHEMA:
        extended = metrics["extended_motion_native"]
        if (not np.isfinite(extended["episode_seconds"]) or extended["episode_seconds"] <= 3
                or metrics.get("extended_motion_seconds") != extended["episode_seconds"]):
            raise ValueError("Rehearsal selection requires motion evaluation beyond three seconds")
        extended_passed = validated_tracking(extended)
        return (quiet_passed, passed and extended_passed, extended["successes"], metrics["successes"],
                extended["mean_survival_seconds"], metrics["mean_survival_seconds"],
                -extended["mean_root_tracking_error_m"], -metrics["mean_root_tracking_error_m"])
    return (metrics["quiet_standing_passed"], passed, metrics["successes"], metrics["mean_survival_seconds"],
            -np.mean([c["horizontal_displacement_rmse_m"] for c in cases]), -metrics["mean_root_tracking_error_m"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if prepare(args.calibration, args.out)["passed"] else 1)
