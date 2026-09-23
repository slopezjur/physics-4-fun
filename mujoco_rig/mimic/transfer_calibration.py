"""Diagnostic slow weight transfers from statically feasible support postures."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from .baseline import sha256
from .contact_projection import StanceProjection
from .force_retarget import refine_support, reference_velocities
from .retarget_steps import validate_motion
from .reference_limits import PENETRATION_M
from .rig import Rig
from .transfer_seed import rotation_offsets


def interpolate_poses(poses, times, samples):
    """Clamped smooth coordinates, with root rotations on the quaternion manifold."""
    offsets = rotation_offsets(np.tile(poses[0, 3:7], (len(poses), 1)), poses[:, 3:7])
    coordinates = np.column_stack((poses[:, :3], offsets, poses[:, 7:]))
    interpolated = CubicSpline(times, coordinates, bc_type="clamped")(samples)
    result = np.empty((len(samples), poses.shape[1]))
    result[:, :3], result[:, 7:] = interpolated[:, :3], interpolated[:, 6:]
    result[:, 3:7] = (Rotation.from_rotvec(interpolated[:, 3:6]) *
                       Rotation.from_quat(poses[0, [4, 5, 6, 3]])).as_quat()[:, [3, 0, 1, 2]]
    return result


def stopped_interpolation(values, times, samples, root_rotations=False):
    """Bounded quintic segments with zero velocity/acceleration at every knot."""
    values, times, samples = np.asarray(values), np.asarray(times), np.asarray(samples)
    if (len(times) < 2 or len(values) != len(times) or not np.isfinite(times).all()
            or not np.isfinite(samples).all() or not np.isfinite(values).all()
            or np.any(np.diff(times) <= 0) or np.any(samples < times[0]) or np.any(samples > times[-1])):
        raise ValueError("Expected finite knots with increasing times and samples inside their interval")
    segments = np.minimum(np.searchsorted(times, samples, side="right") - 1, len(times) - 2)
    u = (samples - times[segments]) / (times[segments + 1] - times[segments])
    weight = u**3 * (10 - 15 * u + 6 * u**2)
    expanded = weight.reshape((-1,) + (1,) * (values.ndim - 1))
    result = values[segments] + expanded * (values[segments + 1] - values[segments])
    if root_rotations:
        first = Rotation.from_quat(values[segments][:, [4, 5, 6, 3]])
        last = Rotation.from_quat(values[segments + 1][:, [4, 5, 6, 3]])
        result[:, 3:7] = (Rotation.from_rotvec(weight[:, None] * (last * first.inv()).as_rotvec()) * first).as_quat()[:, [3, 0, 1, 2]]
    return result


def assemble_transfer(rig, poses, targets, times, phase_seconds, dt=.05):
    """Connect stopped calibration poses, then restore fixed stance anchors."""
    samples = np.linspace(0., times[-1], round(times[-1] / dt) + 1)
    dt = float(samples[1] - samples[0])
    qpos = stopped_interpolation(poses, times, samples, root_rotations=True)
    interpolated_targets = stopped_interpolation(targets, times, samples)
    contact = np.ones((len(samples), 2), dtype=bool)
    for leg in range(2):
        contact[(samples > (leg * 4 + 1) * phase_seconds) &
                (samples < (leg * 4 + 3) * phase_seconds), leg] = False
        interpolated_targets[contact[:, leg], leg] = targets[0, leg]
    projection = StanceProjection(rig, qpos, interpolated_targets, contact)
    qpos = projection.project(qpos)
    return dict(qpos=qpos, qvel=reference_velocities(rig.model, qpos, dt), dt=dt,
                foot_targets=interpolated_targets, contact=contact,
                sole_contact=np.column_stack((contact, contact))), samples


def calibrate(out, max_evaluations=100, phase_seconds=4.):
    if out.exists():
        raise FileExistsError(out)
    if max_evaluations < 1 or not np.isfinite(phase_seconds) or phase_seconds < 1:
        raise ValueError("Require a positive solver budget and phases of at least one second")
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    source = Path(__file__).parent / "assets/stand_reference.npz"
    metadata = json.loads(source.with_suffix(".json").read_text())
    if (metadata["reference_sha256"] != sha256(source) or metadata["model_sha256"] != sha256(rig.path)
            or metadata["source"]["license"] != "CC-BY-4.0"):
        raise ValueError("Standing source provenance mismatch")
    with np.load(source, allow_pickle=False) as arrays:
        rest = arrays["qpos"][0].copy()
    model, data = rig.model, mujoco.MjData(rig.model)
    data.qpos[:] = rest
    mujoco.mj_forward(model, data)
    feet = [model.body("Foot_" + side).id for side in ("L", "R")]
    ground = data.xpos[feet].copy()
    for foot, side in enumerate(("L", "R")):
        geom = model.geom("g_Foot_" + side).id
        ground[foot, 2] = model.geom_size[geom, 2] - model.geom_pos[geom, 2]
    out.mkdir(parents=True, exist_ok=False)
    reports = []

    def static(pose, target, support, label, drop=0., clearance=0.):
        q = np.tile(pose, (3, 1))
        contact = np.tile(support, (3, 1))
        targets = np.tile(target, (3, 1, 1))
        soles = np.column_stack((contact, contact))
        fitted, targets, _, fit = refine_support(rig, q, targets, contact, .05,
                                                max_evaluations=max_evaluations, root_seed_drop_m=drop,
                                                source_soles=soles, stationary=True, linear_solver="exact",
                                                self_clearance_m=clearance)
        checks, metrics = validate_motion(rig, fitted, np.zeros((3, model.nv)), targets, contact, .05, soles)
        # Static calibration poses intentionally have zero foot lift over time.
        passed = all(value for key, value in checks.items() if key != "foot_lift")
        report = dict(label=label, static_feasible=passed, checks=checks, metrics=metrics, fit=fit)
        reports.append(report)
        (out / "static-progress.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
        np.savez_compressed(out / f"{label}.npz", qpos=fitted[0], foot_targets=target, contact=support)
        print(json.dumps(dict(label=label, static_feasible=passed, cost=fit["cost"], seconds=fit["seconds"])), flush=True)
        return fitted[0]

    standing = static(rest, ground, np.ones(2, dtype=bool), "standing", .02)
    all_poses, all_targets, all_times = [], [], []
    for leg in range(2):
        support = np.ones(2, dtype=bool)
        support[leg] = False
        lifted = ground.copy()
        lifted[leg] += [.04, 0., .06]
        endpoint = static(rest, lifted, support, f"endpoint-{leg}", .02)
        shift = [standing]
        for fraction in (.25, .5, .75):
            desired = interpolate_poses(np.array([standing, endpoint]), [0., 1.], np.array([fraction]))[0]
            shift.append(static(desired, ground, np.ones(2, dtype=bool), f"shift-{leg}-{len(shift)}"))
        # Transfer all load before changing the declared contact schedule.
        shift.append(static(endpoint, ground, support, f"unload-{leg}"))
        lift, lift_targets = [shift[-1]], [ground.copy()]
        for fraction in (.25, .5, .75, 1.):
            target = ground + fraction * (lifted - ground)
            lift.append(static(lift[-1], target, support, f"lift-{leg}-{len(lift)}"))
            lift_targets.append(target)
        knots = shift + lift[1:] + lift[-2::-1] + shift[-2::-1]
        targets = [ground] * 5 + lift_targets[1:] + lift_targets[-2::-1] + [ground] * 4
        for index, (pose, target) in enumerate(zip(knots, targets)):
            if leg and not index:
                continue
            all_poses.append(pose)
            all_targets.append(target)
            all_times.append((leg * 4 + index / 4) * phase_seconds)
    poses, targets, times = np.asarray(all_poses), np.asarray(all_targets), np.asarray(all_times)
    arrays, samples = assemble_transfer(rig, poses, targets, times, phase_seconds)
    # One bounded repair pass for colliding double-support segments. Any remaining
    # failure is reported; the geometry and force admission thresholds never move.
    colliding_segments = set()
    for frame, pose in enumerate(arrays["qpos"]):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        if arrays["contact"][frame].all() and any(c.dist < -PENETRATION_M for c in data.contact):
            colliding_segments.add(min(np.searchsorted(times, samples[frame], side="right") - 1, len(times) - 2))
    repairs = []
    for segment in sorted(colliding_segments)[:4]:
        midpoint = float((times[segment] + times[segment + 1]) / 2)
        if any((leg * 4 + 1) * phase_seconds < midpoint < (leg * 4 + 3) * phase_seconds for leg in range(2)):
            continue
        seed = stopped_interpolation(poses, times, [midpoint], root_rotations=True)[0]
        pose = static(seed, ground, np.ones(2, dtype=bool), f"clearance-{segment}", clearance=.015)
        repairs.append((midpoint, pose))
    if repairs:
        times = np.r_[times, [t for t, _ in repairs]]
        poses = np.concatenate((poses, [q for _, q in repairs]))
        targets = np.concatenate((targets, np.tile(ground, (len(repairs), 1, 1))))
        order = np.argsort(times)
        times, poses, targets = times[order], poses[order], targets[order]
        arrays, samples = assemble_transfer(rig, poses, targets, times, phase_seconds)
    checks, metrics = validate_motion(rig, arrays["qpos"], arrays["qvel"], arrays["foot_targets"],
                                      arrays["contact"], arrays["dt"], arrays["sole_contact"])
    path = out / "diagnostic_poses.npz"
    np.savez_compressed(path, **arrays)
    knots_path = out / "knots.npz"
    np.savez_compressed(knots_path, qpos=poses, foot_targets=targets, times=times)
    report = dict(schema="mimic_transfer_calibration_v1", source_reference_sha256=sha256(source),
                  model_sha256=sha256(rig.path), diagnostic_sha256=sha256(path), source=metadata["source"],
                  phase_seconds=phase_seconds, static_poses=reports, checks=checks, metrics=metrics,
                  knots_sha256=sha256(knots_path), clearance_repair_times=[t for t, _ in repairs],
                  colliding_segments_before_repair=sorted(int(i) for i in colliding_segments),
                  interpolation="Piecewise quintic with stops at knots; fixed stance anchors; unload before foot lift",
                  passed=all(checks.values()), admitted_for_training=False,
                  scope="Procedural stepping-in-place calibration derived from a vetted standing pose; not recorded human walking, a trained policy, or a closed-loop tracking result.")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "ATTRIBUTION.md").write_text(
        f"{metadata['source']['credit']}. {metadata['source']['source']}\n\n"
        f"CC-BY-4.0: {metadata['source']['license_url']}\n\n"
        "Adaptation: standing pose retargeted into a procedural slow support-transfer calibration. No endorsement implied.\n",
        encoding="utf-8")
    print(json.dumps(dict(checks=checks, root_failures=len(metrics["root_wrench_consistency"]["infeasible_frames"]),
                          pd_failures=len(metrics["action_window_consistency"]["failing_frames"]["0.25"]))), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-evaluations", type=int, default=100)
    parser.add_argument("--phase-seconds", type=float, default=4.)
    args = parser.parse_args()
    result = calibrate(args.out, args.max_evaluations, args.phase_seconds)
    raise SystemExit(0 if result["passed"] else 1)
