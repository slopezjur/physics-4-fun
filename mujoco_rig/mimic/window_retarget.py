"""Fit overlapping motion intervals with fixed context and full-reference validation."""
import argparse
import json
from pathlib import Path

import numpy as np

from .baseline import sha256
from .contact_projection import StanceProjection
from .force_retarget import refine_support, reference_velocities
from .motion_contacts import validate_sole_schedule, landing_seed_for_window
from .retarget_steps import attribution, validate_motion
from .rig import Rig
from .transfer_seed import load_seed_probe
from mujoco_rig.rl.env_config import ACTION_LATENCY_STEPS


def context_interval(rig, frames, dt, start, stop):
    """Include central derivatives on both sides of delayed interpolated targets."""
    if not np.isfinite(dt) or dt <= 0 or not 0 <= start < stop <= frames:
        raise ValueError("Require a positive timestep and a valid exclusive frame interval")
    delay = ACTION_LATENCY_STEPS * rig.decimation * rig.model.opt.timestep
    halo = int(np.ceil(delay / dt)) + 2
    left, right = max(0, start - halo), min(frames, stop + halo)
    fixed = np.ones(right - left, dtype=bool)
    fixed[start - left:stop - left] = False
    if not fixed.any():
        raise ValueError("Window fitting requires surrounding context; use force_retarget for a whole clip")
    return left, right, fixed


def regression_reasons(before, after):
    """Protect previously feasible dynamics and geometry across the whole clip."""
    old_checks, old = before
    checks, new = after
    reasons = [f"lost check: {name}" for name, passed in old_checks.items() if passed and not checks[name]]
    old_root = set(old["root_wrench_consistency"]["infeasible_frames"])
    new_root = set(new["root_wrench_consistency"]["infeasible_frames"])
    old_pd = set(old["action_window_consistency"]["failing_frames"]["0.25"])
    new_pd = set(new["action_window_consistency"]["failing_frames"]["0.25"])
    if new_root - old_root:
        reasons.append("new root-support failures")
    if (new_root | new_pd) - (old_root | old_pd):
        reasons.append("new combined root/PD failures")
    # Retain global geometry maxima as well as per-frame dynamics feasibility.
    for key in ("max_penetration_m", "max_stance_slide_m_s", "max_foot_target_error_m"):
        if new[key] > old[key] + 1e-6:
            reasons.append(f"increased {key}")
    return reasons


def refine_windows(reference, out, windows, max_evaluations=20, constrained=True, direct_forces=False, seed_probe=None):
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    metadata = json.loads(reference.with_suffix(".json").read_text())
    parent_hash = sha256(reference)
    if metadata["model_sha256"] != sha256(rig.path) or metadata["reference_sha256"] != parent_hash:
        raise ValueError("Reference or rig hash mismatch")
    if metadata["source"]["license"] != "CC-BY-4.0":
        raise ValueError("Expected the vetted CC-BY-4.0 source")
    if out.exists():
        raise FileExistsError(out)
    with np.load(reference, allow_pickle=False) as source:
        poses, targets, contact = (source[k] for k in ("qpos", "foot_targets", "contact"))
        dt = float(source["dt"])
        soles = source["sole_contact"] if "sole_contact" in source else None
    if not windows or not contact.any(axis=1).all():
        raise ValueError("Window fitting requires explicit windows and a completed support schedule")
    intervals = [context_interval(rig, len(poses), dt, *window) for window in windows]
    toe_only = validate_sole_schedule(contact, soles) if soles is not None else None
    rotations = StanceProjection(rig, poses, targets, contact, toe_only).rotations
    offsets = metadata.get("force_fit", {}).get("stance_anchor_offsets_xy")
    clearance = metadata.get("force_fit", {}).get("self_clearance_m", 0.)

    def assess(candidate):
        return validate_motion(rig, candidate, reference_velocities(rig.model, candidate, dt),
                               targets, contact, dt, soles, clearance)

    original = poses.copy()
    baseline = current = assess(poses)
    initial, seed_record = None, None
    if seed_probe is not None:
        initial, seed_record = load_seed_probe(seed_probe, parent_hash, sha256(rig.path), poses, dt, windows[0])
        seed_checks, seed_metrics = assess(initial)
        seed_record.update(checks=seed_checks, metrics=seed_metrics,
                           regression_reasons=regression_reasons(baseline, (seed_checks, seed_metrics)))
    out.mkdir(parents=True, exist_ok=False)
    reports = []
    for index, ((start, stop), (left, right, fixed)) in enumerate(zip(windows, intervals)):
        fitted, _, _, fit = refine_support(
            rig, poses[left:right], targets[left:right], contact[left:right], dt,
            max_evaluations=max_evaluations, root_seed_drop_m=0.,
            source_soles=soles[left:right] if soles is not None else None,
            self_clearance_m=clearance, linear_solver="exact", fixed_frames=fixed, constrained=constrained,
            direct_forces=direct_forces,
            initial_poses=initial[left:right] if initial is not None and index == 0 else None,
            stance_rotations=rotations[left:right],
            landing_seed=landing_seed_for_window(contact, offsets, left, right))
        if not np.array_equal(fitted[fixed], poses[left:right][fixed]):
            raise RuntimeError("Fixed context was modified")
        candidate = poses.copy()
        candidate[start:stop] = fitted[start - left:stop - left]
        checked = assess(candidate)
        reasons = regression_reasons(current, checked)
        if constrained and not fit["constraints_satisfied"]:
            reasons.append("optimizer placement/support constraints not satisfied")
        accepted = not reasons
        if accepted:
            poses, current = candidate, checked
        row = dict(start=start, stop_exclusive=stop, context_start=left, context_stop_exclusive=right,
                   accepted=accepted, rejection_reasons=reasons, fit=fit,
                   checks=checked[0], metrics=checked[1])
        reports.append(row)
        np.savez_compressed(out / f"window-{index:02d}-candidate.npz", qpos=candidate)
        (out / f"window-{index:02d}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(json.dumps(dict(window=[start, stop], accepted=accepted, rejection_reasons=reasons,
                              root_failures=len(checked[1]["root_wrench_consistency"]["infeasible_frames"]),
                              pd_failures=len(checked[1]["action_window_consistency"]["failing_frames"]["0.25"]))), flush=True)
    path = out / "step_reference.npz"
    np.savez_compressed(path, qpos=poses, qvel=reference_velocities(rig.model, poses, dt), dt=dt,
                        foot_targets=targets, contact=contact, **(dict(sole_contact=soles) if soles is not None else {}))
    metadata.update(passed=all(current[0].values()), checks=current[0], metrics=current[1],
                    baseline_checks=baseline[0], baseline_metrics=baseline[1],
                    parent_reference_sha256=parent_hash, reference_sha256=sha256(path))
    metadata["parent_force_fit"] = metadata.get("force_fit", {})
    metadata["force_fit"] = dict(stance_method="Preserved full-stance anchors with fixed-context interval fitting",
                                 self_clearance_m=clearance, stance_anchor_offsets_xy=offsets,
                                 optimizer="SLSQP" if constrained else "least_squares",
                                 force_formulation="explicit" if direct_forces else "projected",
                                 linear_solver=None if constrained else "exact",
                                 seconds=sum(row["fit"]["seconds"] for row in reports))
    metadata["window_fit"] = dict(windows=reports, initialization=seed_record,
                                  changed_frames=np.flatnonzero(np.any(poses != original, axis=1)).tolist(),
                                  scope="Dense local solves with fixed context, immutable full-stance anchors, full-clip derivatives/PD admission and regression rejection. Passing necessary conditions is not closed-loop tracking.")
    metadata["changes"] += " Overlapping interval refinement with fixed surrounding poses and delayed-command context; full-reference regression checks."
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out / "ATTRIBUTION.md").write_text(attribution(metadata), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", action="append", required=True, help="Editable source frames START:STOP, exclusive stop; repeat for overlap")
    parser.add_argument("--max-evaluations", type=int, default=20,
                        help="SLSQP iterations or least-squares evaluations; direct-force restoration consumes this budget first")
    parser.add_argument("--solver", choices=("constrained", "direct-forces", "least-squares"), default="constrained",
                        help="Projected-force SLSQP, explicit-force restoration/SLSQP, or legacy dense least squares")
    parser.add_argument("--seed-probe", type=Path, help="Matching transition_probe directory used only to initialize the first window")
    args = parser.parse_args()
    try:
        windows = [tuple(map(int, value.split(":"))) for value in args.window]
        if any(len(window) != 2 for window in windows) or args.max_evaluations < 1:
            raise ValueError()
    except ValueError:
        parser.error("Use START:STOP windows and a positive evaluation budget")
    report = refine_windows(args.reference, args.out, windows, args.max_evaluations,
                            args.solver != "least-squares", args.solver == "direct-forces", args.seed_probe)
    raise SystemExit(0 if report["passed"] else 1)
