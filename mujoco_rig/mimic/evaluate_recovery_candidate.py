"""Audit and replay a saved recovery-step candidate in native MuJoCo."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .baseline import sha256
from .native_engine import NativeDummyEngine
from .probe_single_support_entry import CONTROLLERS, EntryReference, ScheduledStabilizer
from .recovery_step_motion import (PHYSICAL_CHECKS, FootAnchorController,
                                   TwoFootBalanceController, evaluate,
                                   reference_placement, support_schedule)
from .retarget_steps import validate_motion
from .rig import Rig
from .runtime import activate


SCHEMA = "mimic_recovery_candidate_native_v3"


def run(args):
    if args.out.exists() or not 0 < args.swing_start < args.touchdown:
        raise ValueError("Require a new output directory and ordered swing/touchdown times")
    activate(args.mimickit)
    from util import mp_util
    from .stand_task import StandTask

    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    teacher_path = args.teacher_run / "report.json"
    teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
    if teacher["model_sha256"] != sha256(rig.path):
        raise ValueError("Teacher and native model differ")
    with np.load(args.motion, allow_pickle=False) as arrays:
        motion = {key: arrays[key].copy() for key in arrays.files}
    dt = float(motion["dt"])
    measured_support = motion.get("support_contact")
    checks, metrics = validate_motion(
        rig, motion["qpos"], motion["qvel"], motion["foot_targets"],
        motion["contact"], dt,
        sole_contact=None if measured_support is not None else motion["sole_contact"],
        support_contact=measured_support)
    placement = reference_placement(rig, motion)
    reference = EntryReference(motion["qpos"], rig, dt)
    task = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz",
                     1, "cpu", engine_factory=NativeDummyEngine,
                     control_mode="target_pd", contact_mode="support_v2")
    task.reference = reference
    task.episode_seconds = reference.duration
    times, torques, centers, supports = support_schedule(rig, motion)
    balance = ScheduledStabilizer(task, times, torques, centers,
                                  CONTROLLERS["balanced_axes"])
    balance.stabilizer.k_i_hr = teacher.get("hip_integral_ki", 3.)
    balance = TwoFootBalanceController(
        task, balance, args.touchdown, args.placement_m,
        com_kp=teacher["landing_com_kp"], com_kd=teacher["landing_com_kd"])
    anchor_blend = (teacher["foot_anchor_blend"] if args.anchor_blend is None
                    else args.anchor_blend)
    if not 0. <= anchor_blend <= 1.:
        raise ValueError("Anchor blend must be between zero and one")
    controller = FootAnchorController(
        task, balance, args.swing_start, None, anchor_blend,
        motion["foot_targets"][:, 0])
    native, _, _ = evaluate(task, controller, args.placement_m, args.touchdown)
    required_checks = tuple("support_contact" if name == "sole_contact"
                            and measured_support is not None else name
                            for name in PHYSICAL_CHECKS)
    physical_passed = (all(checks[name] for name in required_checks)
                       and metrics["foot_lift_range_m"][0] >= .03
                       and abs(placement - args.placement_m) < .02
                       and all(item["feasible"] for item in supports))
    args.out.mkdir(parents=True)
    report = dict(schema=SCHEMA, model_sha256=sha256(rig.path),
                  teacher_report_sha256=sha256(teacher_path),
                  motion_sha256=sha256(args.motion), swing_start_seconds=args.swing_start,
                  touchdown_seconds=args.touchdown, placement_m=args.placement_m,
                  reference_placement_m=placement,
                  foot_anchor_blend=anchor_blend,
                  foot_anchor_saturated_fraction=controller.saturated / max(1, controller.samples),
                  physical_checks=checks, physical_metrics=metrics,
                  support_schedule=supports, physical_passed=physical_passed,
                  native=native, passed=bool(physical_passed and native["passed"]),
                  admitted_for_training=False,
                  scope="Saved-candidate physical audit and native teacher diagnostic only; no policy, ball, export or viewer selection")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(passed=report["passed"], physical_passed=physical_passed,
                          root_failures=len(metrics["root_wrench_consistency"]["infeasible_frames"]),
                          penetration_m=metrics["max_penetration_m"],
                          native={key: value for key, value in native.items() if key != "trace"})), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--swing-start", type=float, default=1.35)
    parser.add_argument("--touchdown", type=float, required=True)
    parser.add_argument("--placement-m", type=float, required=True)
    parser.add_argument("--anchor-blend", type=float)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
