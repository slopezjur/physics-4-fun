"""Bounded native search for an early loaded landing on a fixed short reference."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import differential_evolution

from .baseline import sha256
from .native_engine import NativeDummyEngine
from .probe_single_support_entry import CONTROLLERS, EntryReference, ScheduledStabilizer
from .recovery_step_motion import (FootAnchorController, TwoFootBalanceController,
                                   evaluate, support_schedule)
from .rig import Rig
from .runtime import activate


SCHEMA = "mimic_landing_action_search_v3"
JOINTS = ("Thigh_L_rx", "Thigh_L_rz", "Shin_L_rx", "Foot_L_rx", "Foot_L_rz")
DIMENSIONS = JOINTS + ("com_center_y_m", "com_center_x_m")
CORRECTION_START = 2.35
CORRECTION_END = 3.65


class LandingActionCorrection:
    def __init__(self, task, base, offsets, end):
        self.task, self.base = task, base
        names = [task.rig.model.actuator(int(i)).name for i in task.rig.actuators]
        self.indices = [names.index(name) for name in JOINTS]
        self.offsets = torch.as_tensor(offsets, dtype=torch.float32)
        self.end = end

    def __call__(self, observation):
        action = self.base(observation)
        seconds = float(self.task.steps[0] * self.task.dt)
        ramp_in = np.clip((seconds - CORRECTION_START) / .10, 0., 1.)
        ramp_out = np.clip((self.end - seconds) / .25, 0., 1.)
        weight = min(ramp_in, ramp_out)
        if weight:
            action[:, self.indices] = (action[:, self.indices]
                                       + weight * self.offsets).clamp(-1., 1.)
        return action


def landing_score(native, duration):
    rows = [row for row in native["trace"] if 2.5 <= row["seconds"] <= 3.6]
    early_load = np.mean([
        min(row["left_load_n"] / 200., 1.)
        * min(row["right_load_n"] / 200., 1.)
        * np.clip((row["left_foot_shift_m"] - .10) / .03, 0., 1.)
        * row["upright"] for row in rows]) if rows else 0.
    return (10. * min(native["first_loaded_contact_seconds"] / 2., 1.)
            + 3. * early_load
            + 10. * min(native["survival_seconds"] / duration, 1.)
            - 3. * (not native["survived"])
            + 2. * native["survived"] * np.clip((native["final_left_foot_shift_m"] - .13) / .05, 0., 1.)
            + 2. * native["passed"])


def run(args):
    if (args.out.exists() or args.max_iterations < (0 if args.replay_source else 1)
            or args.popsize < 2 or args.correction_end <= CORRECTION_START + .2
            or args.com_release_end <= 2.55
            or (args.com_x_bias is not None and not 0. <= args.com_x_bias <= .10)):
        raise ValueError("Require a new output directory and positive search budgets")
    activate(args.mimickit)
    from util import mp_util
    from .stand_task import StandTask

    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    source_report_path = args.source / "report.json"
    source = json.loads(source_report_path.read_text(encoding="utf-8"))
    motion_path = args.source / "step_reference.npz"
    if (source["model_sha256"] != sha256(rig.path)
            or source["reference_sha256"] != sha256(motion_path)
            or source["phase_seconds"]["touchdown"] != 2.55):
        raise ValueError("Source model, motion, or touchdown mismatch")
    with np.load(motion_path, allow_pickle=False) as arrays:
        motion = {key: arrays[key].copy() for key in arrays.files}
    reference = EntryReference(motion["qpos"], rig, float(motion["dt"]))
    task = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz",
                     1, "cpu", engine_factory=NativeDummyEngine,
                     control_mode="target_pd", contact_mode="support_v2")
    task.reference = reference
    task.episode_seconds = reference.duration
    times, torques, centers, _ = support_schedule(rig, motion)
    history = []
    best = None

    def trial(offsets):
        nonlocal best
        adjusted_centers = centers.copy()
        transfer = np.clip((times - 2.35) / .20, 0., 1.)
        release = np.clip((args.com_release_end - times) / .50, 0., 1.)
        adjusted_centers[:, 1] += offsets[-2] * np.minimum(transfer, release)
        adjusted_centers[:, 0] += offsets[-1] * np.clip((times - 2.55) / .50, 0., 1.)
        balance = ScheduledStabilizer(task, times, torques, adjusted_centers,
                                      CONTROLLERS["balanced_axes"])
        balance.stabilizer.k_i_hr = source["hip_integral_ki"]
        landing = TwoFootBalanceController(task, balance, 2.55, .15,
                                           com_kp=source["landing_com_kp"],
                                           com_kd=source["landing_com_kd"])
        anchor = FootAnchorController(task, landing, 1.35, None,
                                      source["foot_anchor_blend"],
                                      motion["foot_targets"][:, 0])
        controller = LandingActionCorrection(task, anchor, offsets[:len(JOINTS)],
                                              args.correction_end)
        native, rollout, samples = evaluate(task, controller, .15, 2.55)
        score = float(landing_score(native, task.episode_seconds))
        row = dict(index=len(history), offsets=np.asarray(offsets).tolist(),
                   score=score, native_passed=native["passed"],
                   first_loaded_seconds=native["first_loaded_contact_seconds"],
                   survival_seconds=native["survival_seconds"],
                   final_shift_m=native["final_left_foot_shift_m"])
        history.append(row)
        if best is None or score > best["score"]:
            best = dict(**row, native=native, rollout=rollout, samples=samples)
            print(json.dumps(dict(best=row)), flush=True)
        return -score

    zero_score = trial(np.zeros(len(DIMENSIONS)))
    repeat_score = trial(np.zeros(len(DIMENSIONS)))
    if not np.isclose(zero_score, repeat_score, atol=1e-6):
        raise RuntimeError("Native replay is not deterministic across resets")
    seed_source = args.replay_source or args.warm_source
    if seed_source is not None:
        warm = json.loads((seed_source / "report.json").read_text(encoding="utf-8"))
        if warm["model_sha256"] != sha256(rig.path) or warm["source_motion_sha256"] != sha256(motion_path):
            raise ValueError("Warm start uses another rig or source motion")
        prior = np.asarray(warm["best"]["offsets"], dtype=float)
        if prior.shape == (len(JOINTS),):
            prior = np.r_[prior, 0.]
        if prior.shape == (len(JOINTS) + 1,):
            prior = np.r_[prior, 0.]
        if prior.shape != (len(DIMENSIONS),):
            raise ValueError("Warm start has incompatible control dimensions")
        if args.com_x_bias is not None:
            prior[-1] = args.com_x_bias
        trial(prior)
    bounds = [(-.6, .6)] * len(JOINTS) + [(-.08, .02), (0., .10)]
    search = (None if args.replay_source else
              differential_evolution(trial, bounds,
                                     maxiter=args.max_iterations,
                                     popsize=args.popsize, seed=args.seed,
                                     polish=False, updating="immediate"))
    args.out.mkdir(parents=True)
    rollout_path = args.out / "best_native_rollout.npz"
    np.savez_compressed(rollout_path, **best.pop("rollout"))
    samples_path = args.out / "best_teacher_samples.npz"
    np.savez_compressed(samples_path, **best.pop("samples"))
    report = dict(schema=SCHEMA, model_sha256=sha256(rig.path),
                  source_report_sha256=sha256(source_report_path),
                  source_motion_sha256=sha256(motion_path),
                  native_rollout_sha256=sha256(rollout_path),
                  teacher_samples_sha256=sha256(samples_path),
                  dimensions=DIMENSIONS, correction_start=CORRECTION_START,
                  correction_end=args.correction_end,
                  com_release_end=args.com_release_end, bounds=bounds,
                  warm_source=(str(args.warm_source) if args.warm_source else None),
                  replay_source=(str(args.replay_source) if args.replay_source else None),
                  seed=args.seed, max_iterations=args.max_iterations,
                  popsize=args.popsize, evaluations=len(history),
                  optimizer_message=(search.message if search is not None else "Replay only"), best=best,
                  history=history, passed=bool(best["native"]["passed"]),
                  admitted_for_training=False,
                  scope="Native privileged-controller search; no physical reference admission, training, export or viewer selection")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(passed=report["passed"], best={k: v for k, v in best.items() if k != "native"},
                          native={k: v for k, v in best["native"].items() if k != "trace"})), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--popsize", type=int, default=4)
    parser.add_argument("--seed", type=int, default=90727)
    parser.add_argument("--warm-source", type=Path)
    parser.add_argument("--replay-source", type=Path)
    parser.add_argument("--correction-end", type=float, default=CORRECTION_END)
    parser.add_argument("--com-release-end", type=float, default=4.15)
    parser.add_argument("--com-x-bias", type=float)
    run(parser.parse_args())
