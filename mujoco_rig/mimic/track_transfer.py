"""Native closed-loop diagnostics for an unadmitted transfer calibration."""
import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import torch

from .baseline import sha256
from .contact_contract import mode_from_contract
from .evaluate_stand import evaluate
from .force_retarget import reference_velocities
from .motion_contacts import validate_sole_schedule
from .motion_protocol import GATES
from .motion_tracking_audit import TraceEngine, first_time
from .probe_steps import validate_reference_substitution
from .reference import StandReference
from .retarget_steps import validate_motion
from .rig import Rig
from .runtime import activate


class CalibrationReference(StandReference):
    """Reuse runtime sampling without manufacturing a training-admission manifest."""
    def __init__(self, directory, rig, device="cpu"):
        report_path = directory / "report.json"
        metadata = json.loads(report_path.read_text())
        self.placed_step = metadata.get("schema") == "mimic_placed_step_diagnostic_v1"
        self.requested_step_m = metadata.get("requested_step_m") if self.placed_step else None
        self.path = directory / "diagnostic_poses.npz"
        source = Path(__file__).parent / "assets/stand_reference.npz"
        if (metadata.get("schema") not in ("mimic_transfer_calibration_v1", "mimic_placed_step_diagnostic_v1")
                or metadata.get("passed") is not True or metadata.get("admitted_for_training") is not False
                or metadata.get("model_sha256") != sha256(rig.path)
                or metadata.get("source_reference_sha256") != sha256(source)
                or metadata.get("diagnostic_sha256") != sha256(self.path)
                or metadata.get("source", {}).get("license") != "CC-BY-4.0"
                or self.placed_step and (metadata.get("placement_passed") is not True
                    or metadata.get("direction") not in ("forward", "backward")
                    or not isinstance(self.requested_step_m, (float, int))
                    or not np.isfinite(self.requested_step_m)
                    or abs(self.requested_step_m) < .1
                    or np.sign(self.requested_step_m) != (1 if metadata["direction"] == "forward" else -1))):
            raise ValueError("Calibration provenance/admission mismatch")
        self.sha256, self.report_sha256 = sha256(self.path), sha256(report_path)
        with np.load(self.path, allow_pickle=False) as arrays:
            q, v = arrays["qpos"].copy(), arrays["qvel"].copy()
            targets, contact, soles = (arrays[k].copy() for k in ("foot_targets", "contact", "sole_contact"))
            self.dt = float(arrays["dt"])
        self.duration = (len(q) - 1) * self.dt
        if (q.shape != (len(q), rig.model.nq) or v.shape != (len(q), rig.model.nv)
                or targets.shape != (len(q), 2, 3) or contact.shape != (len(q), 2)
                or not np.isfinite(self.dt) or self.dt <= 0 or self.duration < 3.
                or not all(np.isfinite(a).all() for a in (q, v, targets))
                or not np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1., rtol=0, atol=1e-8)
                or np.any(np.sum(q[:-1, 3:7] * q[1:, 3:7], axis=1) <= 0)):
            raise ValueError("Invalid calibration poses, timing or quaternion continuity")
        validate_sole_schedule(contact, soles)
        if not np.allclose(v, reference_velocities(rig.model, q, self.dt), rtol=1e-7, atol=1e-8):
            raise ValueError("Calibration velocities disagree with the saved poses")
        self.checks, self.metrics = validate_motion(rig, q, v, targets, contact, self.dt, soles)
        if self.placed_step:
            from .placed_step import single_step_checks
            data = mujoco.MjData(rig.model)
            foot = rig.model.body("Foot_L").id
            data.qpos[:] = q[0]
            mujoco.mj_kinematics(rig.model, data)
            start_x = float(data.xpos[foot, 0])
            data.qpos[:] = q[-1]
            mujoco.mj_kinematics(rig.model, data)
            measured = float(data.xpos[foot, 0] - start_x)
            gates = single_step_checks(self.checks, self.metrics, measured, self.requested_step_m)
            physical = all(gates.values()) and abs(measured - metadata["measured_step_m"]) < 1e-6
        else:
            physical = all(self.checks.values())
        if not physical:
            raise ValueError("Calibration no longer passes offline physical checks")
        self.qpos = torch.tensor(q, dtype=torch.float32, device=device)
        self.qvel = torch.tensor(v, dtype=torch.float32, device=device)


def summarize_case(task, records, seconds, phase):
    """Use only the first episode, matching post-integration reference times."""
    rig, model = task.rig, task.rig.model
    count = round(seconds / task.dt) * rig.decimation
    records = records[:count]
    arrays = {key: np.asarray([r[key] for r in records]) for key in records[0]}
    times = (np.arange(count) + 1) * model.opt.timestep
    desired, _ = task.reference.sample(torch.tensor(phase + times, dtype=torch.float32))
    desired = desired.numpy()
    actual = arrays["qpos_after"]
    root_error = np.linalg.norm(actual[:, :3] - desired[:, :3], axis=1)
    joints = model.jnt_qposadr[model.actuator_trnid[rig.actuators, 0]]
    feet = [model.body(n).id for n in ("Foot_L", "Foot_R")]
    actual_data, desired_data = mujoco.MjData(model), mujoco.MjData(model)
    errors, positions = [], []
    for frame in range(rig.decimation - 1, count, rig.decimation):
        actual_data.qpos[:], desired_data.qpos[:] = actual[frame], desired[frame]
        mujoco.mj_kinematics(model, actual_data)
        mujoco.mj_kinematics(model, desired_data)
        errors.append(np.linalg.norm(actual_data.xpos[feet] - desired_data.xpos[feet], axis=-1))
        positions.append(actual_data.xpos[feet].copy())
    arrays.update(reference_qpos=desired, root_error_m=root_error)
    row = dict(phase_seconds=phase, survival_seconds=seconds,
               mean_root_error_m=float(root_error.mean()), max_root_error_m=float(root_error.max()),
               mean_foot_error_m=float(np.mean(errors)),
               joint_rmse_rad=float(np.sqrt(np.mean((actual[:, joints] - desired[:, joints])**2))),
               foot_lift_range_m=np.ptp(np.asarray(positions)[..., 2], axis=0).tolist(),
               left_foot_displacement_x_m=float(positions[-1][0, 0] - positions[0][0, 0]),
               final_two_second_min_pelvis_height_m=float(actual[-round(2 / model.opt.timestep):, 2].min()),
               final_two_second_max_horizontal_speed_m_s=float(
                   np.linalg.norm(arrays["qvel_after"][-round(2 / model.opt.timestep):, :2], axis=-1).max()),
               first_root_error_5cm_seconds=first_time(root_error > .05, model.opt.timestep),
               first_self_contact_over_5N_seconds=first_time(arrays["self_contact_max_force_n"] > 5, model.opt.timestep),
               max_self_contact_force_n=float(arrays["self_contact_max_force_n"].max()),
               saturated_motor_sample_fraction=float((np.abs(arrays["control"] / rig.action_scale) >= .999).mean()),
               terminal_root_height_m=float(actual[-1, 2]))
    return row, arrays


def tracking_checks(metrics, cases, require_lift):
    checks = dict(survival=metrics["successes"] == metrics["episodes"],
                  root_tracking=all(c["mean_root_error_m"] < GATES["max_root_error_m"] for c in cases),
                  foot_tracking=all(c["mean_foot_error_m"] < GATES["max_foot_error_m"] for c in cases),
                  joint_tracking=all(c["joint_rmse_rad"] < GATES["max_joint_rmse_rad"] for c in cases))
    if require_lift:
        checks["foot_lift"] = all(min(c["foot_lift_range_m"]) >= GATES["min_each_foot_lift_m"] for c in cases)
    return checks


def run(args):
    activate(args.mimickit)
    from .stand_task import StandTask
    torch.set_num_threads(1)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    reference = CalibrationReference(args.calibration, rig)
    contract = json.loads((args.bundle / "contract.json").read_text())
    if contract.get("diagnostic_only") or sha256(args.bundle / "stand.onnx") != contract["actor_sha256"]:
        raise ValueError("Require an intact trained Stand actor")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.bundle / "stand.onnx"), options, providers=["CPUExecutionProvider"])
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    jobs = (("standing_baseline", 8, 3.), ("transfer_phases", 8, 3.), ("transfer_complete", 1, reference.duration))
    if getattr(args, "interventions_only", False):
        jobs = (("transfer_interventions", 1, 3.),)
    for motion, episodes, seconds in jobs:
        modes = (("reference_pd", False), ("standing_actor", False))
        if motion != "standing_baseline":
            modes += (("reference_pd_prefilled_diagnostic", True),)
        if motion == "transfer_interventions":
            modes = tuple((name, False) for name in ("standing_actor_hold_start_diagnostic",
                          "standing_actor_source_phase_diagnostic", "standing_actor_no_self_collision_diagnostic"))
        for mode, prefill in modes:
            rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
            def factory(*a, **kw):
                return TraceEngine(*a, prefill=prefill, **kw)
            task = StandTask(rig, args.bundle / "stand_reference.npz", episodes, "cpu", engine_factory=factory,
                             control_mode="target_pd", contact_mode=mode_from_contract(contract))
            if motion != "standing_baseline":
                task.reference = reference
            validate_reference_substitution(contract, task.observation_contract())
            intervention = None
            if mode == "standing_actor_hold_start_diagnostic":
                task.reference = copy.copy(reference)
                task.reference.qpos = reference.qpos[:1].repeat(len(reference.qpos), 1)
                task.reference.qvel = torch.zeros_like(reference.qvel)
                intervention = "Hold initial calibration pose with zero reference velocity; same duration/phase input. Not a validated motion replacement."
            elif mode == "standing_actor_no_self_collision_diagnostic":
                rig.model.geom_contype[:] = 2
                rig.model.geom_conaffinity[:] = 1
                rig.model.geom_contype[0], rig.model.geom_conaffinity[0] = 1, 2
                intervention = "Private model disables body/body collision only; floor collisions remain."
            elif mode == "standing_actor_source_phase_diagnostic":
                intervention = "Only the actor's phase scalar uses its original standing-reference duration; all target poses and velocities remain the transfer."
            task.episode_seconds = seconds
            def learned_actor(obs):
                if mode == "standing_actor_source_phase_diagnostic":
                    obs = obs.clone()
                    obs[:, -1] *= reference.duration / contract["reference_duration"]
                return torch.from_numpy(session.run(None, {"observation": obs.numpy()})[0])
            actor = learned_actor if mode.startswith("standing_actor") else lambda obs: torch.zeros((len(obs), len(rig.actuators)))
            metrics, _ = evaluate(task, actor, episodes=episodes)
            cases = []
            for case, duration in enumerate(metrics["survival_seconds"]):
                phase = case / max(1, episodes - 1) * (task.reference.duration - seconds)
                row, arrays = summarize_case(task, task.engine.records[case], duration, phase)
                np.savez_compressed(args.out / f"{motion}-{mode}-{case}.npz", **arrays)
                cases.append(row)
            checks = tracking_checks(metrics, cases,
                                     require_lift=motion == "transfer_complete" and not reference.placed_step)
            if reference.placed_step and motion == "transfer_complete":
                checks["placed_step"] = all(
                    np.sign(c["left_foot_displacement_x_m"]) == np.sign(reference.requested_step_m)
                    and abs(c["left_foot_displacement_x_m"]) >= .1
                    for c in cases)
                checks["settled_finish"] = all(
                    c["final_two_second_min_pelvis_height_m"] >= .75
                    and c["final_two_second_max_horizontal_speed_m_s"] <= .2
                    for c in cases)
            result = dict(motion=motion, controller=mode, seconds=seconds, metrics=metrics,
                          cases=cases, checks=checks, passed=all(checks.values()),
                          intervention=intervention,
                          reset_queue="prefilled diagnostic intervention" if prefill else "production empty queue")
            results.append(result)
            (args.out / "progress.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(json.dumps(dict(motion=motion, controller=mode, survived=metrics["successes"],
                                  episodes=episodes, mean_survival=metrics["mean_survival_seconds"], checks=checks)), flush=True)
    report = dict(schema="mimic_transfer_tracking_v1", calibration_sha256=reference.sha256,
                  calibration_report_sha256=reference.report_sha256, model_sha256=sha256(rig.path),
                  actor_sha256=contract["actor_sha256"], contract=contract,
                  offline_checks=reference.checks, results=results, admitted_for_training=False,
                  scope="Native closed-loop diagnostics only. Default trials retain production physics/control. Optional interventions are labeled per trial and never qualify for promotion. Reference substitution changes motion/phase inputs; the Stand actor is not trained on this transfer. Substep traces end at first failure; short phase tests do not require both feet to lift. No training, export or viewer selection.")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interventions-only", action="store_true", help="Isolate initial pose, phase scaling and self-collisions in three diagnostic trials")
    run(parser.parse_args())
