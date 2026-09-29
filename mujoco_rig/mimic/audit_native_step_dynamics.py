"""Compare coarse reference dynamics with solved native contact at touchdown."""
import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.optimize import linprog

from .baseline import sha256
from .motion_dynamics import frame_dynamics, support_columns
from .native_engine import NativeDummyEngine
from .observation_step_teacher import ObservationStepTeacher
from .probe_single_support_entry import EntryReference
from .recovery_step_motion import evaluate
from .retarget_steps import validate_motion
from .rig import Rig
from .runtime import activate


SCHEMA = "mimic_native_step_dynamics_audit_v2"
FRAMES = (61, 62, 63, 64)


class CapturingNativeEngine(NativeDummyEngine):
    samples = None
    states = None
    pre_states = None

    def _before_physics_step(self, world, data):
        self.pre_states.append((data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()))

    def _after_physics_step(self, world, data):
        inertia = np.zeros(self.rig.model.nv)
        mujoco.mj_mulM(self.rig.model, data, inertia, data.qacc)
        residual = (inertia + data.qfrc_bias - data.qfrc_passive
                    - data.qfrc_actuator - data.qfrc_constraint - data.qfrc_applied)
        forces = self._read_ground_forces(data)
        feet = [self.rig.model.body(name).id - 1 for name in
                ("Foot_L", "Foot_R", "Toe_L", "Toe_R")]
        loads = np.abs(forces[feet, 2])
        contacts = [contact for contact in data.contact[:data.ncon]
                    if 0 in contact.geom]
        self.states.append((data.qpos.copy(), data.qvel.copy(),
                            data.xpos[[self.rig.model.body(name).id for name in
                                       ("Foot_L", "Foot_R")]].copy(),
                            (loads >= 5.).copy(), data.qacc.copy(),
                            data.qfrc_constraint.copy(), data.qfrc_actuator.copy()))
        self.samples.append((float(data.time),
                             float(np.linalg.norm(residual[:6])),
                             float(loads[0] + loads[2]),
                             float(loads[1] + loads[3]),
                             float(max((-contact.dist for contact in contacts),
                                       default=0.)),
                             len(contacts)))


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    activate(args.mimickit)
    from util import mp_util
    from .stand_task import StandTask

    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    source_path = args.source / "report.json"
    search_path = args.search / "report.json"
    motion_path = args.source / "step_reference.npz"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    search = json.loads(search_path.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if (search["source_report_sha256"] != sha256(source_path)
            or search["source_motion_sha256"] != sha256(motion_path)
            or search["model_sha256"] != sha256(rig.path)):
        raise ValueError("Native dynamics audit requires the passing source hashes")
    with np.load(motion_path, allow_pickle=False) as arrays:
        motion = {key: arrays[key].copy() for key in arrays.files}
    task = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz",
                     1, "cpu", engine_factory=CapturingNativeEngine,
                     control_mode="target_pd", contact_mode="support_v2")
    task.reference = EntryReference(motion["qpos"], rig, float(motion["dt"]))
    task.episode_seconds = task.reference.duration
    samples = []
    task.engine.samples = samples
    states = []
    task.engine.states = states
    pre_states = []
    task.engine.pre_states = pre_states
    teacher = ObservationStepTeacher(rig, contract, motion, source, search,
                                     task.dt,
                                     right_arm_clearance=args.right_arm_clearance,
                                     right_arm_clearance_scale=args.right_arm_clearance_scale)
    native, rollout, _ = evaluate(task, teacher, .15, 2.55)
    common = (rig, rollout["qpos"], rollout["qvel"],
              rollout["foot_targets"], rollout["contact"], float(rollout["dt"]))
    projected_checks, projected_metrics = validate_motion(
        *common, sole_contact=rollout["sole_contact"])
    measured_checks, measured_metrics = validate_motion(
        *common, support_contact=rollout["support_contact"])
    trace = np.asarray(samples)
    model, data = rig.model, mujoco.MjData(rig.model)
    weight = model.body_mass.sum() * np.linalg.norm(model.opt.gravity)
    exact_witness = []
    for index in np.flatnonzero((trace[:, 0] >= 3.05) & (trace[:, 0] <= 3.20)):
        pose, velocity, control = pre_states[index]
        data.qpos[:] = pose
        data.qvel[:] = velocity
        data.ctrl[:] = control
        mujoco.mj_forward(model, data)
        columns = []
        for contact in data.contact[:data.ncon]:
            if 0 not in (contact.geom1, contact.geom2):
                continue
            geom = contact.geom2 if contact.geom1 == 0 else contact.geom1
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacobian, None, contact.pos,
                          int(model.geom_bodyid[geom]))
            friction = max(model.geom_friction[0, 0], model.geom_friction[geom, 0])
            for x, y in itertools.product((-1., 1.), repeat=2):
                columns.append(jacobian[:3, :6].T @
                               np.array([x * friction, y * friction, 1.]))
        matrix = np.asarray(columns).T
        required = states[index][5][:6] / weight
        feasible = bool(columns) and linprog(
            np.zeros(len(columns)), A_eq=matrix, b_eq=required,
            bounds=(0., None), method="highs").success
        exact_witness.append(dict(index=int(index), seconds=float(trace[index, 0]),
                                  ground_contacts=len(columns) // 4,
                                  feasible=bool(feasible)))
    foot_bodies = {model.body(name).id for name in
                   ("Foot_L", "Foot_R", "Toe_L", "Toe_R")}
    contact_slip = []
    for index, (pose, velocity, control) in enumerate(pre_states):
        data.qpos[:] = pose
        data.qvel[:] = velocity
        data.ctrl[:] = control
        mujoco.mj_forward(model, data)
        for contact in data.contact[:data.ncon]:
            if 0 not in (contact.geom1, contact.geom2):
                continue
            geom = contact.geom2 if contact.geom1 == 0 else contact.geom1
            if model.geom_bodyid[geom] not in foot_bodies:
                continue
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacobian, None, contact.pos,
                          int(model.geom_bodyid[geom]))
            contact_slip.append((trace[index, 0],
                                 np.linalg.norm((jacobian @ velocity)[:2])))
    contact_slip = np.asarray(contact_slip)

    def slip_metrics(start, stop):
        speeds = contact_slip[(contact_slip[:, 0] >= start)
                              & (contact_slip[:, 0] < stop), 1]
        return dict(contact_points=len(speeds),
                    maximum_m_s=float(speeds.max()) if len(speeds) else None,
                    p95_m_s=float(np.percentile(speeds, 95)) if len(speeds) else None)
    measured_soles = rollout["support_contact"]
    heel_only = measured_soles[:, :2] & ~measured_soles[:, 2:]
    frame_rows = []
    poses = rollout["qpos"]
    for frame in FRAMES:
        required = frame_dynamics(model, data, poses, frame, float(rollout["dt"]))[:6]
        allowed_v6 = support_columns(model, data, rollout["sole_contact"][frame])[:6]
        allowed = support_columns(model, data, measured_soles[frame])[:6]
        center = frame * float(rollout["dt"])
        window = trace[np.abs(trace[:, 0] - center) <= .025]
        frame_rows.append(dict(frame=frame, seconds=center,
                               measured_sole_contact=measured_soles[frame].tolist(),
                               heel_only_contact=heel_only[frame].tolist(),
                               approximate_required_root_wrench_fraction=(required / weight).tolist(),
                               projected_v6_support_columns=allowed_v6.shape[1],
                               approximate_support_columns=allowed.shape[1],
                               native_samples=len(window),
                               native_max_root_residual_fraction=float(window[:, 1].max() / weight),
                               native_min_left_load_n=float(window[:, 2].min()),
                               native_min_right_load_n=float(window[:, 3].min()),
                               native_min_ground_contacts=int(window[:, 5].min()),
                               native_max_penetration_m=float(window[:, 4].max())))
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out / "native_rollout.npz", **rollout)
    np.savez_compressed(args.out / "physics_trace.npz", samples=trace,
                        pre_qpos=np.asarray([row[0] for row in pre_states]),
                        pre_qvel=np.asarray([row[1] for row in pre_states]),
                        ctrl=np.asarray([row[2] for row in pre_states]),
                        qpos=np.asarray([row[0] for row in states]),
                        qvel=np.asarray([row[1] for row in states]),
                        foot_targets=np.asarray([row[2] for row in states]),
                        support_contact=np.asarray([row[3] for row in states]),
                        qacc=np.asarray([row[4] for row in states]),
                        qfrc_constraint=np.asarray([row[5] for row in states]),
                        qfrc_actuator=np.asarray([row[6] for row in states]),
                        columns=("time_s", "root_residual_n", "left_load_n",
                                 "right_load_n", "penetration_m", "ground_contacts"))
    report = dict(schema=SCHEMA, model_sha256=sha256(rig.path),
                  source_report_sha256=sha256(source_path),
                  search_report_sha256=sha256(search_path),
                  native_passed=native["passed"], physics_samples=len(samples),
                  right_arm_clearance=args.right_arm_clearance,
                  right_arm_clearance_scale=args.right_arm_clearance_scale,
                  exact_contact_witness=dict(evaluated_steps=len(exact_witness),
                                             infeasible_steps=[row for row in exact_witness
                                                               if not row["feasible"]]),
                  contact_point_slip=dict(swing=slip_metrics(1.35, 2.55),
                                          impact=slip_metrics(2.55, 2.70),
                                          settled=slip_metrics(2.70, float(trace[-1, 0]) + .001)),
                  projected_v6_checks=projected_checks,
                  measured_support_checks=measured_checks,
                  projected_v6_root_failures=projected_metrics["root_wrench_consistency"]["infeasible_frames"],
                  measured_support_root_failures=measured_metrics["root_wrench_consistency"]["infeasible_frames"],
                  measured_support_penetration_m=measured_metrics["max_penetration_m"],
                  measured_support_slide_m_s=measured_metrics["max_stance_slide_m_s"],
                  measured_support_geometry=measured_metrics["support_geometry"],
                  heel_only_reference_frames=int(heel_only.any(axis=1).sum()),
                  heel_only_frames_by_foot=heel_only.sum(axis=0).tolist(),
                  maximum_native_root_residual_fraction=float(trace[:, 1].max() / weight),
                  frames=frame_rows,
                  scope="Physics-step audit of solved contact; does not change motion-reference admission")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--right-arm-clearance", action="store_true")
    parser.add_argument("--right-arm-clearance-scale", type=float, default=1.)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
