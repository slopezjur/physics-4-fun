"""Quasi-static support/torque budget diagnostic; not a dynamic balance controller."""
import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import linprog, minimize, Bounds, LinearConstraint

from .rig import Rig


def solve_support(rig, qpos, actual_contacts=False, passive_tolerance=0.5, minimum_norm=False,
                  support_names=("Foot_L", "Foot_R", "Toe_L", "Toe_R")):
    model, data = rig.model, mujoco.MjData(rig.model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    weight = model.body_mass.sum() * abs(model.opt.gravity[2])
    feet = {model.body(name).id for name in support_names}
    contacts = []
    if actual_contacts:
        contacts = [(int(model.geom_bodyid[c.geom[1]]), c.pos.copy(), float(c.friction[0]))
                    for c in data.contact[:data.ncon] if c.geom[0] == 0 and model.geom_bodyid[c.geom[1]] in feet]
    else:
        for geom in range(model.ngeom):
            body = model.geom_bodyid[geom]
            if body not in feet or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            corners = np.array(list(itertools.product((-1, 1), repeat=3))) * model.geom_size[geom]
            corners = corners @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            for point in corners[np.argsort(corners[:, 2])[:4]]:
                # Prospective sole contacts: permit only sub-millimetre settling.
                if abs(point[2]) > 0.001:
                    raise ValueError("Reference sole is not within 1 mm of the floor")
                point[2] = 0
                contacts.append((body, point, max(model.geom_friction[geom, 0], model.geom_friction[0, 0])))
    if not contacts:
        return {"feasible": False, "contacts": 0, "solver_message": "No supporting contacts"}
    dofs = model.jnt_dofadr[model.actuator_trnid[rig.actuators, 0]]
    motors = np.zeros((model.nv, len(dofs)))
    motors[dofs, np.arange(len(dofs))] = rig.action_scale
    columns = []
    for body, point, friction in contacts:
        jac = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jac, None, point, int(body))
        for x, y in itertools.product((-1, 1), repeat=2):
            ray = np.array([x * friction / np.sqrt(2), y * friction / np.sqrt(2), 1.])
            columns.append(jac.T @ ray * weight)
    matrix = np.column_stack([motors, np.array(columns).T, np.zeros(model.nv)])
    required = data.qfrc_bias - data.qfrc_passive - data.qfrc_actuator
    controlled = np.r_[np.arange(6), dofs]
    passive = np.setdiff1d(np.arange(model.nv), controlled)
    # Passive wrists/neck can settle slightly; report this explicitly rather than
    # inventing actuators or requiring their unweighted reference angles exactly.
    inequality, bound = [], []
    for index in range(len(dofs)):
        for sign in (-1, 1):
            row = np.zeros(matrix.shape[1]); row[index] = sign; row[-1] = -1
            inequality.append(row); bound.append(0)
    for index in passive if passive_tolerance is not None else []:
        for sign in (-1, 1):
            inequality.append(sign * matrix[index]); bound.append(sign * required[index] + passive_tolerance)
    objective = np.zeros(matrix.shape[1]); objective[-1] = 1
    result = linprog(objective, A_ub=np.array(inequality), b_ub=np.array(bound),
                     A_eq=matrix[controlled] / weight, b_eq=required[controlled] / weight,
                     bounds=[(-1, 1)] * len(dofs) + [(0, None)] * len(columns) + [(0, 1)], method="highs")
    report = {"feasible": bool(result.success), "contacts": len(contacts),
              "com_world": data.subtree_com[1].tolist(), "passive_residual_tolerance_nm": passive_tolerance}
    if result.success:
        if minimum_norm:
            weights = np.r_[np.ones(len(dofs)), np.full(len(columns), 0.001), 0.001]
            refined = minimize(lambda x: 0.5 * np.sum(weights * x*x), result.x,
                               jac=lambda x: weights * x, method="SLSQP",
                               constraints=[LinearConstraint(matrix[controlled] / weight, required[controlled] / weight,
                                                             required[controlled] / weight),
                                            LinearConstraint(np.array(inequality), -np.inf, np.array(bound))],
                               bounds=Bounds(np.r_[np.full(len(dofs), -1.), np.zeros(len(columns) + 1)],
                                             np.r_[np.ones(len(dofs)), np.full(len(columns), np.inf), 1.]),
                               options={"maxiter": 200, "ftol": 1e-10})
            if not refined.success:
                raise RuntimeError("Support torque refinement failed: " + refined.message)
            result = refined
        residual = matrix @ result.x - required
        report.update(max_torque_budget_fraction=float(np.abs(result.x[:len(dofs)]).max()),
                      controlled_equilibrium_residual=float(np.abs(residual[controlled]).max()),
                      passive_residual_nm=float(np.abs(residual[passive]).max()),
                      normalized_motor_torque=result.x[:len(dofs)].tolist())
    else:
        report["solver_message"] = result.message
    return report


def audit(out):
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    reference = Path(__file__).parent / "assets/stand_reference.npz"
    with np.load(reference, allow_pickle=False) as data:
        poses = data["qpos"]
    frames = np.linspace(0, len(poses) - 1, 8).astype(int)
    report = {"scope": "Quasi-static necessary-condition check; prospective two-foot contacts allow <1 mm settling and 0.5 Nm passive residual; not a proof of dynamic tracking",
              "model_sha256": rig.contract()["model_sha256"], "frames": []}
    for frame in frames:
        report["frames"].append({"frame": int(frame), "actual_contacts": solve_support(rig, poses[frame], True),
                                 "prospective_contacts": solve_support(rig, poses[frame])})
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    audit(parser.parse_args().out)
