"""Kinematic reach diagnostic; target poses are not recovery objectives or policies."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .baseline import sha256
from .rig import Rig


def run(args):
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    model, data = rig.model, mujoco.MjData(rig.model)
    joints = model.actuator_trnid[rig.actuators, 0]
    leg = np.array([i for i, a in enumerate(rig.actuators)
                    if model.actuator(int(a)).name.startswith(("Thigh_", "Shin_", "Foot_", "Toe_"))])
    indices = model.jnt_qposadr[joints[leg]]
    bodies = [model.body(name).id for name in ("Foot_L", "Foot_R", "Toe_L", "Toe_R")]
    with np.load(args.reference) as reference:
        postures = reference["qpos"].copy()
    reports, poses = [], []
    for amplitude in (.25, .5):
        for posture, q in enumerate(postures):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            original = data.xpos[bodies].copy()
            original_rotation = data.xmat[bodies].reshape(4, 3, 3).copy()
            for foot in (0, 1):
                for direction in range(4):
                    target = original.copy()
                    offset = np.array([np.cos(direction * np.pi / 2) * .12,
                                       np.sin(direction * np.pi / 2) * .12, .03])
                    target[[foot, foot + 2]] += offset
                    support = original[[1 - foot, 3 - foot], :2].mean(0)
                    lower = np.r_[[-.18, -.18, -.08], np.maximum(model.jnt_range[joints[leg], 0], q[indices] - amplitude)]
                    upper = np.r_[[.18, .18, .03], np.minimum(model.jnt_range[joints[leg], 1], q[indices] + amplitude)]

                    def residual(x):
                        data.qpos[:] = q
                        data.qpos[:3] += x[:3]
                        data.qpos[indices] = x[3:]
                        mujoco.mj_forward(model, data)
                        return np.r_[(50 * (data.xpos[bodies] - target)).ravel(),
                                     (2 * (data.xmat[bodies].reshape(4, 3, 3) - original_rotation)).ravel(),
                                     10 * (data.subtree_com[1, :2] - support),
                                     .02 * (x[3:] - q[indices])]

                    result = least_squares(residual, np.r_[np.zeros(3), q[indices]], bounds=(lower, upper),
                                           max_nfev=150, ftol=1e-6, xtol=1e-6, gtol=1e-6)
                    residual(result.x)
                    error = float(np.linalg.norm(data.xpos[bodies] - target, axis=-1).max())
                    tilt = float(np.max(np.linalg.norm(data.xmat[bodies].reshape(4, 3, 3)[:, :, 2]
                                                        - original_rotation[:, :, 2], axis=-1)))
                    com_error = float(np.linalg.norm(data.subtree_com[1, :2] - support))
                    reports.append(dict(amplitude_rad=amplitude, posture=posture, foot=foot, direction=direction,
                        max_position_error_m=error, foot_normal_error=tilt, support_com_error_m=com_error,
                        feasible=error < .01 and tilt < .1 and com_error < .04))
                    poses.append(data.qpos.copy())
    report = dict(schema="stationary_step_kinematic_authority_v1", model_sha256=sha256(rig.path),
        reference_sha256=sha256(args.reference), clearance_m=.03, displacement_m=.12,
        scope="Kinematic feasibility only; does not establish motor tracking, dynamic recovery or learning",
        summaries=[dict(amplitude_rad=a, feasible=sum(r["feasible"] for r in reports if r["amplitude_rad"] == a),
                        cases=sum(r["amplitude_rad"] == a for r in reports)) for a in (.25, .5)], cases=reports)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "poses.npz", qpos=np.asarray(poses))
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summaries"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
