"""Measure Newton's MJCF import before adopting it for the existing Godot rig."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import newton
import numpy as np
import warp as wp


def probe(asset: Path, output: Path, device: str = "cuda:0") -> dict:
    source = mujoco.MjModel.from_xml_path(str(asset))
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_mjcf(str(asset), ignore_inertial_definitions=False,
                     collapse_fixed_joints=False, enable_self_collisions=True,
                     convert_3d_hinge_to_ball_joints=False, ctrl_direct=True)
    model = builder.finalize(device=device)
    output.mkdir(parents=True, exist_ok=True)
    solver = newton.solvers.SolverMuJoCo(model, njmax=450, nconmax=150,
                                        save_to_mjcf=str(output / "newton_import.xml"))
    imported = solver.mj_model
    checks = {}
    for field in ("nq", "nv", "nu", "nbody", "ngeom", "nexclude"):
        checks[field] = {"source": int(getattr(source, field)), "imported": int(getattr(imported, field))}
    # Name mapping is recorded explicitly: Newton can rename/split joints and shapes.
    for kind, count in (("body", "nbody"), ("joint", "njnt"), ("geom", "ngeom"), ("actuator", "nu")):
        checks[kind + "_names"] = {
            "source": [getattr(source, kind)(i).name for i in range(getattr(source, count))],
            "imported": [getattr(imported, kind)(i).name for i in range(getattr(imported, count))],
        }
    for field in ("body_mass", "body_inertia", "jnt_type", "jnt_axis", "jnt_pos", "jnt_range",
                  "jnt_stiffness", "dof_damping", "dof_armature", "geom_type", "geom_size",
                  "geom_friction", "geom_contype", "geom_conaffinity", "actuator_gainprm",
                  "actuator_biasprm", "actuator_forcerange"):
        a, b = np.asarray(getattr(source, field)), np.asarray(getattr(imported, field))
        checks[field] = {"source": a.tolist(), "imported": b.tolist(),
                         "same_order_equal": bool(a.shape == b.shape and np.allclose(a, b, atol=1e-6, rtol=1e-5))}
    state, next_state = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    for _ in range(8):
        state.clear_forces()
        solver.step(state, next_state, control, None, source.opt.timestep)
        state, next_state = next_state, state
    wp.synchronize()
    checks["finite_after_eight_steps"] = bool(np.isfinite(state.joint_q.numpy()).all())
    (output / "newton_import.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=Path(__file__).resolve().parents[1] / "dummy.xml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = probe(args.asset, args.out, args.device)
    print(json.dumps({k: v.get("same_order_equal", v) if isinstance(v, dict) else v
                      for k, v in result.items() if not k.endswith("_names")}, indent=2))
