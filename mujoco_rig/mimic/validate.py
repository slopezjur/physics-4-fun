"""Compatibility gates, independent of rewards, motion data and learned policies."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .rig import Rig
from .runtime import activate
from .model_contract import model_checks


def validate(mimickit: Path, rig: Rig, output: Path) -> dict:
    activate(mimickit)
    from .kinematics import DummyKinematics
    from .newton_adapter import DummyNewtonEngine

    output.mkdir(parents=True, exist_ok=True)
    engine = DummyNewtonEngine(rig, 2)
    source, imported = rig.model, engine._solver.mj_model
    checks = model_checks(source, imported)
    mapper = DummyKinematics(rig, "cuda:0")
    rng = np.random.default_rng(210921)
    poses = np.tile(rig.rest, (65, 1))
    poses[1:, 7:] = rng.uniform(source.jnt_range[1:, 0], source.jnt_range[1:, 1], (64, source.nq - 7))
    poses[1:, :3] += rng.uniform(-0.2, 0.2, (64, 3))
    root_quats = rng.normal(size=(64, 4))
    poses[1:, 3:7] = root_quats / np.linalg.norm(root_quats, axis=-1, keepdims=True)
    pos, rot = mapper.from_native_qpos(torch.tensor(poses, device="cuda:0", dtype=torch.float32))
    pos, rot = pos.cpu().numpy(), rot.cpu().numpy()
    checks["reference_poses_finite"] = bool(np.isfinite(pos).all() and np.isfinite(rot).all())
    max_mapper_pos = max_mapper_rot = max_newton_pos = 0.0
    contact_mismatches = 0
    for index, pose in enumerate(poses):
        data, other = mujoco.MjData(source), mujoco.MjData(imported)
        data.qpos[:] = other.qpos[:] = pose
        mujoco.mj_forward(source, data)
        mujoco.mj_forward(imported, other)
        expected_rot = data.xquat[1:, [1, 2, 3, 0]]
        max_mapper_pos = max(max_mapper_pos, float(np.max(np.abs(pos[index] - data.xpos[1:]))))
        quat_error = np.minimum(np.linalg.norm(rot[index] - expected_rot, axis=-1),
                                np.linalg.norm(rot[index] + expected_rot, axis=-1))
        max_mapper_rot = max(max_mapper_rot, float(quat_error.max()))
        engine.reset(pose)
        max_newton_pos = max(max_newton_pos, float(np.max(np.abs(engine.get_body_pos(0)[0].cpu().numpy() - data.xpos[1:]))))
        contacts = lambda d: {tuple(sorted(map(int, c.geom))) for c in d.contact[:d.ncon]}
        contact_mismatches += contacts(data) != contacts(other)
    checks["reference_pose_mapper"] = max_mapper_pos < 3e-6 and max_mapper_rot < 3e-6
    checks["newton_pose_mapper"] = max_newton_pos < 3e-6
    checks["sampled_contact_pairs"] = contact_mismatches == 0

    # Matched open-loop controls: this verifies latency/scaling, not balance ability.
    engine.reset()
    datas = [mujoco.MjData(source) for _ in range(2)]
    for data in datas:
        data.qpos[:] = rig.rest
        mujoco.mj_forward(source, data)
    commands = rng.uniform(-0.05, 0.05, (30, 2, len(rig.actuators))).astype(np.float32)
    commands[:, 1] *= -1
    queue = [np.zeros_like(commands[0]) for _ in range(rig.contract()["latency_steps"])]
    ctrl_error = position_error = joint_error = 0.0
    finite = True
    first_trace = []
    for command in commands:
        queue.append(command.clip(-1, 1))
        applied = queue.pop(0)
        expected = np.zeros((2, source.nu))
        expected[:, rig.actuators] = applied * rig.action_scale
        engine.set_cmd(0, torch.tensor(command, device="cuda:0"))
        ctrl_error = max(ctrl_error, float(np.max(np.abs(expected - engine.native_ctrl.cpu().numpy()))))
        engine.step()
        for data, ctrl in zip(datas, expected):
            data.ctrl[:] = ctrl
            for _ in range(rig.decimation):
                mujoco.mj_step(source, data)
        gpu_q = engine.native_qpos().cpu().numpy()
        first_trace.append(gpu_q.copy())
        cpu_q = np.stack([d.qpos for d in datas])
        finite &= bool(np.isfinite(gpu_q).all())
        position_error = max(position_error, float(np.max(np.abs(gpu_q[:, :3] - cpu_q[:, :3]))))
        joint_error = max(joint_error, float(np.max(np.abs(gpu_q[:, 7:] - cpu_q[:, 7:]))))
    checks["delayed_torque_controls"] = ctrl_error < 2e-5
    checks["short_rollout_finite"] = finite
    checks["short_rollout_position"] = position_error < 0.02
    checks["short_rollout_joints"] = joint_error < 0.1
    engine.reset()
    reset_error = 0.0
    reset_root_error = reset_joint_error = 0.0
    reset_finite = True
    first_step_error = None
    reset_state_error = float(np.max(np.abs(engine.native_qpos().cpu().numpy() - rig.rest)))
    for command, expected_q in zip(commands, first_trace):
        engine.set_cmd(0, torch.tensor(command, device="cuda:0"))
        engine.step()
        difference = np.abs(engine.native_qpos().cpu().numpy() - expected_q)
        if not np.isfinite(difference).all():
            reset_finite = False
            break
        if first_step_error is None:
            first_step_error = float(difference.max())
        reset_error = max(reset_error, float(difference.max()))
        reset_root_error = max(reset_root_error, float(difference[:, :3].max()))
        reset_joint_error = max(reset_joint_error, float(difference[:, 7:].max()))
    # GPU factorization/contacts use floating-point reductions. Identical reset
    # inputs can diverge over time; require clean state and first-step agreement,
    # then apply the same trajectory tolerances used for native/GPU parity.
    checks["reset_state"] = reset_state_error < 1e-6
    checks["reset_first_step"] = first_step_error is not None and first_step_error < 1e-6
    checks["reset_replay"] = reset_finite and reset_root_error < 0.02 and reset_joint_error < 0.1
    report = {"passed": all(checks.values()), "checks": checks,
              "metrics": {"pose_count": len(poses), "mapper_position_error_m": max_mapper_pos,
                          "mapper_quaternion_error": max_mapper_rot, "newton_position_error_m": max_newton_pos,
                          "contact_pair_mismatches": contact_mismatches, "control_error_nm": ctrl_error,
                          "rollout_seconds": len(commands) * rig.decimation * source.opt.timestep,
                          "rollout_root_error_m": position_error, "rollout_joint_error_rad": joint_error,
                          "reset_replay_error": reset_error, "reset_state_error": reset_state_error,
                          "reset_first_step_error": first_step_error,
                          "reset_root_error_m": reset_root_error, "reset_joint_error_rad": reset_joint_error},
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "newton", "mujoco", "mujoco-warp", "warp-lang")},
              "scope": "Compatibility only. No motion retargeting, policy training, or Godot policy deployment."}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "control_contract.json").write_text(json.dumps(rig.contract(), indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--asset", type=Path, default=Path(__file__).resolve().parents[1] / "dummy.xml")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.mimickit, Rig.load(args.asset), args.out)
    raise SystemExit(0 if result["passed"] else 1)
