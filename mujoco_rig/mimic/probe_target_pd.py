"""Counterfactual: delayed reference targets with fresh 240 Hz PD torque feedback.

The native torque-actuated plant is unchanged. This tests a different control
contract, not an actor deployable through the existing delayed-torque policy API.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .rig import Rig
from .reference import StandReference
from .native_engine import NativeDummyEngine
from .static_support import solve_support


def rollout(rig, reference, offsets, gain, damping, seconds=3, hold=False, feedforward=None):
    count = len(offsets)
    engine = NativeDummyEngine(rig, count)
    offsets = torch.tensor(offsets, dtype=torch.float32)
    q, v = reference.sample(offsets)
    if hold:
        v.zero_()
    engine.reset_envs(torch.arange(count), q, v)
    joints = rig.model.actuator_trnid[rig.actuators, 0]
    q_indices, v_indices = rig.model.jnt_qposadr[joints], rig.model.jnt_dofadr[joints]
    feet = [rig.model.body(n).id for n in ("Foot_L", "Foot_R")]
    allowed = [rig.model.body(n).id - 1 for n in ("Foot_L", "Foot_R", "Toe_L", "Toe_R")]
    initial_feet = np.stack([d.xpos[feet, :2].copy() for d in engine.datas])
    kp = gain * rig.action_scale
    kd = damping * kp
    targets = np.zeros((2, count, 90))
    active = np.ones(count, dtype=bool)
    steps = np.zeros(count)
    errors = np.zeros((count, 3))
    max_drift = np.zeros(count)
    unstable = np.zeros(count, dtype=bool)
    trace = []
    dt = engine.get_timestep()
    for step in range(round(seconds / dt)):
        phase = offsets if hold else offsets + step * dt
        target_q, target_v = reference.sample(phase)
        if hold:
            target_v.zero_()
        incoming = np.concatenate([target_q[:, q_indices].numpy(), target_v[:, v_indices].numpy()], axis=1)
        ff = np.zeros((count, 30))
        if feedforward is not None:
            frame = (phase.numpy() / reference.dt).clip(0, len(feedforward) - 1)
            index = frame.astype(int).clip(max=len(feedforward) - 2)
            ff = feedforward[index] + (frame - index)[:, None] * (feedforward[index + 1] - feedforward[index])
        incoming = np.concatenate([incoming, ff], axis=1)
        delayed = targets[step % 2].copy()
        targets[step % 2] = incoming
        trace.append(np.stack([d.qpos.copy() for d in engine.datas]))
        clipping = np.zeros(count)
        for i, data in enumerate(engine.datas):
            if not active[i]:
                continue
            for _ in range(rig.decimation):
                torque = (kp * (delayed[i, :30] - data.qpos[q_indices])
                          + kd * (delayed[i, 30:60] - data.qvel[v_indices])
                          + delayed[i, 60:] * rig.action_scale) if step >= 2 else np.zeros(30)
                clipping[i] += np.mean(np.abs(torque) > rig.action_scale) / rig.decimation
                data.ctrl[:] = 0
                data.ctrl[rig.actuators] = np.clip(torque, -rig.action_scale, rig.action_scale)
                mujoco.mj_step(rig.model, data)
            mujoco.mj_forward(rig.model, data)
            unstable[i] |= bool(any(data.warning.number) or not np.isfinite(data.qpos).all()
                                or np.max(np.abs(data.qvel)) > 1000)
        end_phase = offsets if hold else offsets + (step + 1) * dt
        end_q, _ = reference.sample(end_phase)
        for i, data in enumerate(engine.datas):
            if active[i] and not unstable[i]:
                errors[i] += [np.linalg.norm(data.qpos[:3] - end_q[i, :3].numpy()),
                              np.sqrt(np.mean((data.qpos[q_indices] - end_q[i, q_indices].numpy())**2)), clipping[i]]
                max_drift[i] = max(max_drift[i], np.linalg.norm(data.xpos[feet, :2] - initial_feet[i], axis=1).max())
        steps += active
        forces = engine.get_ground_contact_forces(0).numpy()
        forces[:, allowed] = 0
        failed = np.array([d.qpos[2] < 0.65 for d in engine.datas]) | (np.abs(forces).max(axis=(1, 2)) > 1) | unstable
        active &= ~failed
        if not active.any():
            break
    averages = errors / steps[:, None].clip(1)
    return {"successes": int(active.sum()), "episodes": count, "seconds": seconds,
            "survival_seconds": (steps * dt).tolist(), "mean_survival_seconds": float(steps.mean() * dt),
            "mean_root_tracking_error_m": float(averages[:, 0].mean()),
            "mean_joint_tracking_rmse_rad": float(averages[:, 1].mean()),
            "clipped_torque_fraction": float(averages[:, 2].mean()),
            "max_foot_drift_m": float(max_drift.max()), "numerical_failures": int(unstable.sum())}, np.array(trace)


def probe(out, support_feedforward=False):
    out.mkdir(parents=True, exist_ok=False)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    reference = StandReference(Path(__file__).parent / "assets/stand_reference.npz", rig, "cpu")
    end = reference.duration - 3
    candidates = [(g, d) for g in (1., 2., 4., 8.) for d in (0.02, 0.05, 0.1)]
    protocol = {"counterfactual": "Two control-step reference-target delay; fresh PD torque feedback every physics step",
                "production_control_contract": False, "initial_two_commands": "zero motor torque",
                "model_sha256": rig.contract()["model_sha256"], "reference_sha256": reference.sha256,
                "torque_scale_nm": rig.action_scale.tolist(), "candidates": candidates,
                "calibration_phase_indices": [0, 3, 7], "held_out_phase_indices": [1, 2, 4, 5, 6],
                "required_successes": 8, "root_error_gate_m": 0.05, "joint_error_gate_rad": 0.15,
                "foot_drift_gate_m": 0.03, "hold_seconds": 10}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    feedforward = None
    if support_feedforward:
        protocol.update(support_feedforward=True, passive_joint_residuals="unconstrained; unactuated joints may settle")
        (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        support = [solve_support(rig, q, passive_tolerance=None, minimum_norm=True) for q in reference.qpos.numpy()]
        (out / "support.json").write_text(json.dumps(support, indent=2), encoding="utf-8")
        if not all(s["feasible"] for s in support):
            raise RuntimeError("Reference needs unsupported torque/contact forces")
        feedforward = np.array([s["normalized_motor_torque"] for s in support])
    calibration = []
    for gain, damping in candidates:
        result, _ = rollout(rig, reference, [0, end * 3 / 7, end], gain, damping, feedforward=feedforward)
        result.update(gain=gain, damping_time=damping)
        calibration.append(result)
        print(json.dumps(result), flush=True)
    (out / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    best = min(calibration, key=lambda r: (-r["successes"], -r["mean_survival_seconds"], r["mean_root_tracking_error_m"]))
    tracking, trace = rollout(rig, reference, np.linspace(0, end, 8), best["gain"], best["damping_time"], feedforward=feedforward)
    holding, hold_trace = rollout(rig, reference, np.linspace(0, end, 8), best["gain"], best["damping_time"], 10, True, feedforward)
    np.savez_compressed(out / "rollouts.npz", tracking=trace, holding=hold_trace)
    def passed(result):
        return (result["successes"] == 8 and result["mean_root_tracking_error_m"] <= 0.05
                and result["mean_joint_tracking_rmse_rad"] <= 0.15 and result["max_foot_drift_m"] <= 0.03)
    report = {"under_production_control_contract": False, "selected": {k: best[k] for k in ("gain", "damping_time")},
              "tracking": tracking, "hold": holding, "tracking_gate_passed": passed(tracking), "hold_gate_passed": passed(holding)}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--support-feedforward", action="store_true")
    args = parser.parse_args()
    probe(args.out, args.support_feedforward)
