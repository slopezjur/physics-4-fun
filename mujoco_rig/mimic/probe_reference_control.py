"""Bounded controller feasibility probe; never changes the rig, PPO or production actor."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .rig import Rig
from .runtime import activate
from .reference_controller import ReferencePD, ImmediateTorqueDiagnostic


@torch.no_grad()
def rollout(task, controller, offsets, seconds=3.0, hold=False):
    from envs.base_env import EnvMode, DoneFlags
    task.set_mode(EnvMode.TEST)
    task.reset()
    task.offset[:] = torch.tensor(offsets, device=task.device)
    q, v = task.reference.sample(task.offset)
    if hold:
        v.zero_()
    task.engine.reset_envs(task.ids, q, v)
    controller.hold = hold
    count = len(offsets)
    alive = torch.ones(count, device=task.device, dtype=torch.bool)
    lengths = torch.zeros(count, device=task.device)
    sums = torch.zeros((count, 3), device=task.device)
    peaks = torch.zeros(count, device=task.device)
    unstable = torch.zeros(count, device=task.device, dtype=torch.bool)
    initial_feet = task.engine.get_body_pos(0)[:, task.feet, :2].clone()
    trajectory, commands = [], []
    for _ in range(round(seconds / task.dt)):
        if hold:
            # Keep reward/target phase fixed while physical time continues normally.
            task.steps.zero_()
        obs = task.observations()
        command = controller(obs)
        trajectory.append(task.engine.native_qpos().cpu().numpy())
        commands.append(command.clamp(-1, 1).cpu().numpy())
        _, _, done, _ = task.step(command)
        phase = task.offset if hold else task.offset + task.get_env_time()
        target, _ = task.reference.sample(phase)
        actual = task.engine.native_qpos()
        bad = ~torch.isfinite(actual).all(dim=1) | (actual.abs().amax(dim=1) > 1000)
        bad |= task.engine.get_dof_vel(0).abs().amax(dim=1) > 1000
        if hasattr(task.engine, "datas"):
            bad |= torch.tensor([any(d.warning.number) for d in task.engine.datas], device=task.device)
        unstable |= bad & alive
        root_error = torch.linalg.vector_norm(actual[:, :3] - target[:, :3], dim=-1)
        joint_error = torch.sqrt(((actual[:, controller.dofs + 1] - target[:, controller.dofs + 1]) ** 2).mean(dim=1))
        clip_fraction = (command.abs() > 1).float().mean(dim=1)
        sums += torch.nan_to_num(torch.stack([root_error, joint_error, clip_fraction], dim=1)) * (alive & ~bad)[:, None]
        foot_error = torch.linalg.vector_norm(task.engine.get_body_pos(0)[:, task.feet, :2] - initial_feet, dim=-1).amax(dim=1)
        peaks = torch.maximum(peaks, torch.nan_to_num(foot_error) * (alive & ~bad))
        lengths += alive
        alive &= done != DoneFlags.FAIL.value
        alive &= ~bad
        if not alive.any():
            break
        task.reset(torch.nonzero(~alive).flatten())
    averages = sums / lengths.clamp_min(1)[:, None]
    return {"successes": int(alive.sum()), "episodes": count, "seconds": seconds,
            "survival_seconds": (lengths * task.dt).cpu().tolist(),
            "mean_survival_seconds": float(lengths.mean() * task.dt),
            "mean_root_tracking_error_m": float(averages[:, 0].mean()),
            "mean_joint_tracking_rmse_rad": float(averages[:, 1].mean()),
            "clipped_command_fraction": float(averages[:, 2].mean()),
            "numerical_failures": int(unstable.sum()),
            "max_foot_drift_m": float(peaks.max())}, {"qpos": np.asarray(trajectory), "actions": np.asarray(commands)}


def probe(args):
    activate(args.mimickit)
    from .stand_task import StandTask
    from .native_engine import NativeDummyEngine
    from .newton_adapter import DummyNewtonEngine
    args.out.mkdir(parents=True, exist_ok=False)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    reference = Path(__file__).parent / "assets/stand_reference.npz"
    calibration = StandTask(rig, reference, 3, "cpu", engine_factory=NativeDummyEngine)
    def configure_delay(task):
        if args.zero_delay:
            control = ImmediateTorqueDiagnostic(rig, task.get_num_envs(), task.device)
            task.engine.policy_control = control
            if hasattr(task.engine, "datas"):
                task.engine.native_ctrl = control.ctrl
    configure_delay(calibration)
    phase_end = calibration.reference.duration - 3
    candidates = ([dict(gain=g, damping_time=d) for g in (1., 2., 4., 8.) for d in (0.02, 0.05, 0.1)]
                  if args.family == "normalized" else
                  [dict(gain=f, damping_time=z, inertia_scaled=True, gravity_feedforward=ff)
                   for f in (1., 2., 3.) for z in (0.7, 1.) for ff in (False, True)])
    protocol = {"candidates": candidates, "calibration_phase_indices": [0, 3, 7],
                "protocol_version": 2,
                "evaluation_phases": 8, "held_out_phase_indices": [1, 2, 4, 5, 6],
                "family": args.family,
                "zero_delay_counterfactual": args.zero_delay,
                "actual_latency_steps": 0 if args.zero_delay else 2,
                "ranking": "most successes, then longest mean survival, then lowest mean root error",
                "hold_test_seconds": 10, "model_sha256": rig.contract()["model_sha256"],
                "reference_sha256": calibration.reference.sha256,
                "required_successes": 8, "max_mean_root_error_m": 0.05,
                "max_joint_rmse_rad": 0.15, "max_foot_drift_m": 0.03,
                "control_contract": rig.contract(),
                "limitations": "Finite PD search: failure is not a proof of physical infeasibility; no root balance feedback. Inertia-family feedforward cancels static joint bias/passive forces but does not allocate support forces."}
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    start = time.monotonic()
    results = []
    for parameters in candidates:
        result, _ = rollout(calibration, ReferencePD(calibration, **parameters), [0, phase_end * 3 / 7, phase_end])
        result["parameters"] = parameters
        results.append(result)
        (args.out / "calibration.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
    best = min(results, key=lambda r: (-r["successes"], -r["mean_survival_seconds"], r["mean_root_tracking_error_m"]))
    native = StandTask(rig, reference, 8, "cpu", engine_factory=NativeDummyEngine)
    configure_delay(native)
    controller = ReferencePD(native, **best["parameters"])
    offsets = np.linspace(0, phase_end, 8)
    tracking, trace = rollout(native, controller, offsets)
    np.savez_compressed(args.out / "tracking_rollout.npz", **trace)
    holding, hold_trace = rollout(native, controller, offsets, seconds=10, hold=True)
    np.savez_compressed(args.out / "hold_rollout.npz", **hold_trace)
    gpu = StandTask(rig, reference, 8, "cuda:0", engine_factory=DummyNewtonEngine)
    configure_delay(gpu)
    gpu_tracking, _ = rollout(gpu, ReferencePD(gpu, **best["parameters"]), offsets)
    def passed(result):
        return (result["successes"] == 8 and result["mean_root_tracking_error_m"] <= 0.05
                and result["mean_joint_tracking_rmse_rad"] <= 0.15 and result["max_foot_drift_m"] <= 0.03)
    report = {"selected": best["parameters"],
              "under_production_control_contract": not args.zero_delay,
              "native_tracking": tracking, "native_hold": holding, "newton_tracking": gpu_tracking,
              "tracking_gate_passed": passed(tracking), "hold_gate_passed": passed(holding),
              "elapsed_seconds": time.monotonic() - start,
              "production_modified": False}
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--family", choices=("normalized", "inertia"), default="normalized")
    parser.add_argument("--zero-delay", action="store_true", help="Diagnostic counterfactual only; changes the action contract")
    probe(parser.parse_args())
