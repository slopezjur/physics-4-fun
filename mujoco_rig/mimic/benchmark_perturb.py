"""Measure an unchanged Stand ONNX policy under a fixed set of COM force pulses."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .baseline import sha256
from .checkpoints import validate_contract
from .perturb import baseline_trials, RECOVERY, RecoveryMetrics
from .rig import Rig
from .runtime import activate


def sample_metrics(task, metrics, completed_step):
    q = task.engine.native_qpos().cpu().numpy()
    v = np.concatenate((task.engine.get_root_vel(0).cpu().numpy(),
                        task.engine.get_root_ang_vel(0).cpu().numpy()), axis=1)
    feet = task.engine.get_body_pos(0)[:, task.feet].cpu().numpy()
    loads = task.engine.get_ground_contact_forces(0)[:, task.feet, 2].abs().cpu().numpy()
    for index, metric in enumerate(metrics):
        if metric is not None:
            metric.sample(completed_step, q[index], v[index], feet[index], loads[index])


def evaluate_pushes(task, session, trials):
    from envs.base_env import EnvMode, DoneFlags
    task.set_mode(EnvMode.TEST)
    task.episode_seconds = trials[0].trial_steps * task.dt
    task.reset()
    task.offset[:] = torch.tensor([t.phase for t in trials], device=task.device)
    q, v = task.reference.sample(task.offset)
    task.engine.reset_envs(task.ids, q, v)
    metrics = [RecoveryMetrics(t, task.dt) for t in trials]
    active_metrics = list(metrics)
    sample_metrics(task, metrics, 0)
    fallen = np.zeros(len(trials), dtype=bool)
    traces = [[] for _ in trials]
    for step in range(trials[0].trial_steps):
        forces = np.asarray([t.force_at(step) if not fallen[i] else (0, 0, 0) for i, t in enumerate(trials)], dtype=np.float32)
        task.engine.set_external_force(task.rig.model.body("Chest").id, forces)
        obs = task.observations().cpu().numpy()
        action = session.run(None, {"observation": obs})[0]
        _, _, done, _ = task.step(torch.tensor(action, device=task.device))
        sample_metrics(task, active_metrics, step + 1)
        poses = task.engine.native_qpos().cpu().numpy()
        for i, trial in enumerate(trials):
            if not fallen[i] and (step < 2 or trial.start_step - 2 <= step < trial.start_step + trial.duration_steps + 12):
                traces[i].append({"step": step, "force": forces[i].tolist(), "observation": obs[i].tolist(),
                                  "action": action[i].tolist(), "qpos_after": poses[i].tolist()})
        failed = done.cpu().numpy() == DoneFlags.FAIL.value
        fallen |= failed
        for i in np.flatnonzero(failed):
            active_metrics[i] = None
        # Keep terminated worlds finite; never count later state as recovery.
        reset_ids = torch.tensor(np.flatnonzero(fallen), device=task.device, dtype=torch.long)
        task.reset(reset_ids)
        if fallen.all():
            break
    return [{"name": t.name, **m.result(bool(f)), "trace": trace}
            for t, m, f, trace in zip(trials, metrics, fallen, traces)]


def force_preflight(rig, reference):
    from .stand_task import StandTask
    from .native_engine import NativeDummyEngine
    from envs.base_env import EnvMode
    gpu = StandTask(rig, reference, 4, control_mode="target_pd")
    cpu = StandTask(rig, reference, 4, "cpu", engine_factory=NativeDummyEngine, control_mode="target_pd")
    for task in (gpu, cpu):
        task.set_mode(EnvMode.TEST)
        task.reset()
    errors = {"pose": 0., "velocity": 0.}
    body = rig.model.body("Chest").id
    for step in range(24):
        force = np.array([[40, 0, 0], [-40, 0, 0], [0, 40, 0], [0, -40, 0]], dtype=np.float32) if 2 <= step < 8 else np.zeros((4, 3), dtype=np.float32)
        for task in (gpu, cpu):
            task.engine.set_external_force(body, force)
            task.step(torch.zeros((4, 30), device=task.device))
        errors["pose"] = max(errors["pose"], float((gpu.engine.native_qpos().cpu() - cpu.engine.native_qpos()).abs().max()))
        errors["velocity"] = max(errors["velocity"], float((gpu.engine.get_root_vel(0).cpu() - cpu.engine.get_root_vel(0)).abs().max()))
    # Enqueue a pulse then reset one world: its pending force must disappear without
    # erasing another world's pulse. A full reset must clear every external force.
    for task in (gpu, cpu):
        task.engine.set_external_force(body, np.full((4, 3), 10., dtype=np.float32))
        task.reset(torch.tensor([0], device=task.device))
    pending_gpu = gpu.engine._sim_state.body_force[0][:, body - 1, :3].cpu().numpy()
    pending_cpu = np.stack([d.xfrc_applied[body, :3] for d in cpu.engine.datas])
    checks = {"short_pose": errors["pose"] < 0.001, "short_velocity": errors["velocity"] < 0.01,
              "partial_reset_force_isolation": bool(np.array_equal(pending_gpu, pending_cpu)
                                                     and np.all(pending_gpu[0] == 0) and np.all(pending_gpu[1:] == 10))}
    for task in (gpu, cpu):
        task.step(torch.zeros((4, 30), device=task.device))
    checks["force_expires_after_interval"] = bool(torch.count_nonzero(gpu.engine._sim_state.body_force[0]) == 0
        and all(np.count_nonzero(data.xfrc_applied) == 0 for data in cpu.engine.datas))
    for task in (gpu, cpu):
        task.reset()
        task.step(torch.zeros((4, 30), device=task.device))
    checks["reset_then_unforced_step"] = bool((gpu.engine.native_qpos().cpu() - cpu.engine.native_qpos()).abs().max() < 1e-4)
    return {"passed": all(checks.values()), "checks": checks, "max_errors": errors}


def compare_backends(native, newton):
    if [e["name"] for e in native] != [e["name"] for e in newton]:
        raise ValueError("Backend trial ordering mismatch")
    return {"survival_mismatches": [a["name"] for a, b in zip(native, newton) if a["survived"] != b["survived"]],
            "settled_recovery_mismatches": [a["name"] for a, b in zip(native, newton) if a["recovered"] != b["recovered"]],
            "max_survival_difference_seconds": max(abs(a["survival_seconds"] - b["survival_seconds"]) for a, b in zip(native, newton)),
            "scope": "Full-horizon contact-sensitive outcomes, separate from short force-transport parity"}


def run(args):
    activate(args.mimickit)
    from .stand_task import StandTask
    from .native_engine import NativeDummyEngine
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    reference = args.bundle / "stand_reference.npz"
    trials = baseline_trials()
    args.out.mkdir(parents=True, exist_ok=False)
    native = StandTask(rig, reference, len(trials), "cpu", engine_factory=NativeDummyEngine, control_mode="target_pd")
    contract = json.loads((args.bundle / "contract.json").read_text())
    validate_contract(contract, native.observation_contract())
    if sha256(args.bundle / "stand.onnx") != contract["actor_sha256"]:
        raise ValueError("Actor hash mismatch")
    for trial in trials:
        trial.validate(native.dt, native.reference.duration)
    protocol = {"schema": "mimic_push_v1", "force_frame": "MuJoCo_world", "application_point": "body_COM",
                "control_timestep": native.dt, "recovery": RECOVERY, "cases": [t.to_dict() for t in trials],
                "bundle": str(args.bundle.resolve()), "actor_sha256": contract["actor_sha256"],
                "model_sha256": contract["model_sha256"], "reference_sha256": contract["reference_sha256"],
                "scope": "Fixed baseline probes; not training, not held-out validation"}
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    preflight = force_preflight(rig, reference)
    (args.out / "force-parity.json").write_text(json.dumps(preflight, indent=2))
    if not preflight["passed"]:
        raise RuntimeError(f"Push transport parity failed: {preflight}")
    session = ort.InferenceSession(str(args.bundle / "stand.onnx"), providers=["CPUExecutionProvider"])
    results = {}
    for name, task in (("native", native), ("newton", StandTask(rig, reference, len(trials), control_mode="target_pd"))):
        episodes = evaluate_pushes(task, session, trials)
        (args.out / f"{name}-episodes.json").write_text(json.dumps(episodes))
        results[name] = {}
        for strength in (0, 10, 20, 40):
            group = [e for e, t in zip(episodes, trials) if abs(np.linalg.norm(t.force) - strength) < 1e-6]
            results[name][str(strength)] = {"trials": len(group), "survived": sum(e["survived"] for e in group),
                                          "recovered": sum(e["recovered"] for e in group),
                                          "mean_foot_travel_m": float(np.mean([e["foot_travel_m"] for e in group]))}
    results["backend_comparison"] = compare_backends(
        json.loads((args.out / "native-episodes.json").read_text()), json.loads((args.out / "newton-episodes.json").read_text()))
    (args.out / "report.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
