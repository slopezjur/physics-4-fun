"""Compare native root, support and control traces across support checkpoints."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from .baseline import sha256
from .evaluate_stand import evaluate
from .motion_tracking_audit import TraceEngine, first_time
from .policy import StandPolicy
from .probe_steps import validate_reference_substitution
from .rig import Rig
from .runtime import activate
from .support_transition import SupportTransitionReference


STAMPS = (0.5, 1., 1.5, 2., 3., 4., 5., 6.)


def summarize(task, metrics, traces):
    rig = task.rig
    records = task.engine.records[0]
    count = round(metrics["survival_seconds"][0] / rig.model.opt.timestep)
    records = records[:count]
    if not records:
        raise ValueError("Native trace is empty")
    arrays = {key: np.asarray([row[key] for row in records]) for key in records[0]}
    dt = rig.model.opt.timestep
    times = np.arange(1, len(records) + 1) * dt
    target, _ = task.reference.sample(torch.as_tensor(times, dtype=torch.float32))
    target = target.numpy()
    root_error = arrays["qpos_after"][:, :3] - target[:, :3]
    force = arrays["ground_force_solve"][:, :, 2]
    controls = arrays["control"] / rig.action_scale
    actions = traces["actions"][:round(metrics["survival_seconds"][0] / task.dt), 0]
    actual_data, target_data = mujoco.MjData(rig.model), mujoco.MjData(rig.model)
    foot_bodies = [rig.model.body(name).id for name in ("Foot_L", "Foot_R", "Toe_L", "Toe_R")]
    stamps = []
    for stamp in STAMPS:
        index = round(stamp / dt) - 1
        if index >= len(records):
            continue
        actual_data.qpos[:] = arrays["qpos_after"][index]
        target_data.qpos[:] = target[index]
        mujoco.mj_forward(rig.model, actual_data)
        mujoco.mj_forward(rig.model, target_data)
        stamps.append(dict(seconds=float(times[index]),
                           root_xyz_m=arrays["qpos_after"][index, :3].tolist(),
                           target_root_xyz_m=target[index, :3].tolist(),
                           root_error_xyz_m=root_error[index].tolist(),
                           root_velocity_xyz_m_s=arrays["qvel_after"][index, :3].tolist(),
                           body_com_x_m=float(actual_data.subtree_com[1, 0]),
                           target_body_com_x_m=float(target_data.subtree_com[1, 0]),
                           feet_toes_x_m=actual_data.xpos[foot_bodies, 0].tolist(),
                           target_feet_toes_x_m=target_data.xpos[foot_bodies, 0].tolist(),
                           left_right_load_n=[float(force[index, 0] + force[index, 2]),
                                              float(force[index, 1] + force[index, 3])],
                           max_motor_fraction=float(np.abs(controls[index]).max())))
    fraction = (np.abs(controls) >= .999).mean(axis=0)
    return dict(metrics=metrics, first_root_error_5cm_seconds=first_time(
                    np.linalg.norm(root_error, axis=1) > .05, dt),
                final_root_error_xyz_m=root_error[-1].tolist(),
                first_both_feet_unloaded_after_first_control_seconds=first_time(
                    (force.sum(axis=1) < 5.) & (times > task.dt), dt),
                motor_saturated_sample_fraction=float((np.abs(controls) >= .999).mean()),
                peak_motor_fraction=float(np.abs(controls).max()),
                peak_action_fraction=float(np.abs(actions).max()),
                action_clip_fraction=float((np.abs(actions) >= .999).mean()),
                strongest_motor_saturation=[dict(name=rig.model.actuator(int(rig.actuators[i])).name,
                                              fraction=float(fraction[i]))
                                             for i in np.argsort(fraction)[-4:][::-1]],
                samples=stamps), {**arrays, "reference_qpos": target,
                                  "root_error_xyz_m": root_error, "actions": actions}


def run(args):
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .stand_task import StandTask

    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    contract = json.loads(args.source_contract.read_text())
    if contract.get("observation_contract") != "mimic_stand_target_pd_v2":
        raise ValueError("Require support-v2 source contract")
    config = yaml.safe_load(args.agent_config.read_text())
    reference_contract = None
    results = []
    args.out.mkdir(parents=True, exist_ok=False)
    for label, checkpoint in args.checkpoint:
        best_path = checkpoint.with_name("best.json")
        if checkpoint.name == "best.pt":
            best = json.loads(best_path.read_text())
            if best["checkpoint_sha256"] != sha256(checkpoint):
                raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
            validate_reference_substitution(contract, best["contract"])
        task = StandTask(rig, args.reference, 1, "cpu", engine_factory=TraceEngine,
                         control_mode="target_pd", contact_mode="support_v2",
                         reference_factory=lambda p, r, d: SupportTransitionReference(
                             p, r, d, mode="transition"))
        task.episode_seconds = args.episode_seconds
        if args.episode_seconds > task.reference.duration:
            # Diagnose delayed failure while holding the terminal target and phase.
            task.reference_phase = lambda time: (time / task.reference.duration).clamp(max=1.)
        validate_reference_substitution(contract, task.observation_contract())
        reference_contract = task.reference.contract()
        agent = PPOAgent(config, task, "cpu")
        agent.load(str(checkpoint))
        metrics, traces = evaluate(task, StandPolicy(agent).eval(), episodes=1,
                                   capture_actions=True)
        row, arrays = summarize(task, metrics, traces)
        path = args.out / f"{label}.npz"
        np.savez_compressed(path, **arrays)
        results.append(dict(label=label, checkpoint_sha256=sha256(checkpoint),
                            trace_sha256=sha256(path), **row))
        print(json.dumps(dict(label=label, survival=metrics["mean_survival_seconds"],
                              final_root_error_xyz_m=row["final_root_error_xyz_m"])), flush=True)
    report = dict(schema="mimic_support_entry_audit_v2", model_sha256=sha256(rig.path),
                  source_contract_sha256=sha256(args.source_contract),
                  reference=reference_contract, episode_seconds=args.episode_seconds,
                  terminal_phase_held=args.episode_seconds > reference_contract["duration_seconds"],
                  cases=results,
                  sampling="Native qpos/qvel after integration; loads and motor controls from the preceding physics solve",
                  scope="Read-only native checkpoint audit; no training, export or viewer selection")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path, required=True)
    parser.add_argument("--episode-seconds", type=float, default=6.)
    parser.add_argument("--checkpoint", nargs=2, action="append", metavar=("LABEL", "PATH"),
                        type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    options.checkpoint = [(label, Path(path)) for label, path in options.checkpoint]
    run(options)
