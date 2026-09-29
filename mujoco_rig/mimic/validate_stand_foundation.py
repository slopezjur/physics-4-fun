"""Long native standing and seeded unseen velocity-start validation, without episode recycling."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .baseline import sha256
from .continuous_stand import GATES
from .native_engine import NativeDummyEngine
from .policy import StandPolicy
from .rig import Rig
from .runtime import activate


def case_passed(case, seconds):
    """Survival alone must not pass an unloading, drifting, or unsettled stance."""
    scalar_fields = ("survival_seconds", "mean_root_error_m", "final_stable_seconds",
                     "settled_fraction_after_startup", "maximum_foot_error_after_startup_m",
                     "tail_root_drift_m", "tail_foot_motion_m")
    if not set((*scalar_fields, "survived", "minimum_foot_load_after_startup_n")) <= case.keys():
        return False
    if not all(np.isfinite(case[key]) for key in scalar_fields) or not np.isfinite(
            case["minimum_foot_load_after_startup_n"]).all():
        return False
    return (case["survived"] and case["survival_seconds"] >= seconds
            and case["mean_root_error_m"] <= GATES["max_mean_root_error_m"]
            and case["final_stable_seconds"] >= min(10., seconds)
            and case["settled_fraction_after_startup"] >= .99
            and min(case["minimum_foot_load_after_startup_n"]) >= GATES["min_each_foot_load_n"]
            and case["maximum_foot_error_after_startup_m"] <= GATES["max_foot_error_m"]
            and case["tail_root_drift_m"] <= .01
            and case["tail_foot_motion_m"] <= .001)


def step_native(task, action):
    """Advance the production controller without computing unused training rewards."""
    from envs.base_env import DoneFlags
    task.engine.policy_control.set_reference(*task.sample_reference(task.offset + task.steps * task.dt))
    task.engine.set_cmd(0, action)
    task.engine.step()
    task.steps += 1
    done = torch.zeros_like(task.steps, dtype=torch.int)
    done[task.steps * task.dt >= task.episode_seconds] = DoneFlags.TIME.value
    contact = task.engine.get_ground_contact_forces(0).clone()
    contact[:, task.allowed_ground] = 0
    fallen = task.engine.get_root_pos(0)[:, 2] < .65
    fallen |= contact.abs().amax(dim=(1, 2)) > 1.
    finite = torch.isfinite(task.engine.native_qpos()).all(-1)
    finite &= torch.isfinite(task.engine._array("qvel")).all(-1)
    done[fallen | ~finite] = DoneFlags.FAIL.value
    return done


@torch.no_grad()
def evaluate(task, actor, seconds, velocity_noise=0., seed=270928, state_bank=None):
    from envs.base_env import EnvMode, DoneFlags
    count = task.get_num_envs()
    task.episode_seconds = seconds
    task.set_mode(EnvMode.TEST)
    task.reset()
    q, v = task.sample_reference(task.offset)
    initial_v = v.clone()
    rng = np.random.default_rng(seed)
    if velocity_noise:
        delta = torch.tensor(rng.uniform(-1., 1., (count, 6)), dtype=torch.float32)
        delta[:, :3] *= velocity_noise
        delta[:, 3:] *= 2 * velocity_noise
        initial_v[:, :6] += delta
        task.engine.reset_envs(task.ids, q, initial_v)
    targets, _ = task.mapper.from_native_qpos(q)
    target_feet = targets[:, task.feet].numpy()
    active = np.ones(count, dtype=bool)
    survived = np.zeros(count, dtype=bool)
    lengths = np.zeros(count, dtype=int)
    stable = np.zeros(count, dtype=int)
    root_error_sum = np.zeros(count)
    settled_count = np.zeros(count, dtype=int)
    scored_count = np.zeros(count, dtype=int)
    min_load = np.full((count, 2), np.inf)
    max_foot_error = np.zeros(count)
    scalar_traces, root_traces, foot_traces, masks = [], [], [], []
    max_steps = int(np.ceil(seconds / task.dt))
    for step in range(max_steps):
        obs = task.observations()
        done = step_native(task, actor(obs))
        done = done.numpy()
        state = task.engine.native_qpos().numpy()
        velocity = task.engine.get_root_vel(0).numpy()
        angular = task.engine.get_root_ang_vel(0).numpy()
        feet = task.engine.get_body_pos(0)[:, task.feet].numpy()
        foot_error = np.linalg.norm(feet - target_feet, axis=-1).max(-1)
        loads = task.engine.get_ground_contact_forces(0)[:, task.allowed_ground, 2].abs().numpy()
        loads = loads[:, :2] + loads[:, 2:]
        root_error = np.linalg.norm(state[:, :3] - q[:, :3].numpy(), axis=-1)
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (state[:, 4] ** 2 + state[:, 5] ** 2), -1, 1)))
        speed = np.linalg.norm(velocity[:, :2], axis=-1)
        angular_speed = np.linalg.norm(angular, axis=-1)
        settled = ((state[:, 2] >= GATES["min_height_m"]) & (tilt <= GATES["max_tilt_degrees"])
            & (speed <= GATES["max_horizontal_speed_m_s"])
            & (angular_speed <= GATES["max_angular_speed_rad_s"])
            & (loads.min(-1) >= GATES["min_each_foot_load_n"])
            & (foot_error <= GATES["max_foot_error_m"]) & (done != DoneFlags.FAIL.value))
        stable = np.where(active, np.where(settled, stable + 1, 0), stable)
        lengths += active
        root_error_sum += root_error * active
        if (step + 1) * task.dt > 2.:
            scored_count += active
            settled_count += settled & active
            min_load[active] = np.minimum(min_load[active], loads[active])
            max_foot_error[active] = np.maximum(max_foot_error[active], foot_error[active])
        scalar_traces.append(np.column_stack((root_error, tilt, speed, angular_speed,
                                              foot_error, loads, settled)))
        root_traces.append(state[:, :3].copy())
        foot_traces.append(feet.copy())
        masks.append(active.copy())
        if state_bank is not None and (step + 1) % 1800 == 0:
            control = task.engine.policy_control
            valid = active & settled & (foot_error < .025)
            state_bank.append(dict(
                posture=np.flatnonzero(valid) % 8,
                age_seconds=np.full(valid.sum(), (step + 1) * task.dt),
                qpos=np.stack([d.qpos.copy() for d in task.engine.datas])[valid],
                qvel=np.stack([d.qvel.copy() for d in task.engine.datas])[valid],
                ground=task.engine.get_ground_contact_forces(0).numpy()[valid],
                pending=torch.stack([control.pending[(control.cursor + i) % len(control.pending)]
                                     for i in range(len(control.pending))], dim=1).numpy()[valid],
                active=control.active.numpy()[valid].copy(),
                feet=feet[valid].copy(), action=actor(obs).numpy()[valid],
                torque=control.ctrl.numpy()[valid].copy()))
        survived |= active & (done == DoneFlags.TIME.value)
        active &= done == DoneFlags.NULL.value
        if (step + 1) % 3600 == 0:
            print(json.dumps(dict(elapsed_sim_seconds=(step + 1) * task.dt, active=int(active.sum()))), flush=True)
        if not active.any():
            break
        # Reset failed worlds only to keep the batched simulator numerically healthy.
        # Their first-episode masks never become active again.
        task.reset(torch.as_tensor(np.flatnonzero(done != DoneFlags.NULL.value)))
    roots = np.asarray(root_traces)
    foot_positions = np.asarray(foot_traces)
    cases = []
    for i in range(count):
        end = lengths[i]
        window = min(int(10. / task.dt), max(1, end // 4))
        drift = (float(np.linalg.norm(roots[end-window:end, i].mean(0)
                                     - roots[end-2*window:end-window, i].mean(0))) if end >= 2 else 0.)
        foot_window = min(max(1, int(min(60., seconds / 6.) / task.dt)), end)
        foot_motion = float(np.linalg.norm(foot_positions[end-foot_window:end, i]
                                           - foot_positions[end-foot_window, i], axis=-1).max())
        case = dict(case=i, posture=i % 8, survived=bool(survived[i]),
            survival_seconds=float(end * task.dt), mean_root_error_m=float(root_error_sum[i] / max(1, end)),
            final_stable_seconds=float(stable[i] * task.dt),
            settled_fraction_after_startup=float(settled_count[i] / max(1, scored_count[i])),
            minimum_foot_load_after_startup_n=np.where(np.isfinite(min_load[i]), min_load[i], 0).tolist(),
            maximum_foot_error_after_startup_m=float(max_foot_error[i]), tail_root_drift_m=drift,
            tail_foot_motion_m=foot_motion)
        case["passed"] = case_passed(case, seconds)
        cases.append(case)
    report = dict(schema="mimic_standing_foundation_validation_v1", seconds=seconds,
        seed=seed, velocity_noise_m_s=velocity_noise, angular_noise_rad_s=2*velocity_noise,
        initial_velocity=initial_v.tolist(), cases=cases, passed=all(c["passed"] for c in cases),
        successes=sum(c["passed"] for c in cases),
        gate=dict(**GATES, horizon_seconds=seconds, final_settled_seconds=min(10., seconds),
                  startup_seconds=2., min_settled_fraction=.99, max_tail_drift_m=.01,
                  max_tail_foot_motion_m=.001, foot_motion_window_seconds=min(60., seconds / 6.),
                  support="Both feet including toes continuously >=5 N after startup"),
        scope="First uninterrupted episodes only; finite evidence of sustained standing, not proof of infinite stability")
    traces = dict(scalars=np.asarray(scalar_traces), root_position=roots, foot_position=foot_positions, active=np.asarray(masks),
                  columns=np.array(["root_error_m", "tilt_deg", "speed_m_s", "angular_rad_s",
                                    "foot_error_m", "load_l_n", "load_r_n", "settled"]))
    return report, traces


def run(args):
    if args.seconds < 30 or args.cases < 8 or args.cases % 8 or args.velocity_noise < 0:
        raise ValueError("Require >=30 seconds, complete eight-posture replicas, and nonnegative noise")
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .balance_observation import task_classes
    contract_path = args.source_contract or args.checkpoint.parent / "contract.json"
    contract = json.loads(contract_path.read_text())
    ContinuousStandTask, _ = task_classes(contract)
    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    task = ContinuousStandTask(rig, args.reference, args.cases, engine_factory=NativeDummyEngine)
    from .train_continuous_stand import validate_initialization
    migration = validate_initialization(contract, task.observation_contract(), args.allow_reference_migration)
    agent = PPOAgent(yaml.safe_load((args.checkpoint.parent / "agent.yaml").read_text()), task, "cpu")
    checkpoint_hash = sha256(args.checkpoint)
    agent.load(str(args.checkpoint))
    args.out.mkdir(parents=True, exist_ok=False)
    runner_hash = sha256(Path(__file__))
    (args.out / "runner.py").write_text(Path(__file__).read_text())
    bank = [] if args.capture_states else None
    report, traces = evaluate(task, StandPolicy(agent).eval(), args.seconds, args.velocity_noise, args.seed, bank)
    if sha256(args.checkpoint) != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during validation; evidence cannot be attributed safely")
    report.update(checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
                  reference_sha256=sha256(args.reference), model_sha256=sha256(rig.path),
                  runner_sha256=runner_hash, reference_migration=migration)
    np.savez_compressed(args.out / "traces.npz", **traces)
    if bank:
        np.savez_compressed(args.out / "late_states.npz",
            **{key: np.concatenate([entry[key] for entry in bank]) for key in bank[0]},
            reference_sha256=np.array(report["reference_sha256"]),
            model_sha256=np.array(report["model_sha256"]),
            checkpoint_sha256=np.array(report["checkpoint_sha256"]))
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(passed=report["passed"], successes=report["successes"],
                         seconds=[c["survival_seconds"] for c in report["cases"]])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path)
    parser.add_argument("--allow-reference-migration", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=600.)
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--velocity-noise", type=float, default=0.)
    parser.add_argument("--seed", type=int, default=270928)
    parser.add_argument("--capture-states", action="store_true")
    run(parser.parse_args())
