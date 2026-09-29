"""Native physical-hit gate for stationary Perturb without a prescribed step."""
import math

import numpy as np
import torch

from .ball_protocol import VALIDATION_SCHEMA
from .ball_protocol import TARGET_BODIES as ALL_TARGET_BODIES
from .perturb import RECOVERY

TARGET_BODIES = ("Chest", "Spine", "Pelvis")


def selection_cases(speed, seconds, dt=.016668):
    return [dict(name=f"posture-{i%8}-direction-{i//8}", posture=i % 8,
                 direction=i // 8, angle_radians=(i // 8) * math.pi / 2,
                 target_body=TARGET_BODIES[(i + i // 8) % len(TARGET_BODIES)],
                 speed=float(speed), phase=0., launch_step=90,
                 trial_steps=math.ceil(seconds / dt)) for i in range(32)]


def unseen_cases(seed, count, speed_min, speed_max, seconds, dt, *, target_bodies=TARGET_BODIES):
    if count < 32 or count % 8:
        raise ValueError("Unseen suite needs complete posture replicas")
    if not target_bodies or len(set(target_bodies)) != len(target_bodies) or any(
            body not in ALL_TARGET_BODIES for body in target_bodies):
        raise ValueError("Unseen targets must be unique supported body names")
    rng = np.random.default_rng(seed)
    result = []
    for i in range(count):
        direction = (i // 8) % 4
        angle = direction * math.pi / 2 + rng.uniform(-math.pi / 4, math.pi / 4)
        result.append(dict(name=f"unseen-{seed}-{i}", posture=i % 8, direction=direction,
            angle_radians=float(angle), target_body=target_bodies[int(rng.integers(0, len(target_bodies)))],
            speed=float(rng.uniform(speed_min, speed_max)), phase=0.,
            launch_step=int(rng.integers(60, 121)), trial_steps=math.ceil(seconds / dt)))
    return result


def repeated_cases(speed, seconds, dt=.016668):
    if seconds < 11.:
        raise ValueError("Repeated-hit suite needs time to settle after the second shot")
    cases = selection_cases(speed, seconds, dt)
    for case in cases:
        case["second_launch_step"] = 360
        case["second_angle_radians"] = (case["direction"] + 1) * math.pi / 2
        case["second_target_body"] = TARGET_BODIES[(TARGET_BODIES.index(case["target_body"]) + 1) % 3]
        case["second_speed"] = float(speed)
    return cases


def timeout_for_steps(steps, dt):
    if steps < 1 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid control horizon")
    return (steps - .5) * dt


@torch.no_grad()
def evaluate(task, actor, cases, on_step=None):
    from envs.base_env import DoneFlags, EnvMode
    from .recovery_steps import RecoveryStepTracker

    if len(cases) != task.get_num_envs() or any(c["posture"] != i % 8 for i, c in enumerate(cases)):
        raise ValueError("Cases must match world posture order")
    task.set_mode(EnvMode.TEST)
    max_steps = max(c["trial_steps"] for c in cases)
    if any(c["trial_steps"] != max_steps for c in cases):
        raise ValueError("Batched impact cases require the same scored horizon")
    # The task compares float32 elapsed time with a Python float. Put the
    # timeout halfway through the final control interval to avoid a missed TIME.
    task.episode_seconds = timeout_for_steps(max_steps, task.dt)
    task.reset()
    ids = task.ids
    task.shot_steps[:] = torch.tensor([c["launch_step"] for c in cases], device=task.device)
    task.shot_speeds[:] = torch.tensor([c["speed"] for c in cases], device=task.device)
    task.shot_target_body[:] = torch.tensor([task.rig.model.body(c["target_body"]).id - 1
                                             for c in cases], device=task.device)
    angles = torch.tensor([c["angle_radians"] for c in cases], device=task.device)
    task.directions[:] = torch.stack((angles.cos(), angles.sin(), torch.zeros_like(angles)), -1)
    task.shot_enabled[:] = True
    second_enabled = torch.tensor(["second_launch_step" in c for c in cases], device=task.device)
    task.second_shot_enabled[:] = second_enabled
    if second_enabled.any():
        task.second_shot_steps[:] = torch.tensor([c.get("second_launch_step", 0) for c in cases], device=task.device)
        task.second_shot_speeds[:] = torch.tensor([c.get("second_speed", 0.) for c in cases], device=task.device)
        task.second_shot_target_body[:] = torch.tensor([
            task.rig.model.body(c.get("second_target_body", "Chest")).id - 1 for c in cases], device=task.device)
        second_angles = torch.tensor([c.get("second_angle_radians", 0.) for c in cases], device=task.device)
        task.second_directions[:] = torch.stack((second_angles.cos(), second_angles.sin(),
                                                 torch.zeros_like(second_angles)), -1)
    initial_feet = task.engine.get_body_pos(0)[:, task.feet].clone()
    points = task.engine.get_body_pos(0)[:, task.allowed_ground][:, [0, 2, 1, 3]].reshape(len(cases), 2, 2, 3)
    step_tracker = RecoveryStepTracker(points, task.dt)
    active = torch.ones(len(cases), dtype=torch.bool, device=task.device)
    hits = torch.zeros_like(active)
    second_hits = torch.zeros_like(active)
    second_impulse_before = torch.zeros(len(cases), dtype=torch.float64, device=task.device)
    survived = torch.zeros_like(active)
    stable = torch.zeros(len(cases), dtype=torch.long, device=task.device)
    lengths = torch.zeros_like(stable)
    root_error = torch.zeros(len(cases), device=task.device)
    max_foot_shift = torch.zeros(len(cases), device=task.device)
    first_hit = torch.full_like(stable, -1)
    targets, _ = task.sample_reference(task.offset)
    for step in range(max_steps):
        launching_again = second_enabled & (task.steps == task.second_shot_steps) & active
        if launching_again.any():
            second_impulse_before[launching_again] = task.engine.hit_normal_impulse[launching_again]
        _, _, done, _ = task.step(actor(task.observations()))
        if second_enabled.any():
            second_hits |= second_enabled & (task.steps > task.second_shot_steps) & active & (
                task.engine.hit_normal_impulse > second_impulse_before + 1e-6)
        current_hit = task.engine.hit & active
        first_hit[(first_hit < 0) & current_hit] = step + 1
        hits |= current_hit
        q = task.engine.native_qpos()
        loads = task.engine.get_ground_contact_forces(0)[:, task.allowed_ground, 2].abs()
        points = task.engine.get_body_pos(0)[:, task.allowed_ground][:, [0, 2, 1, 3]].reshape(len(cases), 2, 2, 3)
        step_tracker.update(points, torch.stack((loads[:, 0] + loads[:, 2], loads[:, 1] + loads[:, 3]), -1),
                            active, hits)
        feet_supported = (loads[:, 0] + loads[:, 2] >= RECOVERY["each_foot_load_n"]) & (
            loads[:, 1] + loads[:, 3] >= RECOVERY["each_foot_load_n"])
        upright = 1 - 2 * (q[:, 4].square() + q[:, 5].square()) >= math.cos(math.radians(
            RECOVERY["tilt_degrees"]))
        settled = (q[:, 2] >= RECOVERY["height_m"]) & upright & feet_supported
        settled &= torch.linalg.vector_norm(task.engine.get_root_vel(0)[:, :2], dim=-1) <= RECOVERY["horizontal_speed_m_s"]
        settled &= torch.linalg.vector_norm(task.engine.get_root_ang_vel(0), dim=-1) <= RECOVERY["angular_speed_rad_s"]
        stable = torch.where(settled & hits & active, stable + 1, 0)
        root_error += torch.linalg.vector_norm(q[:, :3] - targets[:, :3], dim=-1) * active
        shift = torch.linalg.vector_norm(task.engine.get_body_pos(0)[:, task.feet] - initial_feet, dim=-1).amax(-1)
        max_foot_shift = torch.maximum(max_foot_shift, torch.where(active, shift, max_foot_shift))
        lengths += active
        survived |= active & (done == DoneFlags.TIME.value)
        if on_step is not None:
            on_step(task, step, active, done)
        active &= done == DoneFlags.NULL.value
        if not active.any():
            break
        failed = torch.nonzero(done != DoneFlags.NULL.value).flatten()
        if len(failed):
            task.reset(failed)
            task.shot_enabled[failed] = False
    recovered = hits & survived & (stable * task.dt >= 2.)
    rows = [dict(**case, hit=bool(hits[i]), survived=bool(survived[i]),
                 recovered=bool(recovered[i]), first_hit_step=int(first_hit[i]),
                 second_hit=bool(second_hits[i]) if second_enabled[i] else None,
                 survival_seconds=float(lengths[i] * task.dt),
                 final_settled_seconds=float(stable[i] * task.dt),
                 lifted_feet=step_tracker.lifted[i].tolist(),
                 steps_by_foot=step_tracker.steps[i].tolist(),
                 maximum_clearance_m=step_tracker.max_clearance[i].tolist(),
                 maximum_step_placement_m=step_tracker.max_placement[i].tolist(),
                 recovered_with_step=bool(recovered[i] and step_tracker.steps[i].sum() > 0),
                 max_either_foot_shift_m=float(max_foot_shift[i])) for i, case in enumerate(cases)]
    return dict(evaluation_schema=VALIDATION_SCHEMA, episodes=len(cases),
                successes=int(survived.sum()), confirmed_hits=int(hits.sum()),
                survived_hits=int((hits & survived).sum()), recovered_hits=int(recovered.sum()),
                second_confirmed_hits=int(second_hits.sum()),
                repeated_recoveries=int((recovered & second_hits).sum()),
                recovered_with_step=sum(row["recovered_with_step"] for row in rows),
                step_criteria=step_tracker.contract(),
                mean_survival_seconds=float(lengths.float().mean() * task.dt),
                mean_root_tracking_error_m=float((root_error / lengths.clamp_min(1)).mean()),
                recovery_criteria=dict(RECOVERY, final_settled_seconds=2., either_foot_allowed=True),
                cases=rows), dict(first_hit=first_hit.cpu().numpy(),
                                  foot_shift=max_foot_shift.cpu().numpy())
