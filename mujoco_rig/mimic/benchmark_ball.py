"""Deterministic native/Newton projectile launch, contact and reset parity gate.

The probe deliberately uses one fixed ONNX actor and one shot per world.  It checks the
simulation boundary (launch state, actual ball/character contact, survival timing and
partial reset isolation), not learning quality.  The Newton backend reports contact from
its current contact buffer; the native backend reports MuJoCo's positive contact force.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .baseline import sha256
from .checkpoints import validate_contract
from .runtime import activate
from .rig import Rig


LAUNCH_STEP = 60
TRIAL_STEPS = 180
SPEEDS = (1.0, 2.0)


def make_cases():
    cases = []
    for index in range(8):
        direction = index % 4
        speed = SPEEDS[1 if index >= 4 else 0]
        cases.append({
            "name": f"speed-{speed:g}-direction-{direction}-phase-{0.0 if index < 4 else 0.8:g}",
            "phase": 0.0 if index < 4 else 0.8,
            "speed": speed,
            "direction": direction,
            "launch_step": LAUNCH_STEP,
            "trial_steps": TRIAL_STEPS,
            "target_body": "Chest",
        })
    return cases


def configure_cases(task, cases):
    """Reset a task and replace its randomized shot schedule with the protocol cases."""
    from envs.base_env import EnvMode

    task.set_mode(EnvMode.TEST)
    task.episode_seconds = TRIAL_STEPS * task.dt
    task.reset()
    phases = torch.tensor([case["phase"] for case in cases], device=task.device)
    task.offset[:] = phases
    q, v = task.reference.sample(task.offset)
    task.engine.reset_envs(task.ids, q, v)
    task.shot_steps.fill_(LAUNCH_STEP)
    task.shot_speeds[:] = torch.tensor([case["speed"] for case in cases], device=task.device)
    task.directions.zero_()
    angles = torch.tensor([case["direction"] * np.pi / 2 for case in cases], device=task.device)
    task.directions[:, 0] = torch.cos(angles)
    task.directions[:, 1] = torch.sin(angles)
    if hasattr(task, "shot_target_body") and "target_body" in cases[0]:
        targets = [task.rig.model.body(case.get("target_body", "Chest")).id - 1 for case in cases]
        task.shot_target_body[:] = torch.tensor(targets, device=task.device)
    task.shot_enabled.fill_(True)


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def run_backend(task, session, cases):
    from envs.base_env import DoneFlags

    configure_cases(task, cases)
    initial_observation = task.observations().detach().cpu().numpy()
    initial_ball = _numpy(task.engine.ball_position()).copy()
    active = np.ones(len(cases), dtype=bool)
    fallen = np.zeros(len(cases), dtype=bool)
    first_fall = np.full(len(cases), -1, dtype=int)
    first_hit = np.full(len(cases), -1, dtype=int)
    ball_trace = [initial_ball]

    for step in range(TRIAL_STEPS):
        observation = task.observations()
        action = session.run(None, {"observation": observation.detach().cpu().numpy()})[0]
        _, _, done, _ = task.step(torch.as_tensor(action, device=task.device))
        ball_trace.append(_numpy(task.engine.ball_position()).copy())

        hit = _numpy(task.engine.hit).astype(bool)
        first_hit[(first_hit < 0) & hit & active] = step + 1
        failed = _numpy(done) == DoneFlags.FAIL.value
        new_failures = failed & active
        first_fall[new_failures] = step + 1
        fallen |= new_failures
        active &= ~failed
        if not active.any():
            break
        # Keep fallen worlds numerically healthy without allowing a reset to create another shot.
        reset_ids = np.flatnonzero(new_failures)
        if len(reset_ids):
            ids = torch.as_tensor(reset_ids, device=task.device, dtype=torch.long)
            task.reset(ids)
            task.shot_enabled[ids] = False

    before_peer = _numpy(task.engine.ball_position()[1]).copy()
    q, v = task.reference.sample(task.offset[:1])
    task.engine.reset_envs(torch.tensor([0], device=task.device), q[:1], v[:1])
    after_reset = _numpy(task.engine.ball_position()).copy()
    expected_park = np.asarray(task.engine.rig.rest[46:49], dtype=np.float32)
    peer_unchanged = float(np.max(np.abs(after_reset[1] - before_peer)))
    reset_hit = _numpy(task.engine.hit).astype(bool)
    return {
        "initial_observation": initial_observation,
        "initial_ball": initial_ball,
        "ball_trace": np.asarray(ball_trace),
        "first_hit_step": first_hit.tolist(),
        "first_fall_step": first_fall.tolist(),
        "survived": (first_fall < 0).tolist(),
        "hit": (first_hit >= 0).tolist(),
        "reset": {
            "park_error_m": float(np.max(np.abs(after_reset[0] - expected_park))),
            "peer_ball_error_m": peer_unchanged,
            "reset_hit_flag": bool(reset_hit[0]),
            "peer_hit_preserved": bool(reset_hit[1] == (first_hit[1] >= 0)),
        },
    }


def compare(native, newton):
    native_trace = np.asarray(native["ball_trace"])
    newton_trace = np.asarray(newton["ball_trace"])
    trace_error = float(np.max(np.abs(native_trace - newton_trace)))
    pre_contact_errors = []
    for n, g in zip(native["first_hit_step"], newton["first_hit_step"]):
        if n >= 0 and g >= 0:
            end = min(n, g) + 1
            pre_contact_errors.append(float(np.max(np.abs(native_trace[:end] - newton_trace[:end]))))
    checks = {
        "initial_observations": float(np.max(np.abs(native["initial_observation"] - newton["initial_observation"]))) < 0.01,
        "all_native_hits": all(native["hit"]),
        "all_newton_hits": all(newton["hit"]),
        "hit_outcomes": native["hit"] == newton["hit"],
        "hit_timing": all(abs(a - b) <= 2 for a, b in zip(native["first_hit_step"], newton["first_hit_step"])),
        "survival_outcomes": native["survived"] == newton["survived"],
        "pre_contact_ball_trajectory": max(pre_contact_errors, default=float("inf")) < 0.02,
        "native_reset_parked": native["reset"]["park_error_m"] < 1e-5,
        "newton_reset_parked": newton["reset"]["park_error_m"] < 1e-5,
        "native_reset_isolated": native["reset"]["peer_ball_error_m"] < 1e-5,
        "newton_reset_isolated": newton["reset"]["peer_ball_error_m"] < 1e-5,
        "native_reset_clears_hit": not native["reset"]["reset_hit_flag"],
        "newton_reset_clears_hit": not newton["reset"]["reset_hit_flag"],
        "native_peer_hit_preserved": native["reset"]["peer_hit_preserved"],
        "newton_peer_hit_preserved": newton["reset"]["peer_hit_preserved"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "max_ball_trajectory_error_m": trace_error,
        "max_pre_contact_ball_error_m": max(pre_contact_errors, default=None),
        "hit_timing_difference_steps": [abs(a - b) for a, b in zip(native["first_hit_step"], newton["first_hit_step"])],
        "survival_mismatches": [i for i, (a, b) in enumerate(zip(native["survived"], newton["survived"])) if a != b],
        "scope": "Fixed launch/contact/reset transport; not a learning or generalization score",
    }


def run(args):
    activate(args.mimickit)
    from .ball_task import BallTask

    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    bundle = args.bundle.resolve()
    contract = json.loads((bundle / "contract.json").read_text())
    reference = bundle / "stand_reference.npz"
    cases = make_cases()
    native_task = BallTask(rig, reference, len(cases), "cpu", speed_min=1.0, speed_max=2.0, quiet_fraction=0.0)
    validate_contract(contract, native_task.observation_contract())
    actor_path = bundle / "stand.onnx"
    if sha256(actor_path) != contract["actor_sha256"]:
        raise ValueError("Actor hash mismatch")
    session = ort.InferenceSession(str(actor_path), providers=["CPUExecutionProvider"])
    args.out.mkdir(parents=True, exist_ok=False)
    native = run_backend(native_task, session, cases)
    newton_task = BallTask(rig, reference, len(cases), "cuda:0", speed_min=1.0, speed_max=2.0, quiet_fraction=0.0)
    newton = run_backend(newton_task, session, cases)
    comparison = compare(native, newton)
    protocol = {
        "schema": "mimic_ball_parity_v1",
        "model_sha256": contract["model_sha256"],
        "world_model_sha256": sha256(newton_task.engine.rig.path),
        "actor_sha256": contract["actor_sha256"],
        "reference_sha256": contract["reference_sha256"],
        "control_timestep": native_task.dt,
        "launch_distance_m": 0.65,
        "cases": cases,
        "contact_definition": "positive native MuJoCo ball/character contact; Newton current contact-buffer ball/character pair",
        "scope": "Deterministic native/Newton transport gate; not a training or held-out evaluation",
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (args.out / "native.json").write_text(json.dumps(_jsonable(native), indent=2), encoding="utf-8")
    (args.out / "newton.json").write_text(json.dumps(_jsonable(newton), indent=2), encoding="utf-8")
    report = {"passed": comparison["passed"], "comparison": comparison, "protocol": protocol}
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return comparison["passed"]


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    raise SystemExit(0 if run(parser.parse_args()) else 1)
