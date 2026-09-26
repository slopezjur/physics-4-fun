"""Native closed-loop gate for a physically checked single-step candidate."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from .baseline import sha256
from .com_feedback_teacher import ComFeedbackTeacher
from .evaluate_stand import evaluate
from .native_engine import NativeDummyEngine
from .observation_com import ObservationCom
from .policy import StandPolicy
from .probe_steps import validate_reference_substitution
from .rig import Rig
from .runtime import activate
from .short_step import SCHEMA as REFERENCE_SCHEMA, phase_seconds
from .support_transition import SupportTransitionReference


SCHEMA = "mimic_short_step_native_gate_v1"


def motion_metrics(task, traces, survived):
    poses = traces["qpos_after"][:, 0]
    model = task.rig.model
    data = mujoco.MjData(model)
    foot = model.body("Foot_L").id
    data.qpos[:] = task.reference.qpos[0].numpy()
    mujoco.mj_kinematics(model, data)
    start = data.xpos[foot].copy()
    positions = []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        positions.append(data.xpos[foot].copy())
    positions = np.asarray(positions)
    lift = float(np.max(positions[:, 2] - start[2]))
    shift = float(positions[-1, 0] - start[0])
    q = task.engine.native_qpos()[0]
    loads = task.engine.get_ground_contact_forces(0)[0, task.allowed_ground, 2].abs()
    supported = bool(loads[0] + loads[2] >= 5 and loads[1] + loads[3] >= 5)
    upright = bool(q[2] >= .75 and
                   1 - 2 * (q[4].square() + q[5].square()) >= np.cos(np.deg2rad(15)))
    slow = bool(torch.linalg.vector_norm(task.engine.get_root_vel(0)[0, :2]) <= .2 and
                torch.linalg.vector_norm(task.engine.get_root_ang_vel(0)[0]) <= 1.)
    return dict(peak_left_foot_lift_m=lift, final_left_foot_shift_m=shift,
                final_two_foot_support=supported, final_upright=upright,
                final_slow=slow,
                passed=bool(survived and lift >= .03 and shift >= .1
                            and supported and upright and slow))


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
    source = json.loads(args.source_contract.read_text())
    best = json.loads(args.initialize_from.with_name("best.json").read_text())
    if best["checkpoint_sha256"] != sha256(args.initialize_from):
        raise ValueError("Source checkpoint hash mismatch")
    validate_reference_substitution(source, best["contract"])
    reference_report = json.loads(args.reference.with_suffix(".json").read_text())
    if (reference_report.get("schema") != REFERENCE_SCHEMA
            or reference_report.get("passed") is not True
            or reference_report.get("admitted_for_training") is not False):
        raise ValueError("Require a physically checked, unadmitted short-step candidate")
    phases = phase_seconds(reference_report["motion_seconds"])
    support = StandTask(rig, args.support_reference, 1, "cpu",
                        engine_factory=NativeDummyEngine, control_mode="target_pd",
                        contact_mode="support_v2",
                        reference_factory=lambda p, r, d: SupportTransitionReference(
                            p, r, d, mode="transition"))
    validate_reference_substitution(source, support.observation_contract())
    agent = PPOAgent(yaml.safe_load(args.agent_config.read_text()), support, "cpu")
    agent.load(str(args.initialize_from))
    actor = StandPolicy(agent).eval()
    observer = ObservationCom(rig, source)
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    for label in ("reference_pd", "frozen_actor", "observation_com_teacher"):
        task = StandTask(rig, args.reference, 1, "cpu",
                         engine_factory=NativeDummyEngine, control_mode="target_pd",
                         contact_mode="support_v2")
        validate_reference_substitution(source, task.observation_contract())
        task.episode_seconds = task.reference.duration
        if label == "reference_pd":
            controller = lambda obs: torch.zeros((len(obs), len(rig.actuators)))
        elif label == "frozen_actor":
            controller = actor
        else:
            controller = ComFeedbackTeacher(task, actor, position_gain=1.5,
                                            velocity_gain=1.5, observer=observer)
        metrics, traces = evaluate(task, controller, episodes=1, capture_actions=True)
        motion = motion_metrics(task, traces, metrics["successes"] == 1)
        results.append(dict(label=label, evaluation=metrics, motion=motion,
                            failed_before_swing=bool(metrics["successes"] == 0 and
                                                     metrics["mean_survival_seconds"] <
                                                     phases["swing_start"])))
        print(json.dumps(dict(label=label, survival_seconds=metrics["mean_survival_seconds"],
                              **motion)), flush=True)
    report = dict(schema=SCHEMA, reference_sha256=sha256(args.reference),
                  model_sha256=sha256(rig.path),
                  source_checkpoint_sha256=sha256(args.initialize_from),
                  source_contract_sha256=sha256(args.source_contract),
                  duration_seconds=task.reference.duration,
                  phase_seconds=phases, results=results,
                  scope="Native single-step tracking diagnostic only; no training/export/Godot selection")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--support-reference", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
