"""Bounded native feasibility sweep for COM feedback over a fixed actor."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .baseline import sha256
from .checkpoints import standing_passed
from .com_feedback_teacher import ComFeedbackTeacher
from .evaluate_stand import evaluate
from .policy import StandPolicy
from .probe_steps import validate_reference_substitution
from .rig import Rig
from .runtime import activate
from .support_transition import SupportTransitionReference
from .train_support_transition import gate


CASES = (("actor", 0., 0., 0.),
         ("ankle_1p5_1p5", 1.5, 1.5, 0.),
         ("ankle_2_2", 2., 2., 0.),
         ("ankle_3_1p5", 3., 1.5, 0.),
         ("ankle_hip_2_2", 2., 2., .3))


def run(args):
    activate(args.mimickit)
    from envs.base_env import EnvMode
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .native_engine import NativeDummyEngine
    from .stand_task import StandTask

    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    contract = json.loads(args.source_contract.read_text())
    if args.initialize_from.name == "best.pt":
        best = json.loads(args.initialize_from.with_name("best.json").read_text())
        if best["checkpoint_sha256"] != sha256(args.initialize_from):
            raise ValueError("Source actor hash mismatch")
        validate_reference_substitution(contract, best["contract"])
    config = yaml.safe_load(args.agent_config.read_text())
    task = StandTask(rig, args.reference, 1, "cpu", engine_factory=NativeDummyEngine,
                     control_mode="target_pd", contact_mode="support_v2",
                     reference_factory=lambda p, r, d: SupportTransitionReference(
                         p, r, d, mode="transition"))
    validate_reference_substitution(contract, task.observation_contract())
    task.episode_seconds = 12.
    task.reference_phase = lambda time: (time / task.reference.duration).clamp(max=1.)
    task.set_mode(EnvMode.TEST)
    agent = PPOAgent(config, task, "cpu")
    agent.load(str(args.initialize_from))
    actor = StandPolicy(agent).eval()
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    for label, position_gain, velocity_gain, hip_fraction in (() if args.validation_only else CASES):
        teacher = ComFeedbackTeacher(task, actor, position_gain=position_gain,
                                     velocity_gain=velocity_gain, hip_fraction=hip_fraction)
        metrics, traces = evaluate(task, teacher, episodes=1, capture_actions=True)
        check = gate(metrics, traces, task.reference, task.dt)
        record = {key: np.asarray([row[key] for row in teacher.records]) for key in
                  ("seconds", "com_x_m", "com_vx_m_s", "target_com_x_m",
                   "position_error_m", "velocity_error_m_s", "correction")}
        trace_path = args.out / f"{label}.npz"
        np.savez_compressed(trace_path, **record, **traces)
        result = dict(label=label, controller=teacher.contract(), metrics=metrics,
                      gate=check, trace_sha256=sha256(trace_path),
                      peak_com_error_m=float(np.abs(record["position_error_m"]).max()),
                      peak_correction=float(np.abs(record["correction"]).max()),
                      final_com_error_m=float(record["position_error_m"][-1]))
        results.append(result)
        print(json.dumps(dict(label=label, survival_seconds=metrics["mean_survival_seconds"],
                              gate=check, final_com_error_m=result["final_com_error_m"])), flush=True)
        (args.out / "progress.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    checks = {}
    if args.validation_only:
        static = StandTask(rig, args.reference, 1, "cpu", engine_factory=NativeDummyEngine,
                           control_mode="target_pd", contact_mode="support_v2",
                           reference_factory=lambda p, r, d: SupportTransitionReference(
                               p, r, d, mode="static"))
        validate_reference_substitution(contract, static.observation_contract())
        static.episode_seconds = 12.
        static.reference_phase = lambda time: (time / static.reference.duration).clamp(max=1.)
        quiet = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz",
                          8, "cpu", engine_factory=NativeDummyEngine,
                          control_mode="target_pd", contact_mode="support_v2")
        validate_reference_substitution(contract, quiet.observation_contract())
        teacher_static = ComFeedbackTeacher(static, actor, position_gain=1.5, velocity_gain=1.5)
        teacher_quiet = ComFeedbackTeacher(quiet, actor, position_gain=1.5, velocity_gain=1.5)
        static_base, _ = evaluate(static, actor, episodes=1)
        static_teacher, static_traces = evaluate(static, teacher_static, episodes=1)
        quiet_base, _ = evaluate(quiet, actor)
        quiet_teacher, _ = evaluate(quiet, teacher_quiet)
        checks = dict(teacher_contract=teacher_static.contract(),
                      static_base=static_base, static_teacher=static_teacher,
                      static_teacher_gate=gate(static_teacher, static_traces,
                                               static.reference, static.dt),
                      quiet_base=quiet_base, quiet_teacher=quiet_teacher,
                      quiet_teacher_passed=standing_passed(quiet_teacher))
        print(json.dumps(dict(static_seconds=static_teacher["mean_survival_seconds"],
                              static_gate=checks["static_teacher_gate"],
                              quiet_successes=quiet_teacher["successes"],
                              quiet_passed=checks["quiet_teacher_passed"])), flush=True)
    report = dict(schema="mimic_com_teacher_feasibility_v1", model_sha256=sha256(rig.path),
                  actor_checkpoint_sha256=sha256(args.initialize_from),
                  source_contract_sha256=sha256(args.source_contract),
                  reference=task.reference.contract(), episode_seconds=12., cases=results,
                  validation_only=args.validation_only, checks=checks,
                  scope="Native oracle COM feasibility on one pre-lift transition; no training, export or Godot selection")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation-only", action="store_true")
    run(parser.parse_args())
