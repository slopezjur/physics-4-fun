"""Compare native and observation-derived COM feedback under unseen force pulses."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from .baseline import sha256
from .com_feedback_teacher import ComFeedbackTeacher
from .observation_com import ObservationCom
from .policy import StandPolicy
from .probe_steps import validate_reference_substitution
from .rig import Rig
from .runtime import activate
from .support_transition import SupportTransitionReference
from .train_support_transition import gate


PULSE_START_SECONDS = 2.
PULSE_DURATION_SECONDS = .12
PULSE_FORCE_N = 120.


def trial(task, controller, observer, force_n, *, pulse_start=PULSE_START_SECONDS):
    from envs.base_env import DoneFlags, EnvMode

    task.set_mode(EnvMode.TEST)
    task.reset()
    poses = []
    errors = []
    done = DoneFlags.NULL.value
    for step in range(round(task.episode_seconds / task.dt)):
        observation = task.observations()
        estimate = observer(observation)[0]
        native = task.engine.datas[0]
        mujoco.mj_subtreeVel(task.rig.model, native)
        actual = np.array([native.subtree_com[1, 0], native.subtree_linvel[1, 0]])
        errors.append(estimate - actual)
        poses.append(native.qpos.copy())
        seconds = step * task.dt
        if pulse_start <= seconds < pulse_start + PULSE_DURATION_SECONDS:
            task.engine.set_external_force(1, [[force_n, 0., 0.]])
        with torch.no_grad():
            action = controller(observation)
        _, _, flags, _ = task.step(action)
        done = int(flags[0])
        if done != DoneFlags.NULL.value:
            break
    poses = np.asarray(poses)[:, None, :]
    errors = np.asarray(errors)
    survived = done == DoneFlags.TIME.value
    return dict(survived=survived, survival_seconds=len(poses) * task.dt,
                gate=gate({"successes": int(survived)}, {"qpos": poses}, task.reference, task.dt),
                max_com_position_estimation_error_m=float(np.abs(errors[:, 0]).max()),
                max_com_velocity_estimation_error_m_s=float(np.abs(errors[:, 1]).max()),
                root_x_m=float(poses[-1, 0, 0]))


def run(args):
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .native_engine import NativeDummyEngine
    from .stand_task import StandTask

    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    source = json.loads(args.source_contract.read_text())
    best = json.loads(args.initialize_from.with_name("best.json").read_text())
    if best["checkpoint_sha256"] != sha256(args.initialize_from):
        raise ValueError("Source actor hash mismatch")
    validate_reference_substitution(source, best["contract"])
    observer = ObservationCom(rig, source)
    config = yaml.safe_load(args.agent_config.read_text())

    def make_task(mode):
        task = StandTask(rig, args.reference, 1, "cpu", engine_factory=NativeDummyEngine,
                         control_mode="target_pd", contact_mode="support_v2",
                         reference_factory=lambda p, r, d: SupportTransitionReference(p, r, d, mode=mode))
        validate_reference_substitution(source, task.observation_contract())
        task.episode_seconds = 12.
        task.reference_phase = lambda time: (time / task.reference.duration).clamp(max=1.)
        return task

    first_task = make_task("static")
    agent = PPOAgent(config, first_task, "cpu")
    agent.load(str(args.initialize_from))
    actor = StandPolicy(agent).eval()
    cases = []
    args.out.mkdir(parents=True, exist_ok=False)
    for mode in ("static", "transition"):
        for force_n in (-PULSE_FORCE_N, PULSE_FORCE_N):
            for feedback in ("actor", "native_oracle", "observation"):
                task = make_task(mode)
                if feedback == "actor":
                    controller = actor
                else:
                    controller = ComFeedbackTeacher(
                        task, actor, position_gain=1.5, velocity_gain=1.5,
                        observer=observer if feedback == "observation" else None)
                result = dict(reference_mode=mode, feedback=feedback, force_n=force_n,
                              **trial(task, controller, observer, force_n))
                cases.append(result)
                print(json.dumps(dict(mode=mode, feedback=feedback, force_n=force_n,
                                      survival_seconds=result["survival_seconds"],
                                      gate_passed=result["gate"]["passed"])), flush=True)
                (args.out / "progress.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    report = dict(schema="mimic_observed_com_native_validation_v1",
                  model_sha256=sha256(rig.path), checkpoint_sha256=sha256(args.initialize_from),
                  source_contract_sha256=sha256(args.source_contract),
                  observer_contract=observer.contract(), episode_seconds=12.,
                  force_pulse=dict(start_seconds=PULSE_START_SECONDS,
                                   duration_seconds=PULSE_DURATION_SECONDS,
                                   force_n=[-PULSE_FORCE_N, PULSE_FORCE_N], body="Pelvis"),
                  cases=cases,
                  scope="Native plant only; no training, Godot parity, export, or policy promotion")
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
    run(parser.parse_args())
