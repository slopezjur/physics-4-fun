"""Bounded supervised residual pilot; source actor stays frozen and unselected."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .baseline import sha256
from .checkpoints import standing_passed
from .com_feedback_teacher import ComFeedbackTeacher
from .com_residual_policy import ComResidualPolicy, PILOT_SCHEMA, build_residual_network
from .evaluate_com_observation import PULSE_DURATION_SECONDS, trial
from .evaluate_stand import evaluate
from .native_engine import NativeDummyEngine
from .observation_com import ObservationCom
from .policy import StandPolicy
from .probe_steps import validate_reference_substitution
from .rig import Rig
from .runtime import activate
from .support_transition import SupportTransitionReference


SCHEMA = PILOT_SCHEMA
TRAIN_FORCES_N = (-160., -80., 80., 160.)
HELD_OUT_FORCES_N = (-120., 120.)
TRAIN_PULSE_SECONDS = 2.
HELD_OUT_PULSE_SECONDS = (1.75, 2.25)
RESIDUAL_MAX = .2
QUIET_REHEARSAL_WEIGHT = 4


def collect(task, teacher, force_n):
    from envs.base_env import DoneFlags, EnvMode

    task.set_mode(EnvMode.TEST)
    task.reset()
    observations, labels = [], []
    done = DoneFlags.NULL.value
    for step in range(round(task.episode_seconds / task.dt)):
        obs = task.observations()
        with torch.no_grad():
            action = teacher(obs)
        observations.append(obs[0].numpy().copy())
        labels.append(teacher.records[-1]["correction"])
        seconds = step * task.dt
        if TRAIN_PULSE_SECONDS <= seconds < TRAIN_PULSE_SECONDS + PULSE_DURATION_SECONDS:
            task.engine.set_external_force(1, [[force_n, 0., 0.]])
        _, _, flags, _ = task.step(action)
        done = int(flags[0])
        if done != DoneFlags.NULL.value:
            break
    return np.asarray(observations, dtype=np.float32), np.asarray(labels, dtype=np.float32), dict(
        force_n=force_n, survived=done == DoneFlags.TIME.value,
        survival_seconds=len(observations) * task.dt)


def collect_quiet(task, actor):
    from envs.base_env import DoneFlags, EnvMode

    task.set_mode(EnvMode.TEST)
    task.reset()
    observations = []
    active = torch.ones(task.get_num_envs(), dtype=torch.bool)
    for _ in range(round(task.episode_seconds / task.dt)):
        obs = task.observations()
        observations.append(obs[active].numpy().copy())
        with torch.no_grad():
            action = actor(obs)
        _, _, done, _ = task.step(action)
        active &= done == DoneFlags.NULL.value
        if not active.any():
            break
    samples = np.concatenate(observations).astype(np.float32)
    return samples, np.zeros(len(samples), dtype=np.float32)


def run(args):
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .stand_task import StandTask

    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    source = json.loads(args.source_contract.read_text())
    best = json.loads(args.initialize_from.with_name("best.json").read_text())
    if best["checkpoint_sha256"] != sha256(args.initialize_from):
        raise ValueError("Source checkpoint hash mismatch")
    validate_reference_substitution(source, best["contract"])
    observer = ObservationCom(rig, source)

    def support_task(mode):
        task = StandTask(rig, args.reference, 1, "cpu", engine_factory=NativeDummyEngine,
                         control_mode="target_pd", contact_mode="support_v2",
                         reference_factory=lambda p, r, d: SupportTransitionReference(p, r, d, mode=mode))
        validate_reference_substitution(source, task.observation_contract())
        task.episode_seconds = 12.
        task.reference_phase = lambda time: (time / task.reference.duration).clamp(max=1.)
        return task

    first = support_task("static")
    agent = PPOAgent(yaml.safe_load(args.agent_config.read_text()), first, "cpu")
    agent.load(str(args.initialize_from))
    actor = StandPolicy(agent).eval()
    teacher_probe = ComFeedbackTeacher(first, actor, position_gain=1.5, velocity_gain=1.5,
                                       observer=observer)
    ankles = teacher_probe.ankles
    args.out.mkdir(parents=True, exist_ok=False)
    data, targets, collected = [], [], []
    for mode in ("static", "transition"):
        for force_n in TRAIN_FORCES_N:
            task = support_task(mode)
            teacher = ComFeedbackTeacher(task, actor, position_gain=1.5,
                                         velocity_gain=1.5, observer=observer)
            observations, labels, outcome = collect(task, teacher, force_n)
            data.append(observations)
            targets.append(labels)
            collected.append(dict(mode=mode, **outcome, samples=len(labels)))
            print(json.dumps(collected[-1]), flush=True)
    quiet = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz", 8,
                      "cpu", engine_factory=NativeDummyEngine, control_mode="target_pd",
                      contact_mode="support_v2")
    validate_reference_substitution(source, quiet.observation_contract())
    quiet_observations, quiet_labels = collect_quiet(quiet, actor)
    data.append(np.tile(quiet_observations, (QUIET_REHEARSAL_WEIGHT, 1)))
    targets.append(np.tile(quiet_labels, QUIET_REHEARSAL_WEIGHT))
    x = torch.from_numpy(np.concatenate(data))
    y = torch.from_numpy(np.concatenate(targets))[:, None]
    mean = x.mean(0)
    scale = x.std(0).clamp_min(.1)
    training_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    normalized = ((x - mean) / scale).clamp(-5., 5.).to(training_device)
    target = y.to(training_device)
    model = build_residual_network(x.shape[1]).to(training_device)
    torch.nn.init.zeros_(model[-2].weight)
    torch.nn.init.zeros_(model[-2].bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    losses = []
    for epoch in range(8):
        permutation = torch.randperm(len(x), device=training_device)
        total = 0.
        for indices in permutation.split(512):
            prediction = RESIDUAL_MAX * model(normalized[indices])
            loss = torch.nn.functional.mse_loss(prediction, target[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        losses.append(total / len(x))
        print(json.dumps(dict(epoch=epoch + 1, train_mse=losses[-1])), flush=True)
    model = model.cpu()
    model.eval()

    candidate = ComResidualPolicy(actor, model, mean, scale, ankles, RESIDUAL_MAX)
    held_out = []
    for mode in ("static", "transition"):
        for pulse_start in HELD_OUT_PULSE_SECONDS:
            for force_n in HELD_OUT_FORCES_N:
                task = support_task(mode)
                result = trial(task, candidate, observer, force_n, pulse_start=pulse_start)
                held_out.append(dict(mode=mode, pulse_start_seconds=pulse_start,
                                     force_n=force_n, **result))
                print(json.dumps(dict(mode=mode, pulse_start=pulse_start, force_n=force_n,
                                      survival_seconds=result["survival_seconds"],
                                      passed=result["gate"]["passed"])), flush=True)
    quiet_metrics, _ = evaluate(quiet, candidate)
    passed = all(case["gate"]["passed"] for case in held_out) and standing_passed(quiet_metrics)
    report = dict(schema=SCHEMA, source_checkpoint_sha256=sha256(args.initialize_from),
                  source_contract_sha256=sha256(args.source_contract),
                  observer_contract=observer.contract(), seed=args.seed,
                  train_forces_n=TRAIN_FORCES_N, train_pulse_seconds=TRAIN_PULSE_SECONDS,
                  residual_max=RESIDUAL_MAX, quiet_rehearsal_weight=QUIET_REHEARSAL_WEIGHT,
                  training_device=training_device, training_cases=collected,
                  training_samples=len(x), quiet_rehearsal_samples=len(quiet_labels),
                  training_mse_by_epoch=losses,
                  held_out_forces_n=HELD_OUT_FORCES_N,
                  held_out_pulse_seconds=HELD_OUT_PULSE_SECONDS,
                  held_out=held_out, quiet=quiet_metrics, passed=passed,
                  scope="Supervised residual pilot only; frozen source actor, no export or Godot selection")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(dict(schema=SCHEMA, model=model.state_dict(), mean=mean,
                    scale=scale, residual_max=RESIDUAL_MAX, ankle_indices=ankles,
                    source_contract_sha256=sha256(args.source_contract),
                    source_checkpoint_sha256=sha256(args.initialize_from)),
               args.out / "experimental-residual.pt")
    print(json.dumps(dict(held_out_passed=sum(case["gate"]["passed"] for case in held_out),
                          held_out_total=len(held_out), quiet_successes=quiet_metrics["successes"],
                          passed=passed)), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=210921)
    run(parser.parse_args())
