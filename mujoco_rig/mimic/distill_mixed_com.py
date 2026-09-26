"""One bounded, unselected COM-residual distillation on support and physical hits."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from .baseline import sha256
from .com_feedback_teacher import ComFeedbackTeacher
from .com_residual_policy import MIXED_SCHEMA, build_residual_network
from .distill_com_feedback import (QUIET_REHEARSAL_WEIGHT, TRAIN_FORCES_N,
                                   collect, collect_quiet)
from .evaluate_com_ball import ImpactBallEngine, cases as evaluation_cases
from .native_engine import NativeDummyEngine
from .observation_com import ObservationCom
from .policy import StandPolicy
from .probe_steps import validate_reference_substitution
from .rig import Rig
from .runtime import activate
from .support_transition import SupportTransitionReference


EPISODE_SECONDS = 8.
RESIDUAL_MAX = .5


def training_cases():
    """Fixed impacts offset in direction, speed, time, body, and phase from evaluation."""
    return [dict(angle_radians=math.radians(angle), speed_m_s=speed,
                 launch_seconds=launch, target_body=body, phase_seconds=phase)
            for angle, speed, launch, body, phase in (
                (15, 1.25, 1.75, "Spine", .2),
                (-15, 1.75, 2.25, "Pelvis", .6),
                (165, 1.25, 1.75, "Pelvis", .2),
                (195, 1.75, 2.25, "Spine", .6),
                (30, 1.25, 2.25, "Spine", .2),
                (60, 1.75, 1.75, "Pelvis", .6),
                (120, 1.25, 2.25, "Pelvis", .2),
                (150, 1.75, 1.75, "Spine", .6))]


def assert_disjoint(training):
    evaluation = {(row["angle_radians"], row["speed_m_s"],
                   2., row["target_body"], 0.) for row in evaluation_cases()}
    for row in training:
        signature = (row["angle_radians"], row["speed_m_s"],
                     row["launch_seconds"], row["target_body"], row["phase_seconds"])
        if signature in evaluation:
            raise ValueError("Training impact overlaps final evaluation")


def collect_ball(task, teacher, case):
    from envs.base_env import DoneFlags, EnvMode

    task.set_mode(EnvMode.TEST)
    task.reset()
    task.offset[:] = case["phase_seconds"]
    task.steps.zero_()
    task.engine.reset_envs(task.ids, *task.reference.sample(task.offset))
    task.shot_steps[:] = round(case["launch_seconds"] / task.dt)
    task.shot_speeds[:] = case["speed_m_s"]
    task.shot_target_body[:] = task.rig.model.body(case["target_body"]).id - 1
    angle = case["angle_radians"]
    task.directions[:, 0], task.directions[:, 1] = math.cos(angle), math.sin(angle)
    task.directions[:, 2] = 0.
    task.shot_enabled[:] = True
    teacher.records.clear()
    observations, labels = [], []
    done = DoneFlags.NULL.value
    for _ in range(round(task.episode_seconds / task.dt)):
        obs = task.observations()
        with torch.no_grad():
            action = teacher(obs)
        observations.append(obs[0].numpy().copy())
        labels.append(teacher.records[-1]["correction"])
        _, _, flags, _ = task.step(action)
        done = int(flags[0])
        if done != DoneFlags.NULL.value:
            break
    return (np.asarray(observations, dtype=np.float32),
            np.asarray(labels, dtype=np.float32),
            dict(**case, hit=bool(task.engine.hit[0]),
                 survived=done == DoneFlags.TIME.value,
                 survival_seconds=len(labels) * task.dt,
                 peak_abs_correction=max(map(abs, labels))))


def run(args):
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .ball_task import BallTask
    from .stand_task import StandTask

    if args.out.exists():
        raise FileExistsError(args.out)
    training = training_cases()
    assert_disjoint(training)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    source = json.loads(args.source_contract.read_text())
    best = json.loads(args.initialize_from.with_name("best.json").read_text())
    if best["checkpoint_sha256"] != sha256(args.initialize_from):
        raise ValueError("Source actor hash mismatch")
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

    initial = support_task("static")
    agent = PPOAgent(yaml.safe_load(args.agent_config.read_text()), initial, "cpu")
    agent.load(str(args.initialize_from))
    actor = StandPolicy(agent).eval()
    teacher_kwargs = dict(position_gain=1.5, velocity_gain=1.5, observer=observer)
    ankles = ComFeedbackTeacher(initial, actor, **teacher_kwargs).ankles
    args.out.mkdir(parents=True, exist_ok=False)
    data, targets, outcomes = [], [], []

    for mode in ("static", "transition"):
        for force in TRAIN_FORCES_N:
            task = support_task(mode)
            x, y, result = collect(task, ComFeedbackTeacher(task, actor, **teacher_kwargs), force)
            data.append(x)
            targets.append(y)
            outcomes.append(dict(kind="support_pulse", mode=mode, **result, samples=len(y)))
            print(json.dumps(outcomes[-1]), flush=True)

    asset = Path(__file__).parent / "assets/stand_reference.npz"
    for case in training:
        task = BallTask(rig, asset, 1, "cpu", engine_factory=ImpactBallEngine,
                        contact_mode="support_v2")
        validate_reference_substitution(source, task.observation_contract())
        task.episode_seconds = EPISODE_SECONDS
        task.reference_phase = lambda time: (time / task.reference.duration).clamp(max=1.)
        x, y, result = collect_ball(task, ComFeedbackTeacher(task, actor, **teacher_kwargs), case)
        data.append(x)
        targets.append(y)
        outcomes.append(dict(kind="original_stand_ball", **result, samples=len(y)))
        print(json.dumps(outcomes[-1]), flush=True)

    quiet = StandTask(rig, asset, 8, "cpu", engine_factory=NativeDummyEngine,
                      control_mode="target_pd", contact_mode="support_v2")
    validate_reference_substitution(source, quiet.observation_contract())
    x_quiet, y_quiet = collect_quiet(quiet, actor)
    data.append(np.tile(x_quiet, (QUIET_REHEARSAL_WEIGHT, 1)))
    targets.append(np.tile(y_quiet, QUIET_REHEARSAL_WEIGHT))
    x = torch.from_numpy(np.concatenate(data))
    y = torch.from_numpy(np.concatenate(targets))[:, None]
    mean, scale = x.mean(0), x.std(0).clamp_min(.1)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    normalized = ((x - mean) / scale).clamp(-5., 5.).to(device)
    target = y.to(device)
    model = build_residual_network(x.shape[1]).to(device)
    torch.nn.init.zeros_(model[-2].weight)
    torch.nn.init.zeros_(model[-2].bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    losses = []
    for epoch in range(8):
        permutation = torch.randperm(len(x), device=device)
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

    artifact = args.out / "experimental-residual.pt"
    torch.save(dict(schema=MIXED_SCHEMA, model=model.cpu().state_dict(), mean=mean,
                    scale=scale, residual_max=RESIDUAL_MAX, ankle_indices=ankles,
                    source_contract_sha256=sha256(args.source_contract),
                    source_checkpoint_sha256=sha256(args.initialize_from)), artifact)
    report = dict(schema=MIXED_SCHEMA, source_checkpoint_sha256=sha256(args.initialize_from),
                  source_contract_sha256=sha256(args.source_contract),
                  observer_contract=observer.contract(), teacher=teacher_kwargs | {"observer": "observation_com"},
                  seed=args.seed, training_device=device, training_cases=outcomes,
                  training_samples=len(x), quiet_samples=len(x_quiet),
                  quiet_rehearsal_weight=QUIET_REHEARSAL_WEIGHT,
                  residual_max=RESIDUAL_MAX, train_mse_by_epoch=losses,
                  evaluation_cases_excluded=True,
                  scope="Experimental mixed-reference supervised residual; no export or Godot selection")
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
    parser.add_argument("--seed", type=int, default=210921)
    run(parser.parse_args())
