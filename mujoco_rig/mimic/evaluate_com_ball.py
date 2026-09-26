"""Paired native ballistic-hit stress for the frozen Stand actor and COM residual."""
import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from .baseline import sha256
from .checkpoints import standing_passed
from .com_feedback_teacher import ComFeedbackTeacher
from .capture_step_reflex import CaptureStepReflex, WIDE_ACTION_RADIANS, WideStepTargetControl
from .com_residual_policy import load_experimental_residual
from .evaluate_stand import evaluate
from .native_engine import NativeDummyEngine
from .observation_com import ObservationCom
from .ball_native import BallNativeEngine
from .policy import StandPolicy
from .probe_steps import validate_reference_substitution
from .rig import Rig
from .runtime import activate
from .support_transition import SupportTransitionReference


SCHEMA = "mimic_com_residual_ball_stress_v1"
EPISODE_SECONDS = 33.
SHOT_SECONDS = 2.
DIRECTIONS = (("front", 0.), ("back", math.pi),
              ("front_diagonal", math.pi / 4), ("back_diagonal", 3 * math.pi / 4))
SPEEDS = (1.5, 2.)
REFLEX_SWEEP = {
    "ankle_hip_2_2": dict(position_gain=2., velocity_gain=2.,
                          max_correction=.5, hip_fraction=.3),
    "ankle_hip_2_2_wide": dict(position_gain=2., velocity_gain=2.,
                               max_correction=.75, hip_fraction=.3),
    "ankle_3_1p5_wide": dict(position_gain=3., velocity_gain=1.5,
                              max_correction=.75, hip_fraction=0.),
}


class ImpactBallEngine(BallNativeEngine):
    """Accumulate normal contact impulse without altering ball dynamics."""

    def __init__(self, rig, num_envs, device="cpu", control_factory=None):
        from .target_control import DelayedTargetControl

        super().__init__(rig, num_envs, device, control_factory or DelayedTargetControl)
        self.impact_impulse = torch.zeros(num_envs)

    def reset_envs(self, ids, qpos, qvel):
        super().reset_envs(ids, qpos, qvel)
        self.impact_impulse[ids] = 0

    def _after_physics_step(self, world, data):
        super()._after_physics_step(world, data)
        for index, contact in enumerate(data.contact[:data.ncon]):
            a, b = self.rig.model.geom_bodyid[contact.geom]
            if ((a == self.ball_body and 0 < b < self.ball_body)
                    or (b == self.ball_body and 0 < a < self.ball_body)):
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.rig.model, data, index, wrench)
                self.impact_impulse[world] += max(0., wrench[0]) * self.rig.model.opt.timestep


def cases():
    return [dict(name=f"{direction}-{speed:g}", direction=direction,
                 angle_radians=angle, speed_m_s=speed, target_body="Chest")
            for direction, angle in DIRECTIONS for speed in SPEEDS]


def score(task, controller, specs):
    from envs.base_env import DoneFlags, EnvMode

    task.set_mode(EnvMode.TEST)
    task.reset()
    count = len(specs)
    task.offset.zero_()
    task.steps.zero_()
    task.engine.reset_envs(task.ids, *task.reference.sample(task.offset))
    task.shot_steps[:] = round(SHOT_SECONDS / task.dt)
    task.shot_speeds[:] = torch.tensor([case["speed_m_s"] for case in specs])
    task.shot_target_body[:] = torch.tensor([
        task.rig.model.body(case["target_body"]).id - 1 for case in specs])
    angles = torch.tensor([case["angle_radians"] for case in specs])
    task.directions[:, 0], task.directions[:, 1] = angles.cos(), angles.sin()
    task.directions[:, 2] = 0
    task.shot_enabled[:] = True
    active = torch.ones(count, dtype=torch.bool)
    hit = torch.zeros(count, dtype=torch.bool)
    first_hit = torch.full((count,), -1, dtype=torch.long)
    lengths = torch.zeros(count, dtype=torch.long)
    successes = torch.zeros(count, dtype=torch.bool)
    stable = torch.zeros(count, dtype=torch.long)
    peak_foot_shift = torch.zeros(count)
    impact_impulse = torch.zeros(count)
    prehit_feet = None
    settle_steps = math.ceil(.5 / task.dt)
    for step in range(math.ceil(EPISODE_SECONDS / task.dt)):
        obs = task.observations()
        with torch.no_grad():
            action = controller(obs)
        if step == task.shot_steps[0].item():
            prehit_feet = task.engine.get_body_pos(0)[:, task.feet, :].clone()
        _, _, done, _ = task.step(action)
        hits_now = task.engine.hit & active
        first_hit[(first_hit < 0) & hits_now] = step + 1
        hit |= hits_now
        impact_impulse = torch.maximum(impact_impulse, task.engine.impact_impulse)
        lengths += active
        q = task.engine.native_qpos()
        loads = task.engine.get_ground_contact_forces(0)[:, task.allowed_ground, 2].abs()
        supported = (loads[:, 0] + loads[:, 2] > 5) & (loads[:, 1] + loads[:, 3] > 5)
        upright = 1 - 2 * (q[:, 4].square() + q[:, 5].square()) >= math.cos(math.radians(15))
        settled = (q[:, 2] >= .75) & upright & supported
        settled &= torch.linalg.vector_norm(task.engine.get_root_vel(0)[:, :2], dim=-1) <= .2
        settled &= torch.linalg.vector_norm(task.engine.get_root_ang_vel(0), dim=-1) <= 1.
        stable = torch.where(settled & hit & active, stable + 1, 0)
        if prehit_feet is not None:
            foot_shift = torch.linalg.vector_norm(
                task.engine.get_body_pos(0)[:, task.feet, :] - prehit_feet, dim=-1).amax(dim=1)
            peak_foot_shift = torch.maximum(peak_foot_shift, torch.where(active, foot_shift,
                                                                          peak_foot_shift))
        successes |= active & (done == DoneFlags.TIME.value)
        active &= done == DoneFlags.NULL.value
        if not active.any():
            break
        finished = torch.nonzero(done != DoneFlags.NULL.value).flatten()
        if len(finished):
            if hasattr(controller, "deactivate"):
                controller.deactivate(finished)
            task.reset(finished)
            task.shot_enabled[finished] = False
    result = []
    for i, case in enumerate(specs):
        hit_seconds = float(first_hit[i] * task.dt) if hit[i] else None
        post_hit_seconds = float(lengths[i] * task.dt - hit_seconds) if hit_seconds is not None else None
        recovered = bool(hit[i] and successes[i] and post_hit_seconds >= 30.
                         and stable[i] >= settle_steps)
        result.append(dict(**case, hit=bool(hit[i]), survived=bool(successes[i]),
                           recovered=recovered, survival_seconds=float(lengths[i] * task.dt),
                           first_hit_seconds=hit_seconds, post_hit_seconds=post_hit_seconds,
                           impact_normal_impulse_ns=float(impact_impulse[i]),
                           peak_foot_shift_m=float(peak_foot_shift[i])))
    return result


def summarize(rows):
    confirmed = [row for row in rows if row["hit"]]
    return dict(cases=len(rows), confirmed_hits=len(confirmed),
                survived_hits=sum(row["survived"] for row in confirmed),
                recovered_hits=sum(row["recovered"] for row in confirmed),
                delayed_falls_after_5s=sum(row["survival_seconds"] >= 5 and not row["survived"]
                                            for row in confirmed),
                missed_shots=sum(not row["hit"] for row in rows))


def run(args):
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .ball_task import BallTask
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
    support = StandTask(rig, args.reference, 1, "cpu", engine_factory=NativeDummyEngine,
                        control_mode="target_pd", contact_mode="support_v2",
                        reference_factory=lambda p, r, d: SupportTransitionReference(
                            p, r, d, mode="transition"))
    validate_reference_substitution(source, best["contract"])
    validate_reference_substitution(source, support.observation_contract())
    agent = PPOAgent(yaml.safe_load(args.agent_config.read_text()), support, "cpu")
    agent.load(str(args.initialize_from))
    actor = StandPolicy(agent).eval()
    candidate = load_experimental_residual(args.residual, actor, rig,
                                           args.source_contract, args.initialize_from)
    observer = ObservationCom(rig, source) if (args.include_teachers or args.reflex_sweep
                                               or args.capture_step or args.capture_step_wide) else None
    specs = cases()
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    labels = ["source_actor", "com_residual"]
    if args.include_teachers:
        labels.extend(("native_com_teacher", "observation_com_teacher"))
    if args.reflex_sweep:
        labels.extend(REFLEX_SWEEP)
    if args.capture_step:
        labels.append("capture_step_reflex")
    if args.capture_step_wide:
        labels.append("capture_step_wide")

    def wide_engine(model_rig, count, device, control_factory=None):
        return ImpactBallEngine(model_rig, count, device, control_factory=WideStepTargetControl)

    for label in labels:
        task = BallTask(rig, Path(__file__).parent / "assets/stand_reference.npz",
                        len(specs), "cpu",
                        engine_factory=wide_engine if label == "capture_step_wide" else ImpactBallEngine,
                        contact_mode="support_v2")
        if args.ball_reference == "support_entry":
            task.reference = SupportTransitionReference(args.reference, rig, "cpu", mode="transition")
        expected = ({**source, "residual_radians": WIDE_ACTION_RADIANS}
                    if label == "capture_step_wide" else source)
        validate_reference_substitution(expected, task.observation_contract())
        task.episode_seconds = EPISODE_SECONDS
        task.reference_phase = lambda time: (time / task.reference.duration).clamp(max=1.)
        if label == "source_actor":
            controller = actor
        elif label == "com_residual":
            controller = candidate
        elif label in ("capture_step_reflex", "capture_step_wide"):
            radians = WIDE_ACTION_RADIANS if label == "capture_step_wide" else .25
            controller = CaptureStepReflex(task, actor, observer, action_radians=radians)
        else:
            params = REFLEX_SWEEP.get(label, dict(position_gain=1.5, velocity_gain=1.5))
            controller = ComFeedbackTeacher(task, actor, **params,
                                            observer=None if label == "native_com_teacher" else observer)
        rows = score(task, controller, specs)
        if hasattr(controller, "case_metrics"):
            for row, metrics in zip(rows, controller.case_metrics()):
                row.update(metrics)
        result = dict(label=label, summary=summarize(rows), cases=rows,
                      controller_contract=controller.contract()
                      if isinstance(controller, (ComFeedbackTeacher, CaptureStepReflex)) else None)
        results.append(result)
        (args.out / f"{label}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(dict(label=label, **result["summary"])), flush=True)
    quiet = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz", 8,
                      "cpu", engine_factory=NativeDummyEngine, control_mode="target_pd",
                      contact_mode="support_v2")
    validate_reference_substitution(source, quiet.observation_contract())
    quiet_results = {}
    support_hold_results = {}
    for label, controller in (("source_actor", actor), ("com_residual", candidate)):
        metrics, _ = evaluate(quiet, controller)
        quiet_results[label] = dict(metrics=metrics, passed=standing_passed(metrics))
        hold = StandTask(rig, args.reference, 1, "cpu", engine_factory=NativeDummyEngine,
                         control_mode="target_pd", contact_mode="support_v2",
                         reference_factory=lambda p, r, d: SupportTransitionReference(
                             p, r, d, mode="transition"))
        hold.episode_seconds = EPISODE_SECONDS
        hold.reference_phase = lambda time: (time / hold.reference.duration).clamp(max=1.)
        hold_metrics, _ = evaluate(hold, controller, episodes=1)
        support_hold_results[label] = hold_metrics
    if args.reflex_sweep:
        for label, params in REFLEX_SWEEP.items():
            quiet_teacher = ComFeedbackTeacher(quiet, actor, **params, observer=observer)
            metrics, _ = evaluate(quiet, quiet_teacher)
            quiet_results[label] = dict(metrics=metrics, passed=standing_passed(metrics))
    report = dict(schema=SCHEMA, model_sha256=sha256(rig.path),
                  ball_model_sha256=sha256(task.engine.rig.path),
                  source_checkpoint_sha256=sha256(args.initialize_from),
                  residual_sha256=sha256(args.residual),
                  ball_reference=dict(mode=args.ball_reference,
                                      sha256=task.reference.sha256,
                                      duration_seconds=task.reference.duration),
                  support_reference=support.reference.contract(),
                  episode_seconds=EPISODE_SECONDS, shot_seconds=SHOT_SECONDS,
                  recovery="Confirmed ball impact, 30 seconds afterward, and final uninterrupted 0.5-second upright/loaded-foot settling",
                  phase_behavior="Clamped at reference endpoint as in residual training",
                  include_teachers=args.include_teachers,
                  reflex_sweep=REFLEX_SWEEP if args.reflex_sweep else None,
                  capture_step_wide=args.capture_step_wide,
                  results=results, quiet=quiet_results, support_hold=support_hold_results,
                  scope="Native ballistic controller comparison; no training, export, or Godot policy selection")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(quiet={key: value["passed"] for key, value in quiet_results.items()},
                          support_hold_seconds={key: value["mean_survival_seconds"]
                                                for key, value in support_hold_results.items()})), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path, required=True)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--ball-reference", choices=("support_entry", "original_stand"),
                        default="support_entry")
    parser.add_argument("--include-teachers", action="store_true")
    parser.add_argument("--reflex-sweep", action="store_true")
    parser.add_argument("--capture-step", action="store_true")
    parser.add_argument("--capture-step-wide", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
