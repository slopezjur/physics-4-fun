"""Diagnostic PPO residual on a physical step outcome, not clip imitation."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .baseline import sha256
from .checkpoints import standing_passed
from .distill_recovery_step import StepStudent
from .evaluate_stand import evaluate as evaluate_stand
from .native_engine import NativeDummyEngine
from .policy import StandPolicy
from .probe_single_support_entry import EntryReference
from .recovery_step_motion import evaluate as evaluate_step
from .rig import Rig
from .runtime import activate


SCHEMA = "mimic_step_outcome_residual_pilot_v1"


class PlaybackResidual:
    def __init__(self, task, actions, source, actor, scale):
        self.task, self.actions = task, actions
        self.source, self.actor, self.scale = source, actor, scale

    @torch.no_grad()
    def __call__(self, observation):
        index = self.task.steps.clamp(max=len(self.actions) - 1)
        return (self.actions[index] + self.scale *
                (self.actor(observation) - self.source(observation))).clamp(-1., 1.)


def run(args):
    if (args.out.exists() or args.seconds <= 0 or args.envs < 32
            or args.envs % 32 or not 0. < args.residual_scale <= .5
            or args.evaluation_interval < 1):
        raise ValueError("Require a new output directory and bounded pilot settings")
    activate(args.mimickit)
    from util import mp_util
    from .guarded_ppo import GuardedPPO
    from .newton_adapter import DummyNewtonEngine
    from .stand_task import StandTask

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mp_util.init(0, 1, "cuda:0", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    source_path = args.teacher_run / "report.json"
    search_path = args.teacher_search / "report.json"
    motion_path = args.teacher_run / "step_reference.npz"
    actions_path = args.teacher_search / "best_teacher_samples.npz"
    source_report = json.loads(source_path.read_text(encoding="utf-8"))
    search = json.loads(search_path.read_text(encoding="utf-8"))
    best = json.loads(args.initialize_from.with_name("best.json").read_text(encoding="utf-8"))
    contract = json.loads(args.source_contract.read_text(encoding="utf-8"))
    if (not search["passed"] or search["source_report_sha256"] != sha256(source_path)
            or search["source_motion_sha256"] != sha256(motion_path)
            or search["teacher_samples_sha256"] != sha256(actions_path)
            or search["model_sha256"] != sha256(rig.path)
            or best["checkpoint_sha256"] != sha256(args.initialize_from)
            or contract["model_sha256"] != sha256(rig.path)
            or contract["observation_contract"] != "mimic_stand_target_pd_v2"):
        raise ValueError("Require hash-bound teacher, model and source Stand actor")
    with np.load(motion_path, allow_pickle=False) as arrays:
        poses, dt = arrays["qpos"].copy(), float(arrays["dt"])
    with np.load(actions_path, allow_pickle=False) as arrays:
        actions = torch.as_tensor(arrays["action"].copy(), dtype=torch.float32)
        action_dt = float(arrays["dt"])
    if not np.isclose(dt / action_dt, round(dt / action_dt), rtol=.001):
        raise ValueError("Action playback and reference periods differ")
    source_cpu = StepStudent(args.initialize_from).eval()

    def reference_factory(_path, model_rig, device):
        return EntryReference(poses, model_rig, dt, device)

    class OutcomeTask(StandTask):
        def __init__(self, *task_args, **task_kwargs):
            device = task_kwargs.get("device", task_args[3])
            worlds = task_kwargs.get("num_envs", task_args[2])
            self.playback = actions.to(device)
            self.source_actor = copy.deepcopy(source_cpu).to(device).eval()
            self.residual_scale = args.residual_scale
            self.start_left_x = torch.zeros(worlds, device=device)
            self.start_left_z = torch.zeros(worlds, device=device)
            self.push_force = torch.zeros((worlds, 3), device=device)
            self.push_onset = torch.zeros(worlds, device=device)
            self.chest = rig.model.body("Chest").id
            super().__init__(*task_args, **task_kwargs)
            self.episode_seconds = self.reference.duration

        def reset(self, env_ids=None):
            observation, info = super().reset(env_ids)
            ids = self.ids if env_ids is None else env_ids
            feet = self.engine.get_body_pos(0)[ids, self.feet[0]]
            self.start_left_x[ids] = feet[:, 0]
            self.start_left_z[ids] = feet[:, 2]
            draws = torch.randint(0, 9, (len(ids),), device=self.device,
                                  generator=self.rng)
            axis = torch.div(draws, 2, rounding_mode="floor") % 2
            sign = torch.where(draws % 2 == 0, -1., 1.)
            self.push_force[ids] = 0.
            disturbed = draws < 8
            self.push_force[ids[disturbed], axis[disturbed]] = 10. * sign[disturbed]
            self.push_onset[ids] = torch.where(
                torch.rand(len(ids), device=self.device, generator=self.rng) < .5,
                1.75, 2.75)
            return observation, info

        def reward(self):
            q = self.engine.native_qpos()
            velocity = self.engine.get_root_vel(0)[:, :2]
            feet = self.engine.get_body_pos(0)[:, self.feet[0]]
            loads = self.engine.get_ground_contact_forces(0)[:, self.allowed_ground, 2].abs()
            left, right = loads[:, 0] + loads[:, 2], loads[:, 1] + loads[:, 3]
            shift = feet[:, 0] - self.start_left_x
            lift = feet[:, 2] - self.start_left_z
            upright = torch.clamp((q[:, 2] - .65) / .2, 0., 1.)
            placement = torch.sigmoid((shift - .13) / .025)
            swing = torch.sigmoid((lift - .03) / .015)
            two_foot = torch.sigmoid((left - 20.) / 10.) * torch.sigmoid((right - 20.) / 10.)
            slow = torch.exp(-velocity.square().sum(dim=1) / .2 ** 2)
            seconds = self.steps * self.dt
            before = (seconds < 2.55).float()
            return (.2 * upright + .15 * slow
                    + before * .3 * swing * placement
                    + (1. - before) * (1.2 * placement * two_foot + .2 * slow))

        def step(self, action):
            seconds = self.steps * self.dt
            active = (seconds >= self.push_onset) & (seconds < self.push_onset + .1)
            self.engine.set_external_force(self.chest, self.push_force * active[:, None])
            with torch.no_grad():
                source_action = self.source_actor(self.observations())
                index = self.steps.clamp(max=len(self.playback) - 1)
                applied = (self.playback[index] + self.residual_scale *
                           (action - source_action)).clamp(-1., 1.)
            return super().step(applied)

    train = OutcomeTask(rig, motion_path, args.envs, "cuda:0", seed=args.seed,
                        engine_factory=DummyNewtonEngine, control_mode="target_pd",
                        contact_mode="support_v2", reference_factory=reference_factory)
    native = StandTask(rig, motion_path, 1, "cpu", engine_factory=NativeDummyEngine,
                       control_mode="target_pd", contact_mode="support_v2",
                       reference_factory=reference_factory)
    native.episode_seconds = native.reference.duration
    quiet = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz",
                      8, "cpu", engine_factory=NativeDummyEngine,
                      control_mode="target_pd", contact_mode="support_v2")
    config = yaml.safe_load(args.agent_config.read_text(encoding="utf-8"))
    config.update(actor_max_kl=.02, test_episodes=0,
                  observation_normalization="frozen_loaded_statistics",
                  actor_gradient_clip=1.)
    config["critic_optimizer"] = {**config["critic_optimizer"], "grad_clip": 1.}
    args.out.mkdir(parents=True)
    (args.out / "agent.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    protocol = dict(schema=SCHEMA, model_sha256=sha256(rig.path),
                    source_checkpoint_sha256=sha256(args.initialize_from),
                    source_contract_sha256=sha256(args.source_contract),
                    teacher_search_sha256=sha256(search_path),
                    teacher_actions_sha256=sha256(actions_path),
                    source_reference_sha256=sha256(motion_path),
                    source_reference_admitted=False,
                    residual_scale=args.residual_scale, reward="Observed swing, loaded placement, upright survival and final speed",
                    disturbance="10 N Chest pulse at 1.75 or 2.75 seconds; 1/9 episodes unforced",
                    envs=args.envs, seconds=args.seconds, seed=args.seed,
                    scope="Diagnostic outcome PPO around teacher playback; no motion admission, ball or export")
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    agent = GuardedPPO(config, train, "cuda:0")
    agent.load(str(args.initialize_from))

    def review():
        actor = copy.deepcopy(StandPolicy(agent)).cpu().eval()
        controller = PlaybackResidual(native, actions, source_cpu, actor, args.residual_scale)
        with torch.no_grad():
            step, _, _ = evaluate_step(native, controller, .15, 2.55)
            standing, _ = evaluate_stand(quiet, actor)
        return dict(step={key: value for key, value in step.items() if key != "trace"},
                    quiet=standing, quiet_passed=standing_passed(standing))

    initial = review()
    best = initial
    best_iteration = 0
    learned_passed = False
    best_rank = (initial["quiet_passed"], initial["step"]["passed"],
                 initial["step"]["survival_seconds"])
    agent.save(str(args.out / "initial.pt"))
    agent.save(str(args.out / "best.pt"))
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    agent._init_train()
    curve = []
    started = time.monotonic()
    while time.monotonic() - started < args.seconds:
        info = agent._train_iter()
        agent._sample_count = agent._update_sample_count()
        agent._iter += 1
        if agent._iter % args.evaluation_interval == 0:
            result = review()
            learned_passed |= bool(result["quiet_passed"] and result["step"]["passed"])
            rank = (result["quiet_passed"], result["step"]["passed"],
                    result["step"]["survival_seconds"])
            row = dict(iteration=agent._iter, samples=agent._sample_count,
                       elapsed_seconds=time.monotonic() - started,
                       actor_kl=float(info["actor_kl"]), measurement=result)
            curve.append(row)
            print(json.dumps(dict(iteration=agent._iter, rank=rank,
                                  survival=result["step"]["survival_seconds"])), flush=True)
            if rank > best_rank:
                best, best_rank = result, rank
                best_iteration = agent._iter
                agent.save(str(args.out / "best.pt"))
    agent.save(str(args.out / "final.pt"))
    final = review()
    learned_passed |= bool(final["quiet_passed"] and final["step"]["passed"])
    report = dict(schema=SCHEMA, protocol_sha256=sha256(args.out / "protocol.json"),
                  initial=initial, best=best, final=final, curve=curve,
                  iterations=agent._iter, samples=agent._sample_count,
                  best_iteration=best_iteration,
                  best_checkpoint_sha256=sha256(args.out / "best.pt"),
                  passed=bool(best_iteration and best["quiet_passed"]
                              and best["step"]["passed"]),
                  learned_checkpoint_passed=bool(learned_passed),
                  admitted_for_deployment=False,
                  scope="Diagnostic outcome PPO; no reference admission, impact generalization, export or scene selection")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(iterations=agent._iter, samples=agent._sample_count,
                          best_step=best["step"], best_quiet=best["quiet_passed"])), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--teacher-search", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=180.)
    parser.add_argument("--evaluation-interval", type=int, default=8)
    parser.add_argument("--residual-scale", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=270927)
    run(parser.parse_args())
