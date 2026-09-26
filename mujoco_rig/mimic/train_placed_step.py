"""Isolated guarded PPO pilot for a physically checked single-step reference."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .baseline import sha256
from .checkpoints import standing_passed, validate_contract
from .contact_contract import mode_from_contract
from .evaluate_stand import evaluate
from .policy import StandPolicy
from .probe_steps import validate_reference_substitution
from .motion_tracking_audit import TraceEngine
from .rig import Rig
from .runtime import MIMICKIT_REVISION, activate
from .track_transfer import CalibrationReference, summarize_case


def run(args):
    activate(args.mimickit)
    from util import mp_util
    from envs.base_env import EnvMode
    from .guarded_ppo import GuardedPPO
    from .native_engine import NativeDummyEngine
    from .stand_task import StandTask

    if (args.out.exists() or args.seconds <= 0 or args.envs < 8
            or args.evaluation_interval < 8 or args.evaluation_interval % 8):
        raise ValueError("Require a new output directory, positive budget and valid evaluation interval")
    source_contract = json.loads(args.source_contract.read_text())
    if source_contract.get("diagnostic_only") or mode_from_contract(source_contract) != "support_v2":
        raise ValueError("Require a trained support_v2 checkpoint and its contract")
    if args.initialize_from.name == "best.pt":
        best = json.loads((args.initialize_from.parent / "best.json").read_text())
        if best["checkpoint_sha256"] != sha256(args.initialize_from):
            raise ValueError("Starting checkpoint does not match best.json")
        validate_contract(source_contract, best["contract"])
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mp_util.init(0, 1, "cuda:0", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    reference = CalibrationReference(args.reference, rig)
    if not reference.placed_step:
        raise ValueError("Require a passed placed-step diagnostic reference")

    def task(worlds, device, engine_factory):
        return StandTask(rig, args.reference, worlds, device, seed=args.seed,
                         engine_factory=engine_factory, control_mode="target_pd",
                         contact_mode="support_v2", reference_factory=CalibrationReference)

    class StagedStepTask(StandTask):
        def reset(self, env_ids=None):
            if self._mode != EnvMode.TRAIN:
                return super().reset(env_ids)
            ids = self.ids if env_ids is None else env_ids
            if len(ids):
                self.steps[ids] = 0
                draw = torch.randint(0, 4, (len(ids),), generator=self.rng, device=self.device)
                self.offset[ids] = torch.where(draw < 2, 0., torch.where(draw == 2, 3., 7.))
                q, v = self.sample_reference(self.offset[ids], ids)
                self.engine.reset_envs(ids, q, v)
            return self.observations(), {}

    from .newton_adapter import DummyNewtonEngine
    if args.training_window == "staged8":
        train_task = StagedStepTask(rig, args.reference, args.envs, "cuda:0", seed=args.seed,
                                    engine_factory=DummyNewtonEngine, control_mode="target_pd",
                                    contact_mode="support_v2", reference_factory=CalibrationReference)
        train_task.episode_seconds = 8.
    else:
        train_task = task(args.envs, "cuda:0", DummyNewtonEngine)
    native = task(8, "cpu", NativeDummyEngine)
    native_full = task(1, "cpu", TraceEngine)
    native_full.episode_seconds = reference.duration
    quiet = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz", 8, "cpu",
                      engine_factory=NativeDummyEngine, control_mode="target_pd",
                      contact_mode="support_v2")
    target_contract = train_task.observation_contract()
    validate_reference_substitution(source_contract, target_contract)
    config = yaml.safe_load((args.mimickit / "data/agents/deepmimic_humanoid_ppo_agent.yaml").read_text())
    config.update(iters_per_output=args.evaluation_interval, test_episodes=0,
                  actor_max_kl=args.actor_max_kl,
                  observation_normalization="frozen_loaded_statistics", actor_gradient_clip=1.0)
    config["critic_optimizer"] = {**config["critic_optimizer"], "grad_clip": 1.0}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "agent.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    protocol = dict(schema="mimic_placed_step_pilot_v1", model_sha256=sha256(rig.path),
                    reference_sha256=reference.sha256,
                    reference_report_sha256=reference.report_sha256,
                    starting_checkpoint_sha256=sha256(args.initialize_from),
                    source_contract_sha256=sha256(args.source_contract),
                    target_contract=target_contract, seed=args.seed, envs=args.envs,
                    seconds=args.seconds, evaluation_interval=args.evaluation_interval,
                    training_window=args.training_window,
                    mimickit_revision=MIMICKIT_REVISION, agent_config=config,
                    scope="Experimental motion training only; no policy export or Godot selection.")
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    curve = []

    class BudgetComplete(Exception):
        pass

    class PilotPPO(GuardedPPO):
        def test_model(self, num_episodes):
            if num_episodes == 0:
                return {"mean_return": 0., "mean_ep_len": 0., "num_eps": 0}
            return super().test_model(num_episodes)

        def _build_train_data(self):
            info = super()._build_train_data()
            if self._iter == 0:
                fields = {}
                for name in ("obs", "next_obs", "reward", "tar_val", "adv"):
                    values = self._exp_buffer.get_data(name)
                    fields[name] = dict(finite=bool(torch.isfinite(values).all()),
                                        max_abs=float(torch.nan_to_num(values).abs().max()))
                normalized = self._obs_norm.normalize(self._exp_buffer.get_data("obs"))
                fields["normalized_obs"] = dict(finite=bool(torch.isfinite(normalized).all()),
                                                max_abs=float(torch.nan_to_num(normalized).abs().max()))
                (args.out / "first_rollout_health.json").write_text(
                    json.dumps(fields, indent=2), encoding="utf-8")
            return info

        def _train_iter(self):
            if time.monotonic() >= self.deadline:
                raise BudgetComplete
            result = super()._train_iter()
            invalid = {key: str(float(value)) for key, value in result.items()
                       if not np.isfinite(float(value))}
            invalid_parameters = [name for name, parameter in self.named_parameters()
                                  if not torch.isfinite(parameter).all()]
            if invalid or invalid_parameters:
                (args.out / "numerical_failure.json").write_text(json.dumps(dict(
                    iteration=self._iter + 1, invalid_metrics=invalid,
                    invalid_parameters=invalid_parameters), indent=2), encoding="utf-8")
                raise FloatingPointError("Nonfinite PPO metric")
            completed = self._iter + 1
            if completed % args.evaluation_interval == 0:
                self.save(str(args.out / f"evaluation_{completed:06d}.pt"))
                measurement = review(StandPolicy(self))
                curve.append(dict(iteration=completed, samples=self._update_sample_count(),
                                  measurement=measurement))
                (args.out / "learning_curve.json").write_text(
                    json.dumps(curve, indent=2), encoding="utf-8")
            return result

    agent = PilotPPO(config, train_task, "cuda:0")
    agent.load(str(args.initialize_from))
    if agent._obs_norm.get_count().item() <= 0:
        raise ValueError("Starting checkpoint has no trained normalization")

    def review(policy):
        actor = copy.deepcopy(policy).eval().cpu()
        standing, _ = evaluate(quiet, actor)
        phases, _ = evaluate(native, actor)
        native_full.engine.records = [[]]
        complete, _ = evaluate(native_full, actor, episodes=1)
        duration = float(complete["survival_seconds"][0])
        case, _ = summarize_case(native_full, native_full.engine.records[0], duration, 0.)
        placement = np.sign(case["left_foot_displacement_x_m"]) == np.sign(reference.requested_step_m) \
            and abs(case["left_foot_displacement_x_m"]) >= .1
        settled = (case["final_two_second_min_pelvis_height_m"] >= .75
                   and case["final_two_second_max_horizontal_speed_m_s"] <= .2)
        return dict(quiet=standing, quiet_passed=standing_passed(standing),
                    phase_trials=phases, complete_trial=complete, complete_case=case,
                    step_passed=bool(complete["successes"] == 1 and placement and settled))

    initial = review(StandPolicy(agent))
    agent.save(str(args.out / "initial.pt"))
    started = time.monotonic()
    agent.deadline = started + args.seconds
    try:
        agent.train_model(args.envs * config["steps_per_iter"] * 100000,
                          str(args.out), False, "txt")
    except BudgetComplete:
        pass
    elapsed = time.monotonic() - started
    agent.save(str(args.out / "final.pt"))
    final = review(StandPolicy(agent))
    report = dict(schema="mimic_placed_step_pilot_v1", training_seconds=elapsed,
                  iterations=agent._iter, samples=agent._sample_count,
                  initial=initial, final=final, learning_curve=curve,
                  improved=bool(final["quiet_passed"] and final["step_passed"]
                                and not initial["step_passed"]),
                  scope="Single training seed; full native step and quiet standing measured. No export or viewer selection.")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(iterations=report["iterations"], samples=report["samples"],
                          initial_survival=initial["complete_trial"]["mean_survival_seconds"],
                          final_survival=final["complete_trial"]["mean_survival_seconds"],
                          quiet_passed=final["quiet_passed"], step_passed=final["step_passed"])),
          flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--seconds", type=float, default=300.)
    parser.add_argument("--seed", type=int, default=240924)
    parser.add_argument("--evaluation-interval", type=int, default=64)
    parser.add_argument("--actor-max-kl", type=float, default=.02)
    parser.add_argument("--training-window", choices=("uniform3", "staged8"), default="uniform3")
    run(parser.parse_args())
