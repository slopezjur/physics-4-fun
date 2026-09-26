"""Isolated MimicKit AMP pilot for a licensed single-step motion skill."""
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
from .rig import Rig
from .runtime import MIMICKIT_REVISION, activate
from .track_transfer import CalibrationReference, summarize_case


def load_stand_into_amp(agent, checkpoint):
    """Transfer the complete actor/critic contract; leave only AMP modules new."""
    source = torch.load(checkpoint, map_location=agent._device, weights_only=True)
    if any(key.startswith("_model._disc") or key.startswith("_disc_obs_norm") for key in source):
        raise ValueError("Expected a Stand PPO checkpoint, not an AMP checkpoint")
    missing, unexpected = agent.load_state_dict(source, strict=False)
    if unexpected or any(not key.startswith(("_model._disc", "_disc_obs_norm")) for key in missing):
        raise ValueError(f"Incompatible Stand checkpoint: missing={missing}, unexpected={unexpected}")
    torch.nn.init.uniform_(agent._model._disc_logits.weight, -.01, .01)
    torch.nn.init.zeros_(agent._model._disc_logits.bias)
    agent._sync_optimizer()
    agent._disc_obs_norm.record(agent._env.fetch_disc_obs_demo(8192))
    agent._disc_obs_norm.update()
    if agent._obs_norm.get_count().item() <= 0 or agent._disc_obs_norm.get_count().item() <= 0:
        raise ValueError("AMP requires trained actor and initialized motion normalization")


def run(args):
    activate(args.mimickit)
    from util import mp_util
    from learning.amp_agent import AMPAgent
    from .amp_step import AMPStepTask
    from .guarded_ppo import GuardedPPO
    from .motion_tracking_audit import TraceEngine
    from .native_engine import NativeDummyEngine
    from .newton_adapter import DummyNewtonEngine
    from .stand_task import StandTask

    if args.out.exists() or args.seconds <= 0 or args.envs < 8 or args.evaluation_interval < 1:
        raise ValueError("Require a new output directory and positive training parameters")
    source_contract = json.loads(args.source_contract.read_text())
    if source_contract.get("diagnostic_only") or mode_from_contract(source_contract) != "support_v2":
        raise ValueError("Require a trained support_v2 Stand checkpoint")
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
        raise ValueError("The task reference must be a validated placed step")
    train_task = AMPStepTask(rig, args.reference, args.demonstration, args.envs, "cuda:0",
                             args.seed, DummyNewtonEngine)
    validate_reference_substitution(source_contract, train_task.observation_contract())
    native_full = StandTask(rig, args.reference, 1, "cpu", engine_factory=TraceEngine,
                            control_mode="target_pd", contact_mode="support_v2",
                            reference_factory=CalibrationReference)
    native_full.episode_seconds = reference.duration
    quiet = StandTask(rig, Path(__file__).parent / "assets/stand_reference.npz", 8, "cpu",
                      engine_factory=NativeDummyEngine, control_mode="target_pd", contact_mode="support_v2")

    config = yaml.safe_load((args.mimickit / "data/agents/amp_task_humanoid_agent.yaml").read_text())
    config.update(iters_per_output=args.evaluation_interval, test_episodes=0,
                  actor_max_kl=args.actor_max_kl, normalizer_samples=0,
                  disc_epochs=1, disc_batch_size=8)
    config["critic_optimizer"] = {**config["critic_optimizer"], "grad_clip": 1.0}
    config["disc_optimizer"] = {**config["disc_optimizer"], "grad_clip": 1.0}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "agent.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    protocol = dict(schema="mimic_amp_step_pilot_v1", model_sha256=sha256(rig.path),
                    task_reference_sha256=reference.sha256,
                    task_reference_report_sha256=reference.report_sha256,
                    starting_checkpoint_sha256=sha256(args.initialize_from),
                    source_contract_sha256=sha256(args.source_contract),
                    actor_contract=train_task.observation_contract(),
                    discriminator_contract=train_task.discriminator_contract(),
                    task_reward="Upright survival, reference COM transfer, left foot placement, right foot support, post-landing settling; AMP supplies whole-body motion prior",
                    train_episode_seconds=train_task.episode_seconds,
                    seed=args.seed, envs=args.envs, seconds=args.seconds,
                    evaluation_interval=args.evaluation_interval,
                    mimickit_revision=MIMICKIT_REVISION, agent_config=config,
                    scope="Experimental AMP step training; no export or Godot selection")
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    curve = []

    def review(policy):
        actor = copy.deepcopy(policy).eval().cpu()
        standing, _ = evaluate(quiet, actor)
        native_full.engine.records = [[]]
        complete, _ = evaluate(native_full, actor, episodes=1)
        duration = float(complete["survival_seconds"][0])
        case, _ = summarize_case(native_full, native_full.engine.records[0], duration, 0.)
        placement = np.sign(case["left_foot_displacement_x_m"]) == np.sign(reference.requested_step_m) \
            and abs(case["left_foot_displacement_x_m"]) >= .1
        settled = (case["final_two_second_min_pelvis_height_m"] >= .75
                   and case["final_two_second_max_horizontal_speed_m_s"] <= .2)
        return dict(quiet=standing, quiet_passed=standing_passed(standing),
                    complete_trial=complete, complete_case=case,
                    step_passed=bool(complete["successes"] == 1 and placement and settled))

    class BudgetComplete(Exception):
        pass

    class PilotAMP(GuardedPPO, AMPAgent):
        def test_model(self, num_episodes):
            if num_episodes == 0:
                return {"mean_return": 0., "mean_ep_len": 0., "num_eps": 0}
            return super().test_model(num_episodes)

        def _train_iter(self):
            if time.monotonic() >= self.deadline:
                raise BudgetComplete
            result = super()._train_iter()
            invalid = {key: float(value) for key, value in result.items()
                       if not np.isfinite(float(value))}
            invalid_parameters = [name for name, parameter in self.named_parameters()
                                  if not torch.isfinite(parameter).all()]
            if invalid or invalid_parameters:
                (args.out / "numerical_failure.json").write_text(json.dumps(dict(
                    iteration=self._iter + 1, invalid_metrics=invalid,
                    invalid_parameters=invalid_parameters), indent=2), encoding="utf-8")
                raise FloatingPointError("Nonfinite AMP metric or parameter")
            completed = self._iter + 1
            if completed % args.evaluation_interval == 0:
                measurement = review(StandPolicy(self))
                curve.append(dict(iteration=completed, samples=self._update_sample_count(),
                                  discriminator_reward=float(result["disc_reward_mean"]),
                                  measurement=measurement))
                (args.out / "learning_curve.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")
            return result

    agent = PilotAMP(config, train_task, "cuda:0")
    load_stand_into_amp(agent, args.initialize_from)
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
    report = dict(schema="mimic_amp_step_pilot_v1", training_seconds=elapsed,
                  iterations=agent._iter, samples=agent._sample_count,
                  initial=initial, final=final, learning_curve=curve,
                  improved=bool(final["quiet_passed"] and final["step_passed"]
                                and not initial["step_passed"]),
                  scope="Single-seed native step/quiet gate; no export or viewer selection")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(iterations=report["iterations"], samples=report["samples"],
                          initial_survival=initial["complete_trial"]["mean_survival_seconds"],
                          final_survival=final["complete_trial"]["mean_survival_seconds"],
                          quiet_passed=final["quiet_passed"], step_passed=final["step_passed"])), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--demonstration", type=Path, action="append", required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--seconds", type=float, default=300.)
    parser.add_argument("--seed", type=int, default=240924)
    parser.add_argument("--evaluation-interval", type=int, default=16)
    parser.add_argument("--actor-max-kl", type=float, default=.02)
    run(parser.parse_args())
