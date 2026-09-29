"""Fit a state-only recovery branch from native bilateral discoveries with retention rehearsal."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .baseline import sha256
from .runtime import activate


def validate_demonstration_provenance(report, checkpoint, reference, ball_model, rig_path, trace):
    for key, path in (("checkpoint_sha256", checkpoint), ("reference_sha256", reference),
                      ("ball_sha256", ball_model), ("model_sha256", rig_path)):
        if report[key] != sha256(path):
            raise ValueError("Demonstration provenance mismatch: " + key)
    if "trace_sha256" in report and report["trace_sha256"] != sha256(trace):
        raise ValueError("Demonstration trace changed after evaluation")


def run(args):
    if args.rehearsal_cases and (args.rehearsal_cases < 32 or args.rehearsal_cases % 8):
        raise ValueError("Random rehearsal requires at least 32 cases in complete posture replicas")
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .balance_observation import task_classes
    from .checkpoints import validate_contract, ball_regressions
    from .continuous_stand import evaluate_retention
    from .gated_recovery_net import CAPTURE_NAME, SUPPORT_NAME, support_recovery_gate
    from .native_engine import NativeDummyEngine
    from .policy import StandPolicy
    from .reactive_ball_eval import evaluate, selection_cases, unseen_cases
    from .rig import Rig

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    report = json.loads(args.trace.with_name("report.json").read_text())
    demonstration_checkpoint = args.teacher_checkpoint or args.checkpoint
    demonstration_model = args.demonstration_ball_model or args.ball_model
    validate_demonstration_provenance(report, demonstration_checkpoint, args.reference,
                                     demonstration_model, rig.path, args.trace)
    selected = [i for i, case in enumerate(report["cases"]) if case["recovered_with_step"]]
    if not selected or not all(any(report["cases"][i]["steps_by_foot"][foot] for i in selected)
                               for foot in (0, 1)):
        raise ValueError("Require native successful demonstrations covering both feet")
    contract = json.loads((args.checkpoint.parent / "contract.json").read_text())
    teacher_contract = json.loads((demonstration_checkpoint.parent / "contract.json").read_text())
    validate_contract(contract, teacher_contract)
    config = yaml.safe_load((args.checkpoint.parent / "agent.yaml").read_text())
    if config["model"]["actor_net"] != CAPTURE_NAME:
        raise ValueError("Require a frozen capture recovery source")
    QuietTask, BallTask = task_classes(contract)
    quiet = QuietTask(rig, args.reference, engine_factory=NativeDummyEngine)
    task = BallTask(rig, args.reference, 32, "cpu", ball_path=args.ball_model,
                    speed_max=2., episode_seconds=12., recovery_mode="step_v5")
    validate_contract(contract, task.observation_contract())
    demonstration_task = None
    demonstration_cases = None
    if args.teacher_checkpoint or args.demonstration_ball_model:
        case_keys = ("name", "posture", "direction", "angle_radians", "target_body", "speed",
                     "phase", "launch_step", "trial_steps")
        demonstration_cases = [{key: case[key] for key in case_keys} for case in report["cases"]]
        if any("second_launch_step" in case for case in report["cases"]):
            raise ValueError("Cross-source fitting currently requires single-hit demonstrations")
        demonstration_task = BallTask(rig, args.reference, len(demonstration_cases), "cpu",
            ball_path=demonstration_model, speed_max=2., episode_seconds=12., recovery_mode="step_v5")
        validate_contract(contract, demonstration_task.observation_contract())
    source = PPOAgent(config, task, "cpu")
    source.load(str(args.checkpoint))
    teacher = StandPolicy(source).eval()
    config = copy.deepcopy(config)
    config["model"]["actor_net"] = SUPPORT_NAME
    agent = PPOAgent(config, task, "cpu")
    agent.load_state_dict(source.state_dict())
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "agent.yaml").write_text(yaml.safe_dump(config))
    (args.out / "contract.json").write_text(json.dumps(contract, indent=2))
    protocol = dict(schema="state_only_bilateral_demonstration_fit_v1", seed=args.seed, device=args.device,
        source_checkpoint=str(args.checkpoint), source_checkpoint_sha256=sha256(args.checkpoint),
        demonstration_checkpoint=str(demonstration_checkpoint),
        demonstration_checkpoint_sha256=sha256(demonstration_checkpoint),
        demonstration_ball_model=str(demonstration_model),
        demonstration_ball_model_sha256=sha256(demonstration_model),
        demonstration_cases=demonstration_cases,
        random_rehearsal=dict(cases=args.rehearsal_cases, seed=args.rehearsal_seed,
            seconds=12., speed_m_s=2., purpose="Training rehearsal, never held-out validation",
            fixed_random_loss_mix=[.5, .5] if args.rehearsal_cases else [1., 0.]),
        trace_sha256=sha256(args.trace), trace_report_sha256=sha256(args.trace.with_name("report.json")),
        selected_demonstrations=selected, contract=contract, experiment=task.experiment_contract(),
        architecture=SUPPORT_NAME, correction_limit=2., physical_target_range_radians=.25,
        optimizer=dict(name="Adam", learning_rate=3e-4, gradient_clip=1., batch_size_per_source=128,
                       quiet_weight=args.quiet_weight, rehearsal_weight=args.rehearsal_weight,
                       fit_unclipped_mean=args.fit_unclipped_mean,
                       evaluated_fit_updates=[0, 256, 1024, 4096]),
        hypothesis="Fit both observed step recoveries using current state; expanded internal residual and support gate.",
        scope="Supervised initialization, not PPO or policy admission; no time or selected-foot input.",
        source_hashes={name: sha256(Path(__file__).with_name(name)) for name in
                       ("fit_recovery_demonstrations.py", "gated_recovery_net.py", "reactive_ball_eval.py")})
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (args.out / "source").mkdir()
    for name in protocol["source_hashes"]:
        (args.out / "source" / name).write_bytes(Path(__file__).with_name(name).read_bytes())

    quiet_metrics, quiet_trace = evaluate_retention(quiet, teacher)
    if not quiet_metrics["passed"]:
        raise ValueError("Source failed native quiet gate")
    baseline_observations = []

    def record(obs):
        baseline_observations.append(obs.clone())
        return teacher(obs)

    cases = selection_cases(2., 12., task.dt)
    baseline, _ = evaluate(task, record, cases)
    demonstration_baseline = (evaluate(demonstration_task, teacher, demonstration_cases)[0]
                              if demonstration_task is not None else None)
    random_task, random_cases, random_baseline, random_obs = None, None, None, None
    if args.rehearsal_cases:
        random_task = BallTask(rig, args.reference, args.rehearsal_cases, "cpu",
            ball_path=args.ball_model, speed_min=2., speed_max=2., episode_seconds=12., recovery_mode="step_v5")
        random_cases = unseen_cases(args.rehearsal_seed, args.rehearsal_cases, 2., 2., 12., task.dt)
        random_observations = []

        def record_random(obs):
            random_observations.append(obs.clone())
            return teacher(obs)

        random_baseline, _ = evaluate(random_task, record_random, random_cases)
        successful = torch.tensor([case["recovered"] for case in random_baseline["cases"]])
        random_obs = torch.stack(random_observations)[:, successful].flatten(0, 1)
        random_obs = random_obs[support_recovery_gate(random_obs) > 0]
        if not len(random_obs) or not torch.isfinite(random_obs).all():
            raise ValueError("Random rehearsal has no finite branch-active successful states")
        np.savez_compressed(args.out / "random_rehearsal.npz", observations=random_obs.numpy())
    (args.out / "baseline.json").write_text(json.dumps(dict(quiet=quiet_metrics, hits=baseline,
        demonstration_hits=demonstration_baseline, random_rehearsal=random_baseline), indent=2))
    quiet_obs = torch.from_numpy(quiet_trace["observations"][quiet_trace["active"]].copy())
    quiet_gate = support_recovery_gate(quiet_obs)
    # Rehearse all branch-active standing states and a uniform sample of settled states.
    q_indices = torch.unique(torch.cat((torch.nonzero(quiet_gate > 0).flatten(),
        torch.linspace(0, len(quiet_obs)-1, min(2048, len(quiet_obs))).long())))
    quiet_obs = quiet_obs[q_indices]
    baseline_obs = torch.stack(baseline_observations).flatten(0, 1)
    baseline_obs = baseline_obs[support_recovery_gate(baseline_obs) > 0]
    with np.load(args.trace) as data:
        demo_obs = torch.from_numpy(data["observation"][:, selected].copy()).flatten(0, 1)
        demo_action = torch.from_numpy(data["action"][:, selected].copy()).flatten(0, 1).clamp(-1, 1)
        active = torch.from_numpy(data["active"][:, selected].copy()).flatten()
    active &= support_recovery_gate(demo_obs) > 0
    demo_obs, demo_action = demo_obs[active], demo_action[active]
    if not len(demo_obs) or not torch.isfinite(demo_obs).all() or not torch.isfinite(demo_action).all():
        raise ValueError("Require finite, active demonstration observations and actions")
    with torch.no_grad():
        quiet_action, baseline_action = teacher(quiet_obs), teacher(baseline_obs)
    fit = StandPolicy(agent).to(args.device).eval()
    network = fit.actor
    frozen = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()
              if not key.startswith(("velocity.", "recovery."))}

    def dataset(obs, action):
        with torch.no_grad():
            obs, action = obs.to(args.device), action.to(args.device)
            normalized = fit.obs_norm.normalize(obs)
            raw = normalized * network.observation_std + network.observation_mean
            features = torch.cat((raw[:, 9:11] / .5, raw[:, 12:14] / 2.), -1).clamp(-2, 2)
            base = fit.action_norm.unnormalize(network.base_mean(network.base(normalized[:, :362])))
            gate = support_recovery_gate(raw)[:, None] * network.leg_mask
            return normalized, features, base, gate, action

    demo, quiet_data, rehearsal = (dataset(demo_obs, demo_action), dataset(quiet_obs, quiet_action),
                                  dataset(baseline_obs, baseline_action))
    with torch.no_grad():
        random_data = dataset(random_obs, teacher(random_obs)) if random_obs is not None else None
    bounds = []
    for limit in (.8, 2.):
        _, _, base, gate, target = demo
        nearest = (base + (target-base).clamp(-limit*gate, limit*gate)).clamp(-1, 1)
        bounds.append(dict(correction_limit=limit, minimum_action_rmse=float((nearest-target).square().mean().sqrt())))
    (args.out / "dataset.json").write_text(json.dumps(dict(demonstration_frames=len(demo_obs),
        quiet_frames=len(quiet_obs), baseline_active_frames=len(baseline_obs), representability_bounds=bounds,
        random_rehearsal_frames=len(random_obs) if random_obs is not None else 0,
        random_rehearsal_sha256=sha256(args.out / "random_rehearsal.npz") if random_data is not None else None), indent=2))

    def loss(data, count=128):
        indices = torch.randint(len(data[0]), (min(count, len(data[0])),), device=args.device)
        normalized, features, base, gate, target = [item[indices] for item in data]
        correction = (network.velocity(features) + network.recovery(normalized)).clamp(-2, 2)
        prediction = base + correction*gate
        if not args.fit_unclipped_mean:
            prediction = prediction.clamp(-1, 1)
        return (prediction-target).square().mean()

    optimizer = torch.optim.Adam([p for p in network.parameters() if p.requires_grad], lr=3e-4)
    history = []
    previous = 0
    training_seconds = 0.
    for iteration in (0, 256, 1024, 4096):
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(previous, iteration):
            optimizer.zero_grad()
            demo_loss, quiet_loss, rehearsal_loss = loss(demo), loss(quiet_data), loss(rehearsal)
            if random_data is not None:
                rehearsal_loss = .5 * (rehearsal_loss + loss(random_data))
            total = demo_loss + args.quiet_weight*quiet_loss + args.rehearsal_weight*rehearsal_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.)
            optimizer.step()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        training_seconds += time.perf_counter()-started
        previous = iteration
        for key, value in frozen.items():
            torch.testing.assert_close(network.state_dict()[key].cpu(), value, atol=0, rtol=0)
        agent._model._actor_layers.load_state_dict({key: value.cpu() for key, value in network.state_dict().items()})
        path = args.out / f"candidate_{iteration:06d}.pt"
        agent.save(str(path))
        actor = StandPolicy(agent).eval()
        retention, _ = evaluate_retention(quiet, actor)
        metrics, _ = evaluate(task, actor, cases)
        demonstration_metrics = (evaluate(demonstration_task, actor, demonstration_cases)[0]
                                 if demonstration_task is not None else None)
        random_metrics = evaluate(random_task, actor, random_cases)[0] if random_task is not None else None
        random_regressions = ball_regressions(random_metrics, random_baseline) if random_task is not None else []
        with torch.no_grad():
            errors = dict(demonstration_rmse=float((fit(demo_obs.to(args.device))-demo_action.to(args.device)).square().mean().sqrt()),
                          quiet_rmse=float((fit(quiet_obs.to(args.device))-quiet_action.to(args.device)).square().mean().sqrt()))
            raw_mean = fit.mean(fit.actor(fit.obs_norm.normalize(demo_obs.to(args.device))))
            errors["demonstration_mean_bound_loss"] = float((raw_mean.abs()-1).clamp_min(0).square().sum(-1).mean())
        regressions = ball_regressions(metrics, baseline)
        row = dict(iteration=iteration, checkpoint_sha256=sha256(path), errors=errors, quiet=retention,
                   metrics=metrics, regressions=regressions, training_seconds=training_seconds,
                   demonstration_metrics=demonstration_metrics,
                   random_rehearsal_metrics=random_metrics, random_rehearsal_regressions=random_regressions,
                   eligible_for_further_validation=retention["passed"] and not regressions and not random_regressions
                       and (demonstration_metrics is None or demonstration_metrics["recovered_hits"]
                            > demonstration_baseline["recovered_hits"]),
                   frozen_base_identical=True, godot_selected=False)
        history.append(row)
        (args.out / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(dict(iteration=iteration, quiet=retention["passed"], recovered=metrics["recovered_hits"],
                             recovered_steps=metrics["recovered_with_step"], errors=errors,
                             demonstration_recovered=demonstration_metrics["recovered_hits"] if demonstration_metrics else None,
                             demonstration_steps=demonstration_metrics["recovered_with_step"] if demonstration_metrics else None,
                             random_recovered=random_metrics["recovered_hits"] if random_metrics else None,
                             random_regressions=len(random_regressions),
                             training_seconds=training_seconds, regressions=len(regressions))), flush=True)
    (args.out / "report.json").write_text(json.dumps(dict(protocol=protocol, baseline=baseline,
        demonstration_baseline=demonstration_baseline, random_rehearsal_baseline=random_baseline, history=history,
        training_seconds=training_seconds, godot_selected=False,
        scope="Closed-loop fixed tests only; long quiet and fresh unseen tests required before admission."), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mimickit", "checkpoint", "reference", "ball-model", "trace", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-checkpoint", type=Path,
                        help="Explicit policy that produced the trace, if different from the student source")
    parser.add_argument("--demonstration-ball-model", type=Path,
                        help="Explicit trace projectile model; --ball-model remains the retention model")
    parser.add_argument("--seed", type=int, default=280928)
    parser.add_argument("--quiet-weight", type=float, default=20.)
    parser.add_argument("--rehearsal-weight", type=float, default=1.)
    parser.add_argument("--rehearsal-cases", type=int, default=0,
                        help="Add randomized successful source trajectories to training rehearsal")
    parser.add_argument("--rehearsal-seed", type=int, default=281010)
    parser.add_argument("--fit-unclipped-mean", action="store_true")
    run(parser.parse_args())
