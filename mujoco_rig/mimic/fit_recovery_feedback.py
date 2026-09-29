"""Fit discovered state-feedback reactions into the existing MimicKit actor readout."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .baseline import sha256
from .runtime import activate


def run(args):
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .checkpoints import BestCheckpoint, validate_contract
    from .continuous_stand import evaluate_retention
    from .continuous_stand_task import ContinuousStandTask
    from .native_engine import NativeDummyEngine
    from .policy import StandPolicy
    from .reactive_ball_task import ReactiveBallTask
    from .reactive_ball_eval import evaluate, selection_cases
    from .rig import Rig
    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    trace_report = json.loads(args.trace.with_name("report.json").read_text())
    for key, path in (("checkpoint_sha256", args.checkpoint), ("reference_sha256", args.reference),
                      ("ball_sha256", args.ball_model), ("model_sha256", rig.path)):
        if trace_report[key] != sha256(path):
            raise ValueError("Discovery trace provenance mismatch: " + key)
    if not trace_report.get("feedback_sha256"):
        raise ValueError("Require a measured state-feedback teacher trace")
    config = yaml.safe_load((args.checkpoint.parent / "agent.yaml").read_text())
    contract = json.loads((args.checkpoint.parent / "contract.json").read_text())
    quiet = ContinuousStandTask(rig, args.reference, engine_factory=NativeDummyEngine)
    native = ReactiveBallTask(rig, args.reference, 32, "cpu", ball_path=args.ball_model,
                              speed_max=2., episode_seconds=12., recovery_mode="step_v4")
    validate_contract(contract, native.observation_contract())
    agent = PPOAgent(config, native, "cpu")
    agent.load(str(args.checkpoint))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "agent.yaml").write_text(yaml.safe_dump(config))
    (args.out / "contract.json").write_text(json.dumps(contract, indent=2))
    protocol = dict(schema="mimic_state_feedback_initialization_v1",
                    source_checkpoint_sha256=sha256(args.checkpoint),
                    teacher_trace_sha256=sha256(args.trace), teacher_report=trace_report,
                    source_sha256=sha256(Path(__file__)),
                    experiment=native.experiment_contract(), contract=contract,
                    fit="ridge-regularized actor readout residual; quiet outputs retained by weighted rehearsal",
                    runtime="unchanged MimicKit MLP; no feedback controller, clock, or added actor inputs")
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (args.out / "runner.py").write_bytes(Path(__file__).read_bytes())
    best = BestCheckpoint(args.out, contract)
    cases = selection_cases(2., 12., native.dt)
    quiet_metrics, quiet_trace = evaluate_retention(quiet, StandPolicy(agent).eval())
    baseline, _ = evaluate(native, StandPolicy(agent).eval(), cases)
    baseline.update(quiet_standing_passed=bool(quiet_metrics["passed"]), quiet_native=quiet_metrics)
    best.consider(agent.save, baseline, iteration=0, samples=0)
    if not quiet_metrics["passed"]:
        raise ValueError("Source failed native quiet gate")
    quiet_obs = torch.from_numpy(quiet_trace["observations"][quiet_trace["active"]])
    quiet_obs = quiet_obs[torch.linspace(0, len(quiet_obs)-1, min(4096, len(quiet_obs))).long()]
    with np.load(args.trace) as trace:
        observation = torch.from_numpy(trace["observation"][trace["active"]].copy())
        target = torch.from_numpy(trace["action"][trace["active"]].copy())
    readout = agent._model._action_dist._mean_net
    weight, bias = readout.weight.detach().clone(), readout.bias.detach().clone()

    @torch.no_grad()
    def features(obs):
        hidden = agent._model._actor_layers(agent._obs_norm.normalize(obs))
        return torch.cat((hidden, torch.ones(len(hidden), 1)), -1).double()

    with torch.no_grad():
        x, q = features(observation), features(quiet_obs)
        residual = (agent._a_norm.normalize(target) -
                    agent._model.eval_actor(agent._obs_norm.normalize(observation)).mean).double()
        xx, qq = x.T @ x / len(x), q.T @ q / len(q)
        xy = x.T @ residual / len(x)
        eye = torch.eye(x.shape[1], dtype=torch.float64)
    history = []
    settings = [(100., .001), (100., .01), (1000., .001)]
    for index, (quiet_weight, ridge) in enumerate(settings, 1):
        delta = torch.linalg.solve(xx + quiet_weight * qq + ridge * eye, xy)
        with torch.no_grad():
            readout.weight.copy_(weight + delta[:-1].T.float())
            readout.bias.copy_(bias + delta[-1].float())
        fit = dict(quiet_weight=quiet_weight, ridge=ridge,
                   reaction_rmse=float((x @ delta-residual).square().mean().sqrt()),
                   quiet_output_rmse=float((q @ delta).square().mean().sqrt()))
        agent.save(str(args.out / f"candidate_{index:06d}.pt"))
        quiet_metrics, _ = evaluate_retention(quiet, StandPolicy(agent).eval())
        row = dict(iteration=index, fit=fit, quiet=quiet_metrics)
        if quiet_metrics["passed"]:
            metrics, _ = evaluate(native, StandPolicy(agent).eval(), cases)
            metrics.update(quiet_standing_passed=True, quiet_native=quiet_metrics)
            row.update(metrics=metrics, accepted=best.consider(agent.save, metrics,
                       iteration=index, samples=len(observation)))
        else:
            row["accepted"] = False
        history.append(row)
        (args.out / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(dict(iteration=index, fit=fit, quiet=quiet_metrics["passed"],
            recovered=row.get("metrics", {}).get("recovered_hits"), accepted=row["accepted"])), flush=True)
    report = dict(schema="mimic_state_feedback_initialization_report_v1", baseline=baseline,
                  selected_checkpoint=best.metadata, samples=len(observation), history=history,
                  godot_selected=False, scope="Training initialization; long quiet and unseen gates outstanding")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mimickit", "checkpoint", "reference", "ball-model", "trace", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
