"""Measure action-bound pressure and native return changes on the recovery seed's own states."""
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
    from .balance_observation import task_classes
    from .rig import Rig

    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    config = yaml.safe_load((args.checkpoint.parent / "agent.yaml").read_text())
    contract = json.loads((args.checkpoint.parent / "contract.json").read_text())
    _, BallTask = task_classes(contract)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    task = BallTask(rig, args.reference, 32, "cpu", ball_path=args.ball_model,
                    speed_max=2., episode_seconds=12., recovery_mode="step_v5")
    seed = PPOAgent(config, task, "cpu")
    seed.load(str(args.checkpoint))
    candidate = PPOAgent(config, task, "cpu")
    candidate.load(str(args.candidate))
    traces, reports = [], []
    for trace, checkpoint in ((args.seed_trace, args.checkpoint), (args.candidate_trace, args.candidate)):
        report = json.loads(trace.with_name("report.json").read_text())
        for key, path in (("checkpoint_sha256", checkpoint), ("reference_sha256", args.reference),
                          ("ball_sha256", args.ball_model), ("model_sha256", rig.path)):
            if report[key] != sha256(path):
                raise ValueError("Trace provenance mismatch: " + key)
        reports.append(report)
        with np.load(trace) as data:
            traces.append({key: data[key].copy() for key in ("observation", "active", "reward")})
    mask = traces[0]["active"].copy()
    for i, case in enumerate(reports[0]["cases"]):
        hit = case["first_hit_step"]
        mask[:hit, i] = False
        mask[hit+180:, i] = False
    obs = torch.from_numpy(traces[0]["observation"][mask])
    normalized = seed._obs_norm.normalize(obs)
    legs = seed._model._actor_layers.leg_mask.bool()
    parameters = [p for p in seed._model._actor_layers.parameters() if p.requires_grad]
    candidate_parameters = [p for p in candidate._model._actor_layers.parameters() if p.requires_grad]
    distribution = seed._model.eval_actor(normalized)
    violation = (distribution.mean.abs()-1).clamp_min(0).square()
    bound = violation.sum(-1).mean()
    gradient = torch.autograd.grad(bound, parameters)
    gradient = torch.cat([value.flatten() for value in gradient])
    delta = torch.cat([(b-a).detach().flatten() for a, b in zip(parameters, candidate_parameters)])
    alignment = torch.dot(-gradient, delta)/(gradient.norm()*delta.norm()).clamp_min(1e-12)
    with torch.no_grad():
        candidate_mean = candidate._model.eval_actor(normalized).mean
        candidate_bound = (candidate_mean.abs()-1).clamp_min(0).square()
    rows = []
    for i, (before, after) in enumerate(zip(reports[0]["cases"], reports[1]["cases"])):
        if before["name"] != after["name"]:
            raise ValueError("Native case suites disagree")
        returns = []
        for report, trace in zip(reports, traces):
            hit = report["cases"][i]["first_hit_step"]
            rewards = trace["reward"][hit:, i] * trace["active"][hit:, i]
            returns.append(float(np.dot(rewards, .99**np.arange(len(rewards)))))
        rows.append(dict(name=before["name"], source_recovered=before["recovered"],
            candidate_recovered=after["recovered"], source_stepped=before["recovered_with_step"],
            candidate_stepped=after["recovered_with_step"], source_discounted_return=returns[0],
            candidate_discounted_return=returns[1]))
    result = dict(schema="recovery_regularization_audit_v1", checkpoint_sha256=sha256(args.checkpoint),
        candidate_sha256=sha256(args.candidate), seed_trace_sha256=sha256(args.seed_trace),
        candidate_trace_sha256=sha256(args.candidate_trace), source_sha256=sha256(Path(__file__)),
        samples=len(obs), bound_weight=config["action_bound_weight"],
        source_mean_bound_loss=float(bound.detach()),
        source_leg_bound_loss=float(violation[:, legs].sum(-1).mean().detach()),
        source_frozen_upper_bound_loss=float(violation[:, ~legs].sum(-1).mean().detach()),
        candidate_bound_loss_on_source_states=float(candidate_bound.sum(-1).mean()),
        candidate_leg_bound_loss_on_source_states=float(candidate_bound[:, legs].sum(-1).mean()),
        bound_gradient_norm=float(gradient.norm()), actual_actor_delta_norm=float(delta.norm()),
        cosine_actual_update_vs_negative_bound_gradient=float(alignment), cases=rows,
        scope="Same-state regularization diagnostic and measured native return changes; not a causal ablation.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({key:value for key,value in result.items() if key != "cases"}), flush=True)
    print(json.dumps([row for row in rows if row["source_stepped"] or row["source_recovered"] != row["candidate_recovered"]]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mimickit", "checkpoint", "candidate", "reference", "ball-model", "seed-trace", "candidate-trace", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
