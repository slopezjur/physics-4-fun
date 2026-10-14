"""Recheck complete posture blocks containing previously identified regressions."""
import argparse
import json
from pathlib import Path

import torch
import yaml

from .baseline import sha256
from .runtime import activate


def run(args):
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .balance_observation import task_classes
    from .policy import StandPolicy
    from .reactive_ball_eval import evaluate
    from .rig import Rig
    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    previous = json.loads(args.previous_report.read_text())
    contract = json.loads((args.checkpoint.parent / "contract.json").read_text())
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    for field, path in (("character_model_sha256", rig.path),
                        ("reference_sha256", args.reference), ("ball_model_sha256", args.ball_model)):
        if previous[field] != sha256(path):
            raise ValueError("Retention recheck requires identical plant and reference")
    _, Task = task_classes(contract)
    task = Task(rig, args.reference, 8, "cpu", ball_path=args.ball_model,
                speed_min=2., speed_max=2., episode_seconds=30., target_set="all")
    agent = PPOAgent(yaml.safe_load((args.checkpoint.parent / "agent.yaml").read_text()), task, "cpu")
    checkpoint_hash = sha256(args.checkpoint)
    agent.load(str(args.checkpoint))
    actor = StandPolicy(agent).eval()
    results = []
    for block in sorted({index // 8 for index in args.case_ids}):
        offset = block * 8
        original = previous["cases"][offset:offset + 8]
        if len(original) != 8:
            raise ValueError("Missing complete posture block")
        fields = ("name", "posture", "direction", "angle_radians", "target_body", "speed",
                  "phase", "launch_step", "trial_steps")
        specifications = [{key: case[key] for key in fields} for case in original]
        metrics, _ = evaluate(task, actor, specifications)
        for local, (before, after) in enumerate(zip(original, metrics["cases"])):
            results.append(dict(original_case=offset + local, requested=offset + local in args.case_ids,
                previous_recovered=before["recovered"], previous_survived=before["survived"],
                current_recovered=after["recovered"], current_survived=after["survived"], case=after))
    if checkpoint_hash != sha256(args.checkpoint):
        raise RuntimeError("Checkpoint changed during retention recheck")
    report = dict(schema="reactive_recovery_case_retention_v1", checkpoint=str(args.checkpoint),
        checkpoint_sha256=checkpoint_hash, previous_report=str(args.previous_report),
        previous_report_sha256=sha256(args.previous_report), requested_case_ids=args.case_ids,
        cases=results, runner_sha256=sha256(Path(__file__)),
        scope="Exact prior impact protocols, complete eight-posture blocks; no policy updates")
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([row for row in results if row["requested"]]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--previous-report", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--ball-model", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
