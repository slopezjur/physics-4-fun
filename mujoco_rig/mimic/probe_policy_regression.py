"""Separate actor-weight and observation-normalizer changes in a saved regression."""
import argparse
import copy
import json
from pathlib import Path

import torch
import yaml

from .runtime import activate
from .rig import Rig
from .policy import StandPolicy
from .evaluate_stand import evaluate
from .baseline import sha256


def probe(args):
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .stand_task import StandTask
    from .native_engine import NativeDummyEngine
    mp_util.init(0, 1, "cpu", None)
    task = StandTask(Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml"),
                     args.reference, 8, "cpu", engine_factory=NativeDummyEngine, control_mode="target_pd")
    agent = PPOAgent(yaml.safe_load(args.config.read_text()), task, "cpu")
    policies = {}
    for name, checkpoint in (("before", args.before), ("after", args.after)):
        agent.load(str(checkpoint))
        policies[name] = StandPolicy(agent).eval()
    report = {"checkpoints": {name: {"path": str(path), "sha256": sha256(path)} for name, path in
                              (("before", args.before), ("after", args.after))}, "cases": {}}
    for weights, normalizer in (("before", "before"), ("after", "after"), ("before", "after"), ("after", "before")):
        actor = copy.deepcopy(policies[weights])
        actor.obs_norm = copy.deepcopy(policies[normalizer].obs_norm)
        metrics, _ = evaluate(task, actor)
        report["cases"][f"weights_{weights}_normalizer_{normalizer}"] = metrics
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mimickit", "before", "after", "config", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path(__file__).parent / "assets/stand_reference.npz")
    probe(parser.parse_args())
