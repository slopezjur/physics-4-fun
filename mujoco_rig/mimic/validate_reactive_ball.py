"""Independent native validation of a selected actor or explicitly hash-bound candidate."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from .baseline import sha256
from .ball_protocol import TARGET_BODIES as ALL_TARGET_BODIES
from .checkpoints import validate_contract
from .policy import StandPolicy
from .reactive_ball_eval import TARGET_BODIES, evaluate, unseen_cases
from .rig import Rig
from .runtime import activate


def validate_checkpoint_identity(actual_hash, selected_hash, candidate_hash=None):
    if candidate_hash is not None:
        if candidate_hash != actual_hash:
            raise ValueError("Candidate checkpoint does not match the explicit SHA-256")
    elif actual_hash != selected_hash:
        raise ValueError("Validation requires the selected checkpoint")
    return actual_hash == selected_hash


def checkpoint_selection_hash(checkpoint, candidate_hash=None):
    metadata_path = checkpoint.parent / "best.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())["checkpoint_sha256"]
    if candidate_hash is None:
        raise ValueError("Validation requires selected checkpoint metadata or an explicit candidate SHA-256")
    return None


def run(args):
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .balance_observation import task_classes
    contract = json.loads((args.checkpoint.parent / "contract.json").read_text())
    _, ReactiveBallTask = task_classes(contract)

    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    checkpoint_hash = sha256(args.checkpoint)
    selected_checkpoint = validate_checkpoint_identity(checkpoint_hash,
        checkpoint_selection_hash(args.checkpoint, args.candidate_sha256), args.candidate_sha256)
    task = ReactiveBallTask(rig, args.reference, args.cases, "cpu", ball_path=args.ball_model,
                            speed_min=args.speed_min, speed_max=args.speed_max,
                            episode_seconds=min(30., args.seconds))
    validate_contract(contract, task.observation_contract())
    config = yaml.safe_load((args.checkpoint.parent / "agent.yaml").read_text())
    agent = PPOAgent(config, task, "cpu")
    agent.load(str(args.checkpoint))
    cases = unseen_cases(args.seed, args.cases, args.speed_min, args.speed_max,
                         args.seconds, task.dt, target_bodies=args.target_bodies)
    evaluator_hash = sha256(Path(__file__).with_name("reactive_ball_eval.py"))
    runner_source = Path(__file__).read_bytes()
    runner_hash = hashlib.sha256(runner_source).hexdigest()
    actual_bodies = [set() for _ in cases]
    normal_impulse = [0.] * len(cases)

    def record_contacts(task, step, active, done):
        for i in torch.nonzero(active).flatten().tolist():
            actual_bodies[i].update(task.engine.hit_bodies[i])
            normal_impulse[i] = max(normal_impulse[i], float(task.engine.hit_normal_impulse[i]))

    metrics, _ = evaluate(task, StandPolicy(agent).eval(), cases, record_contacts)
    for i, case in enumerate(metrics["cases"]):
        case["contacted_bodies"] = [rig.model.body(body).name for body in sorted(actual_bodies[i])]
        case["intended_target_contacted"] = case["target_body"] in case["contacted_bodies"]
        case["normal_impulse_n_s"] = normal_impulse[i]
    if checkpoint_hash != sha256(args.checkpoint):
        raise RuntimeError("Checkpoint changed during impact validation")
    metrics.update(schema="mimic_reactive_ball_unseen_validation_v1", seed=args.seed,
        checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
        selected_checkpoint=selected_checkpoint, explicit_candidate_sha256=args.candidate_sha256,
        character_model_sha256=sha256(rig.path), ball_model_sha256=sha256(args.ball_model),
        reference_sha256=sha256(args.reference), evaluator_sha256=evaluator_hash,
        runner_sha256=runner_hash, seconds=args.seconds, target_bodies=list(args.target_bodies),
        intended_target_contacts=sum(c["intended_target_contacted"] for c in metrics["cases"]),
        intended_target_recoveries=sum(c["intended_target_contacted"] and c["recovered"]
                                      for c in metrics["cases"]),
        passed=metrics["confirmed_hits"] == args.cases
               and metrics["recovered_hits"] == args.cases,
        scope="Frozen policy, fresh randomized angles/speeds within explicit target set; first episodes only")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "runner.py").write_bytes(runner_source)
    (args.out / "report.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in
        ("passed", "confirmed_hits", "survived_hits", "recovered_hits")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-sha256",
                        help="Evaluate an unselected candidate with this exact hash; does not select or promote it")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--ball-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=270935)
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=33.)
    parser.add_argument("--speed-min", type=float, default=1.)
    parser.add_argument("--speed-max", type=float, default=2.)
    parser.add_argument("--target-bodies", nargs="+", choices=ALL_TARGET_BODIES, default=TARGET_BODIES,
                        help="Body targets for this unseen suite; default preserves the original torso cases")
    run(parser.parse_args())
