"""Paired native curriculum audit varying projectile mass/radius and actual body contacts."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from .baseline import sha256
from .runtime import activate


def run(args):
    if not math.isfinite(args.seconds) or not 12 <= args.seconds <= 60:
        raise ValueError("Require a 12..60 second diagnostic horizon")
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .balance_observation import task_classes
    from .ball_protocol import TARGET_BODIES
    from .checkpoints import validate_contract
    from .policy import StandPolicy
    from .reactive_ball_eval import evaluate
    from .reactive_ball_task import write_ball_variant
    from .rig import Rig

    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    contract = json.loads((args.checkpoint.parent / "contract.json").read_text())
    config = yaml.safe_load((args.checkpoint.parent / "agent.yaml").read_text())
    _, BallTask = task_classes(contract)
    args.out.mkdir(parents=True, exist_ok=False)
    settings = [(5., .09), (6., .09), (8., .09), (5., .06), (5., .12)]
    protocol = dict(schema="paired_native_projectile_curriculum_audit_v1", settings=settings,
        checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
        reference_sha256=sha256(args.reference), source_model_sha256=sha256(args.ball_model),
        seed=args.seed, seconds=args.seconds, speed_m_s=2.,
        purpose="Measure failure distribution before curriculum changes; not held-out admission.",
        source_hashes={name: sha256(Path(__file__).with_name(name)) for name in
            ("audit_ball_curriculum.py", "reactive_ball_task.py", "reactive_ball_eval.py", "gated_recovery_net.py")})
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (args.out / "source").mkdir()
    for name in protocol["source_hashes"]:
        (args.out / "source" / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    summaries = []
    for mass, radius in settings:
        label = f"mass-{mass:g}-radius-{radius:g}"
        model = write_ball_variant(args.ball_model, args.out / f"{label}.xml", mass, radius_m=radius)
        task = BallTask(rig, args.reference, 96, "cpu", ball_path=model,
                        speed_min=2., speed_max=2., episode_seconds=min(30., args.seconds), recovery_mode="step_v5")
        validate_contract(contract, task.observation_contract())
        agent = PPOAgent(config, task, "cpu")
        agent.load(str(args.checkpoint))
        rng = np.random.default_rng(args.seed)
        cases = []
        for body_index, body in enumerate(TARGET_BODIES):
            for posture in range(8):
                direction = (posture // 2 + body_index) % 4
                cases.append(dict(name=f"{body}-posture-{posture}-direction-{direction}", posture=posture,
                    direction=direction, angle_radians=direction*math.pi/2+float(rng.uniform(-math.pi/6, math.pi/6)),
                    target_body=body, speed=2., phase=0., launch_step=int(rng.integers(75, 106)),
                    trial_steps=math.ceil(args.seconds/task.dt)))
        actual_bodies = [set() for _ in cases]
        normal_impulse = np.zeros(len(cases))

        def record(task, step, active, done):
            for i in torch.nonzero(active).flatten().tolist():
                actual_bodies[i].update(task.engine.hit_bodies[i])
                normal_impulse[i] = max(normal_impulse[i], float(task.engine.hit_normal_impulse[i]))

        metrics, _ = evaluate(task, StandPolicy(agent).eval(), cases, record)
        for i, case in enumerate(metrics["cases"]):
            case["contacted_bodies"] = [rig.model.body(body).name for body in sorted(actual_bodies[i])]
            case["intended_target_contacted"] = case["target_body"] in case["contacted_bodies"]
            case["normal_impulse_n_s"] = float(normal_impulse[i])
        metrics.update(mass_kg=mass, radius_m=radius, model_sha256=sha256(model),
                       checkpoint_sha256=protocol["checkpoint_sha256"],
                       intended_target_contacts=sum(c["intended_target_contacted"] for c in metrics["cases"]))
        (args.out / f"{label}.json").write_text(json.dumps(metrics, indent=2))
        summary = {key: metrics[key] for key in ("mass_kg", "radius_m", "confirmed_hits", "survived_hits",
                    "recovered_hits", "recovered_with_step", "intended_target_contacts")}
        summaries.append(summary)
        (args.out / "progress.json").write_text(json.dumps(summaries, indent=2))
        print(json.dumps(summary), flush=True)
    (args.out / "report.json").write_text(json.dumps(dict(protocol=protocol, results=summaries,
        godot_selected=False, scope="Paired diagnostic suite, not held-out promotion evidence"), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mimickit", "checkpoint", "reference", "ball-model", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=280929)
    parser.add_argument("--seconds", type=float, default=12.,
                        help="Uninterrupted diagnostic duration; use 33 seconds to expose late settling/falls")
    run(parser.parse_args())
