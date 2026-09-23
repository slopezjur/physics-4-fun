"""Frozen randomized ball tests, independent of PPO and checkpoint selection."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .baseline import sha256
from .ball_protocol import TARGET_BODIES
from .checkpoints import validate_contract
from .contact_contract import mode_from_contract
from .evaluate_stand import evaluate
from .rig import Rig
from .runtime import activate

SCHEMA = "mimic_ball_random_test_v1"


def make_cases(seed, count, dt, reference_duration, speed_min=1., speed_max=2.5):
    if count < len(TARGET_BODIES) or count % len(TARGET_BODIES):
        raise ValueError("Use an equal positive number of cases for each target body")
    if not 1 <= speed_min <= speed_max <= 8 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid speed range or timestep")
    steps = math.ceil(5. / dt)
    phase_max = min(.9, reference_duration - steps * dt)
    if not math.isfinite(phase_max) or phase_max <= 0:
        raise ValueError("Reference must accommodate the complete five-second test")
    rng = np.random.default_rng(seed)
    cases = [dict(name=f"random-{i:03d}-{body}", target_body=body,
                  angle_radians=float(rng.uniform(0, 2 * math.pi)),
                  speed=float(rng.uniform(speed_min, speed_max)),
                  phase=float(rng.uniform(0, phase_max)),
                  launch_step=int(rng.integers(60, 121)), trial_steps=steps)
             for i, body in enumerate(list(TARGET_BODIES) * (count // len(TARGET_BODIES)))]
    rng.shuffle(cases)
    return cases


def validate_protocol(protocol):
    if protocol.get("schema") != SCHEMA:
        raise ValueError("Unsupported randomized test schema")
    dt, duration = protocol["control_timestep"], protocol["reference_duration"]
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(duration):
        raise ValueError("Invalid test timing")
    cases = protocol["cases"]
    if not cases or len({case["name"] for case in cases}) != len(cases):
        raise ValueError("Test cases must have unique names")
    for case in cases:
        if case["target_body"] not in TARGET_BODIES or not all(math.isfinite(case[k]) for k in
                ("angle_radians", "speed", "phase")) or not 1 <= case["speed"] <= 8:
            raise ValueError("Invalid ball test case")
        if type(case["launch_step"]) is not int or not 60 <= case["launch_step"] <= 120 \
                or case["trial_steps"] != math.ceil(5. / dt) or case["phase"] < 0 \
                or case["phase"] + case["trial_steps"] * dt > duration:
            raise ValueError("Test case exceeds its timing/reference bounds")


def create_protocol(path, source_bundle, seed=230923, count=120):
    contract = json.loads((source_bundle / "contract.json").read_text())
    experiment = json.loads((source_bundle / "experiment.json").read_text())
    if experiment["task"] != "ball":
        raise ValueError("Use a ball export to establish the test physics")
    protocol = dict(schema=SCHEMA, seed=seed, created_utc=datetime.now(timezone.utc).isoformat(),
                    purpose="Frozen evaluation only; never passed to training or checkpoint selection",
                    model_sha256=contract["model_sha256"], reference_sha256=contract["reference_sha256"],
                    world_model_sha256=experiment["world_model_sha256"],
                    control_timestep=contract["control_timestep"], reference_duration=contract["reference_duration"],
                    cases=make_cases(seed, count, contract["control_timestep"], contract["reference_duration"],
                                     experiment["horizontal_speed_min"], experiment["horizontal_speed_max"]))
    validate_protocol(protocol)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(protocol, file, indent=2)
    return protocol


def make_task(rig, reference, protocol, contact_mode="legacy"):
    from envs.base_env import EnvMode
    from .ball_task import BallTask

    validate_protocol(protocol)
    cases = protocol["cases"]

    class ProtocolBallTask(BallTask):
        """Change only the evaluation reset schedule; use the normal collision dynamics."""
        def reset(self, env_ids=None):
            result = super().reset(env_ids)
            if not hasattr(self, "shot_steps") or self._mode == EnvMode.TRAIN:
                return result
            ids = self.ids if env_ids is None else env_ids
            specs = [cases[i] for i in ids.tolist()]
            if specs:
                self.offset[ids] = torch.tensor([c["phase"] for c in specs])
                self.shot_steps[ids] = torch.tensor([c["launch_step"] for c in specs])
                self.shot_speeds[ids] = torch.tensor([c["speed"] for c in specs])
                self.shot_target_body[ids] = torch.tensor([rig.model.body(c["target_body"]).id - 1 for c in specs])
                angles = torch.tensor([c["angle_radians"] for c in specs])
                self.directions[ids] = torch.stack((torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), -1)
                self.shot_enabled[ids] = True
                self.engine.reset_envs(ids, *self.reference.sample(self.offset[ids]))
            return self.observations(), {}

    task = ProtocolBallTask(rig, reference, len(cases), "cpu", contact_mode=contact_mode)
    task.validation_cases = cases
    if task.dt != protocol["control_timestep"] or task.reference.sha256 != protocol["reference_sha256"] \
            or sha256(rig.path) != protocol["model_sha256"] \
            or sha256(task.engine.rig.path) != protocol["world_model_sha256"]:
        raise ValueError("Test physics/reference differs from the frozen protocol")
    return task


def run(args):
    activate(args.mimickit)
    torch.set_num_threads(1)
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    args.out.mkdir(parents=True, exist_ok=False)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    results = []
    for bundle in args.bundle:
        bundle = bundle.resolve()
        contract = json.loads((bundle / "contract.json").read_text())
        actor_hash = sha256(bundle / "stand.onnx")
        if actor_hash != contract["actor_sha256"]:
            raise ValueError("Actor hash mismatch")
        task = make_task(rig, bundle / "stand_reference.npz", protocol, mode_from_contract(contract))
        validate_contract(contract, task.observation_contract())
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(bundle / "stand.onnx"), options, providers=["CPUExecutionProvider"])
        metrics, _ = evaluate(task, lambda obs: torch.from_numpy(session.run(None, {"observation": obs.numpy()})[0]))
        metrics["evaluation_schema"] = SCHEMA
        result = dict(bundle=str(bundle), actor_sha256=actor_hash, metrics=metrics)
        results.append(result)
        (args.out / f"actor-{len(results):02d}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"bundle": str(bundle), **{k: metrics[k] for k in
              ("episodes", "confirmed_hits", "survived_hits", "recovered_hits", "mean_survival_seconds")}}), flush=True)
    report = dict(schema=SCHEMA, protocol_sha256=sha256(args.protocol), protocol=protocol, results=results,
                  scope="Independent randomized five-second cases; no selection or viewer changes; not long-horizon proof")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--create-protocol", action="store_true")
    parser.add_argument("--seed", type=int, default=230923)
    parser.add_argument("--cases", type=int, default=120)
    args = parser.parse_args()
    if args.create_protocol:
        if len(args.bundle) != 1:
            parser.error("Protocol creation requires one source bundle")
        create_protocol(args.protocol, args.bundle[0], args.seed, args.cases)
        print(f"Frozen {args.cases} cases: {args.protocol}")
    else:
        if args.mimickit is None or args.out is None:
            parser.error("Evaluation requires --mimickit and --out")
        run(args)
