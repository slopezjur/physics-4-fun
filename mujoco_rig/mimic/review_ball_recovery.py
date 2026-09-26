"""Paired front/back recovery review, including explicitly out-of-training runtime stress."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .baseline import sha256
from .ball_protocol import validation_cases
from .checkpoints import validate_contract
from .contact_contract import mode_from_contract
from .evaluate_stand import evaluate
from .rig import Rig
from .runtime import activate


def review_cases(seconds, dt, speed_min=1., speed_max=2.):
    if not math.isfinite(seconds) or seconds < 5 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Review requires at least five seconds and a positive timestep")
    if not 1 <= speed_min <= speed_max <= 8:
        raise ValueError("Review ball speeds must be in [1, 8]")
    steps = math.ceil(seconds / dt)
    cases = [dict(c, quiet=False, trial_steps=steps) for c in validation_cases(speed_min, speed_max)
             if c["direction"] in (0, 2) and c["target_body"] in ("Chest", "Spine", "Pelvis")]
    return cases + [dict(name=f"quiet-{i}", target_body="Chest", direction=0, speed=speed_min,
                        phase=i / 7 * .9, launch_step=60, trial_steps=steps, quiet=True)
                    for i in range(8)]


def make_task(rig, bundle, contract, seconds, speed_min, speed_max):
    from envs.base_env import EnvMode
    from .ball_task import BallTask
    cases = review_cases(seconds, contract["control_timestep"], speed_min, speed_max)

    class ReviewTask(BallTask):
        def reset(self, env_ids=None):
            result = super().reset(env_ids)
            if not hasattr(self, "shot_steps") or self._mode != EnvMode.TEST:
                return result
            ids = self.ids if env_ids is None else env_ids
            specs = [cases[i] for i in ids.tolist()]
            if specs:
                self.offset[ids] = torch.tensor([c["phase"] for c in specs])
                self.shot_steps[ids] = torch.tensor([c["launch_step"] for c in specs])
                self.shot_speeds[ids] = torch.tensor([c["speed"] for c in specs])
                self.shot_enabled[ids] = torch.tensor([not c["quiet"] for c in specs])
                self.shot_target_body[ids] = torch.tensor([rig.model.body(c["target_body"]).id - 1 for c in specs])
                angles = torch.tensor([c["direction"] * math.pi / 2 for c in specs])
                self.directions[ids] = torch.stack((torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), -1)
                self.engine.reset_envs(ids, *self.reference.sample(self.offset[ids]))
            return self.observations(), {}

    task = ReviewTask(rig, bundle / "stand_reference.npz", len(cases), "cpu",
                      speed_min=speed_min, speed_max=speed_max, contact_mode=mode_from_contract(contract))
    validate_contract(contract, task.observation_contract())
    task.validation_cases = cases
    task.episode_seconds = seconds
    return task


def summarize(metrics):
    impact = [c for c in metrics["cases"] if not c["quiet"]]
    quiet = [c for c in metrics["cases"] if c["quiet"]]
    result = dict(impact_cases=len(impact), confirmed_hits=sum(c["hit"] for c in impact),
        survived_hits=sum(c["hit"] and c["survived"] for c in impact),
        recovered_hits=sum(c["recovered"] for c in impact),
        delayed_falls_after_5s=sum(c["hit"] and c["survival_seconds"] >= 5. and not c["survived"] for c in impact),
        quiet_survived=sum(c["survived"] for c in quiet), quiet_cases=len(quiet))
    result["directions"] = {name: dict(
        survived=sum(c["hit"] and c["survived"] for c in impact if c["direction"] == direction),
        recovered=sum(c["recovered"] for c in impact if c["direction"] == direction), cases=6)
        for name, direction in (("forward", 0), ("backward", 2))}
    return result


def run(args):
    activate(args.mimickit)
    torch.set_num_threads(1)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    args.out.mkdir(parents=True, exist_ok=False)
    results, physics = [], None
    for bundle in args.bundle:
        bundle = bundle.resolve()
        contract = json.loads((bundle / "contract.json").read_text())
        if sha256(bundle / "stand.onnx") != contract["actor_sha256"]:
            raise ValueError("Actor hash mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(bundle / "stand.onnx"), options, providers=["CPUExecutionProvider"])
        for seconds in (5., 30.):
            task = make_task(rig, bundle, contract, seconds, args.speed_min, args.speed_max)
            identity = (contract["reference_sha256"], contract["model_sha256"],
                        sha256(task.engine.rig.path), task.dt, mode_from_contract(contract))
            if physics is not None and identity != physics:
                raise ValueError("Paired review requires identical physics, reference and observation semantics")
            physics = identity
            metrics, traces = evaluate(task, lambda obs: torch.from_numpy(
                session.run(None, {"observation": obs.numpy()})[0]))
            row = dict(bundle=str(bundle), actor_sha256=contract["actor_sha256"], seconds=seconds,
                       outside_training_reference=seconds > task.reference.duration,
                       summary=summarize(metrics), metrics=metrics)
            results.append(row)
            (args.out / f"actor-{len(results):02d}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            np.savez_compressed(args.out / f"actor-{len(results):02d}-traces.npz", **traces)
            print(json.dumps({k: v for k, v in row.items() if k != "metrics"}), flush=True)
    report = dict(schema="mimic_front_back_review_v1", results=results, runner_sha256=sha256(Path(__file__)),
        scope="Five-second matched recovery plus 30-second runtime stress. Long stress preserves the existing Godot semantics: reference pose/velocity clamp at the clip endpoint while phase increases beyond training. Long stress is not an in-distribution continuous-standing test. Fixed cases inform development, not an untouched generalization test. Foot movement alone is not evidence of deliberate stepping.")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--speed-min", type=float, default=1.)
    parser.add_argument("--speed-max", type=float, default=2.)
    run(parser.parse_args())
