"""Matched native torso probes; record commands and physics without modifying the controller."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .baseline import sha256
from .ball_native import BallNativeEngine
from .ball_protocol import VALIDATION_EPISODES
from .checkpoints import TORSO_BODIES, validate_contract
from .contact_contract import mode_from_contract
from .evaluate_stand import evaluate
from .rig import Rig
from .runtime import activate
from .target_control import DelayedTargetControl


class TraceTargetControl(DelayedTargetControl):
    """Observe the original PD computation at every physics substep."""
    def apply(self, action):
        self.substeps = []
        return super().apply(action)

    def physics_control(self, q, v):
        result = super().physics_control(q, v)
        error = self.active[:, :30] - q[:, self.q_indices]
        self.substeps.append(dict(
            joint_error=error.numpy().copy(),
            torque_fraction=(result[:, self.indices] / self.scale).numpy().copy(),
            target=self.active[:, :30].numpy().copy()))
        return result


def make_task(rig, reference, speed_min, speed_max, quiet=False, contact_mode="legacy"):
    from .ball_task import BallTask

    def engine_factory(rig, count, device, control_factory):
        return BallNativeEngine(rig, count, device, TraceTargetControl)

    class ProbeTask(BallTask):
        def reset(self, env_ids=None):
            result = super().reset(env_ids)
            if quiet and hasattr(self, "shot_enabled"):
                self.shot_enabled[self.ids if env_ids is None else env_ids] = False
            return result

        def step(self, action):
            result = super().step(action)
            control = self.engine.policy_control
            model = self.engine.rig.model
            loads = self.engine.get_ground_contact_forces(0).numpy()
            allowed = self.allowed_ground.numpy()
            frame = {key: np.stack([s[key][self.probe_ids] for s in control.substeps])
                     for key in ("joint_error", "torque_fraction", "target")}
            foot_ids = [model.body(n).id for n in ("Foot_L", "Foot_R")]
            frame.update(action=action[self.probe_ids].numpy().copy(),
                         qpos=self.engine.native_qpos()[self.probe_ids].numpy(),
                         foot_load=np.stack((np.abs(loads[self.probe_ids][:, allowed[0], 2]) + np.abs(loads[self.probe_ids][:, allowed[2], 2]),
                                             np.abs(loads[self.probe_ids][:, allowed[1], 2]) + np.abs(loads[self.probe_ids][:, allowed[3], 2])), -1),
                         foot_position=np.stack([self.engine.datas[i].xpos[foot_ids].copy() for i in self.probe_ids]),
                         root_velocity=self.engine.get_root_vel(0)[self.probe_ids].numpy())
            other = loads[self.probe_ids].copy()
            other[:, allowed] = 0
            frame["nonfoot_load"] = np.abs(other).max(axis=2)
            self.frames.append(frame)
            return result

    task = ProbeTask(rig, reference, VALIDATION_EPISODES, "cpu", engine_factory=engine_factory,
                     speed_min=speed_min, speed_max=speed_max, contact_mode=contact_mode)
    task.probe_ids = [i for i, c in enumerate(task.validation_cases)
                      if c["target_body"] in TORSO_BODIES and c["direction"] in (0, 2)]
    task.frames = []
    return task


def summarize_case(case, index, traces, dt, joint_names, *, start_step=None):
    """Exclude reset/replayed frames after termination, and only measure post-hit commands."""
    end = round(case["survival_seconds"] / dt)
    start = max(0, (case["first_hit_step"] if start_step is None else start_step) - 1)
    if start >= end:
        raise ValueError("No post-impact diagnostic window")
    window = slice(start, end)
    action = traces["action"][window, index]
    torque = traces["torque_fraction"][window, :, index]
    error = traces["joint_error"][window, :, index]
    feet = traces["foot_position"][window, index]
    loads = traces["foot_load"][window, index]
    qpos = traces["qpos"][window, index]
    saturation = (np.abs(torque) >= .95).mean(axis=(0, 1))
    order = np.argsort(-saturation)[:5]
    contacts = traces["nonfoot_load"][end - 1, index]
    root_velocity = traces["root_velocity"][window, index]
    # Loaded horizontal travel indicates sliding/stance adjustment, not a confirmed step.
    movement = np.linalg.norm(np.diff(feet[..., :2], axis=0), axis=-1)
    loaded = (loads[1:] > 5) & (loads[:-1] > 5)
    return dict(**case, post_hit_seconds=(end - start) * dt,
                action_near_limit_fraction=float((np.abs(action) >= .95).mean()),
                torque_near_limit_fraction=float((np.abs(torque) >= .95).mean()),
                highest_torque_saturation=[dict(joint=joint_names[j], fraction=float(saturation[j]),
                                               mean_abs_target_error_rad=float(np.abs(error[..., j]).mean())) for j in order],
                foot_max_lift_from_impact_m=(feet[..., 2] - feet[0, :, 2]).max(axis=0).tolist(),
                foot_max_x_displacement_m=np.abs(feet[..., 0] - feet[0, :, 0]).max(axis=0).tolist(),
                foot_loaded_travel_m=(movement * loaded).sum(axis=0).tolist(),
                foot_unloaded_fraction=(loads <= 5).mean(axis=0).tolist(),
                root_x_displacement_m=float(qpos[-1, 0] - qpos[0, 0]),
                final_root_velocity_m_s=root_velocity[-1].tolist(),
                final_root_height_m=float(qpos[-1, 2]),
                final_root_tilt_degrees=float(np.rad2deg(np.arccos(np.clip(1 - 2 * (qpos[-1, 4]**2 + qpos[-1, 5]**2), -1, 1)))),
                terminal_height_failure=bool(not case["survived"] and qpos[-1, 2] < .65),
                terminal_nonfoot_bodies=np.flatnonzero(contacts > 1).tolist())


def run(args):
    activate(args.mimickit)
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    results = []
    for actor_index, bundle in enumerate(args.bundle):
        bundle = bundle.resolve()
        contract = json.loads((bundle / "contract.json").read_text())
        if sha256(bundle / "stand.onnx") != contract["actor_sha256"]:
            raise ValueError("Actor hash mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(bundle / "stand.onnx"), options, providers=["CPUExecutionProvider"])
        actor = lambda obs: torch.from_numpy(session.run(None, {"observation": obs.numpy()})[0])
        result = dict(bundle=str(bundle), actor_sha256=contract["actor_sha256"], contract=contract)
        for quiet in (False, True):
            task = make_task(rig, bundle / "stand_reference.npz", args.speed_min, args.speed_max, quiet,
                             mode_from_contract(contract))
            validate_contract(contract, task.observation_contract())
            metrics, _ = evaluate(task, actor)
            traces = {key: np.stack([f[key] for f in task.frames]) for key in task.frames[0]}
            mode = "quiet" if quiet else "impact"
            np.savez_compressed(args.out / f"actor-{actor_index:02d}-{mode}.npz", **traces)
            summaries = []
            for index, case_id in enumerate(task.probe_ids):
                case = metrics["cases"][case_id]
                start = result["impact"][index]["first_hit_step"] if quiet else None
                summary = summarize_case(case, index, traces, task.dt, contract["actuator_names"], start_step=start)
                summary["terminal_nonfoot_bodies"] = [rig.model.body(b + 1).name for b in summary["terminal_nonfoot_bodies"]]
                summaries.append(summary)
            result[mode] = summaries
            if not quiet:
                result["full_suite_metrics"] = metrics
                result["world_model_sha256"] = sha256(task.engine.rig.path)
            print(json.dumps(dict(bundle=str(bundle), mode=mode, successes=metrics["successes"],
                                  survived_hits=metrics["survived_hits"], recovered=metrics["recovered_hits"])), flush=True)
        results.append(result)
        (args.out / f"actor-{actor_index:02d}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "report.json").write_text(json.dumps(dict(
        schema="mimic_torso_diagnostics_v1", results=results,
        scope="Matched five-second native torso/quiet probes. Foot motion is not proof of deliberate stepping. No training or viewer changes."), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--speed-min", type=float, default=1.)
    parser.add_argument("--speed-max", type=float, default=2.5)
    run(parser.parse_args())
