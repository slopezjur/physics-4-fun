"""Trace early weight-shift failure without changing training or viewer policies."""
import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import onnx
import onnxruntime as ort
import torch

from .baseline import sha256
from .contact_contract import mode_from_contract
from .evaluate_stand import evaluate
from .force_retarget import reference_velocities
from .motion_tracking_audit import TraceEngine
from .probe_steps import validate_reference_substitution
from .rig import Rig
from .runtime import activate
from .track_transfer import summarize_case


def normalization_summary(observations, mean, std, layout):
    """Report clipping in the exported actor's original observation coordinates."""
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Normalizer standard deviations must be finite and positive")
    z = (observations - mean) / std
    groups = []
    for group in layout:
        start, width = group["offset"], group["width"]
        values = z[:, start:start + width]
        groups.append(dict(name=group["name"], max_abs_z=float(np.abs(values).max()),
                           clipped_fraction=float((np.abs(values) > 10).mean()),
                           clipped_indices=(start + np.flatnonzero((np.abs(values) > 10).any(axis=0))).tolist()))
    return groups


def scaled_reference(reference, rig, amplitude):
    """Private kinematic intervention; never saved as an admitted reference."""
    result = copy.copy(reference)
    source = reference.qpos.numpy().astype(np.float64)
    poses = np.empty_like(source)
    difference = np.zeros(rig.model.nv)
    for index, pose in enumerate(source):
        mujoco.mj_differentiatePos(rig.model, difference, 1., source[0], pose)
        poses[index] = source[0]
        mujoco.mj_integratePos(rig.model, poses[index], difference, amplitude)
    result.qpos = torch.tensor(poses, dtype=torch.float32)
    result.qvel = torch.tensor(reference_velocities(rig.model, poses, reference.dt), dtype=torch.float32)
    return result


def remove_observation_clip(model):
    """Private ONNX intervention; preserve action clipping and every learned weight."""
    model = copy.deepcopy(model)
    divisions = {node.output[0] for node in model.graph.node
                 if node.op_type == "Div" and node.input[1] == "obs_norm._std"}
    clips = [node for node in model.graph.node if node.op_type == "Clip" and node.input[0] in divisions]
    if len(clips) != 1:
        raise ValueError("Require exactly one identified observation-normalization clip")
    node = clips[0]
    node.op_type = "Identity"
    del node.input[1:]
    del node.attribute[:]
    onnx.checker.check_model(model)
    return model


def run(args):
    activate(args.mimickit)
    from .stand_task import StandTask
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    for bundle in args.bundle:
        contract = json.loads((bundle / "contract.json").read_text())
        if sha256(bundle / "stand.onnx") != contract["actor_sha256"]:
            raise ValueError("Actor hash mismatch")
        model = onnx.load(bundle / "stand.onnx")
        weights = {v.name: onnx.numpy_helper.to_array(v) for v in model.graph.initializer}
        mean, std = weights["obs_norm._mean"], weights["obs_norm._std"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        unclipped = getattr(args, "normalizer_unclipped", False)
        if unclipped:
            model = remove_observation_clip(model)
        session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
        jobs = (("quiet", None), ("hold_start", 0.), ("quarter_shift", .25), ("half_shift", .5), ("full_shift", 1.))
        if unclipped:
            jobs = (("quiet", None), ("full_shift", 1.))
        for label, amplitude in jobs:
            rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
            reference = Path(__file__).parent / "assets/stand_reference.npz" if label == "quiet" else args.reference
            task = StandTask(rig, reference, 1, "cpu", engine_factory=TraceEngine,
                             control_mode="target_pd", contact_mode=mode_from_contract(contract))
            validate_reference_substitution(contract, task.observation_contract())
            if amplitude is not None and amplitude != 1:
                task.reference = scaled_reference(task.reference, rig, amplitude)
            task.episode_seconds = 3.
            actor = lambda obs: torch.from_numpy(session.run(None, {"observation": obs.numpy()})[0])
            metrics, traces = evaluate(task, actor, episodes=1, capture_actions=True)
            seconds = metrics["survival_seconds"][0]
            row, substeps = summarize_case(task, task.engine.records[0], seconds, 0.)
            early_seconds = min(.8, seconds)
            control_frames = max(1, int(early_seconds / task.dt))
            physics_frames = control_frames * rig.decimation
            actions = traces["actions"][:control_frames, 0]
            controls = substeps["control"][:physics_frames] / rig.action_scale
            groups = normalization_summary(traces["observations"][:control_frames, 0], mean, std,
                                            contract["observation_layout"])
            q = substeps["qpos_after"]
            target = substeps["reference_qpos"]
            stamps = []
            for time in (.25, .5, .8, 1., 1.5):
                index = round(time / rig.model.opt.timestep) - 1
                if index >= len(q):
                    continue
                force = substeps["ground_force_solve"][index, :, 2]
                stamps.append(dict(seconds=(index + 1) * rig.model.opt.timestep,
                    root_xyz=q[index, :3].tolist(), reference_root_xyz=target[index, :3].tolist(),
                    root_velocity=substeps["qvel_after"][index, :3].tolist(),
                    foot_toe_load_n=[float(force[0] + force[2]), float(force[1] + force[3])]))
            max_actions = np.abs(actions).max(axis=0)
            max_controls = np.abs(controls).max(axis=0)
            joint_indices = task.engine.policy_control.q_indices.numpy()
            desired, _ = task.reference.sample(torch.arange(control_frames) * task.dt)
            control = task.engine.policy_control
            issued_unbounded = desired.numpy()[:, joint_indices] + actions * control.residual_radians
            target_clipped = (issued_unbounded < control.lower.numpy()) | (issued_unbounded > control.upper.numpy())
            result = dict(bundle=str(bundle), actor_sha256=contract["actor_sha256"], case=label,
                amplitude=amplitude, metrics=metrics, trace=row, early_seconds=control_frames * task.dt,
                early_normalization=groups, early_action_max=float(max_actions.max()),
                early_motor_fraction_max=float(max_controls.max()),
                early_target_limit_fraction=float(target_clipped.mean()),
                strongest_actions=[dict(name=contract["actuator_names"][i], max_abs_action=float(max_actions[i]),
                                        max_abs_motor_fraction=float(max_controls[i]))
                                   for i in np.argsort(max_actions)[-6:][::-1]], samples=stamps,
                diagnostic_only=unclipped or amplitude in (0., .25, .5), normalizer_clip_removed=unclipped)
            index = len(results)
            np.savez_compressed(args.out / f"case-{index:02d}.npz", **traces,
                                **{f"physics_{k}": v for k, v in substeps.items()})
            results.append(result)
            print(json.dumps(dict(case=label, bundle=str(bundle), survival=seconds,
                early_motor_max=result["early_motor_fraction_max"],
                first_root_error_5cm=row["first_root_error_5cm_seconds"])), flush=True)
    report = dict(schema="mimic_weight_shift_failure_audit_v1", reference_sha256=sha256(args.reference),
        runner_sha256=sha256(Path(__file__)), results=results,
        scope="Paired phase-zero diagnostics. Amplitude interventions are unadmitted in-memory poses. Optional normalizer clip removal is labeled per case and remains in memory. Learned weights, normalization statistics and runtime motor controls are unchanged. No training or viewer changes.")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, action="append", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--normalizer-unclipped", action="store_true", help="Private diagnostic removing only the observation clip")
    run(parser.parse_args())
