"""Explicit, behavior-gated normalization migration for a new motion curriculum."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, helper
import onnxruntime as ort
import torch

from .baseline import sha256
from .checkpoints import standing_passed, validate_contract
from .contact_contract import mode_from_contract
from .evaluate_stand import evaluate
from .native_engine import NativeDummyEngine
from .probe_steps import validate_reference_substitution
from .rig import Rig
from .runtime import activate
from .weight_shift import EVALUATION_SCHEMA, tracking_metrics
from .motion_features import SCHEMA as FEATURE_SCHEMA

SCHEMA = "mimic_motion_normalization_v1"
GROUPS = ("pending_targets_q_v_valid_oldest_first", "future_reference_poses_1_2_3")


def validate_preparation(contract_path, checkpoint, reference_sha256):
    contract = json.loads(contract_path.read_text())
    marker = contract.get("normalization_preparation", {})
    if marker.get("schema") != SCHEMA or marker.get("manifest") != "preparation.json":
        raise ValueError("Unsupported normalization preparation contract")
    path = contract_path.parent / "preparation.json"
    if marker.get("sha256") != sha256(path):
        raise ValueError("Normalization preparation hash mismatch")
    report = json.loads(path.read_text())
    if (report.get("schema") != SCHEMA or report.get("ready_for_training") is not True
            or report.get("checkpoint_sha256") != sha256(checkpoint)
            or report.get("actor_sha256") != contract.get("actor_sha256")
            or reference_sha256 not in report.get("calibration_reference_sha256", [])[1:]):
        raise ValueError("Normalization checkpoint/reference is not admitted by this preparation")
    rows, initial = report["validation"], report["initial_native"]
    if len(rows) != len(initial) + 1 or len(rows) < 3:
        raise ValueError("Incomplete normalization validation")
    quiet = rows[:2]
    if ([r["seconds"] for r in quiet] != [3., 5.]
            or any(r["metrics"]["episodes"] != 8 or not standing_passed(r["metrics"]) for r in quiet)
            or any(r["motion_input_clipped_fraction"] != 0. for r in rows)
            or any(r["metrics"]["successes"] < before["successes"]
                   or r["metrics"]["mean_survival_seconds"] < before["mean_survival_seconds"]
                   for r, before in zip(rows[2:], initial[1:]))):
        raise ValueError("Normalization preparation failed measured retention/coverage")
    if report["migration_mode"] == "additive_features":
        expected_indices = [i for group in contract["observation_layout"] if group["name"] in GROUPS
                            for i in range(group["offset"], group["offset"] + group["width"])]
        if report.get("motion_feature_indices") != expected_indices or not expected_indices:
            raise ValueError("Expanded motion features disagree with the observation layout")
        error = report.get("calibration_max_action_difference", float("inf"))
        if not np.isfinite(error) or error > 1e-5:
            raise ValueError("Expanded features changed the initial actor beyond parity tolerance")
    elif report["migration_mode"] not in ("compensated", "calibration_only"):
        raise ValueError("Unknown normalization migration mode")
    return {"schema": SCHEMA, "manifest_sha256": marker["sha256"],
            "migration_mode": report["migration_mode"], "checkpoint_sha256": report["checkpoint_sha256"],
            "feature_schema": FEATURE_SCHEMA if report["migration_mode"] == "additive_features" else None,
            "motion_feature_indices": report["motion_feature_indices"]}


def migrate(state, observations, indices, compensate=True):
    """Widen selected scales and compensate both first layers; means remain fixed.

    Compensation preserves preactivations where the old clip was inactive.
    Broader clipping boundaries deliberately change behavior outside that domain,
    so the candidate requires native standing validation before training admission.
    """
    result = copy.deepcopy(state)
    mean, old = state["_obs_norm._mean"], state["_obs_norm._std"]
    if observations.ndim != 2 or observations.shape[1] != len(mean) or not torch.isfinite(observations).all():
        raise ValueError("Require finite calibration observations of the contracted width")
    if not torch.isfinite(old).all() or (old <= 0).any():
        raise ValueError("Source normalization scales must be finite and positive")
    new = old.clone()
    new[indices] = torch.maximum(old[indices], (observations[:, indices] - mean[indices]).abs().amax(0) / 6.)
    ratio = new / old
    if compensate:
        for branch in ("actor", "critic"):
            key = f"_model._{branch}_layers.0.weight"
            result[key] = state[key] * ratio.unsqueeze(0)
    result["_obs_norm._std"] = new
    return result


def update_graph(model, original, candidate):
    """Bind source ONNX weights to the checkpoint before changing a private copy."""
    model = copy.deepcopy(model)
    prefixes = (("obs_norm.", "_obs_norm."), ("action_norm.", "_a_norm."),
                ("actor.", "_model._actor_layers."), ("mean.", "_model._action_dist._mean_net."))
    changed = set()
    for value in model.graph.initializer:
        for exported, saved in prefixes:
            if value.name.startswith(exported):
                key = saved + value.name[len(exported):]
                if key not in original or not np.array_equal(numpy_helper.to_array(value), original[key].numpy()):
                    raise ValueError(f"Checkpoint and source ONNX disagree: {key}")
                if not torch.equal(original[key], candidate[key]):
                    changed.add(key)
                    value.CopyFrom(numpy_helper.from_array(candidate[key].numpy(), value.name))
                break
    expanded = "_obs_norm._motion_std" in candidate
    required = set() if expanded else {"_obs_norm._std"}
    if not torch.equal(original["_model._actor_layers.0.weight"], candidate["_model._actor_layers.0.weight"]):
        required.add("_model._actor_layers.0.weight")
    if changed != required:
        raise ValueError("Expected exactly the normalization scale and compensated actor input layer to change")
    if expanded:
        divisions = [n for n in model.graph.node if n.op_type == "Div" and n.input[1] == "obs_norm._std"]
        if len(divisions) != 1:
            raise ValueError("Cannot identify the original normalization path")
        division = divisions[0]
        clip = next(n for n in model.graph.node if n.op_type == "Clip" and n.input[0] == division.output[0])
        layer = next(n for n in model.graph.node if "actor.0.weight" in n.input)
        old_input = layer.input[0]
        layer.input[0] = "motion_features"
        model.graph.initializer.extend([
            numpy_helper.from_array(candidate["_obs_norm._motion_std"].numpy(), "obs_norm._motion_std"),
            numpy_helper.from_array(candidate["_obs_norm._motion_indices"].numpy(), "obs_norm._motion_indices")])
        nodes = [helper.make_node("Div", [division.input[0], "obs_norm._motion_std"], ["motion_z"]),
                 helper.make_node("Gather", ["motion_z", "obs_norm._motion_indices"], ["motion_subset"], axis=1),
                 helper.make_node("Clip", ["motion_subset", clip.input[1], clip.input[2]], ["motion_clipped"]),
                 helper.make_node("Concat", [old_input, "motion_clipped"], ["motion_features"], axis=1)]
        position = list(model.graph.node).index(layer)
        for node in reversed(nodes):
            model.graph.node.insert(position, node)
    onnx.checker.check_model(model)
    return model


def run(args):
    activate(args.mimickit)
    from .stand_task import StandTask
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    contract = json.loads((args.bundle / "contract.json").read_text())
    if contract.get("normalization_preparation") or contract.get("diagnostic_only"):
        raise ValueError("Prepare from an original trained Stand bundle")
    if sha256(args.bundle / "stand.onnx") != contract["actor_sha256"]:
        raise ValueError("Source actor hash mismatch")
    original = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = onnx.load(args.bundle / "stand.onnx")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    def actor_for(graph):
        session = ort.InferenceSession(graph.SerializeToString(), options, providers=["CPUExecutionProvider"])
        return lambda obs: torch.from_numpy(session.run(None, {"observation": obs.numpy()})[0])
    original_actor = actor_for(model)
    quiet_path = Path(__file__).parent / "assets/stand_reference.npz"
    references = [quiet_path, *args.reference]
    tasks, observations, initial = [], [], []
    for reference in references:
        task = StandTask(rig, reference, 8, "cpu", engine_factory=NativeDummyEngine,
                         control_mode="target_pd", contact_mode=mode_from_contract(contract))
        validate_reference_substitution(contract, task.observation_contract())
        if reference == quiet_path:
            validate_contract(contract, task.observation_contract())
        metrics, traces = evaluate(task, original_actor)
        initial.append(metrics)
        observations.extend(traces["observations"][:round(t / task.dt), i]
                            for i, t in enumerate(metrics["survival_seconds"]))
        # Include the entire target distribution, even beyond an early policy fall.
        for start in range(0, len(task.reference.qpos), 8):
            times = (torch.arange(8) + start).clamp(max=len(task.reference.qpos)-1) * task.reference.dt
            task.offset[:] = times
            task.steps[:] = 0
            q, v = task.reference.sample(times)
            task.engine.reset_envs(task.ids, q, v)
            control = task.engine.policy_control
            packet = torch.cat((q[:, control.q_indices], v[:, control.v_indices], torch.ones((8, 1))), -1)
            control.pending[:] = packet
            observations.append(task.observations().numpy())
        tasks.append(task)
    data = torch.tensor(np.concatenate(observations), dtype=torch.float32)
    indices = torch.tensor([i for g in contract["observation_layout"] if g["name"] in GROUPS
                            for i in range(g["offset"], g["offset"] + g["width"])])
    compensate = args.migration_mode == "compensated"
    candidate = migrate(original, data, indices, compensate=compensate)
    if args.migration_mode == "additive_features":
        motion_std = candidate["_obs_norm._std"].clone()
        candidate = copy.deepcopy(original)
        candidate["_obs_norm._motion_std"] = motion_std
        candidate["_obs_norm._motion_indices"] = indices
        for branch in ("actor", "critic"):
            key = f"_model._{branch}_layers.0.weight"
            candidate[key] = torch.cat((original[key], torch.zeros((original[key].shape[0], len(indices)))), dim=1)
    graph = update_graph(model, original, candidate)
    migrated_actor = actor_for(graph)
    with torch.no_grad():
        action_error = float((migrated_actor(data) - original_actor(data)).abs().max())
    results, coverage = [], []
    for index, task in enumerate(tasks):
        for seconds in ((3., 5.) if index == 0 else (3.,)):
            task.episode_seconds = seconds
            metrics, traces = evaluate(task, migrated_actor)
            if index:
                metrics.update(evaluation_schema=EVALUATION_SCHEMA,
                               weight_shift_tracking=tracking_metrics(task, metrics, traces))
            observed = np.concatenate([traces["observations"][:round(t / task.dt), i]
                                      for i, t in enumerate(metrics["survival_seconds"])])
            selected_std = candidate.get("_obs_norm._motion_std", candidate["_obs_norm._std"])
            normalized = (observed - candidate["_obs_norm._mean"].numpy()) / selected_std.numpy()
            clip = float((np.abs(normalized[:, indices.numpy()]) > 10).mean())
            coverage.append(clip)
            results.append(dict(reference_sha256=task.reference.sha256, seconds=seconds, metrics=metrics,
                                motion_input_clipped_fraction=clip))
            np.savez_compressed(args.out / f"validation-{index}-{int(seconds)}.npz", **traces)
    retained = all(standing_passed(row["metrics"]) for row in results[:2])
    motion_retained = all(row["metrics"]["successes"] >= before["successes"]
                          and row["metrics"]["mean_survival_seconds"] >= before["mean_survival_seconds"]
                          for row, before in zip(results[2:], initial[1:]))
    ready = retained and motion_retained and max(coverage) == 0.
    if args.migration_mode == "additive_features":
        ready = ready and action_error <= 1e-5
    path = args.out / "candidate.pt"
    torch.save(candidate, path)
    onnx.save(graph, args.out / "candidate.onnx")
    np.savez_compressed(args.out / "normalization.npz", mean=candidate["_obs_norm._mean"].numpy(),
                        old_std=original["_obs_norm._std"].numpy(), new_std=selected_std.numpy(),
                        indices=indices.numpy())
    report = dict(schema=SCHEMA, ready_for_training=ready, standing_retained=retained,
        motion_baseline_retained=motion_retained, migration_mode=args.migration_mode,
        calibration_max_action_difference=action_error, motion_feature_indices=indices.tolist(),
        checkpoint_sha256=sha256(path), source_checkpoint_sha256=sha256(args.checkpoint),
        source_actor_sha256=contract["actor_sha256"], actor_sha256=sha256(args.out / "candidate.onnx"),
        calibration_reference_sha256=[t.reference.sha256 for t in tasks], calibration_observations=len(data),
        modified_groups=list(GROUPS), changed_scale_count=int((selected_std != original["_obs_norm._std"]).sum()),
        initial_native=initial, validation=results,
        method="Motion scales cover calibration within +/-6; fixed +/-10 clipping remains. Compensated/calibration_only modes replace scales and require behavioral validation. Additive_features retains the entire old normalization path and appends broader motion features with zero-initialized actor/critic input weights, preserving initial outputs while allowing new motion-feature weights to learn.",
        scope="Prepared initialization only, no PPO or viewer selection. Validation covers the listed references; not general recovery or motion success.")
    manifest = args.out / "preparation.json"
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    source_contract = {**contract, "actor_sha256": report["actor_sha256"], "diagnostic_only": not ready,
                       "normalization_preparation": dict(schema=SCHEMA, manifest="preparation.json", sha256=sha256(manifest))}
    (args.out / "contract.json").write_text(json.dumps(source_contract, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--migration-mode", choices=("compensated", "calibration_only", "additive_features"), default="additive_features")
    run(parser.parse_args())
