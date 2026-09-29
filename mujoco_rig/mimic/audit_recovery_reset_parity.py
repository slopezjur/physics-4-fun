"""Compare native and GPU continuation from identical physical post-hit states."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .baseline import sha256
from .runtime import activate


def run(args):
    if not 1 <= args.steps <= 60:
        raise ValueError("Require 1..60 control steps for local reset parity")
    activate(args.mimickit)
    from learning.ppo_agent import PPOAgent
    from util import mp_util
    from .balance_observation import task_classes
    from .checkpoints import validate_contract
    from .policy import StandPolicy
    from .rig import Rig

    torch.set_num_threads(1)
    mp_util.init(0, 1, "cpu", None)
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    contract = json.loads(args.checkpoint.with_name("contract.json").read_text())
    _, Task = task_classes(contract)
    args.out.mkdir(parents=True, exist_ok=False)
    with np.load(args.bank, allow_pickle=False) as data:
        if set(data["posture"].tolist()) != set(range(8)):
            raise ValueError("Require all eight postures")
        support_bodies = [rig.model.body(name).id - 1 for name in ("Foot_L", "Toe_L", "Foot_R", "Toe_R")]
        support_loads = np.abs(data["ground"][:, support_bodies, 2]).reshape(-1, 2, 2).sum(-1)
        rows = []
        for posture in range(8):
            candidates = np.flatnonzero(data["posture"] == posture)
            selected = (candidates[np.argmin(support_loads[candidates].min(-1))]
                        if args.state_selection == "least-supported" else candidates[0])
            rows.append(selected)
        rows = np.asarray(rows)
        fields = {key: data[key][rows] if key in
                  ("qpos", "qvel", "ground", "pending", "control_active", "posture", "direction")
                  else data[key].copy() for key in data.files}
    subset = args.out / "eight_posture_states.npz"
    np.savez_compressed(subset, **fields)
    tasks = [Task(rig, args.reference, 8, device, ball_path=args.ball_model,
                  quiet_fraction=0., recovery_reset_fraction=1., speed_min=2., speed_max=2.,
                  episode_seconds=12., recovery_mode="step_v5") for device in ("cpu", "cuda:0")]
    for task in tasks:
        validate_contract(contract, task.observation_contract())
        task.load_recovery_resets(subset)
        task.reset()
    agent = PPOAgent(yaml.safe_load(args.checkpoint.with_name("agent.yaml").read_text()), tasks[0], "cpu")
    agent.load(str(args.checkpoint))
    actor = StandPolicy(agent).eval()
    metrics = []
    with torch.no_grad():
        for step in range(args.steps + 1):
            native, gpu = tasks
            qpos_error = (native.engine.native_qpos() - gpu.engine.native_qpos().cpu()).abs().max()
            forces = [task.engine.get_ground_contact_forces(0)[:, :rig.model.nbody - 1].cpu() for task in tasks]
            observations = [task.observations().cpu() for task in tasks]
            actions = [actor(observation) for observation in observations]
            loads = [observation[:, 93:95] * 1000. for observation in observations]
            action_error = float((actions[0] - actions[1]).abs().max())
            metrics.append(dict(step=step, qpos_error=float(qpos_error),
                                contact_force_error_n=float((forces[0] - forces[1]).abs().max()),
                                observation_error=float((observations[0] - observations[1]).abs().max()),
                                action_error=action_error, target_error_radians=.25 * action_error,
                                native_foot_loads_n=loads[0].tolist(), gpu_foot_loads_n=loads[1].tolist(),
                                load_threshold_disagreements={str(threshold): int(
                                    ((loads[0] >= threshold) != (loads[1] >= threshold)).sum())
                                    for threshold in (5., 30., 50.)},
                                finite=bool(all(torch.isfinite(obs).all() for obs in observations))))
            if step < args.steps:
                action = actions[0]
                for task in tasks:
                    task.step(action.to(task.device))
    checks = dict(qpos=max(row["qpos_error"] for row in metrics) < 1e-4,
                  contacts=max(row["contact_force_error_n"] for row in metrics) < .1,
                  observations=max(row["observation_error"] for row in metrics) < 1e-4,
                  finite=all(row["finite"] for row in metrics))
    report = dict(schema="physical_post_hit_reset_parity_v1", checks=checks, passed=all(checks.values()),
                  checkpoint_sha256=sha256(args.checkpoint), bank_sha256=sha256(args.bank),
                  subset_sha256=sha256(subset), selected_bank_rows=rows.tolist(),
                  state_selection=args.state_selection, initial_bank_foot_loads_n=support_loads[rows].tolist(),
                  reference_sha256=sha256(args.reference), ball_model_sha256=sha256(args.ball_model),
                  character_model_sha256=sha256(rig.path), metrics=metrics,
                  scope="Local physics continuation under identical native-policy actions; not long-horizon policy admission")
    (args.out / "runner.py").write_bytes(Path(__file__).read_bytes())
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(passed=report["passed"], checks=checks, final=metrics[-1])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mimickit", "checkpoint", "reference", "ball-model", "bank", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--state-selection", choices=("first", "least-supported"), default="first")
    run(parser.parse_args())
