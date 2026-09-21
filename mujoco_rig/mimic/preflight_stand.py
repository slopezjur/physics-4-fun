"""Gate motion/task/reset/native parity before spending a training budget."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from .runtime import activate
from .rig import Rig


def preflight(checkout: Path, reference: Path, out: Path, control_mode="torque"):
    activate(checkout)
    from envs.base_env import EnvMode
    from .stand_task import StandTask
    from .native_engine import NativeDummyEngine
    rig = Rig.load(Path(__file__).resolve().parents[1] / "dummy.xml")
    gpu = StandTask(rig, reference, 2, control_mode=control_mode)
    cpu = StandTask(rig, reference, 2, "cpu", engine_factory=NativeDummyEngine, control_mode=control_mode)
    for task in (gpu, cpu):
        task.set_mode(EnvMode.TEST)
        task.reset()
    checks = {"reference_reward": bool(torch.allclose(gpu.reward(), torch.ones(2, device="cuda:0"), atol=1e-5)),
              "observation_width": gpu.observations().shape[-1] == gpu.observation_contract()["num_obs"]}
    difference = np.abs(gpu.observations().cpu().numpy() - cpu.observations().numpy())
    channels = gpu.observation_contract()["observation_layout"]
    channel_errors = {c["name"]: float(difference[:, c["offset"]:c["offset"] + c["width"]].max()) for c in channels}
    checks["initial_observations"] = bool(difference.max() < 0.003)
    actions = np.random.default_rng(918).uniform(-0.02, 0.02, (12, 2, 30)).astype(np.float32)
    reward_error = pose_error = torque_error = 0.0
    for action in actions:
        _, rg, dg, _ = gpu.step(torch.tensor(action, device="cuda:0"))
        _, rc, dc, _ = cpu.step(torch.tensor(action))
        reward_error = max(reward_error, float(np.max(np.abs(rg.cpu().numpy() - rc.numpy()))))
        pose_error = max(pose_error, float(np.max(np.abs(gpu.engine.native_qpos().cpu().numpy() - cpu.engine.native_qpos().numpy()))))
        torque_error = max(torque_error, float((gpu.engine.native_ctrl.cpu() - cpu.engine.native_ctrl).abs().max()))
    checks["short_reward_parity"] = reward_error < 0.01
    checks["short_pose_parity"] = pose_error < 0.01
    checks["terminal_flags"] = bool(torch.equal(dg.cpu(), dc))
    checks["motor_torque_parity"] = torque_error < 0.1
    preserved = gpu.engine.native_qpos()[1].clone()
    pending = gpu.engine.policy_control.pending[:, 1].clone()
    steps, offset = gpu.steps[1].clone(), gpu.offset[1].clone()
    gpu.reset(torch.tensor([0], device="cuda:0"))
    checks["partial_reset_peer_state"] = bool(torch.equal(preserved, gpu.engine.native_qpos()[1]))
    checks["partial_reset_peer_queue"] = bool(torch.equal(pending, gpu.engine.policy_control.pending[:, 1]))
    checks["partial_reset_peer_time"] = bool(steps == gpu.steps[1] and offset == gpu.offset[1])
    checks["partial_reset_own_queue"] = bool(torch.count_nonzero(gpu.engine.policy_control.pending[:, 0]) == 0)
    if gpu.target_pd:
        checks["partial_reset_active_target"] = bool(torch.count_nonzero(gpu.engine.policy_control.active[0]) == 0)
    checks["partial_reset_reward"] = bool(abs(gpu.reward()[0].item() - 1) < 1e-5)
    # Rewards must distinguish tracking from a visibly wrong pose.
    q, v = gpu.reference.sample(gpu.offset[:1])
    q[:, 0] += 0.4
    gpu.engine.reset_envs(torch.tensor([0], device="cuda:0"), q, v)
    checks["displaced_pose_scores_less"] = bool(gpu.reward()[0] < 0.95)
    report = {"passed": all(checks.values()), "checks": checks, "control_mode": control_mode,
              "initial_channel_errors": channel_errors, "reward_error": reward_error, "pose_error": pose_error,
              "torque_error_nm": torque_error}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path(__file__).parent / "assets/stand_reference.npz")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control", choices=("torque", "target_pd"), default="torque")
    args = parser.parse_args()
    result = preflight(args.mimickit, args.reference, args.out, args.control)
    raise SystemExit(0 if result["passed"] else 1)
