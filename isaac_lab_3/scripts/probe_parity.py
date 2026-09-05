"""Does the ONNX graph Godot runs answer like the PyTorch policy Isaac trained?

**Nothing on this project has ever checked this on a real observation.** `export.verify()` checks
the graph's shapes and its response to a synthetic rest pose; it never compares the exported graph
against the policy it came from. So a normaliser left outside the graph, a dtype narrowing, an
opset regression, or simply a `Models/locomotion_policy.onnx` exported from a DIFFERENT checkpoint
than the one being measured would all pass export and then look exactly like "the physics did not
transfer" - which is the conclusion this project has reached nine times.

Rolls the policy out in Isaac, and at every step feeds the SAME observation to both the torch actor
and the ONNX session. Any disagreement here invalidates every physics comparison downstream,
because the two engines are then not running the same controller at all.

    python isaac_lab_3/scripts/probe_parity.py --checkpoint <model.pt> --onnx Models/locomotion_policy.onnx
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import p4f_newton.tasks  # noqa: F401,E402

from run_conditions import restore  # noqa: E402

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="P4F-Dummy-Walk-Newton-v0")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--onnx", default=str(PROJECT_ROOT / "Models" / "locomotion_policy.onnx"))
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    import gymnasium as gym
    import onnxruntime
    from importlib import metadata
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    restore(env_cfg, args.checkpoint, label="parity")

    env = gym.make(args.task, cfg=env_cfg)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=args.device)

    sess = onnxruntime.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    def tensor_of(x):
        """The wrapper returns either a tensor or a {'policy': tensor} dict depending on version."""
        # TensorDict is NOT a dict subclass, so an isinstance check misses it and the failure
        # only surfaces three calls later as `.numpy()` returning a dict of arrays.
        while not isinstance(x, torch.Tensor) and hasattr(x, "keys"):
            keys = list(x.keys())
            x = x["policy"] if "policy" in keys else x[keys[0]]
        return x

    reset_out = wrapped.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    worst = 0.0
    worst_rel = 0.0
    diffs = []
    for _ in range(args.steps):
        with torch.inference_mode():
            act_torch = policy(obs)
        flat = tensor_of(obs).detach().cpu().numpy().astype(np.float32)
        # One env at a time: the exported graph has a FIXED batch of 1 (dynamic_axes={}).
        for row in range(flat.shape[0]):
            act_onnx = sess.run(None, {in_name: flat[row : row + 1]})[0][0]
            ref = act_torch[row].detach().cpu().numpy()
            d = float(np.abs(act_onnx - ref).max())
            scale = max(float(np.abs(ref).max()), 1e-6)
            diffs.append(d)
            worst = max(worst, d)
            worst_rel = max(worst_rel, d / scale)
        with torch.inference_mode():
            obs = wrapped.step(act_torch)[0]

    diffs_np = np.array(diffs)
    print(f"\n[parity] {len(diffs)} observation vectors through both paths")
    print(f"[parity] |onnx - torch|  median {np.median(diffs_np):.3e}  "
          f"p99 {np.percentile(diffs_np, 99):.3e}  max {worst:.3e}")
    print(f"[parity] worst relative to the action's own magnitude: {worst_rel:.3%}")
    if worst < 1e-4:
        print("[parity] PASS - the two engines are running the same controller.")
    else:
        print("[parity] *** FAIL *** the ONNX Godot runs does NOT match the trained policy. "
              "Every physics comparison made against this export is void.")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
