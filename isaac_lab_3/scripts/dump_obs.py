"""Dump Isaac's full 143-float observation under a trained policy, in the same CSV layout Godot writes.

Pairs with `IsaacPolicyDriver.DofTracePath`. Two files in one format make "which of the 143 floats
does Godot present differently?" a diff instead of a guess - which matters because every previous
answer to that question on this project was reached by looking at one slice that had been chosen in
advance, and was wrong.

    python isaac_lab_3/scripts/dump_obs.py --checkpoint <model.pt> --out isaac_walk.csv
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="P4F-Dummy-Walk-Newton-v0")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=720)
    p.add_argument("--env_index", type=int, default=0, help="which env's trace to write")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no_reset_noise", action="store_true",
                   help="zero the reset joint/height noise so the start pose is deterministic and "
                        "matches Godot's, which spawns at exact rest")
    p.add_argument("--command", type=float, default=None,
                   help="force cmd (vx,0,0) every step, to match what Godot sends")
    args = p.parse_args()

    import gymnasium as gym
    from importlib import metadata
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    restore(env_cfg, args.checkpoint, label="dump")
    if args.no_reset_noise:
        # **Otherwise the open-loop plant comparison is confounded at t=0.** Measured: Isaac's
        # reset scatters joints by up to 0.099 rad while Godot spawns at exact rest, so the two
        # rigs start 0.081 rad RMS apart - most of the "divergence" attributed to the plant.
        env_cfg.reset_joint_noise = 0.0
        env_cfg.reset_height_noise = 0.0
        print("[dump] reset noise zeroed; start pose is deterministic")

    env = gym.make(args.task, cfg=env_cfg)
    base = env.unwrapped
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=args.device)

    def tensor_of(x):
        while not isinstance(x, torch.Tensor) and hasattr(x, "keys"):
            keys = list(x.keys())
            x = x["policy"] if "policy" in keys else x[keys[0]]
        return x

    dofs = list(base.robot.joint_names)
    header = ["t", "grav_x", "grav_y", "grav_z", "linVel_x", "linVel_y", "linVel_z",
              "angVel_x", "angVel_y", "angVel_z", "height"]
    header += [f"pos_{d}" for d in dofs] + [f"vel_{d}" for d in dofs]
    header += ["c_HandL", "c_HandR", "c_FootL", "c_FootR"]
    header += [f"act{i}" for i in range(36)] + ["cmd_x", "cmd_y", "cmd_yaw"]

    reset_out = wrapped.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    # **The command ceiling is NOT restored from the checkpoint.** `_cmd_speed_ceiling` is runtime
    # state initialised from cfg, so an eval of a policy trained to a 1.0 m/s ceiling resamples from
    # the STARTING 0.25 - and every Isaac-side reference number taken this way describes a slower
    # walk than the one Godot is asked to reproduce. Forcing the command removes that entirely.
    forced = None
    if args.command is not None:
        forced = torch.zeros((args.num_envs, 3), device=args.device)
        forced[:, 0] = args.command

    rows = []
    for step in range(args.steps):
        if forced is not None:
            base._command[:] = forced
        flat = tensor_of(obs).detach().cpu().numpy()
        rows.append([step / 60.0] + list(flat[args.env_index].astype(float)))
        with torch.inference_mode():
            obs = wrapped.step(policy(obs))[0]

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(f"{v:.4f}" for v in r) + "\n")
    print(f"[dump] {len(rows)} steps x {len(header) - 1} floats -> {out}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
