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

from run_conditions import apply_overrides, restore  # noqa: E402


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
    p.add_argument("--playback", action="store_true",
                   help="Pin action_scale_range and effort_scale_range to nominal (1.0, 1.0). "
                        "**Measure the policy, not the per-episode actuator draw.** Both ranges are "
                        "resampled every reset, and measured 2026-09-06 three unpinned runs of ONE "
                        "checkpoint scored WALK 3.62 m, HOP 0.28 m and WALK 3.12 m - the verdict "
                        "itself flipped. `evaluate_walk.py` has always set this; this script did "
                        "not, so every single-run score taken through it carried that spread.")
    p.add_argument("--set", dest="sets", action="append", default=[],
                   help="Override one env-cfg field on top of the checkpoint's restored conditions, "
                        "e.g. --set sim.dt=0.0041667 --set decimation=4. Same parser as train.py.")
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
    if args.playback:
        env_cfg.playback = True
    if args.sets:
        apply_overrides(env_cfg, args.sets, label="dump")
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
    # **World foot/pelvis height, matching the columns `IsaacPolicyDriver.DofTracePath` writes.**
    # The 143-float observation carries only contact FLAGS, which are a height threshold — and the
    # two engines use different ones (Godot 0.06 m, Isaac 0.05 m), so the flags cannot be compared
    # directly and cannot show how a foot approaches the ground. These are the raw heights.
    header += ["footZ_L", "footZ_R", "pelvisZ", "footX_L", "footX_R", "pelvisX", "footZfwd_L", "footZfwd_R", "pelvisZfwd"]

    # **Whole-body linear and ORBITAL angular momentum about the system centre of mass.** Positions
    # and angles already agree between the engines and the torso still diverges, so the discriminating
    # quantity is momentum: dL/dt is the net external torque, which separates a reaction-dynamics
    # difference from a kinematic one. Orbital term only — the spin term would need each body's
    # inertia tensor in a common convention, a second unverified mapping, whereas the masses are
    # already verified identical across the rigs.
    #
    # `body_com_lin_vel_w` and `body_com_pos_w` are both on the LIVE side of the XPBD buffer split
    # documented in `p4f_newton/state.py`; the frozen `root_*` buffers are not touched here.
    header += ["comX", "comY", "comZ", "comVx", "comVy", "comVz", "Lx", "Ly", "Lz"]

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

    # Masses are read once: they are static, and `default_mass` is one of the few plant buffers
    # Newton/XPBD reports faithfully because it is never written back by the solver.
    masses = base.robot.data.default_mass
    if hasattr(masses, "torch"):
        masses = masses.torch
    masses = masses.to(args.device).float()
    if masses.dim() == 2:
        masses = masses[args.env_index]
    total_mass = float(masses.sum())

    # Godot uses each body's ORIGIN as its centre of mass, because the rig's bodies are single
    # primitive shapes. Assert that here rather than assume it: if the two disagree the orbital
    # momenta are being computed about different points and are not comparable.
    _com = base.robot.data.body_com_pos_w.torch[args.env_index]
    _link = base.robot.data.body_link_pos_w.torch[args.env_index]
    _off = float((_com - _link).norm(dim=-1).max())
    print(f"[dump] total mass {total_mass:.2f} kg over {masses.numel()} bodies; "
          f"max |body_com_pos_w - body_link_pos_w| = {_off * 1000.0:.2f} mm")

    rows = []
    for step in range(args.steps):
        if forced is not None:
            base._command[:] = forced
        flat = tensor_of(obs).detach().cpu().numpy()
        com = base.robot.data.body_com_pos_w.torch
        z = (com[:, base._contact_ids, 2] - base.scene.env_origins[:, 2].unsqueeze(-1))
        z_xy = (com[:, base._contact_ids, 1] - base.scene.env_origins[:, 1].unsqueeze(-1))
        z_fw = (com[:, base._contact_ids, 0] - base.scene.env_origins[:, 0].unsqueeze(-1))
        # `_contact_ids` order is Hand_L, Hand_R, Foot_L, Foot_R — see `StandEnv._contacts`.
        foot_l = float(z[args.env_index, 2]); foot_r = float(z[args.env_index, 3])
        pelvis_z = float(base.rig.root_pos_w[args.env_index, 2]
                         - base.scene.env_origins[args.env_index, 2])
        # Godot's +X is Isaac's -Y under the rig frame map to_usd(p) = (-p[2], -p[0], p[1]);
        # negated here so "forward" has the same sign in both traces.
        foot_lx = -float(z_xy[args.env_index, 2]); foot_rx = -float(z_xy[args.env_index, 3])
        pelvis_x = -float(base.rig.root_pos_w[args.env_index, 1]
                          - base.scene.env_origins[args.env_index, 1])
        rows.append([step / 60.0] + list(flat[args.env_index].astype(float))
                    + [foot_l, foot_r, pelvis_z, foot_lx, foot_rx, pelvis_x,
                       float(z_fw[args.env_index, 2]), float(z_fw[args.env_index, 3]),
                       float(base.rig.root_pos_w[args.env_index, 0]
                             - base.scene.env_origins[args.env_index, 0])])

        # Momentum is translation-invariant about the system centre of mass, so no env-origin
        # subtraction is needed for comVel or L; the COM position is shifted to env-local so it
        # lines up with the other position columns.
        b_pos = base.robot.data.body_com_pos_w.torch[args.env_index]
        b_vel = base.robot.data.body_com_lin_vel_w.torch[args.env_index]
        mw = masses.unsqueeze(-1)
        com_w = (mw * b_pos).sum(0) / total_mass
        com_v = (mw * b_vel).sum(0) / total_mass
        ang_mom = (mw * torch.cross(b_pos - com_w, b_vel - com_v, dim=-1)).sum(0)
        com_local = com_w - base.scene.env_origins[args.env_index]
        rows[-1].extend([float(v) for v in com_local]
                        + [float(v) for v in com_v]
                        + [float(v) for v in ang_mom])
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
