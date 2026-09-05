"""Drive ISAAC open-loop from a recorded action CSV — the mirror of Godot's `ReplayActionsPath`.

**Without this, no plant comparison is clean.** Godot can be driven from a recorded action sequence
(`IsaacPolicyDriver.ReplayActionsPath`) but Isaac could only be run closed-loop, so every
"Godot vs Isaac" trace differed by BOTH the body and the policy's response to it. That confounded
the geometry A/B on 2026-09-05: the open-loop divergence looked worse after the collider fix, and the
comparison could not say whether that was the geometry or a policy still adapting to it.

With both engines replayable, a plant change can be tested properly: hold the action sequence fixed,
change one thing, and see what the BODY does.

    python isaac_lab_3/scripts/replay_isaac.py --actions replay.csv --out iso_replay.csv

The CSV format is the one `godot_walk_score`/`dump_obs` already write: a header of `act0..act35` and
one row per policy step. Output matches `dump_obs.py` column-for-column so the existing analysis
scripts read it unchanged.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import p4f_newton.tasks  # noqa: F401,E402

from run_conditions import restore  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="P4F-Dummy-Walk-Newton-v0")
    p.add_argument("--actions", required=True, help="CSV with act0..act35, one row per policy step")
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default="", help="only to restore the trained plant conditions")
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--command", type=float, default=0.30)
    args = p.parse_args()

    import gymnasium as gym
    from isaaclab_tasks.utils import load_cfg_from_registry

    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    # Deterministic start, so the replay is reproducible and starts where Godot's does.
    cfg.reset_joint_noise = 0.0
    cfg.reset_height_noise = 0.0
    # **Randomisation off for a plant comparison.** `action_scale_range` defaults to (0.8, 1.25),
    # so every episode scales the commanded target by a random draw - which Godot has no equivalent
    # of, and which silently inflated Isaac's excursions by up to 25% in the first A/B run here.
    cfg.action_scale_range = (1.0, 1.0)
    cfg.effort_scale_range = (1.0, 1.0)
    if args.checkpoint:
        restore(cfg, args.checkpoint, label="replay")

    rows = list(csv.DictReader(open(args.actions, encoding="utf-8")))
    n_act = cfg.action_space
    actions = torch.tensor(
        [[float(r[f"act{i}"]) for i in range(n_act)] for r in rows],
        dtype=torch.float32, device=args.device,
    )
    print(f"[replay] {len(actions)} recorded action rows, {n_act} DOF")

    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    env.reset()

    forced = torch.zeros((args.num_envs, 3), device=args.device)
    forced[:, 0] = args.command

    dofs = list(base.robot.joint_names)
    header = ["t", "grav_x", "grav_y", "grav_z", "linVel_x", "linVel_y", "linVel_z",
              "angVel_x", "angVel_y", "angVel_z", "height"]
    header += [f"pos_{d}" for d in dofs] + [f"vel_{d}" for d in dofs]
    header += ["c_HandL", "c_HandR", "c_FootL", "c_FootR"]
    header += [f"act{i}" for i in range(n_act)] + ["cmd_x", "cmd_y", "cmd_yaw"]

    out_rows = []
    for step in range(len(actions)):
        base._command[:] = forced
        a = actions[step].unsqueeze(0).expand(args.num_envs, -1)
        obs = env.step(a)[0]
        while not isinstance(obs, torch.Tensor) and hasattr(obs, "keys"):
            keys = list(obs.keys())
            obs = obs["policy"] if "policy" in keys else obs[keys[0]]
        out_rows.append([step / 60.0] + list(obs[0].detach().cpu().numpy().astype(float)))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(header) + "\n")
        for r in out_rows:
            fh.write(",".join(f"{v:.4f}" for v in r) + "\n")
    print(f"[replay] {len(out_rows)} steps -> {out}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
