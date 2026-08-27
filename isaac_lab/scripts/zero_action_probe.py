"""How long does the dummy stay up in Isaac with the policy commanding nothing?

The Isaac half of a controlled comparison against Godot. `IsaacPolicyDriver.ZeroActionBaseline`
answers the same question in Godot; this answers it here, on the same body, from the same pose,
with the same command - all-zero actions, which the action mapping defines as the rest pose.

The point is to separate two very different diagnoses. If Isaac's rest pose holds indefinitely and
Godot's collapses in a second, the engines disagree about static equilibrium and the actuator model
has to be reconciled before any policy can transfer. If BOTH fall on a similar timescale, then the
rest pose is merely a slow topple in both and the trained policy is genuinely balancing in Isaac -
which makes the transfer failure a control problem, not a statics one.

    python isaac_lab/scripts/zero_action_probe.py [--seconds 8] [--task stand]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="stand", choices=["stand", "walk", "perturb", "run"])
parser.add_argument("--seconds", type=float, default=8.0)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument(
    "--action_scale_probe",
    type=float,
    default=0.0,
    help="Constant action applied to every joint. 0 is the rest pose; use a small value to probe "
    "how much authority the actuators actually have.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import p4f_isaac.tasks  # noqa: F401, E402

TASKS = {
    "stand": ("P4F-Dummy-Stand-Direct-v0", "stand.stand_env_cfg", "StandEnvCfg"),
    "walk": ("P4F-Dummy-Walk-Direct-v0", "walk.walk_env_cfg", "WalkEnvCfg"),
    "perturb": ("P4F-Dummy-Perturb-Direct-v0", "perturb.perturb_env_cfg", "PerturbEnvCfg"),
    "run": ("P4F-Dummy-Run-Direct-v0", "run.run_env_cfg", "RunEnvCfg"),
}


def main() -> None:
    import importlib

    task_id, cfg_mod, cfg_cls = TASKS[args.task]
    env_cfg = getattr(importlib.import_module(f"p4f_isaac.tasks.{cfg_mod}"), cfg_cls)()
    env_cfg.scene.num_envs = args.num_envs

    env = gym.make(task_id, cfg=env_cfg)
    base = env.unwrapped
    robot = base.robot

    env.reset()

    steps = int(args.seconds / base.step_dt)
    action = torch.full(
        (base.num_envs, base.cfg.action_space), args.action_scale_probe, device=base.device
    )

    print("=" * 64)
    print(f"  ZERO-ACTION PROBE - {args.task}, {base.num_envs} bodies, action={args.action_scale_probe}")
    print("=" * 64)
    print(f"{'t (s)':>8}{'head (m)':>11}{'pelvis (m)':>12}{'upright':>10}{'standing %':>12}")

    report_every = max(1, int(0.5 / base.step_dt))
    for step in range(steps):
        with torch.inference_mode():
            env.step(action)

        if step % report_every == 0 or step == steps - 1:
            data = robot.data
            head = (data.body_com_pos_w[:, base._head_id[0], 2] - base.scene.env_origins[:, 2])
            pelvis = data.root_pos_w[:, 2] - base.scene.env_origins[:, 2]
            upright = -data.projected_gravity_b[:, 2]
            # 1.35 is STANDING_HEAD_HEIGHT; the same bar the Godot arena and the evaluators use.
            standing = ((head >= 1.35) & (upright >= 0.86)).float().mean().item() * 100.0
            print(f"{step * base.step_dt:>8.2f}{head.mean().item():>11.3f}"
                  f"{pelvis.mean().item():>12.3f}{upright.mean().item():>10.3f}{standing:>11.1f}%")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    os._exit(0)
