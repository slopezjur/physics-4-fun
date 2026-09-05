"""Isaac's actuator telemetry under a trained policy, in Godot's units.

**The regime is the whole point.** Standing and walking are different mechanical problems, and the
two engines' actuators agree in one and not the other. Measured in Godot under the walk policy:

    standing       fvScale 0.97-1.00      demand 0.04-0.15
    diverging      fvScale 0.38           demand 0.83

Every earlier comparison between the engines was taken at rest, where Godot's Hill derating and
effort clamp are both inert, so they were dismissed as "not the blocker". That dismissal cannot be
transferred to a gait: both bind hard during the transient, which is exactly the regime that fails.

This runs the SAME policy in Isaac and reports the same two numbers, so the actuators can be
compared under motion. `fvScale` is the worst Hill derating on the body; `demand` is the peak torque
request as a fraction of a bone's ceiling. Both come from `StandEnv._effort_limited`, which is where
Isaac reproduces Godot's clamp.

    python isaac_lab_3/scripts/probe_actuator.py --checkpoint <model.pt> --task P4F-Dummy-Walk-Newton-v0

Run it with `P4F_XPBD_ITERATIONS` set to the value the checkpoint trained at - iterations are part of
the plant, and probing at the module default while training ran at 8 has produced a wrong conclusion
here before.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import statistics
import sys

import gymnasium as gym
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import p4f_newton.tasks  # noqa: F401,E402  registers the tasks with gymnasium

from run_conditions import restore  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=str, default="P4F-Dummy-Walk-Newton-v0")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    from importlib import metadata

    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    # The plant the checkpoint trained on, not the task defaults - see run_conditions.
    restore(env_cfg, args.checkpoint, label="probe")

    env = gym.make(args.task, cfg=env_cfg)
    base = env.unwrapped

    # Same construction as `evaluate_walk.py`. The rsl-rl config needs the deprecation shim for this
    # library version; building the runner without it raises on `MLPModel(stochastic=...)`.
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=args.device)

    reset_out = wrapped.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    steps = int(args.seconds * 60)
    hill, demand, jv = [], [], []
    for _ in range(steps):
        with torch.inference_mode():
            step_out = wrapped.step(policy(obs))
        obs = step_out[0]
        hill.append(float(base._last_hill))
        demand.append(float(base._last_demand))
        # Worst joint speed on the body, the same quantity Godot's driver prints as `jointVel`.
        # It is also observation slice [55:100], so a mismatch here is a mismatch in the policy's
        # INPUT, not just in the plant.
        jv.append(float(base.rig.joint_state()[1].abs().amax()))

    def pct(values: list[float], q: float) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    print(f"\n[actuator] {len(hill)} policy steps, {args.num_envs} envs, "
          f"xpbd iterations {os.environ.get('P4F_XPBD_ITERATIONS', '2 (module default!)')}")
    print(f"[actuator] fvScale  min {min(hill):.3f}  p05 {pct(hill, 0.05):.3f}  "
          f"median {statistics.median(hill):.3f}  max {max(hill):.3f}")
    print(f"[actuator] demand   median {statistics.median(demand):.3f}  "
          f"p95 {pct(demand, 0.95):.3f}  max {max(demand):.3f}")
    print(f"[actuator] jointVel median {statistics.median(jv):.2f}  p95 {pct(jv, 0.95):.2f}  "
          f"max {max(jv):.2f} rad/s   (observation clip is 15.0)")
    print("[actuator] GODOT, same policy: standing fvScale 0.97-1.00 demand 0.04-0.15; "
          "diverging fvScale 0.38 demand 0.83; jointVel PINNED at the 15.0 clip")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
