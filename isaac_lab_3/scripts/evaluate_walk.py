"""Score a Walk checkpoint against a commanded velocity.

Walk has **no success condition, deliberately** - `docs/RL-SESSION-INVARIANTS.md` records why:
walking is sustained behaviour with no goal state, and a distance threshold would end the episode at
the exact moment the agent is doing the thing being trained. So this reports a profile rather than a
pass mark, and the numbers only mean something read together:

* **tracked speed** - metres per second along the commanded direction. The headline.
* **distance** - net displacement. A policy that oscillates in place can hold a respectable
  instantaneous speed and go nowhere.
* **upright fraction** - share of steps above the fall thresholds. Diving forward produces speed
  too; this is what separates walking from falling with style.
* **steps taken** - touchdowns after a real flight, from the same height proxy the reward uses.
  **Near zero with a healthy tracked speed means the dummy is SLIDING**, which is the failure the
  air-time term exists to prevent and the one an additive reward produced in 2.3.2.

Every environment is given the SAME command, so the numbers are a profile of one behaviour rather
than an average over a random command distribution that hides which commands fail.

    python isaac_lab_3/scripts/evaluate_walk.py --checkpoint <abs path> --speed 0.8
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import math
import pathlib
import sys

import gymnasium as gym
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import p4f_newton.tasks  # noqa: F401,E402  registers the tasks with gymnasium

FALL_HEAD_HEIGHT = 0.9
FALL_TILT_DEG = 70.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=str, default="P4F-Dummy-Walk-Newton-v0")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--zero_action", action="store_true")
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--speed", type=float, default=0.8, help="Commanded forward speed, m/s.")
    p.add_argument("--yaw", type=float, default=0.0, help="Commanded yaw rate, rad/s.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--json", type=str, default="")
    return p.parse_args()


def main() -> dict:
    args = parse_args()
    if not args.checkpoint and not args.zero_action:
        raise SystemExit("Pass --checkpoint or --zero_action.")

    from isaaclab_tasks.utils import load_cfg_from_registry
    from p4f_newton.tasks.walk.walk_env import WalkEnv

    # Unbounded, as evaluate_stand.py runs: what training reports is bounded by episode ends, and a
    # gait is only as good as what it does when nothing stops it.
    def _no_dones(self):
        never = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._fell = never
        return never, never

    # One fixed command for every environment, so this profiles a behaviour rather than averaging
    # over a random command distribution that would hide which commands fail.
    def _fixed_command(self, env_ids) -> None:
        self._command[env_ids, 0] = args.speed
        self._command[env_ids, 1] = 0.0
        self._command[env_ids, 2] = args.yaw

    WalkEnv._get_dones = _no_dones
    WalkEnv._resample_commands = _fixed_command

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.playback = True  # measure the policy, not the observation noise

    env = gym.make(args.task, cfg=env_cfg)
    policy = None
    if args.checkpoint:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from rsl_rl.runners import OnPolicyRunner

        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg.device = args.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=args.device)

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    base = env.unwrapped
    head_id = base.robot.body_names.index("Head")
    pelvis_id = base.robot.body_names.index("Pelvis")
    zeros = torch.zeros(base.num_envs, base.cfg.action_space, device=base.device)

    start = base.rig.root_pos_w.clone()
    tilt_cos = math.cos(math.radians(FALL_TILT_DEG))
    steps = int(args.seconds / base.step_dt)

    tracked = torch.zeros(base.num_envs, device=base.device)
    upright_steps = torch.zeros(base.num_envs, device=base.device)
    touchdowns = torch.zeros(base.num_envs, device=base.device)
    # Mirrors WalkEnv's own bookkeeping so the two cannot disagree about what a step is. The env
    # updates `_feet_grounded` inside its reward, which still runs here - only `_get_dones` was
    # replaced.
    was_grounded = base._feet_grounded.clone()
    action_abs = 0.0
    counted = 0

    for _ in range(steps):
        with torch.no_grad():
            action = zeros if policy is None else policy(obs)
            obs = env.step(action)[0]

        data = base.robot.data
        vel = data.body_com_lin_vel_w.torch[:, pelvis_id, :2]
        # Along the commanded direction in WORLD terms: the command is +x in the heading frame and
        # the dummy starts facing +x, so for a straight-line command these agree. A yaw command
        # makes this an underestimate, which is why `distance` is reported beside it.
        tracked += vel[:, 0]

        head = data.body_com_pos_w.torch[:, head_id, 2] - base.scene.env_origins[:, 2]
        ok = (head >= FALL_HEAD_HEIGHT) & (base.rig.upright() >= tilt_cos)
        upright_steps += ok.float()

        grounded = base._feet_grounded
        touchdowns += (grounded & ~was_grounded).float().sum(dim=1)
        was_grounded = grounded.clone()
        action_abs += float(action.abs().mean().item())
        counted += 1

    travelled = (base.rig.root_pos_w - start)[:, :2]
    result = {
        "checkpoint": pathlib.Path(args.checkpoint).name if args.checkpoint else "ZERO-ACTION",
        "commanded_speed": args.speed,
        "tracked_speed": float((tracked / counted).mean().item()),
        "distance": float(torch.linalg.vector_norm(travelled, dim=1).mean().item()),
        "upright_fraction": float((upright_steps / counted).mean().item()),
        "steps_taken": float(touchdowns.mean().item()),
        "mean_abs_action": action_abs / max(1, counted),
        "num_envs": base.num_envs,
        "seconds": args.seconds,
    }

    print(f"  commanded speed  : {result['commanded_speed']:6.2f} m/s")
    print(f"  tracked speed    : {result['tracked_speed']:6.3f} m/s")
    print(f"  distance         : {result['distance']:6.3f} m   over {args.seconds:.0f}s")
    print(f"  upright fraction : {result['upright_fraction'] * 100:6.1f} %")
    print(f"  steps taken      : {result['steps_taken']:6.1f}     "
          f"(near zero with real speed means SLIDING, not walking)")
    print(f"  mean |action|    : {result['mean_abs_action']:6.3f}")

    if args.json:
        with open(args.json, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(result) + "\n")

    env.close()
    return result


if __name__ == "__main__":
    main()
