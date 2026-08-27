"""Score a Stand checkpoint against the criterion ported from `UprightTermination`.

**Do not promote a checkpoint on mean episode length.** Stand terminates only below 0.9 m of head
height or past 70 degrees of tilt, so a policy scoring 470/480 may be crouching, wobbling, or
standing on one leg - the exact failure `UprightTermination` was written to catch, and the reason a
Godot Stand run once read as healthy while scoring 36% on the strict criterion.

The criterion, identical to the 2.3.2 evaluator so the numbers are comparable:

* head >= 1.35 m
* tilt <= 30 degrees (uprightness >= cos 30)
* centre-of-mass speed <= 0.6 m/s
* **held continuously for 1.5 s**

The hold is what makes it a specification rather than a snapshot. At 0.75 s the 2.3.2 track measured
a policy scoring 0.96 that fell over immediately afterwards - the number was accurate and measured
nothing past the moment it stopped looking.

Runs the arena **unbounded** (no terminations, no resets), because that is the honest test: what
training reports is bounded by episode ends, and a policy is only as good as what it does when
nothing stops it.

Read three numbers together, per the 2.3.2 notes:
* **standing success** - the gate.
* **mean head height** - must sit near the 1.540 m rest height. Materially below it and the policy
  is surviving in a crouch, which satisfies tilt and speed while never really standing.
* **mean |action|** - a policy holding itself up on a few percent of its range is balancing; one
  near saturation is bracing, and bracing is what stops transferring.

    python isaac_lab_3/scripts/evaluate_stand.py --checkpoint <abs path to model_N.pt>
    python isaac_lab_3/scripts/evaluate_stand.py --zero_action        # the baseline
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

STANDING_HEAD_HEIGHT = 1.35
STANDING_TILT_DEG = 30.0
STANDING_MAX_SPEED = 0.6
STANDING_HOLD_SECONDS = 1.5
REST_HEAD_HEIGHT = 1.54


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=str, default="P4F-Dummy-Stand-Newton-v0")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--zero_action", action="store_true", help="Score the baseline instead.")
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--json", type=str, default="", help="Also append the result as one JSON line here.")
    p.add_argument(
        "--action_rate_limit",
        type=float,
        default=-1.0,
        help="Per-policy-step cap on each action component. -1 uses the task default. **A policy "
        "trained with a cap must be scored with the same cap** - without it the commands arrive as "
        "steps the policy never learned to issue, and it reads as a total failure.",
    )
    p.add_argument(
        "--effort_scale",
        type=float,
        default=-1.0,
        help="Pin every joint's torque budget to this fraction of nominal. -1 uses the task "
        "default (nominal when measuring). Sweep it to estimate Godot's effective authority.",
    )
    p.add_argument(
        "--push",
        type=float,
        default=-1.0,
        help="Spawn push in m/s. -1 keeps the task's own setting; 0 disables it. **A policy scored "
        "without a push is not being asked to balance** - a statue satisfies every criterion.",
    )
    return p.parse_args()


def main() -> dict:
    args = parse_args()
    if not args.checkpoint and not args.zero_action:
        raise SystemExit("Pass --checkpoint or --zero_action.")

    from isaaclab_tasks.utils import load_cfg_from_registry
    from p4f_newton.tasks.stand.stand_env import StandEnv

    # Unbounded: no terminations, therefore no resets. See the module docstring.
    def _no_dones(self):
        never = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._fell = never
        return never, never

    StandEnv._get_dones = _no_dones

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.playback = True  # measure the policy, not the observation noise
    if args.action_rate_limit >= 0.0:
        env_cfg.action_rate_limit = args.action_rate_limit
    if args.push >= 0.0:
        env_cfg.push_velocity = args.push
        env_cfg.push_ang_velocity = args.push * (5.0 / 3.0)
    if args.effort_scale > 0.0:
        # Pin the torque budget to a fixed fraction of nominal. Sweeping this answers a question
        # nothing else can: how weak is Godot's actuator EFFECTIVELY? Matching `MaxTorque` numbers
        # is not the same as matching authority, and the scale at which Isaac starts failing the way
        # Godot does is the honest estimate of the gap.
        env_cfg.effort_scale_range = (args.effort_scale, args.effort_scale)
        env_cfg.playback = False  # honour the range above; playback would pin it to nominal
        env_cfg.obs_joint_vel_noise = 0.0  # ... but still measure the policy, not the sampler

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

    tilt_cos = math.cos(math.radians(STANDING_TILT_DEG))
    hold_steps = int(STANDING_HOLD_SECONDS / base.step_dt)
    steps = int(args.seconds / base.step_dt)

    consecutive = torch.zeros(base.num_envs, device=base.device)
    succeeded = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    ever_fell = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    head_sum = torch.zeros(base.num_envs, device=base.device)
    action_abs_sum = 0.0
    valid_steps = 0

    for _ in range(steps):
        with torch.no_grad():
            action = zeros if policy is None else policy(obs)
            out = env.step(action)
            obs = out[0]

        data = base.robot.data
        head = data.body_com_pos_w.torch[:, head_id, 2] - base.scene.env_origins[:, 2]
        upright = base.rig.upright()
        speed = torch.linalg.vector_norm(
            data.body_com_lin_vel_w.torch[:, pelvis_id], dim=-1
        )

        # A diverged environment is not a policy failure and must not be scored as one; it is
        # excluded rather than counted as a fall. See StandEnv._diverged.
        finite = torch.isfinite(head) & torch.isfinite(upright) & torch.isfinite(speed)

        ok = (head >= STANDING_HEAD_HEIGHT) & (upright >= tilt_cos) & (speed <= STANDING_MAX_SPEED)
        ok &= finite
        consecutive = torch.where(ok, consecutive + 1, torch.zeros_like(consecutive))
        succeeded |= consecutive >= hold_steps
        ever_fell |= (head < 0.9) & finite
        head_sum += torch.where(finite, head, torch.zeros_like(head))
        # Masked, not a plain mean: this arena never resets, so a single diverged environment's
        # NaN would poison the statistic for the whole run - and it did, reporting `mean |action|`
        # as nan where the previous day's identical run read 0.456.
        finite_action = action[torch.isfinite(action).all(dim=1)]
        if finite_action.numel():
            action_abs_sum += float(finite_action.abs().mean().item())
        valid_steps += 1

    n = float(base.num_envs)
    result = {
        "checkpoint": pathlib.Path(args.checkpoint).name if args.checkpoint else "ZERO-ACTION",
        "standing_success": float(succeeded.float().mean().item()),
        "ever_fell": float(ever_fell.float().mean().item()),
        "mean_head_height": float((head_sum / valid_steps).mean().item()),
        "mean_abs_action": action_abs_sum / max(1, valid_steps),
        "num_envs": base.num_envs,
        "seconds": args.seconds,
        "push": float(env_cfg.push_velocity),
    }

    print(f"  standing success : {result['standing_success'] * 100:6.1f} %   "
          f"(head>={STANDING_HEAD_HEIGHT}, tilt<={STANDING_TILT_DEG:.0f}deg, "
          f"speed<={STANDING_MAX_SPEED}, held {STANDING_HOLD_SECONDS}s)")
    print(f"  ever fell        : {result['ever_fell'] * 100:6.1f} %")
    print(f"  mean head height : {result['mean_head_height']:6.3f} m   (rest {REST_HEAD_HEIGHT})")
    print(f"  mean |action|    : {result['mean_abs_action']:6.3f}     (bracing if near 1.0)")
    print(f"  spawn push       : {env_cfg.push_velocity:6.3f} m/s (0 means the policy is never disturbed)")

    if args.json:
        with open(args.json, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(result) + "\n")

    env.close()
    return result


if __name__ == "__main__":
    main()
