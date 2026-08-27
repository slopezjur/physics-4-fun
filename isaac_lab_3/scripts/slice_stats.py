"""Print the same per-slice observation diagnostic Godot's IsaacPolicyDriver prints.

The first-step observation already matches between the engines to three decimals, and the dummy
still falls in Godot while the same policy holds 100% in Isaac. A single matching step proves the
mapping and proves nothing about the trajectory, so this produces the one thing that can be
compared directly against Godot's running diagnostic: the largest magnitude in each observation
slice, at the same wall-clock offsets.

The point is to localise the divergence to a slice. The observation normaliser is baked into the
exported graph, so a slice whose values run far outside what training saw is amplified into a
saturated action - and the action magnitude alone cannot say which slice did it. Godot reaches
|a| > 3 within a second and a half against a clamp of 1; Isaac's stays near 0.46 forever. One of
these slices explains that.

    python isaac_lab_3/scripts/slice_stats.py --checkpoint <abs path to model_N.pt>

Read it beside a Godot run of Scenes/RL/Isaac3/Stand/IsaacStandCheckNewton.tscn - the columns are in the
same order and mean the same thing.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import pathlib
import sys

import gymnasium as gym
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import p4f_newton.tasks  # noqa: F401,E402  registers the tasks with gymnasium

# Contract slice boundaries, matching IsaacObservation.cs. Named rather than sliced inline so a
# layout change is a one-line edit here instead of a silent off-by-three.
SLICES = {
    "gravity": (0, 3),
    "linVel": (3, 6),
    "angVel": (6, 9),
    "height": (9, 10),
    "jointPos": (10, 55),
    "jointVel": (55, 100),
    "contacts": (100, 104),
    "prevAct": (104, 140),
    "command": (140, 143),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=str, default="P4F-Dummy-Stand-Newton-v0")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--zero_action", action="store_true")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--interval", type=float, default=0.5, help="Seconds between report lines.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Remove every source of spawn randomness, for an open-loop comparison against Godot.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint and not args.zero_action:
        raise SystemExit("Pass --checkpoint or --zero_action.")

    from isaaclab_tasks.utils import load_cfg_from_registry
    from p4f_newton.tasks.stand.stand_env import StandEnv

    # Unbounded, exactly as evaluate_stand.py runs it: terminations would reset a falling body and
    # hide the very trajectory being compared. Godot's arena has no resets either.
    def _no_dones(self):
        never = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._fell = never
        return never, never

    StandEnv._get_dones = _no_dones

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.playback = True  # measure the policy, not the observation noise
    if args.deterministic:
        # Open-loop plant comparison: both engines must start from the SAME body, or the
        # trajectories are not comparable step by step.
        env_cfg.reset_joint_noise = 0.0
        env_cfg.reset_height_noise = 0.0
        env_cfg.push_velocity = 0.0

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
    zeros = torch.zeros(base.num_envs, base.cfg.action_space, device=base.device)

    # Whole-body centre of mass against the foot midpoint - the same quantity Godot's driver
    # reports as `comOff`. **This is the balance question the joint-level diagnostics miss.**
    # Zero-action traces show the two engines failing differently: Isaac sags, Godot tips, and a
    # topple means the centre of mass is leaving the support polygon regardless of joint control.
    masses = base.robot.data.default_mass.torch[0].to(base.device)
    total_mass = masses.sum()
    foot_ids = [base.robot.body_names.index(n) for n in ("Foot_L", "Foot_R")]

    steps = int(args.seconds / base.step_dt)
    every = max(1, int(args.interval / base.step_dt))

    for step in range(steps):
        with torch.no_grad():
            action = zeros if policy is None else policy(obs)
            out = env.step(action)
            obs = out[0]

        if step % every:
            continue

        # Mean across environments of each environment's own worst slot. Godot reports one body's
        # max; averaging the per-body maxima keeps the same meaning rather than reporting the single
        # unluckiest environment, which would drift upward purely with num_envs.
        flat = obs if isinstance(obs, torch.Tensor) else obs["policy"]
        cols = []
        for name, (lo, hi) in SLICES.items():
            cols.append(f"{name}={flat[:, lo:hi].abs().amax(dim=1).mean().item():.2f}")
        head = (base.robot.data.body_com_pos_w.torch[:, head_id, 2]
                - base.scene.env_origins[:, 2]).mean().item()
        worst = action.abs().amax(dim=1).mean().item()

        # The same quantity Godot's driver reports as `trackErr`: how far each joint sits from the
        # angle the policy commanded. **This is the acceptance test for whether the two plants are
        # the same**, and it is far more diagnostic than a standing percentage - a policy can score
        # well in both engines while its commands are reaching one body and not the other.
        # Grouped per bone so the magnitude is comparable with Godot's per-bone vector error.
        actuated = base._actuated_ids
        error = (base._joint_target[:, actuated] - base._joint_pos[:, actuated])
        per_bone = error.view(base.num_envs, -1, 3).norm(dim=-1)
        com = (base.robot.data.body_com_pos_w.torch * masses.view(1, -1, 1)).sum(dim=1) / total_mass
        feet = base.robot.data.body_com_pos_w.torch[:, foot_ids]
        mid = feet.mean(dim=1)
        com_off = torch.linalg.vector_norm(com[:, :2] - mid[:, :2], dim=1).mean().item()
        foot_h = (feet[:, :, 2].amin(dim=1) - base.scene.env_origins[:, 2]).mean().item()

        print(f"[isaac t={step * base.step_dt:5.2f}s] " + " ".join(cols)
              + f" | maxAction={worst:.2f} head={head:.3f}"
              + f" trackErr={per_bone.mean().item():.2f}/{per_bone.amax(dim=1).mean().item():.2f}rad"
              + f" comOff={com_off:.3f}m feet={foot_h:.3f}m"
              + f" mass={total_mass.item():.1f}kg comY={(com[:, 2] - base.scene.env_origins[:, 2]).mean().item():.3f}m",
              flush=True)

    env.close()


if __name__ == "__main__":
    main()
