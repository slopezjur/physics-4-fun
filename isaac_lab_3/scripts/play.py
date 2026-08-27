"""Watch a trained policy — or the zero-action baseline — under Newton/XPBD.

The arena. Normally driven by `watch.ps1`, which resolves the newest checkpoint from `config.ps1`.

Unlike the 2.3.2 `play.py` this needs no Kit, no `--kit_args=--/app/vulkan=false`, and no
several-minute shader-cache build on first launch. `--viewer newton` opens a native OpenGL window;
`--viewer viser` serves one in the browser.

**`--zero_action` is worth using often.** It is the baseline every result on this track is measured
against: commanding nothing but the rest pose, the body is on the floor in under 2 s, exactly as it
is in Godot and unlike PhysX, which holds ~84% standing for 8 s. A trained policy that looks
impressive is only impressive relative to that, and the 2.3.2 track recorded a case where a policy
scored *worse* than doing nothing while every quality metric read healthy.

    python isaac_lab_3/scripts/play.py --checkpoint <abs path to model_N.pt> --num_envs 4
    python isaac_lab_3/scripts/play.py --zero_action --num_envs 4
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import gymnasium as gym
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import p4f_newton.tasks  # noqa: F401,E402  registers the tasks with gymnasium


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=str, default="P4F-Dummy-Stand-Newton-v0")
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--checkpoint", type=str, default="", help="Full path to a model_*.pt.")
    p.add_argument(
        "--zero_action",
        action="store_true",
        help="Command the rest pose instead of loading a policy — the baseline every result here "
        "is measured against.",
    )
    p.add_argument("--viewer", type=str, default="newton", choices=["newton", "viser", "none"])
    p.add_argument(
        "--terminate",
        action="store_true",
        help="Let episodes end and reset. Off by default: the arena is unbounded, which is the "
        "honest test of a policy and also avoids the reset bug in README section 3b.",
    )
    p.add_argument("--seconds", type=float, default=0.0, help="0 runs until the window is closed.")
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def enable_playback_mode() -> None:
    """Disable every termination, so nothing ever resets. The default for the arena.

    Ported in spirit from the Godot track's `PlaybackMode`, which exists for the same reason: what
    training measures is bounded by episode ends, and the honest test of a policy is what it does
    when nothing stops it. The 2.3.2 notes record a policy scoring 0.96 success while, in the
    unbounded arena, it stood 1-2 s and went down.

    Here it also isolates the arena from the reset bug in README §3b — `write_root_pose_to_sim_index`
    does not round-trip, so every reset after the first respawns the dummy 0.82 m in the air. With
    no terminations there are no resets, and what you watch is the physics alone.
    """
    from p4f_newton.tasks.stand.stand_env import StandEnv

    def _no_dones(self):
        never = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._fell = never
        return never, never

    StandEnv._get_dones = _no_dones


def attach_viewer(env_cfg, which: str) -> None:
    if which == "none":
        return
    if which == "newton":
        from isaaclab_visualizers.newton import NewtonVisualizerCfg

        env_cfg.sim.visualizer_cfgs = [NewtonVisualizerCfg()]
    elif which == "viser":
        from isaaclab_visualizers.viser import ViserVisualizerCfg

        env_cfg.sim.visualizer_cfgs = [ViserVisualizerCfg()]


def find_viewer(base):
    """The live `NewtonViewerGL`, or None when running headless or on another backend.

    Reached through `SimulationContext._visualizers`, which is where the sim keeps the visualizer
    instances it built from `sim.visualizer_cfgs`. Neither the cfg object nor `NewtonManager`
    exposes the viewer, so this is the only route - and it is private API, hence the defensive
    lookup: a viser or headless run has no `_viewer` at all, and this must degrade to "no hotkey"
    rather than taking the arena down with an AttributeError.
    """
    for visualizer in getattr(base.sim, "_visualizers", []) or []:
        viewer = getattr(visualizer, "_viewer", None)
        if viewer is not None and hasattr(viewer, "is_key_down"):
            return viewer
    return None


def main() -> None:
    args = parse_args()
    if not args.checkpoint and not args.zero_action:
        raise SystemExit("Pass --checkpoint <model_N.pt> or --zero_action.")

    from isaaclab_tasks.utils import load_cfg_from_registry

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.playback = True  # measure the policy, not the observation noise
    attach_viewer(env_cfg, args.viewer)

    if not args.terminate:
        enable_playback_mode()

    env = gym.make(args.task, cfg=env_cfg)

    policy = None
    if args.checkpoint:
        import importlib.metadata as metadata

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from rsl_rl.runners import OnPolicyRunner

        agent_cfg.device = args.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        wrapped = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=args.device)
        env = wrapped
        print(f"[play] policy: {args.checkpoint}")
    else:
        print("[play] ZERO-ACTION baseline — commanding the rest pose, no policy loaded.")

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    base = env.unwrapped
    zeros = torch.zeros(base.num_envs, base.cfg.action_space, device=base.device)

    # R restarts the episode without leaving the viewer, matching what R does in the Godot arenas.
    # Edge-triggered: `is_key_down` reports the key's STATE, so without tracking the previous frame
    # a single press spanning several frames would fire a reset on each of them.
    viewer = find_viewer(base)
    was_reset_down = False
    if viewer is not None:
        print("[play] press R in the viewer to restart the episode")

    head_id = base.robot.body_names.index("Head")
    steps = int(args.seconds / base.step_dt) if args.seconds > 0 else None
    step = 0
    report_every = max(1, int(0.5 / base.step_dt))
    print(f"  {'t (s)':>7}{'head (m)':>11}{'upright':>10}{'standing':>11}")
    try:
        while steps is None or step < steps:
            # `no_grad`, NOT `inference_mode`. Tensors produced under inference_mode are marked as
            # such, and the env's persistent buffers (`_action`, `_episode_sums`) then refuse the
            # in-place writes `_reset_idx` performs: "Inplace update to inference tensor outside
            # InferenceMode is not allowed". It only bites once an episode actually ends, so an
            # arena with terminations disabled hides it entirely. rsl_rl collects rollouts under
            # no_grad for the same reason.
            with torch.no_grad():
                action = zeros if policy is None else policy(obs)
                # The rsl_rl wrapper returns 4 values; a bare gym env returns 5. Take the first
                # element either way rather than unpacking a fixed arity.
                out = env.step(action)
                obs = out[0]

            if viewer is not None:
                # Polled after the step so the reset is the last thing to touch the state this
                # frame; resetting before `env.step` would have the step immediately overwrite it.
                is_down = bool(viewer.is_key_down("R"))
                if is_down and not was_reset_down:
                    reset_out = env.reset()
                    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                    step = 0
                    print("[play] R - episode restarted")
                was_reset_down = is_down
            if step % report_every == 0:
                com = base.robot.data.body_com_pos_w.torch
                head = com[:, head_id, 2] - base.scene.env_origins[:, 2]
                upright = base.rig.upright()
                standing = ((head >= 1.35) & (upright >= 0.86)).float().mean().item()
                print(f"  {step * base.step_dt:>7.2f}{head.mean().item():>11.3f}"
                      f"{upright.mean().item():>10.3f}{standing * 100.0:>10.1f}%")
            step += 1
    except KeyboardInterrupt:
        print("\n[play] interrupted")

    env.close()


if __name__ == "__main__":
    main()
