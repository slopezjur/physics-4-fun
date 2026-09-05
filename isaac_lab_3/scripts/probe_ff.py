"""Per-bone gravity feed-forward in Isaac, for diffing against Godot's.

`gravity_ff.py` reproduces `ActiveBone.ComputeLoadCompensationTorque`. A whole-body outcome cannot
tell a correct feed-forward from one that is half the right size or points somewhere plausible but
wrong - both of those were live bugs in this module on 2026-09-03, and one of them made the term
actively harmful while still looking like "a gravity compensator". So compare per bone, against the
values Godot prints for the same pose.

    # Godot side (writes `[PLANT] ... ffMag=` for every bone):
    Godot_v4.7.1-stable_mono_win64.exe --path . \
        "res://Scenes/RL/Isaac3/Walk/IsaacWalkCheckNewton.tscn" > godot.log 2>&1

    # Isaac side:
    python isaac_lab_3/scripts/probe_ff.py --godot godot.log

Both must be in the SAME pose. The default here is the reset pose held for a moment under zero
action, which is the standing rest pose both engines spawn in.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

import gymnasium as gym
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import p4f_newton.tasks  # noqa: F401,E402  registers the tasks with gymnasium

_GODOT_RE = re.compile(r"\[PLANT\] bone=(?P<bone>\S+).* ffMag=(?P<ff>\S+) supportShare=(?P<share>\S+)")


def read_godot(path: pathlib.Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for match in _GODOT_RE.finditer(path.read_text(encoding="utf-8", errors="replace")):
        out[match.group("bone")] = float(match.group("ff").replace(",", "."))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=str, default="P4F-Dummy-Stand-Newton-v0")
    p.add_argument("--godot", type=pathlib.Path, default=None, help="Godot log with [PLANT] lines.")
    p.add_argument("--settle", type=int, default=20, help="Zero-action steps before reading.")
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    from isaaclab_tasks.utils import load_cfg_from_registry

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 16
    env_cfg.sim.device = args.device
    env_cfg.push_velocity = 0.0
    if hasattr(env_cfg, "balance_assist"):
        env_cfg.balance_assist = 0.0
    env_cfg.gravity_feedforward = 1.0

    env = gym.make(args.task, cfg=env_cfg)
    base = env.unwrapped
    env.reset()

    zero = torch.zeros(base.num_envs, base.cfg.action_space, device=base.device)
    for _ in range(args.settle):
        env.step(zero)

    ff = base._gravity_ff
    if ff is None:
        raise SystemExit("gravity_feedforward is off; nothing to probe.")

    torque = ff.torques(
        base.rig.body_link_pos_w_all(), base.rig.body_quat_w_all(), base._contacts()
    )
    magnitude = torque.norm(dim=-1).mean(dim=0)

    contacts = base._contacts().mean(dim=0)
    print("[ff] contacts (mean over envs): "
          + ", ".join(f"{n}={v:.2f}" for n, v in zip(base._contact_ids and
                                                     [base.robot.body_names[i] for i in base._contact_ids],
                                                     contacts.tolist())))

    # **Pose check before torque check.** Two rigs that disagree about where a limb is disagree
    # about every lever arm, and that is indistinguishable from a broken feed-forward. Godot's rest
    # pose is exactly symmetric (arms +/-0.360, legs +/-0.140), which is why its torso feed-forward
    # is ~0: left and right cancel. If Isaac's is not symmetric, no pivot fix will make the torques
    # agree.
    link = base.rig.body_link_pos_w_all()[0] - base.scene.env_origins[0]
    print("\n[ff] Isaac link positions (x, y, z), env 0:")
    for name in ("Pelvis", "Spine", "Chest", "UpperArm_L", "UpperArm_R", "Thigh_L", "Thigh_R",
                 "Foot_L", "Foot_R"):
        if name in base.robot.body_names:
            p_ = link[base.robot.body_names.index(name)]
            print(f"    {name:<12} {p_[0]:+.4f} {p_[1]:+.4f} {p_[2]:+.4f}")
    print()

    godot = read_godot(args.godot) if args.godot else {}
    header = f"{'bone':<12}{'isaac':>10}{'godot':>10}{'ratio':>9}"
    print(header)
    print("-" * len(header))
    worst = (None, 0.0)
    for row, bone in enumerate(ff._bones):
        isaac = magnitude[row].item()
        if bone in godot:
            ref = godot[bone]
            ratio = isaac / ref if ref > 1e-6 else float("inf") if isaac > 1e-6 else 1.0
            # Only bones carrying a real load are worth ranking: a 40x ratio on two torques that are
            # both a thousandth of a N.m says nothing about whether the model is right.
            if ref > 1.0 and abs(ratio - 1.0) > abs(worst[1] - 1.0):
                worst = (bone, ratio)
            print(f"{bone:<12}{isaac:>10.3f}{ref:>10.3f}{ratio:>9.2f}")
        else:
            print(f"{bone:<12}{isaac:>10.3f}{'-':>10}{'-':>9}")

    if worst[0]:
        print(f"\n[ff] worst load-bearing mismatch: {worst[0]} at {worst[1]:.2f}x Godot")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
