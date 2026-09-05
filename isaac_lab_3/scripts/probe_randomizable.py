"""Which plant properties can actually be randomised per environment on the Newton/XPBD backend?

**Nothing here is assumed from an API's existence.** On this backend half the state buffers are
frozen and read a plausible zero, and `rand_*` randomisation was dropped from the 2.3.2 config
because it went through `robot.root_physx_view`, which Newton does not have. A knob that accepts a
write and changes nothing is worse than no knob: it produces a training run that looks randomised
and is not, which is exactly the class of mistake `assets.py`'s inert `P4F_GAIN_SCALE` was.

So each candidate is written with DIFFERENT values per environment, read back, and then the envs are
stepped to see whether their behaviour actually diverges. Only a candidate that passes both is
usable.

    python isaac_lab_3/scripts/probe_randomizable.py
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import p4f_newton.tasks  # noqa: F401,E402


def report(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name:34s} {detail}")


def main() -> int:
    import gymnasium as gym
    from isaaclab_tasks.utils import load_cfg_from_registry

    cfg = load_cfg_from_registry("P4F-Dummy-Walk-Newton-v0", "env_cfg_entry_point")
    cfg.scene.num_envs = 8
    cfg.sim.device = "cuda:0"
    env = gym.make("P4F-Dummy-Walk-Newton-v0", cfg=cfg)
    base = env.unwrapped
    robot = base.robot
    n = cfg.scene.num_envs

    print(f"\nrobot type: {type(robot).__name__}")
    print("write-ish methods:", [m for m in dir(robot) if m.startswith("write_")][:20])

    # --- masses -------------------------------------------------------------------
    try:
        masses = robot.data.default_mass.torch.clone()
        scale = torch.linspace(0.7, 1.3, n, device=masses.device).view(n, 1)
        target = masses * scale
        wrote = False
        for method in ("write_mass_to_sim", "set_mass", "write_body_mass_to_sim"):
            if hasattr(robot, method):
                getattr(robot, method)(target)
                wrote = True
                break
        if not wrote:
            report("body mass", False, "no write method found on the articulation")
        else:
            back = robot.data.default_mass.torch
            spread = float((back[0] - back[-1]).abs().max())
            report("body mass", spread > 1e-4, f"per-env spread after readback: {spread:.4f} kg")
    except Exception as e:  # noqa: BLE001
        report("body mass", False, f"{type(e).__name__}: {e}")

    # --- joint stiffness / damping -------------------------------------------------
    for label, attr, writer in (
        ("joint stiffness", "joint_stiffness", "write_joint_stiffness_to_sim"),
        ("joint damping", "joint_damping", "write_joint_damping_to_sim"),
        ("joint effort limit", "joint_effort_limits", "write_joint_effort_limit_to_sim"),
    ):
        try:
            if not hasattr(robot, writer):
                report(label, False, f"no {writer}")
                continue
            cur = getattr(robot.data, attr).torch.clone()
            scale = torch.linspace(0.5, 1.5, n, device=cur.device).view(n, 1)
            getattr(robot, writer)(cur * scale)
            back = getattr(robot.data, attr).torch
            spread = float((back[0] - back[-1]).abs().max())
            report(label, spread > 1e-6, f"per-env spread after readback: {spread:.4f}")
        except Exception as e:  # noqa: BLE001
            report(label, False, f"{type(e).__name__}: {e}")

    # --- friction -------------------------------------------------------------------
    try:
        mats = getattr(robot.root_physx_view, "get_material_properties", None)
        report("friction (root_physx_view)", False,
               "root_physx_view exists" if mats else "no root_physx_view - as recorded for Newton")
    except Exception as e:  # noqa: BLE001
        report("friction (root_physx_view)", False, f"{type(e).__name__}: {e}")

    # --- does anything actually BITE? -------------------------------------------------
    # **A divergence metric does not work here.** With zero actions the body falls, and a falling
    # body is chaotic: measured, 8 environments with reset noise OFF and nothing randomised still
    # spread 0.238 rad after 180 steps, purely from non-deterministic GPU reductions amplified by
    # the fall. Any real effect is invisible under that.
    #
    # So hold the body against a CONSTANT target instead and ask whether the steady deviation
    # varies MONOTONICALLY with the scale written to each environment. A knob that bites produces a
    # near-perfect rank correlation with its own scale; an inert one produces noise around zero.
    print("  --- mechanical effect: rank correlation of joint deviation with the written scale ---")
    env.close()

    def bite(writer_name: str, attr: str, lo: float, hi: float) -> str:
        c = load_cfg_from_registry("P4F-Dummy-Walk-Newton-v0", "env_cfg_entry_point")
        c.scene.num_envs, c.sim.device = 16, "cuda:0"
        c.reset_joint_noise = 0.0
        c.reset_height_noise = 0.0
        c.effort_scale_range = (1.0, 1.0)
        e = gym.make("P4F-Dummy-Walk-Newton-v0", cfg=c)
        b = e.unwrapped
        e.reset()
        k = c.scene.num_envs
        if not hasattr(b.robot, writer_name):
            e.close()
            return "no writer"
        cur = getattr(b.robot.data, attr).torch.clone()
        scale = torch.linspace(lo, hi, k, device=cur.device)
        getattr(b.robot, writer_name)(cur * scale.view(k, 1))
        # A constant, sizeable, one-sided command so every joint is pushed off rest and HELD.
        a = torch.full((k, c.action_space), 0.6, device=b.device)
        for _ in range(90):
            e.step(a)
        q, _ = b.rig.joint_state()
        dev = q.abs().mean(dim=1)                      # mean |deviation| per environment
        e.close()
        x = scale.double() - scale.double().mean()
        y = dev.double() - dev.double().mean()
        r = float((x * y).sum() / ((x * x).sum().sqrt() * (y * y).sum().sqrt() + 1e-12))
        return f"r = {r:+.3f}   {'BITES' if abs(r) > 0.7 else 'INERT'}   (dev {float(dev.min()):.4f}..{float(dev.max()):.4f})"

    for label, writer, attr in (
        ("joint stiffness", "write_joint_stiffness_to_sim", "joint_stiffness"),
        ("joint damping", "write_joint_damping_to_sim", "joint_damping"),
        ("joint effort limit", "write_joint_effort_limit_to_sim", "joint_effort_limits"),
        ("joint armature", "write_joint_armature_to_sim", "joint_armature"),
    ):
        try:
            print(f"    {label:22s}: {bite(writer, attr, 0.5, 1.5)}")
        except Exception as exc:  # noqa: BLE001
            print(f"    {label:22s}: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
