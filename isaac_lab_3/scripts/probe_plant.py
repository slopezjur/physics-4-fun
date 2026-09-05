"""Step response of ONE joint, in Isaac. The Godot half lives in `Scenes/RL/Isaac3/Probe`.

**Why this exists.** `godot_plant.py` scales Isaac's `stiffness`/`damping` by the factor Godot's
Stable PD applies to its own authored gains. That corrects a real, measured discrepancy - but it
assumes `ke` in Newton's XPBD drive and `Kp` in Godot's SPD are the same currency, and they are not
obviously so: XPBD applies the drive as a compliant positional constraint INSIDE the solve, while
Godot computes an explicit torque and applies it as a force. `assets.py` says as much in its own
comment and the scaling was done anyway.

Consequence of getting it wrong, measured 2026-09-03: with the gains scaled, Isaac's body CROUCHES
(52.7% standing, head 1.459 against 1.540 upright) while Godot's, stripped of its gravity
feed-forward, COLLAPSES outright. Both are "too weak to hold the body up", but not equally, and no
comparison of parameter NAMES can say by how much.

So compare DELIVERED dynamics: command one joint a step from rest and record where the angle
actually goes. Rise time gives the effective natural frequency, overshoot the damping ratio, and the
steady-state offset the stiffness against whatever load is present. Run the matching Godot probe
with the same joint, amplitude and gravity setting, and the two traces are directly comparable.

Gravity is OFF by default so the drive is the only thing moving the limb; pass `--gravity` to
include the gravity load, which is the condition the feed-forward exists to handle.

    python isaac_lab_3/scripts/probe_plant.py --joint joint_Forearm_L:0 --target 0.30
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import gymnasium as gym
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import p4f_newton.tasks  # noqa: F401,E402  registers the tasks with gymnasium


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=str, default="P4F-Dummy-Stand-Newton-v0")
    p.add_argument("--joint", type=str, default="joint_Forearm_L:0")
    p.add_argument("--target", type=float, default=0.30, help="Commanded joint angle (rad).")
    p.add_argument("--settle", type=int, default=150, help="Zero-action steps before the step.")
    p.add_argument("--steps", type=int, default=200, help="Policy steps recorded after the step.")
    p.add_argument("--gravity", action="store_true", help="Leave gravity on (default: off).")
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    from isaaclab_tasks.utils import load_cfg_from_registry

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 16
    env_cfg.sim.device = args.device

    # Nothing may move the body except the joint under test.
    env_cfg.push_velocity = 0.0
    for field, value in (("balance_assist", 0.0), ("push_impulse_range", (0.0, 0.0))):
        if hasattr(env_cfg, field):
            setattr(env_cfg, field, value)
    if not args.gravity:
        env_cfg.sim.gravity = (0.0, 0.0, 0.0)

    env = gym.make(args.task, cfg=env_cfg)
    base = env.unwrapped
    env.reset()

    names = list(base.robot.joint_names)
    if args.joint not in names:
        raise SystemExit(f"unknown joint {args.joint!r}; have {names[:6]} ...")
    joint_id = names.index(args.joint)

    actuated = list(base._actuated_ids.tolist())
    if joint_id not in actuated:
        raise SystemExit(f"{args.joint} is not actuated; the action vector cannot command it.")
    slot = actuated.index(joint_id)

    # Invert `_apply_action`'s mapping so the probe commands a known ANGLE rather than a raw action:
    #   target = default + action_scale * action * span
    default = base._act_default[slot].item()
    upper = base._act_upper[slot].item()
    lower = base._act_lower[slot].item()
    span = (upper - default) if args.target >= default else (default - lower)
    action_value = (args.target - default) / (env_cfg.action_scale * span)

    clipped = max(-1.0, min(1.0, action_value))
    reachable = default + env_cfg.action_scale * clipped * span
    print(f"[probe] joint {args.joint} (id {joint_id}, action slot {slot})")
    print(f"[probe] default {default:+.4f}  limits [{lower:+.4f}, {upper:+.4f}]  span {span:.4f}")
    # **Print the solver iteration count, and default it to the trained value.** XPBD iterations are
    # part of the plant, not a performance dial: measured on `joint_Thigh_L:1`, overshoot is +10.5%
    # at 2 iterations and +0.4% at 8, against Godot's +0.9%. Probing at the module default of 2 while
    # training runs at 8 produced a "structural damping mismatch that no scaling can close" - a
    # conclusion about the SOLVER SETTING, not about the engines.
    iterations = int(os.environ.get("P4F_XPBD_ITERATIONS", "0"))
    print(f"[probe] action_scale {env_cfg.action_scale}  gravity {'ON' if args.gravity else 'OFF'}"
          f"  xpbd iterations {iterations or '2 (module default - set P4F_XPBD_ITERATIONS to match training)'}")
    if abs(action_value) > 1.0:
        # Not a probe failure - it is the answer to a different question, and worth saying plainly:
        # at this action_scale the policy simply cannot command that angle, however hard it tries.
        print(f"[probe] WARNING target {args.target:+.4f} needs action {action_value:+.3f}, outside "
              f"[-1,1]. Clamped; the reachable target is {reachable:+.4f} rad.")
    print(f"[probe] commanding target {reachable:+.4f} rad (action {clipped:+.3f})")

    zero = torch.zeros(base.num_envs, base.cfg.action_space, device=base.device)
    recent = []
    for _ in range(args.settle):
        env.step(zero)
        recent.append(base.rig.joint_state()[0][:, joint_id].clone())
        recent = recent[-10:]

    # **Gate the measurement on the body actually being at rest.** A step response measured from a
    # moving start is not a step response, and the error is not small: an unsettled probe scattered
    # the pre-step angle by +/-0.005 rad against a 0.05 rad amplitude - 10% - and produced a sweep in
    # which MORE stiffness gave a SLOWER rise and MORE damping gave MORE overshoot. The same
    # parameters re-measured 217 ms and then 300 ms. Both readings were meaningless, and nothing in
    # the output said so, which is how a calibration gets baked from noise.
    spread = recent[-1].std().item()
    drift = (recent[-1] - recent[0]).abs().mean().item()
    print(f"[probe] settle check: across-env std {spread:.6f} rad, drift over last 10 steps "
          f"{drift:.6f} rad")
    if spread > 0.002 or drift > 0.002:
        print("[probe] WARNING NOT SETTLED - rise time and overshoot below are unreliable. "
              "Increase --settle, or find what is still moving the body.")

    # `base.rig.joint_state()`, NOT `robot.data.joint_pos`. Under XPBD that buffer is frozen and
    # reads exactly 0.0000 - which is the healthy rest pose, so it looks like a body holding
    # position rather than a dead read. `p4f_newton/state.py` documents it; this probe hit it
    # anyway and reported a joint that never moved, under gravity, for 90 steps.
    start = base.rig.joint_state()[0][:, joint_id].mean().item()

    action = zero.clone()
    action[:, slot] = clipped

    dt = base.cfg.sim.dt * base.cfg.decimation
    trace = []
    resets = 0
    for i in range(args.steps):
        _, _, terminated, truncated, _ = env.step(action)
        # **A reset mid-trace silently snaps the joint back to its default and the trace reads as a
        # slow, badly damped response that is really two responses spliced together.** Measured
        # before this check existed: a calibration sweep produced a HIGHER stiffness giving a SLOWER
        # rise and more damping giving MORE overshoot - both impossible, and both explained by
        # episodes resetting underneath the probe. Counted and reported rather than assumed absent.
        resets += int(terminated.sum().item()) + int(truncated.sum().item())
        angle = base.rig.joint_state()[0][:, joint_id].mean().item()
        trace.append(((i + 1) * dt, angle))

    if resets:
        print(f"[probe] WARNING {resets} env-reset(s) DURING the trace across {base.num_envs} envs. "
              "The response below is spliced across resets and is NOT a step response. Raise the "
              "episode length or relax terminations before trusting rise time or overshoot.")
    else:
        print(f"[probe] no env resets during the trace ({base.num_envs} envs) - trace is clean.")

    final = trace[-1][1]
    travel = final - start
    print(f"[probe] start {start:+.5f}  final {final:+.5f}  travel {travel:+.5f} rad")
    print(f"[probe] steady-state error vs command: {reachable - final:+.5f} rad")

    # **Measured against the COMMANDED step, not the last sample.** Using `final` assumes the trace
    # reached steady state, and when it has not the metric inverts: a slower, softer plant that is
    # still climbing at the end of the window gets a tiny `travel`, so its 63.2% threshold lands in
    # the first few milliseconds and its peak reads as enormous overshoot. Measured that way, a
    # genuinely slower plant reported "83 ms rise, +522% overshoot" - faster and wilder than the
    # baseline it was four times softer than.
    commanded = reachable - start
    if abs(commanded) > 1e-6:
        threshold = start + 0.632 * commanded
        rise = next((t for t, a in trace if (a >= threshold) == (commanded > 0)), None)
        print(f"[probe] time to 63.2% of COMMAND ({threshold:+.5f}): "
              + (f"{rise * 1000:.1f} ms" if rise else "NOT REACHED in window"))
        peak = max(trace, key=lambda ta: abs(ta[1] - start))
        overshoot = (abs(peak[1] - start) / abs(commanded) - 1.0) * 100.0
        print(f"[probe] peak {peak[1]:+.5f} at {peak[0] * 1000:.1f} ms  overshoot vs command "
              f"{overshoot:+.1f}%")
        print(f"[probe] settled fraction of command: {(final - start) / commanded:.3f} "
              "(1.000 means it got there)")

    print("[probe] trace (ms, rad):")
    for t, a in trace[:30]:
        print(f"    {t * 1000:7.1f}  {a:+.5f}")

    env.close()


if __name__ == "__main__":
    main()
