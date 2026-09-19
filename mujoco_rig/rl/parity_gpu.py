"""Does the GPU engine agree with the CPU engine we actually ship?

**This gate exists because of Isaac.** Every previous attempt on this project trained on one engine
and deployed on another, and the policy died crossing the boundary. The MuJoCo architecture fixed
that by making training and runtime the same engine - so moving training to the GPU deliberately
reopens the wound, and it is only acceptable while the size of the gap is a number somebody looks at.

`mujoco_warp` computes in **float32**; MuJoCo's C engine computes in **float64**, and there is no
float64 option. So the two WILL differ. What this measures is how much, and - critically - whether
the difference is real or just the system amplifying rounding.

**The control run is the point.** Two CPU trajectories are started 1 nanometre apart. If those
diverge as fast as CPU-vs-GPU does, the system is chaotic and the comparison says nothing about the
engines. Measured 2026-09-09: the CPU pair does NOT diverge at all over 2.5 s, so the GPU departure
is genuinely the engine and cannot be excused as chaos.

It also asserts the constraint buffers have headroom. `nefc` overflow is reported only as a line on
stderr from inside a kernel, and it silently degrades contact resolution - raising `njmax` from the
default to 256 improved 2.5 s divergence from 21 degrees to 4.8. That is far too large an effect to
leave to a warning nobody reads.

    python mujoco_rig/rl/parity_gpu.py
"""
from __future__ import annotations

import argparse
import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Divergence permitted at the EARLY horizon, where float32 rounding has not yet been amplified by
# contact. A regression that matters - a silently unsupported feature, an overflowing buffer, a
# mismatched model - shows up here immediately. Later divergence is expected and only reported.
EARLY_SECONDS = 0.5
EARLY_QPOS_TOL = 1.0e-4
EARLY_TILT_TOL = 0.05        # degrees


def tilt_between(r_a, r_b):
    """Angle between two pelvis up-axes, in degrees."""
    up_a = r_a @ np.array([0.0, 0.0, 1.0])
    up_b = r_b @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(up_a @ up_b, -1.0, 1.0))))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="dummy_ball.xml")
    p.add_argument("--seconds", type=float, default=2.5)
    p.add_argument("--njmax", type=int, default=256)
    p.add_argument("--naconmax", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import mujoco_warp as mjw

    mjm = mujoco.MjModel.from_xml_path(str(ROOT / args.model))
    pelvis = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "Pelvis")
    qadr = [mjm.jnt_qposadr[mjm.actuator_trnid[i, 0]] for i in range(mjm.nu)]
    steps = int(args.seconds / mjm.opt.timestep)

    rng = np.random.default_rng(args.seed)
    # A non-trivial but gentle control sequence. Zero control would leave the body in a symmetric
    # rest pose where the two engines agree for uninteresting reasons.
    ctrl = rng.uniform(-0.05, 0.05, size=(steps, mjm.nu))

    def fresh(nudge=0.0):
        d = mujoco.MjData(mjm)
        mujoco.mj_resetData(mjm, d)
        pose = np.random.default_rng(args.seed + 1).uniform(-0.03, 0.03, size=mjm.nu)
        d.qpos[qadr] += pose
        d.qpos[0] += nudge
        mujoco.mj_forward(mjm, d)
        return d

    cpu_a = fresh()
    cpu_b = fresh(1.0e-9)          # the control: same engine, 1 nanometre apart
    cpu_g = fresh()                # the CPU half of the CPU-vs-GPU comparison

    m = mjw.put_model(mjm)
    gpu = mjw.put_data(mjm, cpu_g, nworld=1, njmax=args.njmax, naconmax=args.naconmax)
    gpu.qpos.assign(cpu_g.qpos.reshape(1, -1))
    gpu.qvel.assign(cpu_g.qvel.reshape(1, -1))

    print(f"[parity] {args.model}  {steps} steps @ dt={mjm.opt.timestep}  "
          f"njmax={args.njmax} naconmax={args.naconmax}")
    print(f"{'t (s)':>7} | {'CPU vs CPU (1e-9 nudge)':>26} | {'CPU vs GPU (mujoco_warp)':>26}")
    print(f"{'':>7} | {'|dqpos|':>12} {'tilt (deg)':>13} | {'|dqpos|':>12} {'tilt (deg)':>13}")

    peak_nefc, peak_nacon = 0, 0
    early = None
    report_every = max(1, steps // 5)

    for k in range(steps):
        for d in (cpu_a, cpu_b, cpu_g):
            d.ctrl[:] = ctrl[k]
            mujoco.mj_step(mjm, d)
        gpu.ctrl.assign(ctrl[k].reshape(1, -1))
        mjw.step(m, gpu)

        peak_nefc = max(peak_nefc, int(gpu.nefc.numpy().max()))
        peak_nacon = max(peak_nacon, int(gpu.nacon.numpy().max()))

        t = (k + 1) * mjm.opt.timestep
        gq = gpu.qpos.numpy()[0]
        gr = gpu.xmat.numpy()[0][pelvis].reshape(3, 3)
        dq = float(np.abs(cpu_g.qpos - gq).max())
        dt_deg = tilt_between(cpu_g.xmat[pelvis].reshape(3, 3), gr)
        if early is None and t >= EARLY_SECONDS:
            early = (t, dq, dt_deg)

        if (k + 1) % report_every == 0:
            print(f"{t:7.2f} | {np.abs(cpu_a.qpos - cpu_b.qpos).max():12.2e} "
                  f"{tilt_between(cpu_a.xmat[pelvis].reshape(3, 3), cpu_b.xmat[pelvis].reshape(3, 3)):13.4f}"
                  f" | {dq:12.2e} {dt_deg:13.4f}")

    ok = True
    print(f"\n  constraint buffer peak: nefc {peak_nefc}/{args.njmax}, "
          f"contacts {peak_nacon}/{args.naconmax}")
    if peak_nefc >= args.njmax or peak_nacon >= args.naconmax:
        # Overflow is only ever reported as a stderr line from inside a kernel, and it silently
        # degrades contact rather than failing - exactly the kind of quiet wrongness this project
        # has been burned by. Fail loudly instead.
        print("  FAIL: constraint buffer overflowed; contact resolution was silently degraded")
        ok = False

    t, dq, dt_deg = early
    print(f"  early agreement at {t:.2f}s: |dqpos| {dq:.2e} (tol {EARLY_QPOS_TOL:.0e}), "
          f"tilt {dt_deg:.4f} deg (tol {EARLY_TILT_TOL})")
    if dq > EARLY_QPOS_TOL or dt_deg > EARLY_TILT_TOL:
        print("  FAIL: the engines disagree before float32 amplification could explain it")
        ok = False

    print("\n  Reminder: this gate does NOT license scoring on the GPU engine. Every checkpoint is\n"
          "  scored by eval.py on CPU MuJoCo, which is the engine Godot drives through P/Invoke.")
    print("  PASS" if ok else "  FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
