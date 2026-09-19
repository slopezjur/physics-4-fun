"""Gate on the generated model: is this the body we meant to build?

Run after every `build_mjcf.py`. It checks the four things that have each silently been wrong at
some point and would have made every later measurement meaningless.

**What this does NOT check any more.** It used to require that the model, with every actuator
commanded to zero, still stood - "COLLAPSED" was the failure message. That gate belonged to the
POSITION-actuated plant, where a zero command meant "hold the rest pose" and standing was free. The
body is now driven by `motor` actuators over passive ligaments, so zero command is zero muscle and
an unpowered body SHOULD collapse. Keeping the old assertion would have this script fail forever on
a correct model, which is worse than having no gate at all.

It also no longer compares against Godot's 80.60 kg / 0.8298 m pelvis. That rig was 1.68 m and
BMI 28; this one is deliberately 1.75 m at BMI 22.7, so the two are not supposed to match.
"""
import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent

TARGET_HEIGHT = 1.750        # m, build_mjcf.SCALE_TO_HEIGHT
TARGET_MASS = 70.0           # kg, build_mjcf.TARGET_MASS
HEIGHT_TOL = 0.005           # m
MASS_TOL = 0.5               # kg


def geom(m, name):
    return next(g for g in range(m.ngeom)
                if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) == name)


def check(model_path):
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    failures = []

    print(f"\n{model_path.name}")
    print(f"  bodies {m.nbody - 1}  joints {m.njnt}  actuators {m.nu}  dofs {m.nv}")

    # --- 1. the rest pose is the KEYFRAME, and it must not put the body inside itself.
    # At qpos0 the arms hang inside the legs: 5.5 cm of forearm inside the thigh, 331.8 N.m per
    # shoulder to hold. Everything that starts an episode resets to `rest` instead.
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "rest")
    if key < 0:
        failures.append("no `rest` keyframe - every env and MjBridge.ResetData expects one")
        mujoco.mj_resetData(m, d)
    else:
        mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)

    self_contacts = [c for c in d.contact[:d.ncon]
                     if m.geom_type[c.geom1] != mujoco.mjtGeom.mjGEOM_PLANE
                     and m.geom_type[c.geom2] != mujoco.mjtGeom.mjGEOM_PLANE]
    if self_contacts:
        pairs = {tuple(sorted((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                               mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2))))
                 for c in self_contacts}
        failures.append(f"rest pose self-intersects: {sorted(pairs)}")
    print(f"  rest pose: {d.ncon} contacts, {len(self_contacts)} body-on-body")

    # --- 2. stature and mass
    head = geom(m, "g_Head")
    foot = geom(m, "g_Foot_L")
    crown = d.geom_xpos[head][2] + m.geom_size[head][0]
    sole = d.xpos[m.geom_bodyid[foot]][2] - m.geom_size[foot][2]
    height = crown - sole
    mass = float(sum(m.body_mass))
    ball = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
    if ball >= 0:
        mass -= float(m.body_mass[ball])
    print(f"  height {height:.3f} m (target {TARGET_HEIGHT})   "
          f"mass {mass:.2f} kg (target {TARGET_MASS})   BMI {mass / height ** 2:.1f}")
    if abs(height - TARGET_HEIGHT) > HEIGHT_TOL:
        failures.append(f"height {height:.3f} m is off target by more than {HEIGHT_TOL} m")
    if abs(mass - TARGET_MASS) > MASS_TOL:
        failures.append(f"mass {mass:.2f} kg is off target by more than {MASS_TOL} kg")

    # --- 3. the feet start ON the floor, not sunk into it or hovering
    for side in ("L", "R"):
        g = geom(m, f"g_Foot_{side}")
        z = d.xpos[m.geom_bodyid[g]][2] - m.geom_size[g][2]
        if abs(z) > 0.01:
            failures.append(f"foot {side} sole starts at z {z:+.4f}, not on the floor")

    # --- 4. an UNPOWERED body must crumple, not fall as a plank. This is the property the whole
    # torque-actuator change exists to produce, so it is the thing worth asserting.
    hinges = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    d2 = mujoco.MjData(m)
    if key >= 0:
        mujoco.mj_resetDataKeyframe(m, d2, key)
    d2.qvel[0] = -0.4
    for _ in range(int(6.0 / m.opt.timestep)):
        d2.ctrl[:] = 0.0
        mujoco.mj_step(m, d2)
    bend = np.degrees([abs(d2.qpos[m.jnt_qposadr[j]]) for j in hinges]).mean()
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Pelvis")
    print(f"  unpowered fall: mean joint bend {bend:5.1f} deg, pelvis settles at "
          f"{d2.xpos[pel][2]:.3f} m")
    if bend < 10.0:
        failures.append(f"unpowered body barely bends ({bend:.1f} deg) - it is falling as a plank, "
                        "which means the actuators are servos rather than motors")
    if not np.isfinite(d2.qpos).all():
        failures.append("the unpowered fall diverged - check JOINT_ARMATURE")

    return failures


def main() -> int:
    bad = 0
    for name in ("dummy.xml", "dummy_ball.xml", "dummy_limp.xml"):
        path = ROOT / name
        if not path.exists():
            print(f"\n{name}: MISSING - run build_mjcf.py")
            bad += 1
            continue
        failures = check(path)
        for f in failures:
            print(f"  FAIL: {f}")
        bad += len(failures)

    print("\n-> " + ("OK" if bad == 0 else f"{bad} FAILURES - do not proceed"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
