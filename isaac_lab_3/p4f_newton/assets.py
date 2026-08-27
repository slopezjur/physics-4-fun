"""Articulation config for the Godot dummy under Isaac Lab 3 / Newton.

Deliberately independent of `isaac_lab/p4f_isaac/assets.py`. That module is written against Isaac
Lab 2.3.2 and carries seven attempts' worth of PhysX-specific knobs — stiffness scaling, the
Stable-PD actuator, the URDF/D6 rig switch — none of which mean anything to a Newton solver. This
one starts clean and only grows a knob when a measurement demands it.

**The rig is not regenerated here.** `dummy_d6.usd` is read from `isaac_lab/assets/`, where
`isaac_lab/scripts/build_d6_usd.py` writes it from `Scenes/ActiveRagdoll.tscn`. It is the same
mechanism Godot simulates — 16 bodies, 15 D6 joints, 45 DOF, 80.60 kg — and the Godot scene stays
the single source of truth for masses, limits and gains. Copying the artefact into this directory
would be a second copy that drifts the first time the scene is retuned.

Every number below is read out of `dummy_d6_rig.json`. Nothing is restated by hand; the project has
already paid for prose that disagreed with the generator (see `isaac_lab/obs_action_contract.md`).
"""

from __future__ import annotations

import os

import json
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# The 2.3.2 tree owns the generated assets; this track consumes them.
ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "isaac_lab" / "assets"
RIG_PATH = ASSETS_DIR / "dummy_d6_rig.json"
USD_PATH = ASSETS_DIR / "dummy_d6.usd"

RIG: dict = json.loads(RIG_PATH.read_text(encoding="utf-8"))

ACTUATED_JOINTS: list[str] = RIG["actuated_joints"]
PASSIVE_JOINTS: list[str] = RIG["passive_joints"]
CONTACT_BONES: list[str] = RIG["contact_bones"]
TOTAL_MASS: float = RIG["total_mass"]

# Pelvis height of the Godot rest pose. The D6 rig's root frame is the pelvis centre, so spawning
# the root here puts the soles exactly on the ground.
REST_PELVIS_HEIGHT = 0.82


def _root_rest_quat_xyzw() -> tuple[float, float, float, float]:
    """The root spawn orientation, in Isaac Lab 3's quaternion order.

    **Isaac Lab 3 changed the root quaternion convention from wxyz to xyzw.**
    `InitialStateCfg().rot` now defaults to `(0, 0, 0, 1)`; in 2.3.2 it was `(1, 0, 0, 0)`. Both
    spell "identity" in their own order, so the 2.3.2 value copies across looking entirely correct
    and is read as **x = 1**: a 180 degree rotation about X.

    Measured, before this was found. The whole articulation spawns mirrored in y and z about the
    pelvis — Head at 0.100 m instead of 1.540, Foot at 1.600 instead of 0.040 — while
    `root_quat_w` still reports identity, because the read path applies the same swap and the error
    round-trips. `projected_gravity_b` reads (0, 0, +1) for a body that looks upright in every log.
    Nothing raises. The body simply falls, which is exactly the answer this track is looking for,
    so it would have been read as a passing feasibility gate.

    The rig contract stores the key in wxyz if it stores it at all, so it is converted here rather
    than restated, and the default is written in xyzw to match the version actually in use.
    """
    wxyz = RIG.get("root_rest_quat_wxyz")
    if wxyz is None:
        return (0.0, 0.0, 0.0, 1.0)
    w, x, y, z = (float(v) for v in wxyz)
    return (x, y, z, w)


ROOT_REST_QUAT_XYZW = _root_rest_quat_xyzw()

_stiffness = {name: spec["stiffness"] for name, spec in RIG["joints"].items()}
_damping = {name: spec["damping"] for name, spec in RIG["joints"].items()}
_effort = {name: spec["effort"] for name, spec in RIG["joints"].items()}


# `ImplicitActuatorCfg` is the correct choice here and it is worth saying why, because the 2.3.2
# track spent an attempt on the alternative.
#
# Under Newton, an implicit actuator's gains are written straight to the solver as Newton's
# `joint_target_ke` / `joint_target_kd`, and XPBD supports exactly those — the drive is applied as a
# compliant positional constraint inside the position-based solve rather than as a force computed
# from the current state. That is the same structural trick that makes Godot's Stable PD stable at
# 1800 N.m/rad and a 1/120 s step.
#
# The explicit path (`IdealPDActuatorCfg`) computes `kp*(q_des - q) + kd*(qd_des - qd)` in Python and
# applies it as `joint_f`. That was attempt #5 in the 2.3.2 track and it diverged in BOTH engines:
# joint velocities reached ~1e10 in Isaac and PPO crashed on a NaN. Do not reach for it here.
#
# **Known XPBD gap:** `SolverXPBD` does not support `joint_effort_limit`, `joint_velocity_limit` or
# `joint_armature` (its own docstring says so). The `effort_limit` below is therefore expected to be
# IGNORED by the XPBD solver — the torque ceilings that are part of the Godot contract do not bind.
# That is measured rather than assumed by `probe_xpbd.py`, and it is the first thing to look at if
# the body holds its pose by bracing at implausible torque.
#
# `effort_limit_sim`, not `effort_limit`. The 2.3.2 track recorded that `effort_limit` is deprecated
# and `velocity_limit` on an implicit actuator is silently ignored — the `_sim` suffixed parameters
# are the ones that reach the solver. Using the deprecated name here would stack a second silent
# no-op on top of XPBD's own, and make the effort-limit measurement unreadable.
# Scale applied to EVERY joint gain, stiffness and damping alike.
#
# **This is the largest single mismatch between the two engines, and it is a hard limit on Godot's
# side rather than a tuning choice.** Godot drives its joints through `ActiveBone` ->
# `PidController3D`, which uses the Tan-Liu-Turk SPD form and divides both gains by
# `1 + kd*dt/I + kp*dt^2/I`. Measured live in the stand check across the 12 controlled bones:
#
#     authored kp 882  ->  effective 98-116        ceiling I/dt^2 = 630-725
#
# So Godot applies about a fifth of the gain the rig contract authors, while Isaac's XPBD applies
# the drive inside the solve at the full value. Worse, the ceiling is real: as kp tends to infinity
# the effective gain tends to I/dt^2, which for these limbs at 120 Hz is BELOW the authored 882.
# No gain reproduces the contract in Godot.
#
# Raising kp alone was tried and backfired - both effective gains are divided by the same
# denominator, so lifting kp without kd took the damping ratio from 0.64 to 0.19 and the body rang
# itself apart. That shared denominator is also the useful part: Godot's actuator is simply the
# authored gains scaled DOWN by a common factor, which is a plant Isaac can reproduce exactly by
# scaling its own the same way.
#
# 0.176 = 155/882, the measured effective stiffness over the authored one.
GAIN_SCALE = float(os.environ.get("P4F_GAIN_SCALE", "1.0"))

ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*"],
    stiffness={name: value * GAIN_SCALE for name, value in _stiffness.items()},
    damping={name: value * GAIN_SCALE for name, value in _damping.items()},
    effort_limit_sim=_effort,
)


DUMMY_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        # No PhysX solver-iteration properties here, unlike the 2.3.2 config. Those are PhysX
        # articulation settings; under Newton the equivalent knob is the solver's own `iterations`,
        # which lives on `XPBDSolverCfg` and is swept by the probe.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Off for the same reason the 2.3.2 rig has it off: the rest pose has the arms inside
            # the torso's swept volume, so self-collision starts the body interpenetrating and the
            # solver spends the first frames shoving it apart.
            enabled_self_collisions=False,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, REST_PELVIS_HEIGHT),
        # The D6 builder carries the Godot->USD rotation on the JOINT frames, not the body frames,
        # so bodies stay conventionally USD Z-up and the root spawns unrotated. Putting it on the
        # bodies instead makes `projected_gravity_b` read (0,-1,0), which silently computes the
        # `upright` reward as exactly 0.0 for a perfectly standing body.
        rot=ROOT_REST_QUAT_XYZW,
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={"all": ACTUATOR_CFG},
    soft_joint_pos_limit_factor=1.0,
)
