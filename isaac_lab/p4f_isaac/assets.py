"""Articulation config for the Godot dummy, built from the generated rig contract.

Every number here is read out of `assets/dummy_rig.json`, which `tools/tscn_to_urdf.py` writes
from the Godot scene. Nothing is restated by hand: a gain typed in twice is a gain that will
disagree with itself the first time the scene is retuned, and the disagreement would show up as a
policy that behaves differently in Isaac than in Godot for no visible reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg

from .actuators import StablePDActuatorCfg
from isaaclab.assets import ArticulationCfg

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

# Which rig encoding to load. Set P4F_RIG=d6 for the native-D6 rig.
#
# `urdf` is the original: URDF has no 3-DOF joint, so every one of Godot's `Generic6DOFJoint3D`
# became three stacked hinges separated by two massless links - 46 bodies and 45 hinges for a
# 16-bone skeleton. It trains well and does not transfer; the Godot body collapses in under two
# seconds under a policy that holds 98.4% standing here, and neither the actuator model, the gain
# scale nor observation randomisation closed the gap.
#
# `d6` is built by `scripts/build_d6_usd.py` straight to USD, which HAS the joint type URDF lacks:
# 16 bodies and 15 D6 joints, the same mechanism Godot actually simulates. The DOF count is
# identical either way (15 x 3 = 45), so the 143-float observation and 36-float action contract are
# unchanged - only the rig and the DOF ordering differ.
_RIG_KIND = os.environ.get("P4F_RIG", "urdf").lower()

if _RIG_KIND == "d6":
    RIG_PATH = ASSETS_DIR / "dummy_d6_rig.json"
    USD_PATH = ASSETS_DIR / "dummy_d6.usd"
else:
    RIG_PATH = ASSETS_DIR / "dummy_rig.json"
    USD_PATH = ASSETS_DIR / "dummy.usd"

RIG: dict = json.loads(RIG_PATH.read_text(encoding="utf-8"))

# Joint order for the policy's action vector. Frozen: index 3*i+{0,1,2} is bone i's x/y/z axis.
#
# The bone order is the URDF's TREE order - Spine, Thigh_L, Thigh_R, Chest, ... - NOT
# RagdollRLBridge.ControlledBoneNames order, because tscn_to_urdf.py appends to this list inside
# its tree-order emission loop. This comment used to claim ControlledBoneNames order and was
# wrong; only index 0 coincides between the two. Read the list, never restate it. See
# obs_action_contract.md §2.
ACTUATED_JOINTS: list[str] = RIG["actuated_joints"]
PASSIVE_JOINTS: list[str] = RIG["passive_joints"]
CONTACT_BONES: list[str] = RIG["contact_bones"]

# Pelvis height of the Godot rest pose. The URDF root frame is the pelvis centre, so spawning the
# root here puts the soles exactly on the ground: Foot_L sits at 0.04 with a 0.08-tall box.
REST_PELVIS_HEIGHT = 0.82

# Which actuator model drives the joints. Set P4F_ACTUATOR=explicit to switch.
#
# **This is the sim-to-sim knob, and it is not cosmetic.** Godot and Isaac carry byte-identical
# gains - every bone's kp, kd and torque ceiling matches the scene exactly - and still disagree
# about whether the rest pose is an equilibrium. Measured with all-zero actions: Isaac holds 98.4%
# standing after 8 seconds, wobbling to 1.487 m and recovering to 1.537; Godot is flat on the floor
# inside 2 seconds. Same body, same numbers, opposite outcome.
#
# The difference is WHEN the torque is computed. `ImplicitActuatorCfg` hands the gains to PhysX,
# which folds the PD into the articulation solve and evaluates it at the END of the timestep - so
# it is unconditionally stable no matter how stiff, and the knee's 1800 N.m/rad behaves like 1800.
# `IdealPDActuatorCfg` computes `kp*(q_des - q) + kd*(qd_des - qd)` from the CURRENT state and
# clips it, which is exactly what `ActiveBone` does in Godot - and explicit integration at that
# stiffness and a 1/120 s step is at the edge of its own stability limit, so the effective
# stiffness is far below the nominal one.
#
# A policy trained against the implicit model learns to balance a body whose joints hold. Replayed
# in Godot it inherits joints that sag, which is not the body it was evaluated on.
# `spd` reproduces Godot's Stable PD - see actuators.py. `explicit` is kept only to document that
# a plain explicit PD diverges here; it is not a usable setting.
_ACTUATOR_CHOICE = os.environ.get("P4F_ACTUATOR", "implicit").lower()
_ACTUATOR_CLS = {
    "explicit": IdealPDActuatorCfg,
    "spd": StablePDActuatorCfg,
}.get(_ACTUATOR_CHOICE, ImplicitActuatorCfg)

# Global multiplier on every joint's stiffness. Set P4F_STIFFNESS_SCALE=0.3 to weaken.
#
# This is the sim-to-sim knob that actually works, after IdealPDActuatorCfg turned out not to be.
#
# Godot does not run a naive explicit PD; `ActiveBone` uses Stable PD (Tan-Liu-Turk), which stays
# stable at high gains by evaluating against the predicted next-step state. The price is a velocity
# factor below 1 that reduces the stiffness actually delivered - the class's own comments put the
# loss at 2.6x on the ankle and 5.4x on the knee. So Godot's effective gains are well under the
# authored ones, while Isaac's implicit actuator delivers them in full.
#
# Reproducing that by switching Isaac to IdealPDActuatorCfg (a true explicit PD) does not work: at
# the knee's 1800 N.m/rad and a 1/120 s step it diverges outright - joint velocities reached ~1e10,
# reward terms hit -1e21, and PPO crashed on a NaN action std within 16 seconds. Explicit PD is
# LESS stable than Godot's SPD, not equivalent to it.
#
# Scaling the implicit gains keeps the solver stable while matching the authority Godot has, which
# is the property that actually decides whether the rest pose is an equilibrium.
_STIFFNESS_SCALE = float(os.environ.get("P4F_STIFFNESS_SCALE", "1.0"))

_stiffness = {name: spec["stiffness"] * _STIFFNESS_SCALE for name, spec in RIG["joints"].items()}
_damping = {name: spec["damping"] for name, spec in RIG["joints"].items()}
_effort = {name: spec["effort"] for name, spec in RIG["joints"].items()}


DUMMY_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        # Required for the ContactSensor on the hands and feet: the contact reporter API is opt-in
        # per rigid body, and a sensor pointed at bodies without it fails at init with
        # "could not find any bodies with contact reporter API" rather than reporting zeros.
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Off for the same reason the asset conversion leaves it off: the rest pose has the
            # arms inside the torso's swept volume, so self-collision starts the body
            # interpenetrating and PhysX spends the first frames shoving it apart.
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, REST_PELVIS_HEIGHT),
        # Identity for the URDF rig, whose bodies are authored directly in USD axes. The D6 rig
        # instead keeps every body's frame equal to GODOT's frame - that is what lets its joint
        # limits copy across with no sign conversion - so its root has to spawn with the
        # Godot-to-USD rotation or the Godot-frame joint anchors are read as world offsets and the
        # body assembles lying down. Read from the contract rather than restated.
        rot=tuple(RIG.get("root_rest_quat_wxyz", (1.0, 0.0, 0.0, 0.0))),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # One group over every DOF, with per-joint values. Grouping by body part would be tidier
        # to read but would silently drop any joint whose name stopped matching its pattern; an
        # explicit per-joint dict cannot lose one.
        "all": _ACTUATOR_CLS(
            joint_names_expr=[".*"],
            stiffness=_stiffness,
            damping=_damping,
            effort_limit=_effort,
            velocity_limit=20.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
