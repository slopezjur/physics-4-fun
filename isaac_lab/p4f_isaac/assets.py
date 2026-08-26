"""Articulation config for the Godot dummy, built from the generated rig contract.

Every number here is read out of `assets/dummy_rig.json`, which `tools/tscn_to_urdf.py` writes
from the Godot scene. Nothing is restated by hand: a gain typed in twice is a gain that will
disagree with itself the first time the scene is retuned, and the disagreement would show up as a
policy that behaves differently in Isaac than in Godot for no visible reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
RIG_PATH = ASSETS_DIR / "dummy_rig.json"
USD_PATH = ASSETS_DIR / "dummy.usd"

RIG: dict = json.loads(RIG_PATH.read_text(encoding="utf-8"))

# Joint order for the policy's action vector. Frozen: index 3*i+{0,1,2} is bone i's x/y/z axis, in
# the order RagdollRLBridge.ControlledBoneNames declares. See obs_action_contract.md.
ACTUATED_JOINTS: list[str] = RIG["actuated_joints"]
PASSIVE_JOINTS: list[str] = RIG["passive_joints"]
CONTACT_BONES: list[str] = RIG["contact_bones"]

# Pelvis height of the Godot rest pose. The URDF root frame is the pelvis centre, so spawning the
# root here puts the soles exactly on the ground: Foot_L sits at 0.04 with a 0.08-tall box.
REST_PELVIS_HEIGHT = 0.82

_stiffness = {name: spec["stiffness"] for name, spec in RIG["joints"].items()}
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
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # One group over every DOF, with per-joint values. Grouping by body part would be tidier
        # to read but would silently drop any joint whose name stopped matching its pattern; an
        # explicit per-joint dict cannot lose one.
        "all": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=_stiffness,
            damping=_damping,
            effort_limit=_effort,
            velocity_limit=20.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
