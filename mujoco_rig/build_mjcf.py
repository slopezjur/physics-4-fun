"""Generate a MuJoCo model of the dummy from GODOT's own rig dump.

**Godot stays the single source of truth.** `IsaacDriverDiagnostics.LogPlantTable` already emits every
body's mass, parent, world position and joint-anchor offset on the first physics step, precisely so
the rig is not hand-copied into a second place and allowed to drift. This reads that dump plus the
collider shapes from `Scenes/ActiveRagdoll.tscn`, and the per-DOF limits from `dummy_rig.json`.

Frame: Godot is Y-up, MuJoCo is Z-up. The mapping is the one the rest of this project already uses,
`IsaacObservation.ToIsaacFrame` -> (-z, -x, y), a proper rotation, so traces stay comparable.

Emits TWO models. A free projectile in the model shortens the walk's stable horizon even when it is
parked and excluded from the centre of mass - measured with a 3 kg ball, the Python walk fell from
180 s to about 60 s purely from its presence - so STAND and WALK load the clean rig and PERTURB loads
the one that carries the ball.

    python mujoco_rig/build_mjcf.py                                  # dummy.xml, dummy_ball.xml
    python mujoco_rig/build_mjcf.py --no_actuators --out dummy_limp  # the unpowered ragdoll
    python mujoco_rig/validate.py                                    # after every rebuild
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DUMP = ROOT / "mujoco_rig" / "plant_dump.txt"
SCENE = ROOT / "Scenes" / "ActiveRagdoll.tscn"
RIG = ROOT / "isaac_lab" / "assets" / "dummy_rig.json"
OUT = ROOT / "mujoco_rig" / "dummy.xml"

SHAPE_FOR = {
    "Pelvis": "Pelvis", "Spine": "Spine", "Chest": "Chest", "Head": "Head",
    "UpperArm_L": "UpperArm", "UpperArm_R": "UpperArm",
    "Forearm_L": "Forearm", "Forearm_R": "Forearm",
    "Hand_L": "Hand", "Hand_R": "Hand",
    "Thigh_L": "Thigh", "Thigh_R": "Thigh",
    "Shin_L": "Shin", "Shin_R": "Shin",
    "Foot_L": "Foot", "Foot_R": "Foot",
}

# Joint axes must go through the SAME frame map as the positions: to_mj(x)=(0,-1,0),
# to_mj(y)=(0,0,1), to_mj(z)=(-1,0,0). Writing the raw axis here is a silent bug - zero targets
# still give the correct rest pose, and only commanded motion reveals it.
AXIS_VEC = {"x": "0 -1 0", "y": "0 0 1", "z": "-1 0 0"}

# Godot's BallGun small ball is 3.0 kg at 6 m/s = 22.5 N.s, which this body absorbs with only
# 1.0 deg of tilt (measured) - correct, but invisible to watch. Heavier here so the perturbation
# reads on screen; set back to 3.0 to reproduce Godot's own numbers exactly.
#
# 15 kg, raised from 10 on 2026-09-09. The measured single-impact envelope with no policy at all:
# 3 kg tilts the body 1.0 deg and it survives 100% of the time, 6 kg tilts it 23.5 deg and it
# survives 75%, 10 kg tilts it 24.3 deg. Below ~6 kg there is nothing for a policy to learn,
# because standing still already works.
# **5 kg, not 15.** A 15 kg ball at the 6 m/s the scene fires is 90 N.s into a 70 kg body - it
# hands the whole body 1.29 m/s of centre-of-mass velocity. A standing person can arrest roughly
# 0.3-0.5 m/s with ankles alone and about 1.0 m/s by stepping, so that projectile is unrecoverable
# by construction, for any controller, human strength or not. Measured on the trained policy: a
# SINGLE ball at 6 m/s was survived 0 times in 12, and even at the 2.92 m/s the curriculum had
# reached, only 17%.
#
# For scale: a boxer's straight punch is 30-50 N.s, a football at 25 m/s is 11 N.s. At 5 kg and
# 6 m/s this is 30 N.s - a hard medicine-ball hit, 0.43 m/s of COM change, inside what a stepping
# recovery can absorb. The projectile looks identical; only its mass changes.
#
# **8 kg from 2026-09-11**, once the balance brain survived 92.4% of single 5 kg hits: 48 N.s at the
# scene's 6 m/s, 0.69 m/s of COM change - harder, and still inside the ~1.0 m/s a stepping recovery
# can absorb (15 kg, 1.29 m/s, was not). Every perturb score before that date is at 5 kg.
BALL_MASS = 8.0

# Emit Godot's EFFECTIVE gains rather than the authored ones - see parse_effective_gains().
# Set by --gains on the command line; the default is chosen there.
# **Default is EFFECTIVE, not authored.** The authored gains are 2.5-6x what Godot's Stable PD
# applies, and a body that stiff stands for 40 s at 100% upright with NO policy at all - which makes
# "do nothing" the correct answer to every task and is why four training runs in a row produced a
# statue. Measured cliff: gain scale x0.50 still stands, x0.35 collapses; Godot's real thigh gain
# (533 of an authored 1600) is just past it. The soft body is the character.
EFFECTIVE_GAINS = True

# Uniform multiplier on every actuator gain, applied after the authored/effective choice. The
# authored rig is 2.5-6x stiffer than Godot actually applies, which is enough for the body to stand
# with no policy at all - and a body that stands for free hands every training run a statue as its
# free optimum. This is the knob for finding a plant that still needs a controller.
GAIN_SCALE = 1.0

# Clamp every actuator to HUMAN_TORQUE instead of the rig's authored effort.
HUMAN_STRENGTH = True

# Emit NO actuators at all - a limp ragdoll.
#
# This is NOT the same as commanding zero. MuJoCo's `position` actuator servos to its target, so a
# zero command means "hold the authored rest pose": the body falls as a rigid plank, still fighting
# to keep its shape all the way down. With the actuators removed only gravity, the joint limits and
# the passive springs on the head and hands remain, and the body actually crumples.
NO_ACTUATORS = False

# Peak isometric joint torque for an adult male, N.m (Winter 2009; Anderson & Pandy). The authored
# rig is about TWICE these everywhere - hip 400 vs 200, knee 400 vs 250, ankle 250 vs 150 - and a
# body with twice human strength does not need the behaviours we are trying to train. It can catch
# a fall on ankle torque alone instead of stepping, which is exactly the unnatural stiffness the
# dummy shows on screen, and it makes the protective step a more expensive option than brute force.
HUMAN_TORQUE = {
    "Pelvis": 200.0, "Spine": 200.0, "Chest": 200.0, "Head": 30.0,
    "UpperArm": 90.0, "Forearm": 60.0, "Hand": 15.0,
    "Thigh": 200.0, "Shin": 250.0, "Foot": 150.0,
}

# Joints a human drives on ONE axis only. The rig gives every limb three actuated DOF, so the knee
# can currently twist and splay - anatomically wrong, visually wrong, and 26 extra actuators for the
# policy to search over. `rx` is the flexion axis for both (verified by driving each joint and
# measuring: the elbow closes hand-to-shoulder 0.540 -> 0.319 m).
HINGE_ONLY = {"Shin", "Forearm"}

# --- muscle and ligament, for the WHOLE body ----------------------------------------------------
#
# A `position` actuator is a spring to a target pose, so its zero command does not mean "no muscle" -
# it means "hold the rest pose", and it holds it hard. Measured on this rig over a 6 s passive fall:
#
#   actuators on, commanding zero   mean joint bend  1.8 deg, peak actuator force 118.7 N.m
#   no actuators (dummy_limp.xml)   mean joint bend 38.9 deg
#
# The body flexed by under two degrees while it toppled, because kp 533 N.m/rad at the hip answers
# 10 deg of bend with 93 N.m - half a human hip's entire capacity - spent purely on holding still.
# That is the whole "stiff with a brain, rag without one" problem, and no amount of training fixes
# it: the plant simply has no limp state.
#
# So the neck's two-layer model becomes the whole body's:
#
#   ligament   a weak passive spring on every joint. What is left when the muscle is off.
#   muscle     a `motor` actuator - the policy outputs NEWTON-METRES, clamped to HUMAN_TORQUE.
#              Zero output is zero torque, so an unpowered body is a ragdoll and "dead" is a scale
#              on the policy's output rather than a separate model.
#
# 5% was chosen by measuring the unpowered fall: it crumples (mean bend 24.5 deg, pelvis settles at
# 0.165 m) and then stays still (max |qd| 0.03 rad/s at 6 s). At 10% the plank starts coming back,
# at 0% the joints are slightly loose. Standing costs almost nothing to hold - 1.5 N.m at the hip,
# 3.9 at the knee, measured - so the muscle layer is for CORRECTIONS, not for fighting gravity.
ACTUATOR_MODE = "torque"      # "torque" -> <motor>, "position" -> the old <position kp kv> servo
LIGAMENT = 0.05               # joint spring, as a fraction of that joint's peak torque, per radian
LIGAMENT_DAMP = 0.05          # joint damping, as a fraction of its own ligament stiffness

# The arms hang INSIDE the legs at the authored rest pose: forearm 5.5 cm inside the thigh, hand
# 4.4 cm, forearm 1.1 cm inside the pelvis - 11 contacts at qpos0, costing 331.8 N.m at each
# shoulder (a human shoulder peaks at 90) just to hold. Every episode of every run started there.
# 8 deg of abduction clears everything by 3-5 cm; measured signs are L +, R -.
#
# It cannot be done with the joint's `ref`, which relabels the joint value without moving the
# geometry, so it is emitted as a KEYFRAME the environments and Godot reset to, with `springref` on
# the same joints so the ligament rests in the cleared pose rather than pulling back into the leg.
REST_ABDUCTION = 0.1396       # rad = 8 deg

# Reflected inertia at the joint. Without it the torque plant is numerically unstable: 10 rollouts
# of 20 s under random torque produced 10 `Nan, Inf or huge value in QACC` divergences and drove the
# constraint count to 622, and MuJoCo silently resets the state when that happens - a trainer would
# have been learning from garbage. At 0.02 the same 10 rollouts give ZERO divergences and peak nefc
# 117, while the unpowered crumple is unchanged (mean joint bend 22.7 deg against 24.3, pelvis
# settles at the same 0.164 m). Every MuJoCo humanoid carries armature for exactly this reason.
#
# The head is excluded: its two-layer neck was tuned and measured without it, and 0.02 against the
# head's own 0.05 kg.m^2 would be a 40% change to how it flops.
JOINT_ARMATURE = 0.02

# --- the neck, in two layers -------------------------------------------------------------------
#
# A living human never holds their head up passively; the neck is muscle, and head stabilisation is
# a reflex. A relaxed or unconscious neck genuinely flops. Neither a stiff spring nor a soft one is
# right on its own, so the neck is modelled as BOTH layers and the difference between alive and dead
# becomes the actuator gain rather than a separate ragdoll:
#
#   ligament   the passive spring below. What is left when the muscle is off - the head flops and
#              the skull reaches the ground.
#   muscle     a position actuator held at zero, torque-limited to HUMAN_TORQUE["Head"] = 30 N.m.
#              It holds the head up against the 4.7 N.m gravity applies to a 4.34 kg head on a
#              0.11 m arm, but an impact OVERPOWERS it and the head whips, which is the behaviour
#              that neither a weld nor a rag can produce.
#
# Passive joints - Head and the Hands - were emitted at the AUTHORED gain while every actuated joint
# moved to Godot's effective one. That left the wrists 31x too stiff (60 against a measured kpEff of
# 1.93) and the neck a weld: 120 N.m/rad gives the head 3.1 deg of travel, so after a fall the skull
# hovered 4.0 cm off the floor instead of resting on it.
HUMAN_PASSIVE = {"Head": (5.0, 0.6)}     # N.m/rad, N.m.s/rad - ligament only, muscle is the actuator
NECK_ACTUATED = True
NECK_GAIN = (15.0, 1.5)                  # kp, kv - measured, not guessed; see below
#
# Why 15 and not the 60 that "holds the head firmly": standing, the head's CoM sits ON the
# neck axis, so gravity applies ~no torque and almost any gain holds it up. The gain only
# shows when the torso is NOT upright, and there a high one is wrong twice over. Measured,
# 8 s after a backward fall and with the torso held at 30 deg of forward pitch:
#
#   kp 60 -> skull 3.3 cm off the floor,  5.8 deg of head lag
#   kp 30 -> skull 2.0 cm off the floor, 10.6 deg
#   kp 15 -> skull 0.1 cm off the floor, 17.9 deg   <-
#
# A body on the ground must have its head ON the ground; at 60 it hovers, which is what the
# stiff passive spring did. And 17.9 deg of neck extension against 30 deg of torso pitch
# leaves the head only 12 deg off vertical - head-in-space stabilisation, which is what a
# person actually does. The +-30 N.m force limit, not the gain, is what makes an impact whip
# the head and then bring it back.

# The neck actuator is NOT in the policy's action space. The environments derive both the action
# and the observation width from `mjModel.nu`, so leaving it there would cost 3 action dimensions
# and 9 observations for the policy to explore - to rediscover the value a position actuator
# already starts at, since a zero command IS "head straight". There is no balance information in a
# 6%-mass segment sitting on the rotation axis. See env_config.POLICY_EXCLUDE.

# Neck range of motion. Parent-child contact is disabled in MuJoCo, so the joint limit is the ONLY
# thing keeping the skull out of the shoulder - and measured against the yoke it was not enough:
# Head_rx cleared only to +40.9 deg of its +57.3 limit and Head_rz to +-46.2 of +-57.3, so at full
# range the head sank up to 1.9 cm into the shoulder. It never showed because the stiff spring never
# let the head travel that far. Both trimmed values are still inside human range (flexion ~50 deg,
# lateral bend ~45 deg). Applied AFTER the `reversed` flip, so these are the emitted numbers.
NECK_LIMIT = {"Head_rx": (None, 0.6981), "Head_rz": (-0.7854, 0.7854)}   # radians

# MuJoCo's joint limits are SOFT constraints, and at the default compliance a ball to the head drove
# the neck 21 deg past its stop - the skull sank 5.8 cm into the shoulder for 138 ms at 18 m/s.
# Trimming the range further buys almost nothing (0.5 cm of clearance per 5 deg), so the stop itself
# is stiffened instead: timeconst 0.008 is ~2x the 4.2 ms timestep, the stiffest value that stays
# stable. Overlap at 18 m/s falls 5.8 -> 1.9 cm, and 2.9 -> 1.2 cm at 12 m/s.
NECK_LIMIT_SOLVER = ' solreflimit="0.008 1" solimplimit="0.95 0.999 0.001 0.5 2"'

# A TOE, and the reason it matters more than it looks.
#
# The authored foot is a single 0.22 m box. A box cannot roll heel-to-toe and has no joint ahead of
# the ankle, so the body has **no push-off** - and push-off at toe-off is what actually drives human
# walking. Without it a policy can only ever shuffle, which is a large part of why the gait looks
# wrong however it is trained.
#
# The footprint is unchanged; it is split. The foot keeps the rear 0.16 m and the toe takes the
# front 0.06 m on a metatarsophalangeal hinge, which is roughly where a human's is.
ADD_TOE = True
TOE_LENGTH = 0.06           # metres, along the foot's long axis
TOE_MASS = 0.2              # kg, taken out of the foot rather than added to the body
TOE_TORQUE = 30.0           # N.m, human MTP peak
TOE_RANGE = (-0.50, 1.00)   # rad: 29 deg of curl, 57 deg of extension at toe-off
# The authored foot is also 1.7x a human foot (2.0 kg against ~1.2). A heavy foot drags the swing
# leg and is felt most exactly where we are trying to train a step.
FOOT_MASS = 1.2

# **Attach the arms to the torso.** The rig places each shoulder 0.36 m from the midline while the
# chest only reaches 0.21, so the upper arm's inner surface sits 9 cm clear of the chest and the arm
# visibly floats. The same rig leaves a 3 cm gap at the wrist. Both are inherited from the Jolt/Isaac
# era, where the colliders were authored separately from the bone positions and nothing forced them
# to meet.
#
# The correction is computed from the geometry, not nudged by eye: the shoulder is placed where the
# chest surface actually is, plus the arm's own radius, and the hand is drawn back along the forearm
# until the two touch. Limb LENGTHS are untouched - only where the chain attaches moves.
ATTACH_ARMS = True

# **Torso breadth.** The authored chest is 0.42 m wide (half-extent 0.21); an adult male's is about
# 0.32. Because the shoulders are now derived from the chest surface, an over-wide torso pushes the
# whole arm chain out with it - the elbows ended up 0.54 m apart against a human's ~0.40-0.44.
#
# Only the LATERAL half-extent changes. Depth, height and mass are untouched, so the torso keeps its
# weight and MuJoCo recomputes the inertia from the new shape, which is the more correct one.
CHEST_HALF_WIDTH = 0.16      # metres -> 0.32 m breadth

# A full anthropometric rebuild - every segment LENGTH on its Winter (2009) fraction of stature -
# was written, measured and REMOVED. It put the landmarks where a human's are and still looked
# wrong: rounded ellipsoid segments read as stacked discs, and no arrangement of ~14 primitives
# matches a modelled mannequin. Physics colliders and the visual body are different objects, and
# trying to make one do both cost several rounds. The colliders keep the AUTHORED proportions,
# uniformly scaled to SCALE_TO_HEIGHT below, which read acceptably; a skinned mesh is the right
# answer for how the dummy LOOKS. See isaac_lab_3/SIM-TO-SIM.md for the measurements.

# Uniformly scale the authored rig to this standing height, keeping every proportion it already has.
# Mass is set directly rather than scaled by s^3: cubing 80.6 kg for a 4% taller body gives 91 kg
# (BMI 30), where the point of the change was a normal build.
SCALE_TO_HEIGHT = 1.75
# **Chest top on the shoulder line, and a neck in the space that frees.**
# The authored chest reaches 10.4 cm ABOVE the shoulder joint, which is the boxy collar either side
# of the head and why the skull looks like it rests on a crate. Dropping the top to the shoulder
# leaves exactly that 10.4 cm for a neck, so one change fixes both.
CHEST_TOP_AT_SHOULDER = True
NECK = True
NECK_RADIUS = 0.060

# **A shoulder yoke - the trapezius.** With the chest top cut back to the shoulder line, the torso
# ends in a flat plate 33 cm across and the neck rises out of the middle of it like a pipe. A real
# upper body has no flat top: it slopes up from the shoulders into the neck, and that slope is what
# stops the head reading as bolted on.
#
# A rounded mound on the chest, narrower than the chest and rising above its top, so the silhouette
# goes shoulder -> slope -> neck instead of shoulder -> plate -> pipe.
# **Anchor each limb capsule at its own joint.** A capsule centred on the body origin has its
# proximal cap 4.2 cm BELOW the shoulder, so when the arm swings about that joint the cap arcs away
# and opens a gap - measured up to 9.1 cm in abduction, which is why the arms looked detached
# whenever the dummy fell with its arms out. Extending the proximal end onto the joint means the cap
# centre IS the centre of rotation, so it cannot swing away from anything sitting there.
ANCHOR_CAPSULES = True

# **Put the shoulder joint on the arm's own axis.** The authored rig anchors it 12.5 cm INBOARD of
# the upper arm, so the arm rotates like a spoke about a point nowhere near its own top end: in
# abduction the capsule swings 9.1 cm clear of anything covering the joint, which is what made the
# arms look detached whenever the dummy fell with its arms out.
#
# A real glenohumeral joint sits at the top of the humerus, on its axis - so removing the lateral
# offset is the anatomically correct move, not a cosmetic one. The joint keeps its height.
SHOULDER_ON_ARM_AXIS = True

# Head diameter. The authored sphere scales to 29.2 cm, nearly twice a human head's 15.5 cm width,
# and it made every other part look small beside it. 20 cm is the middle ground: closer to human
# than the sphere, without the elongation that made a correctly-proportioned ellipsoid read as an
# egg. The head is RAISED as it shrinks so the top still lands on standing height.
#
# It is set 1 cm below the 21 cm that read well on its own because head and neck are COUPLED: the
# crown is pinned to standing height, so every millimetre taken off the neck is handed back to the
# head by the final normalisation. Shortening the neck alone grew the head to 22.2 cm.
HEAD_DIAMETER = 0.20

# Neck between the shoulder line and the base of the skull. Shrinking the head while holding the
# crown at standing height pushed all the freed space into the neck - 18.6 cm of it, against a
# human's ~11 - so the neck is set DIRECTLY and the body is rescaled afterwards to restore height.
#
# Most of this is NOT visible: the shaped chest top (the yoke, YOKE_RISE above the chest) fills
# 7.8 cm of it, so 11 cm left only 3.7 cm of bare column and 9.5 cm leaves 2.1 cm. Below ~8.5 cm the
# skull sits straight on the yoke and the neck disappears, which is the look this replaced.
NECK_LENGTH = 0.095

SHOULDER_YOKE = True
YOKE_WIDTH = 0.62        # fraction of the chest's own width
YOKE_RISE = 0.075        # metres it stands above the chest top
NECK_MASS = 1.0              # kg, taken out of the chest
# **Deltoids.** Without them the arm capsule only touches the torso in the REST pose: rotate the
# shoulder and the capsule swings clear, opening a visible gap at the joint - most obvious when the
# dummy is on the ground with its arms thrown out. A sphere centred on the shoulder joint belongs to
# the CHEST, so it stays put while the arm moves and covers the joint at every angle.
DELTOIDS = True
TARGET_MASS = 70.0           # kg -> BMI 22.9
BALL_RADIUS = 0.09

BALL_BLOCK = """    <!-- The perturbation projectile. Godot's BallGun fires a 3.0 kg ball of radius 0.06 m; this
         one is heavier and larger - BALL_MASS and BALL_RADIUS in build_mjcf.py say why - and is
         fired from 2.0 m at one of twelve target bones. `solref` is negative so the contact pair
         is given directly as (stiffness, damping) and the impact is springy rather than
         near-plastic. Parked far away and ABOVE the floor when idle - parking below an infinite
         plane means penetrating it, which MuJoCo resolves by ejecting the ball violently. -->
    <body name="ball" pos="40 40 2">
      <freejoint name="ball_free"/>
      <geom name="g_ball" type="sphere" size="{radius}" mass="{mass}" solref="-8000 -30"/>
    </body>
"""


def num(s: str) -> float:
    """Godot prints under a Spanish locale, so the decimal separator is a comma."""
    return float(s.replace(",", "."))


def to_mj(v):
    """Godot (x, y, z) -> MuJoCo (-z, -x, y). Same proper rotation the Isaac side uses."""
    x, y, z = v
    return (-z, -x, y)


def parse_shapes():
    t = SCENE.read_text(encoding="utf-8", errors="replace")
    shapes = {}
    for m in re.finditer(r'\[sub_resource type="(\w+)" id="Shape_(\w+)"\]\n((?:(?!\[)[^\n]*\n)*)', t):
        kind, name, body = m.group(1), m.group(2), m.group(3)
        if v := re.search(r"size = Vector3\(([^)]+)\)", body):
            sx, sy, sz = [float(x) for x in v.group(1).split(",")]
            mx, my, mz = to_mj((sx, sy, sz))
            shapes[name] = ("box", (abs(mx) / 2, abs(my) / 2, abs(mz) / 2))
        elif (r := re.search(r"radius = ([\d.]+)", body)) and (h := re.search(r"height = ([\d.]+)", body)):
            rad, height = float(r.group(1)), float(h.group(1))
            # Godot capsule height INCLUDES the two caps; MuJoCo wants the cylinder half-length.
            shapes[name] = ("capsule", (rad, max(1e-4, (height - 2 * rad) / 2)))
        elif r := re.search(r"radius = ([\d.]+)", body):
            shapes[name] = ("sphere", (float(r.group(1)),))
    return shapes


def parse_plant():
    bones = {}
    for line in DUMP.read_text(encoding="utf-8", errors="replace").splitlines():
        if "bone=" not in line:
            continue
        f = dict(re.findall(r"(\w+)=([^\s]+)", line))
        bones[f["bone"]] = {
            "mass": num(f["mass"]),
            "parent": f["parent"],
            "pos": tuple(num(x) for x in f["posWorld"].split("|")),
            "pivot": tuple(num(x) for x in f["pivotOffset"].split("|")),
        }
    return bones


def parse_effective_gains():
    """Godot's EFFECTIVE per-bone gains, from its own [PLANT] dump.

    `ActiveBone` drives every joint through Tan-Liu-Turk Stable PD, which divides the authored gain
    by `1 + Kd*dt/I + Kp*dt^2/I`. Godot therefore never applies the authored number - it applies
    `kpEff`, which the dump reports alongside. Measured 2026-09-09 the two differ by 2.5x at the
    spine, 3.0x at the thigh, 4.8x at the shin and 6.0x at the foot.

    Copying the AUTHORED gain into MuJoCo's position actuator builds a body several times stiffer
    than the character in the Godot scene: it lands like a plank, and - the part that matters for
    training - it holds its rest pose so rigidly that a zero action stands for 40 s at 100% upright,
    which hands every policy a statue as the free optimum.
    """
    gains = {}
    for line in DUMP.read_text(encoding="utf-8", errors="replace").splitlines():
        if "bone=" not in line:
            continue
        f = dict(re.findall(r"(\w+)=([^\s]+)", line))
        gains[f["bone"]] = (num(f["kpEff"]), num(f["kdEff"]))
    return gains


def add_rest_keyframe(path):
    """Append the `rest` keyframe - qpos0 with the shoulders abducted out of the legs.

    This has to happen after the file is written rather than during emission, because the qpos
    vector's layout is only known once MuJoCo has compiled the model, and it differs between the
    plain and the ball-carrying variants. Everything that starts an episode - both environments and
    the Godot bridge - resets to this keyframe instead of to qpos0. See REST_ABDUCTION.
    """
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(path))
    q = m.qpos0.copy()
    for side, sign in (("L", +1.0), ("R", -1.0)):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"UpperArm_{side}_rz")
        if j >= 0:
            q[m.jnt_qposadr[j]] = sign * REST_ABDUCTION
    text = path.read_text(encoding="utf-8")
    key = ('  <keyframe>' + chr(10)
           + '    <key name="rest" qpos="' + " ".join(f"{v:.6f}" for v in q) + '"/>' + chr(10)
           + '  </keyframe>' + chr(10))
    path.write_text(text.replace("</mujoco>", key + "</mujoco>"), encoding="utf-8")

    # The pose must be self-clearing: a start pose with the arms buried in the legs is what this
    # whole keyframe exists to fix, so verify it rather than assume it.
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    body = [c for c in d.contact[:d.ncon]
            if m.geom_type[c.geom1] != mujoco.mjtGeom.mjGEOM_PLANE
            and m.geom_type[c.geom2] != mujoco.mjtGeom.mjGEOM_PLANE]
    if body:
        names = {tuple(sorted((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                               mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2))))
                 for c in body}
        raise SystemExit(f"{path.name}: the rest keyframe still self-intersects: {sorted(names)}")


class Proportions:
    """Dimensions DERIVED while the rig is reshaped - scaled, retargeted, measured - and read when
    the model is emitted.

    They used to be the module constants above, rewritten in place through `globals()` halfway
    through a build, so the same name held its authored value or its scaled one depending on when
    it was read.
    """

    def __init__(self):
        self.chest_half_width = CHEST_HALF_WIDTH
        self.toe_length = TOE_LENGTH
        self.neck_radius = NECK_RADIUS
        self.yoke_rise = YOKE_RISE
        self.neck_gap = 0.0         # set by the chest-top retarget
        self.shoulder_half = 0.0    # measured once the arms are placed


def standing_height(shapes, bones):
    """Top of the head to the sole, in the rig's current state."""
    head_r = shapes[SHAPE_FOR["Head"]][1][0]
    foot_half_h = shapes[SHAPE_FOR["Foot_L"]][1][2]
    return (bones["Head"]["pos"][1] + head_r) - (bones["Foot_L"]["pos"][1] - foot_half_h)


def scale_rig(shapes, bones, s):
    """Scale every position, joint anchor and collider by `s`, about the origin."""
    for b in bones.values():
        b["pos"] = tuple(v * s for v in b["pos"])
        b["pivot"] = tuple(v * s for v in b["pivot"])
    for key, (kind, dims) in list(shapes.items()):
        shapes[key] = (kind, tuple(v * s for v in dims))


def scale_to_height(shapes, bones, p):
    """Uniformly scale the AUTHORED rig to SCALE_TO_HEIGHT, and set its mass to TARGET_MASS."""
    s = SCALE_TO_HEIGHT / standing_height(shapes, bones)
    scale_rig(shapes, bones, s)
    total = sum(b["mass"] for b in bones.values())
    for b in bones.values():
        b["mass"] = b["mass"] / total * TARGET_MASS
    p.chest_half_width = p.chest_half_width * s
    print(f"  scaled x{s:.4f} to {SCALE_TO_HEIGHT:.2f} m, mass set to {TARGET_MASS:.1f} kg")


def reshape_torso_and_head(shapes, bones, p):
    """Narrow the chest, shrink the head onto its neck, and cut the chest top to the shoulders."""
    if p.chest_half_width:
        kind, dims = shapes[SHAPE_FOR["Chest"]]
        shapes[SHAPE_FOR["Chest"]] = (kind, (dims[0], p.chest_half_width, dims[2]))

    if HEAD_DIAMETER:
        kind, dims = shapes[SHAPE_FOR["Head"]]
        new_r = HEAD_DIAMETER / 2.0
        shapes[SHAPE_FOR["Head"]] = (kind, (new_r,) + tuple(dims[1:]))
        # Place the skull by the NECK it should leave, not by the crown height. Height is restored
        # by normalise_height, which scales the whole body rather than stretching one gap.
        shoulder_y = bones["UpperArm_L"]["pos"][1] + bones["UpperArm_L"]["pivot"][1]
        hx, _, hz = bones["Head"]["pos"]
        bones["Head"]["pos"] = (hx, shoulder_y + NECK_LENGTH + new_r, hz)
        print(f"  head {2*new_r:.3f} m across, neck set to {NECK_LENGTH:.3f} m")

    if CHEST_TOP_AT_SHOULDER:
        # Godot frame: Y up. `shapes` holds MuJoCo half-extents, so the chest's vertical half is
        # dims[2]. The shoulder anchor is the upper arm's own pivot, in world terms.
        kind, dims = shapes[SHAPE_FOR["Chest"]]
        cy = bones["Chest"]["pos"][1]
        chest_bottom = cy - dims[2]
        shoulder_y = bones["UpperArm_L"]["pos"][1] + bones["UpperArm_L"]["pivot"][1]
        if shoulder_y > chest_bottom:
            new_half = (shoulder_y - chest_bottom) / 2.0
            shapes[SHAPE_FOR["Chest"]] = (kind, (dims[0], dims[1], new_half))
            bones["Chest"]["pos"] = (bones["Chest"]["pos"][0],
                                     (shoulder_y + chest_bottom) / 2.0,
                                     bones["Chest"]["pos"][2])
            head_r = shapes[SHAPE_FOR["Head"]][1][0]
            head_bottom = bones["Head"]["pos"][1] - head_r
            p.neck_gap = max(0.0, head_bottom - shoulder_y)
            print(f"  chest top -> shoulder line (half {dims[2]:.3f} -> {new_half:.3f}), "
                  f"neck gap {p.neck_gap:.3f} m")


def place_arms(shapes, bones, p):
    """Put each shoulder joint on its arm's own axis, and draw the arm chain in to meet the chest."""
    if SHOULDER_ON_ARM_AXIS:
        for side in ("L", "R"):
            arm = f"UpperArm_{side}"
            if arm in bones:
                px, py, pz = bones[arm]["pivot"]
                bones[arm]["pivot"] = (0.0, py, pz)      # drop the lateral component only

    if ATTACH_ARMS:
        # Godot frame here: X is lateral, Y is up. `shapes` holds MuJoCo half-extents, and the
        # chest's MuJoCo Y half-extent is its Godot X half-width.
        chest_half_x = shapes[SHAPE_FOR["Chest"]][1][1]
        for side in ("L", "R"):
            arm, fore, hand = f"UpperArm_{side}", f"Forearm_{side}", f"Hand_{side}"
            if arm not in bones:
                continue
            arm_radius = shapes[SHAPE_FOR[arm]][1][0]
            want = chest_half_x + arm_radius            # inner surface just touching the chest
            x, y, z = bones[arm]["pos"]
            shift = abs(x) - want
            if shift > 0.0:
                # Move the whole chain inward together; children are emitted relative to the parent,
                # but their WORLD positions are read here, so each is shifted explicitly.
                for b_name in (arm, fore, hand):
                    if b_name in bones:
                        bx, by, bz = bones[b_name]["pos"]
                        bones[b_name]["pos"] = (bx - (shift if bx > 0 else -shift), by, bz)
            # Close the wrist along the arm's long axis (Godot Y, pointing down).
            if hand in bones and fore in bones:
                fr, fh = shapes[SHAPE_FOR[fore]][1][0], shapes[SHAPE_FOR[fore]][1][1]
                hand_half = shapes[SHAPE_FOR[hand]][1][2]     # Godot Y half-extent
                span = abs(bones[hand]["pos"][1] - bones[fore]["pos"][1])
                gap = span - (fh + fr) - hand_half
                if gap > 0.0:
                    hx, hy, hz = bones[hand]["pos"]
                    bones[hand]["pos"] = (hx, hy + gap, hz)   # draw the hand up toward the elbow

    # The deltoids need to know where the shoulder ended up.
    p.shoulder_half = abs(bones["UpperArm_L"]["pos"][0])


def normalise_height(shapes, bones, p):
    """Make the standing height exact again, and put the soles on the floor."""
    # Every earlier step may have moved the crown - shrinking the head, retargeting the neck. One
    # uniform scale at the end makes standing height exact without disturbing any of the
    # attachments, because positions, colliders and the derived dimensions all scale together.
    s2 = SCALE_TO_HEIGHT / standing_height(shapes, bones)
    if abs(s2 - 1.0) > 1e-6:
        scale_rig(shapes, bones, s2)
        p.toe_length = p.toe_length * s2
        p.neck_radius = p.neck_radius * s2
        p.yoke_rise = p.yoke_rise * s2
        p.neck_gap = p.neck_gap * s2
        p.shoulder_half = p.shoulder_half * s2
        print(f"  normalised x{s2:.4f} -> standing height exactly {SCALE_TO_HEIGHT:.3f} m")

    # **Put the soles ON the floor.** The scale above is about the origin, so a sole that was
    # not already at y = 0 is scaled away from it: the body spawned 1.09 cm in the air and
    # dropped at the start of every episode, adding an impact the policy had to absorb before
    # it could do anything. Measured, not assumed - `validate.py` asserts it.
    drop = bones["Foot_L"]["pos"][1] - shapes[SHAPE_FOR["Foot_L"]][1][2]
    if abs(drop) > 1e-6:
        for b in bones.values():
            x, y, z = b["pos"]
            b["pos"] = (x, y - drop, z)
        print(f"  lowered {drop * 100:+.2f} cm so the soles rest on the floor")


class Emitter:
    """Writes the MJCF body tree depth first, and collects the actuators in the same order."""

    def __init__(self, shapes, bones, joints, effective, p):
        self.shapes, self.bones, self.joints, self.effective, self.p = (
            shapes, bones, joints, effective, p)
        self.lines, self.actuators = [], []
        self.children = {}
        for name, b in bones.items():
            self.children.setdefault(b["parent"], []).append(name)

    def tree(self):
        root = next(n for n, b in self.bones.items() if b["parent"] == "-")
        self.body(root, 3)

    def body(self, name, depth):
        b = self.bones[name]
        pad = "  " * depth
        parent = b["parent"]
        wp = to_mj(b["pos"])
        pp = to_mj(self.bones[parent]["pos"]) if parent in self.bones else (0.0, 0.0, 0.0)
        rel = tuple(a - c for a, c in zip(wp, pp))
        self.lines.append(f'{pad}<body name="{name}" pos="{rel[0]:.6f} {rel[1]:.6f} {rel[2]:.6f}">')
        if parent == "-":
            self.lines.append(f'{pad}  <freejoint name="root"/>')
        else:
            for ax in ("x", "y", "z"):
                self.joint(name, ax, pad)
        self.geoms(name, pad)
        for c in sorted(self.children.get(name, [])):
            self.body(c, depth + 1)
        self.lines.append(f"{pad}</body>")

    # ------------------------------------------------------------ joints
    def joint(self, name, ax, pad):
        """One hinge of `name` about `ax`: its range, its spring, and its actuator if it has one."""
        key = f"{name}_r{ax}"
        stem = name.split("_")[0]
        # A hinge joint keeps only its flexion axis; the other two are dropped entirely rather
        # than locked, so they cost nothing in the solver either.
        if key not in self.joints or (stem in HINGE_ONLY and ax != "x"):
            return
        spec = self.joints[key]
        lo, hi = self.joint_range(key, spec)
        stop = NECK_LIMIT_SOLVER if stem == "Head" else ""
        arm = "" if stem == "Head" else f' armature="{JOINT_ARMATURE:.4f}"'
        jp = to_mj(self.bones[name]["pivot"])
        self.lines.append(
            f'{pad}  <joint name="{key}" type="hinge" axis="{AXIS_VEC[ax]}" '
            f'pos="{jp[0]:.6f} {jp[1]:.6f} {jp[2]:.6f}" '
            f'range="{lo:.4f} {hi:.4f}" limited="true"{arm}{self.spring(name, key, spec)}{stop}/>')
        self.actuate(name, key, spec)

    @staticmethod
    def joint_range(key, spec):
        lo, hi = float(spec["lower"]), float(spec["upper"])
        # `dummy_rig.json` states limits in ISAAC's convention; `reversed` marks the axes
        # where isaac = -godot, so those flip to stay in the Godot sense used here.
        if spec.get("reversed"):
            lo, hi = -hi, -lo
        # Parent-child contact is off, so a joint limit is the only thing stopping a
        # segment entering its parent. See NECK_LIMIT.
        if key in NECK_LIMIT:
            nlo, nhi = NECK_LIMIT[key]
            lo = lo if nlo is None else max(lo, nlo)
            hi = hi if nhi is None else min(hi, nhi)
        return lo, hi

    def spring(self, name, key, spec):
        """The joint's passive spring: a ligament under a torque actuator, or a passive joint's own."""
        stem = name.split("_")[0]
        if not spec.get("actuated"):
            # **Passive joints still carry gains.** `dummy_rig.json` marks Head and Hand joints
            # actuated=False, but gives them stiffness 120 / damping 12, and Godot holds them at
            # rest through NeutraliseUncommandedBones. Emitting them as FREE hinges left the head a
            # 5 kg sphere on a frictionless pivot: gravity rotated it down and, since MuJoCo
            # disables parent-child contact, it sank into the chest after ~3 s.
            # They get the SAME treatment as an actuated gain: Godot's effective value rather than
            # the authored one, then the human override.
            ks, kd = float(spec["stiffness"]), float(spec["damping"])
            if EFFECTIVE_GAINS and name in self.effective:
                ks, kd = self.effective[name]
            if HUMAN_STRENGTH and stem in HUMAN_PASSIVE:
                ks, kd = HUMAN_PASSIVE[stem]
            return f' stiffness="{ks:.3f}" damping="{kd:.3f}"'
        if ACTUATOR_MODE != "torque":
            # A position actuator carries the damping as `kv`, so the joint must not also be
            # damped here or the damping is applied twice.
            return ""
        # Ligament. In torque mode the actuator carries no damping of its own, so this is the only
        # thing that stops a joint ringing - see LIGAMENT.
        ks = LIGAMENT * HUMAN_TORQUE.get(stem, float(spec["effort"]))
        spring = f' stiffness="{ks:.3f}" damping="{LIGAMENT_DAMP * ks:.3f}"'
        if key in ("UpperArm_L_rz", "UpperArm_R_rz"):
            ref = REST_ABDUCTION * (1.0 if key.startswith("UpperArm_L") else -1.0)
            spring += f' springref="{ref:.4f}"'
        return spring

    def actuate(self, name, key, spec):
        """The joint's actuator, if it has one: the neck's muscle, or the policy's own."""
        stem = name.split("_")[0]
        if NECK_ACTUATED and stem == "Head" and not spec.get("actuated"):
            # The muscle layer. Held at zero by whoever owns the model - the env writes nothing to
            # these indices, and a zero position target is "head straight".
            kp, kv = NECK_GAIN
            effort = HUMAN_TORQUE["Head"]
            self.actuators.append(
                f'    <position name="{key}" joint="{key}" '
                f'kp="{kp:.1f}" kv="{kv:.3f}" '
                f'forcerange="{-effort:.1f} {effort:.1f}"/>')
        if not spec.get("actuated"):
            return
        effort = HUMAN_TORQUE.get(stem, float(spec["effort"]))
        if not HUMAN_STRENGTH:
            effort = float(spec["effort"])
        if ACTUATOR_MODE == "torque":
            # The policy's output IS the torque.
            self.actuators.append(
                f'    <motor name="{key}" joint="{key}" gear="1" '
                f'ctrllimited="true" ctrlrange="{-effort:.1f} {effort:.1f}" '
                f'forcerange="{-effort:.1f} {effort:.1f}"/>')
            return
        # Godot drives `tau = kp(target - q) - kd*qd`. MuJoCo's position actuator is exactly that
        # with kp AND kv, so the damping belongs on the ACTUATOR, not as joint damping: kp alone is
        # a spring with no velocity feedback, which at the authored 450-1800 oscillates the limb
        # apart.
        kp, kv = float(spec["stiffness"]), float(spec["damping"])
        if EFFECTIVE_GAINS and name in self.effective:
            kp, kv = self.effective[name]
        kp *= GAIN_SCALE
        kv *= GAIN_SCALE
        self.actuators.append(
            f'    <position name="{key}" joint="{key}" '
            f'kp="{kp:.1f}" kv="{kv:.3f}" '
            f'forcerange="{-effort:.1f} {effort:.1f}"/>')

    # ------------------------------------------------------------ colliders
    def geoms(self, name, pad):
        """The body's collider, and the chest's extra shapes: yoke, neck and deltoids."""
        b = self.bones[name]
        kind, dims = self.shapes[SHAPE_FOR[name]]
        mass = b["mass"]
        chest = name == "Chest"
        if SHOULDER_YOKE and chest:
            # MuJoCo half-extents on the chest's own frame: x fwd, y lateral, z up. Centred ON the
            # chest top so half of it sits inside the chest and half rises above, which is what
            # makes the join read as a slope rather than a step.
            self.lines.append(
                f'{pad}  <geom name="g_Yoke" type="ellipsoid" '
                f'size="{dims[0] * 0.86:.4f} {dims[1] * YOKE_WIDTH:.4f} {self.p.yoke_rise:.4f}" '
                f'pos="0 0 {dims[2]:.4f}" mass="1.2"/>')
        if NECK and chest:
            mass = max(0.1, mass - NECK_MASS)     # the neck's mass comes out of the chest
        if SHOULDER_YOKE and chest:
            mass = max(0.1, mass - 1.2)           # and so does the yoke's

        if ADD_TOE and name.startswith("Foot"):
            self.foot_and_toe(name, pad, kind, dims)
        elif ANCHOR_CAPSULES and kind == "capsule" and b["parent"] in self.bones:
            # Extend the proximal end onto this body's own joint. `pivot` is that joint relative to
            # the body origin, in Godot's frame; the capsule runs along MuJoCo's local Z, which is
            # Godot's Y, so only the vertical component matters.
            p_dist = abs(b["pivot"][1])
            half = max(dims[1], (p_dist + dims[1]) / 2.0)
            offset = (p_dist - dims[1]) / 2.0
            self.lines.append(f'{pad}  <geom name="g_{name}" type="capsule" '
                              f'size="{dims[0]:.4f} {half:.4f}" pos="0 0 {offset:.4f}" '
                              f'mass="{mass:.4f}"/>')
        else:
            size = " ".join(f"{d:.4f}" for d in dims)
            self.lines.append(f'{pad}  <geom name="g_{name}" type="{kind}" size="{size}" '
                              f'mass="{mass:.4f}"/>')

        if NECK and chest:
            # A NECK. The head is placed by its own landmark (its top at standing height), which
            # correctly leaves ~9 cm between the chest and the skull - the neck. Nothing occupied
            # that space, so the head rendered as though it were floating. This fills it, and gives
            # the neck a collider, without adding a joint: the head's own joint IS the neck joint.
            half = self.p.neck_gap / 2.0
            self.lines.append(f'{pad}  <geom name="g_Neck" type="capsule" '
                              f'size="{self.p.neck_radius:.4f} {half:.4f}" '
                              f'pos="0 0 {dims[2] + half:.4f}" mass="{NECK_MASS:.4f}"/>')
        if DELTOIDS and chest:
            # Just larger than the arm it caps - nothing more.
            #
            # The first version also added the chest-to-shoulder distance, on the assumption there
            # was a gap to bridge. There is not: ATTACH_ARMS already pulls the arm in until its
            # inner surface meets the chest. That phantom term produced a 23.8 cm sphere - 81% of
            # the head diameter - which is what made the shoulders look like shoulder pads.
            r = self.shapes[SHAPE_FOR["UpperArm_L"]][1][0] * 1.15
            # Chest frame: +Y is the body's left, +Z up. The shoulder joint sits at the chest top.
            for sy in (+1.0, -1.0):
                self.lines.append(
                    f'{pad}  <geom name="g_Deltoid_{"L" if sy > 0 else "R"}" type="sphere" '
                    f'size="{r:.4f}" pos="0 {sy * self.p.shoulder_half:.4f} {dims[2]:.4f}" '
                    f'mass="0.35"/>')

    def foot_and_toe(self, name, pad, kind, dims):
        """The foot as a heel box plus a toe on its own hinge - see ADD_TOE."""
        # Shorten the foot along its long axis (MuJoCo X) and hand the front to the toe.
        half = self.p.toe_length / 2.0
        dims = (dims[0] - half, dims[1], dims[2])
        mass = FOOT_MASS - TOE_MASS
        size = " ".join(f"{d:.4f}" for d in dims)
        self.lines.append(f'{pad}  <geom name="g_{name}" type="{kind}" size="{size}" '
                          f'pos="{-half:.4f} 0 0" mass="{mass:.4f}"/>')
        side = name.split("_")[1]
        # The FRONT FACE, not the half-extent: the geom is offset back by `half`, so its face
        # is at (-half + dims[0]). Using the half-extent left a 3 cm gap between foot and toe
        # and stretched the footprint from the authored 0.22 m to 0.25.
        hinge = -half + dims[0]
        self.lines.append(f'{pad}  <body name="Toe_{side}" pos="{hinge:.4f} 0 0">')
        toe_lig = (f' stiffness="{LIGAMENT * TOE_TORQUE:.3f}" '
                   f'damping="{LIGAMENT_DAMP * LIGAMENT * TOE_TORQUE:.3f}"'
                   if ACTUATOR_MODE == "torque" else "")
        self.lines.append(f'{pad}    <joint name="Toe_{side}_rx" type="hinge" '
                          f'axis="{AXIS_VEC["x"]}" pos="0 0 0" '
                          f'range="{TOE_RANGE[0]:.4f} {TOE_RANGE[1]:.4f}" '
                          f'limited="true" armature="{JOINT_ARMATURE:.4f}"{toe_lig}/>')
        self.lines.append(f'{pad}    <geom name="g_Toe_{side}" type="box" '
                          f'size="{half:.4f} {dims[1]:.4f} {dims[2]:.4f}" '
                          f'pos="{half:.4f} 0 0" mass="{TOE_MASS:.4f}"/>')
        self.lines.append(f'{pad}  </body>')
        self.actuators.append(
            f'    <motor name="Toe_{side}_rx" joint="Toe_{side}_rx" gear="1" '
            f'ctrllimited="true" ctrlrange="{-TOE_TORQUE:.1f} {TOE_TORQUE:.1f}" '
            f'forcerange="{-TOE_TORQUE:.1f} {TOE_TORQUE:.1f}"/>'
            if ACTUATOR_MODE == "torque" else
            f'    <position name="Toe_{side}_rx" joint="Toe_{side}_rx" '
            f'kp="120.0" kv="1.200" '
            f'forcerange="{-TOE_TORQUE:.1f} {TOE_TORQUE:.1f}"/>')


def write_models(emitter, bones):
    """Write the clean model and the one carrying the projectile, then give both the rest pose."""
    total = sum(b["mass"] for b in bones.values())
    nl = chr(10)
    header = (
        '<mujoco model="p4f_dummy">\n'
        '  <!-- Generated by mujoco_rig/build_mjcf.py from Godot\'s own [PLANT] dump. Do not\n'
        '       hand-edit: regenerate instead, so the Godot scene stays the single source of truth\n'
        f'       for the rig. {len(bones)} bodies, {total:.2f} kg total. -->\n'
        '  <compiler angle="radian" autolimits="true"/>\n'
        '  <option timestep="0.004167" integrator="implicitfast"/>\n'
        '  <worldbody>\n'
        '    <geom name="floor" type="plane" size="60 60 0.1" pos="0 0 0"/>\n'
        + nl.join(emitter.lines) + nl)
    if NO_ACTUATORS:
        footer = '  </worldbody>' + nl + '</mujoco>' + nl
    else:
        footer = ('  </worldbody>\n'
                  '  <actuator>\n'
                  + nl.join(emitter.actuators) + nl
                  + '  </actuator>\n'
                  '</mujoco>\n')

    OUT.write_text(header + footer, encoding="utf-8")
    with_ball = OUT.with_name(OUT.stem + "_ball.xml")
    with_ball.write_text(
        header + BALL_BLOCK.format(radius=BALL_RADIUS, mass=BALL_MASS) + footer,
        encoding="utf-8")
    for path in (OUT, with_ball):
        add_rest_keyframe(path)
    print(f"wrote {OUT.name} (no projectile) and {with_ball.name} (with projectile)")
    print(f"  {len(bones)} bodies, {total:.2f} kg, {len(emitter.actuators)} actuators")


def main() -> int:
    shapes, bones = parse_shapes(), parse_plant()
    if not bones:
        print("no [PLANT] lines found - regenerate mujoco_rig/plant_dump.txt", file=sys.stderr)
        return 1
    joints = json.loads(RIG.read_text(encoding="utf-8"))["joints"]
    effective = parse_effective_gains()

    p = Proportions()
    if SCALE_TO_HEIGHT:
        scale_to_height(shapes, bones, p)
    reshape_torso_and_head(shapes, bones, p)
    place_arms(shapes, bones, p)
    if SCALE_TO_HEIGHT:
        normalise_height(shapes, bones, p)

    emitter = Emitter(shapes, bones, joints, effective, p)
    emitter.tree()
    write_models(emitter, bones)
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gains", choices=("authored", "effective"), default="effective",
                    help="authored = the rig's raw kp (2.5-6x stiffer than Godot actually applies); "
                         "effective = the kpEff Godot's Stable PD really uses")
    ap.add_argument("--no_actuators", action="store_true",
                    help="emit a limp ragdoll: no actuators, only gravity and the joint limits")
    ap.add_argument("--gain_scale", type=float, default=1.0,
                    help="uniform multiplier on every actuator gain")
    ap.add_argument("--out", default="", help="override the output stem, e.g. dummy_limp")
    cli = ap.parse_args()
    EFFECTIVE_GAINS = cli.gains == "effective"
    GAIN_SCALE = cli.gain_scale
    NO_ACTUATORS = cli.no_actuators
    if cli.out:
        OUT = ROOT / "mujoco_rig" / (cli.out + ".xml")
    raise SystemExit(main())
