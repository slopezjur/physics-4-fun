"""Projectile-world validation and ballistic launches; the character contract stays fixed."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

from .rig import Rig


def set_curriculum_ball_mass(engine, mass):
    """Scale only the projectile mass and inertia in an isolated native world."""
    if not np.isfinite(mass) or not .1 <= mass <= 10.:
        raise ValueError("Curriculum ball mass must be within 0.1–10 kg")
    model = engine.rig.model
    body = model.body("ball").id
    ratio = mass / model.body_mass[body]
    model.body_mass[body] = mass
    model.body_inertia[body] *= ratio
    mujoco.mj_setConst(model, engine.datas[0])
    for data in engine.datas:
        mujoco.mj_forward(model, data)


def load_ball_rig(character, path=None):
    path = Path(path) if path is not None else character.path.with_name("dummy_ball.xml")
    plain, world = ET.parse(character.path).getroot(), ET.parse(path).getroot()
    ball = world.find("./worldbody/body[@name='ball']")
    if ball is None:
        raise ValueError("Projectile model has no ball")
    world.find("worldbody").remove(ball)
    # Compare the authoring trees, ignoring only the appended projectile and keyframe tail.
    for root in (plain, world):
        root.remove(root.find("keyframe"))
        root.attrib.pop("model", None)
        for node in root.iter():
            node.text = node.tail = None
    if ET.tostring(plain) != ET.tostring(world):
        raise ValueError("Projectile world changes the character or its physics")
    rig = Rig.load(path)
    joint = rig.model.joint("ball_free").id
    if rig.model.nq != character.model.nq + 7 or rig.model.nv != character.model.nv + 6 \
            or rig.model.jnt_qposadr[joint] != character.model.nq or rig.model.jnt_dofadr[joint] != character.model.nv:
        raise ValueError("Projectile must be appended after the complete character state")
    return rig


def launch_state(target, direction, speed, gravity, distance=.65):
    """Aim a ballistic shot at the launch-time target; do not cancel gravity or guide it."""
    target, direction = np.asarray(target, dtype=np.float32), np.asarray(direction, dtype=np.float32)
    if target.shape[-1] != 3 or not np.isfinite(target).all() or not np.isfinite(direction).all() \
            or not np.isfinite(speed).all() or np.any(np.asarray(speed) <= 0) or distance <= 0:
        raise ValueError("Invalid ballistic launch")
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    if np.any(norm < 1e-6) or np.any(np.abs(direction[..., 2]) > 1e-6):
        raise ValueError("Launch direction must be horizontal and nonzero")
    direction = direction / norm
    speed = np.asarray(speed, dtype=np.float32)[..., None]
    time = distance / speed
    position = target - direction * distance
    velocity = direction * speed - .5 * np.asarray(gravity, dtype=np.float32) * time
    return position, velocity
