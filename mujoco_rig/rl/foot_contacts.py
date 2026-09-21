"""Floor load and tangential slip of foot/toe contact points on MuJoCo C.

Measurements are read-only and taken from the final physics substep, like cvel
and xpos. Neither a ball contact nor a raised foot near the floor is support.
"""
import mujoco
import numpy as np


LOAD_ON_NEWTONS = 5.0
LOAD_OFF_NEWTONS = 1.0


def loaded_feet(xp, load, was_grounded):
    """Load hysteresis in newtons; keep a light existing contact until it unloads."""
    return load > xp.where(was_grounded, LOAD_OFF_NEWTONS, LOAD_ON_NEWTONS)


def foot_geom_sides(model):
    """Map every foot/toe geometry to left=0, right=1; other geometries to -1."""
    sides = np.full(model.ngeom, -1, dtype=np.int32)
    for side, suffix in enumerate(('_L', '_R')):
        bodies = [model.body(bone + suffix).id for bone in ('Foot', 'Toe')]
        sides[np.isin(model.geom_bodyid, bodies)] = side
    return sides


def tangential_speed_squared(spatial_velocity, reference, point, normal):
    """cvel is based at the root subtree COM, not at the foot or contact point."""
    velocity = spatial_velocity[..., 3:] + np.cross(spatial_velocity[..., :3], point - reference)
    tangent = velocity - (velocity * normal).sum(-1, keepdims=True) * normal
    return (tangent * tangent).sum(-1)


class FootContacts:
    def __init__(self, model):
        self.model = model
        self.floor = model.geom('floor').id
        self.sides = foot_geom_sides(model)

    def read(self, data):
        """Return per-foot normal load and load-weighted mean squared slip speed.

        Average squared speeds, not velocity vectors: opposite contact velocities
        during a twist must not cancel. Weighting avoids dependence on the number
        of manifold points. Units are N and m^2/s^2, respectively.
        """
        ids, geoms, loads = [], [], []
        force = np.zeros(6)
        for i in range(data.ncon):
            contact = data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if self.floor not in (g1, g2):
                continue
            geom = g2 if g1 == self.floor else g1
            if geom < 0 or self.sides[geom] < 0:
                continue
            mujoco.mj_contactForce(self.model, data, i, force)
            normal_load = max(0.0, float(force[0]))
            if normal_load == 0:
                continue
            ids.append(i)
            geoms.append(geom)
            loads.append(normal_load)
        if not ids:
            return np.zeros(2), np.zeros(2)
        # Batch the kinematic transforms; a per-contact np.cross dominates CPU
        # scoring time across hundreds of thousands of world steps.
        bodies = self.model.geom_bodyid[geoms]
        reference = data.subtree_com[self.model.body_rootid[bodies]]
        speed_sq = tangential_speed_squared(data.cvel[bodies], reference,
                                            data.contact.pos[ids], data.contact.frame[ids, :3])
        sides = self.sides[geoms]
        load = np.bincount(sides, weights=loads, minlength=2)
        weighted_slip = np.bincount(sides, weights=np.asarray(loads) * speed_sq, minlength=2)
        return load, weighted_slip / load.clip(1e-9, None)
