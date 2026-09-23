"""Conservative, stable box separation for offline clearance constraints."""
import mujoco
import numpy as np


def box_separation(position_a, rotation_a, size_a, position_b, rotation_b, size_b):
    """SAT separation lower bound, vectorized over pairs; negative means overlap.

    A positive projection gap is no larger than Euclidean distance. Requiring it
    to exceed a clearance is conservative. Near-parallel cross axes are omitted;
    the two boxes' six face normals always remain. Rotation columns are box axes.
    """
    a, b = rotation_a.swapaxes(-1, -2), rotation_b.swapaxes(-1, -2)
    cross = np.cross(a[..., :, None, :], b[..., None, :, :]).reshape(a.shape[:-2] + (9, 3))
    axes = np.concatenate((a, b, cross), axis=-2)
    norm = np.linalg.norm(axes, axis=-1)
    valid = norm > 1e-7
    axes = axes / np.maximum(norm[..., None], 1e-7)
    distance = np.abs(np.sum(axes * (position_b - position_a)[..., None, :], axis=-1))
    radius_a = (np.abs(axes @ rotation_a) * size_a[..., None, :]).sum(axis=-1)
    radius_b = (np.abs(axes @ rotation_b) * size_b[..., None, :]).sum(axis=-1)
    return np.where(valid, distance - radius_a - radius_b, -np.inf).max(axis=-1)


class CollisionClearance:
    """Use stable SAT bounds for box pairs and native distance for other shapes."""
    def __init__(self, model, pairs):
        self.model = model
        self.pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
        self.box = (model.geom_type[self.pairs] == mujoco.mjtGeom.mjGEOM_BOX).all(axis=1)

    def distances(self, data, limit=.05):
        result = np.full(len(self.pairs), limit)
        for i in np.flatnonzero(~self.box):
            result[i] = mujoco.mj_geomDistance(self.model, data, *self.pairs[i], limit, None)
        if self.box.any():
            a, b = self.pairs[self.box].T
            result[self.box] = np.minimum(limit, box_separation(
                data.geom_xpos[a], data.geom_xmat[a].reshape(-1, 3, 3), self.model.geom_size[a],
                data.geom_xpos[b], data.geom_xmat[b].reshape(-1, 3, 3), self.model.geom_size[b]))
        return result
