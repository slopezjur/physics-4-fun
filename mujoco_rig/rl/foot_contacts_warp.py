"""GPU floor-contact adapter; no host transfers or per-world Python loop."""
import numpy as np
import warp as wp
import mujoco_warp as mjw

from foot_contacts import foot_geom_sides


@wp.kernel
def _aggregate_floor_contacts(
    count: wp.array[int],
    geom: wp.array[wp.vec2i],
    world: wp.array[int],
    point: wp.array[wp.vec3],
    frame: wp.array[wp.mat33],
    force: wp.array[wp.spatial_vector],
    floor: int,
    sides: wp.array[int],
    geom_body: wp.array[int],
    root_body: wp.array[int],
    cvel: wp.array2d[wp.spatial_vector],
    subtree_com: wp.array2d[wp.vec3],
    load: wp.array2d[float],
    weighted_slip: wp.array2d[float],
):
    i = wp.tid()
    # Capacity includes stale slots after contacts disappear. Never read them.
    if i >= count[0]:
        return
    pair = geom[i]
    other = int(-1)
    if pair[0] == floor:
        other = pair[1]
    elif pair[1] == floor:
        other = pair[0]
    if other < 0:
        return
    side = sides[other]
    normal_load = wp.max(force[i][0], 0.0)
    if side < 0 or normal_load <= 0.0:
        return
    w = world[i]
    body = geom_body[other]
    v = cvel[w, body]
    arm = point[i] - subtree_com[w, root_body[body]]
    velocity = wp.spatial_bottom(v) + wp.cross(wp.spatial_top(v), arm)
    f = frame[i]
    normal = wp.vec3(f[0, 0], f[0, 1], f[0, 2])
    tangent = velocity - wp.dot(velocity, normal) * normal
    wp.atomic_add(load, w, side, normal_load)
    wp.atomic_add(weighted_slip, w, side, normal_load * wp.dot(tangent, tangent))


class FootContactsWarp:
    def __init__(self, model, warp_model, data, num_envs, device):
        self.model, self.data = warp_model, data
        self.floor = model.geom('floor').id
        self.sides = wp.array(foot_geom_sides(model), dtype=wp.int32, device=device)
        self.geom_body = wp.array(model.geom_bodyid, dtype=wp.int32, device=device)
        self.root_body = wp.array(model.body_rootid, dtype=wp.int32, device=device)
        capacity = data.contact.geom.shape[0]
        self.ids = wp.array(np.arange(capacity, dtype=np.int32), device=device)
        self.force = wp.zeros(capacity, dtype=wp.spatial_vector, device=device)
        self.load = wp.zeros((num_envs, 2), dtype=float, device=device)
        self.weighted_slip = wp.zeros((num_envs, 2), dtype=float, device=device)
        self._load_view = wp.to_torch(self.load)
        self._slip_view = wp.to_torch(self.weighted_slip)

    def read(self):
        """Same units, averaging and physics stage as FootContacts.read."""
        self.load.zero_()
        self.weighted_slip.zero_()
        d = self.data
        mjw.contact_force(self.model, d, self.ids, False, self.force)
        wp.launch(_aggregate_floor_contacts, dim=self.ids.size, inputs=[
            d.nacon, d.contact.geom, d.contact.worldid, d.contact.pos,
            d.contact.frame, self.force, self.floor, self.sides, self.geom_body,
            self.root_body, d.cvel, d.subtree_com, self.load, self.weighted_slip],
            device=self.load.device)
        return self._load_view, self._slip_view / self._load_view.clamp_min(1e-9)
