"""Godot's gravity feed-forward, reproduced in Isaac.

**Why this is not optional.** Measured body-wide in Godot under the walk policy, the feed-forward
supplies `ffShare = 0.45..0.68` of the holding torque - half to two-thirds. Isaac's
`ImplicitActuatorCfg` is a PD and nothing else. While Isaac trained at the full authored gains its
joints were stiff enough to hold a pose on the PD term alone, so the omission barely showed; once
`godot_plant.py` matched the gains DOWN onto Godot's Stable-PD plant the substitute was removed and
Isaac's body began to crouch (52.7% standing, head 1.459 against 1.540 upright). Godot without its
feed-forward does not crouch, it COLLAPSES - head 0.136 m, flat and motionless at every authority -
which is the cleanest evidence that the term is load-bearing rather than a refinement.

This mirrors `ActiveBone.ComputeLoadCompensationTorque` and `PlantedLimbLoadDistribution` term for
term, rather than inventing a gravity compensator of its own, because the goal is to make the two
plants the SAME, not to make Isaac's plant good.

Two terms, exactly as Godot has them:

* **open chain** - hold up everything distal to the joint: ``tau = -sum_i r_i x m_i g``, with
  ``r_i`` measured from the joint pivot.
* **closed chain** - a planted limb also carries body weight up from the ground: the planted
  end-effectors split the TOTAL body mass evenly, and that share is applied at the lowest contact
  point of the chain. Godot notes this is a deliberate over-estimate (a leg is asked to hold the
  other leg's mass too) and bounds it with the same half-of-ceiling clamp used here; it is copied
  as-is, over-estimate included, because a feed-forward that differs from Godot's is the bug this
  module exists to remove.

The result is applied as an equal-and-opposite pair - ``+tau`` on the bone, ``-tau`` on its parent -
which is what a joint torque is, and what Godot does explicitly via its own reaction term.
"""

from __future__ import annotations

import torch

from . import godot_plant

#: Godot's `PlantedLimbLoadDistribution.SupportEndEffectors`.
SUPPORT_END_EFFECTORS = ("Forearm_L", "Forearm_R", "Foot_L", "Foot_R")

#: Godot's `PlantedLimbLoadDistribution.TorsoStops` - the share stops here walking up a limb.
TORSO_STOPS = ("Chest", "Spine", "Pelvis")

#: Godot's `ActiveBone.GravityMagnitude`. Hard-coded there, so hard-coded here to match.
GRAVITY_MAGNITUDE = 9.81


def _support_chain(end_effector: str) -> list[str]:
    """Bones from `end_effector` up to (not including) the first torso stop."""
    chain: list[str] = []
    cursor: str | None = end_effector
    for _ in range(32):
        if cursor is None or cursor in TORSO_STOPS:
            break
        chain.append(cursor)
        cursor = godot_plant.BONE_PARENT.get(cursor)
    return chain


class GravityFeedForward:
    """Per-bone gravity compensation torques, batched over envs.

    Built once; `torques()` is called every physics substep from `_apply_action`.
    """

    def __init__(self, robot, device: torch.device | str, contact_bones: list[str]) -> None:
        self._device = device
        body_names = list(robot.body_names)
        index = {name: i for i, name in enumerate(body_names)}

        # Only bones that HAVE a parent produce a joint torque; the root has nothing to react against.
        self._bones = [
            b for b in godot_plant.BONE_PARENT
            if godot_plant.BONE_PARENT[b] is not None and b in index
        ]
        missing = [b for b in godot_plant.BONE_PARENT if b not in index]
        if missing:
            raise RuntimeError(
                f"bones in godot_plant with no Isaac body: {missing}. The plant table and the USD "
                "rig disagree; regenerate with scripts/derive_plant.py against the current scene."
            )

        self._body_ids = [index[b] for b in self._bones]
        self._parent_ids = [index[godot_plant.BONE_PARENT[b]] for b in self._bones]

        n_bones = len(self._bones)
        n_bodies = len(body_names)

        # `weights[b, j]` = mass of body j when j is distal to bone b, else 0. Encoding the ragged
        # distal chains as a dense matrix keeps the whole feed-forward to a couple of matmuls
        # instead of a Python loop over 15 joints on every physics substep.
        weights = torch.zeros(n_bones, n_bodies, device=device)
        for row, bone in enumerate(self._bones):
            for member in godot_plant.distal_chain(bone):
                if member in index:
                    weights[row, index[member]] = godot_plant.BONE_MASS[member]
        self._weights = weights
        self._chain_mass = weights.sum(dim=1)

        # **Godot's measured joint anchor, converted into Isaac's frame at generation time.**
        #
        # Two wrong answers preceded this one. Applying Godot's bone-LOCAL triple directly put every
        # lever arm somewhere else entirely and the feed-forward drove the body into the floor
        # (zero-action mean head height 0.257 m without the term, 0.058 m with it). Falling back to
        # the bone/parent midpoint fixed the direction but not the position: the hip is LATERAL to
        # the pelvis centre, so the midpoint sits medial of the real joint, and the per-bone diff
        # against Godot showed the thigh at 2.3-2.5x and the foot at 0.37x.
        #
        # The anchor is now emitted as a WORLD offset by the `[PLANT]` dump and mapped through the
        # D6 builder's own `to_usd` in `derive_plant.py`. USD bodies spawn unrotated, so that offset
        # at the rest pose is exactly the body-local offset, and rotating it by the live body
        # quaternion tracks the joint as the limb moves.
        self._pivot_local = torch.tensor(
            [godot_plant.BONE_PIVOT_LOCAL[b] for b in self._bones],
            device=device, dtype=torch.float32,
        )

        # Half the bone's own ceiling, matching `ActiveBone.LoadCompensationTorqueFraction = 0.5`.
        self._max_ff = torch.tensor(
            [godot_plant.BONE_MAX_TORQUE[b] * 0.5 for b in self._bones],
            device=device, dtype=torch.float32,
        )

        self._total_mass = float(sum(godot_plant.BONE_MASS.values()))

        # Support bookkeeping: `support[b, s]` marks that contact slot `s` plants bone `b`, i.e. the
        # slot's limb chain passes through it. The force is applied AT THE CONTACT POINT, not at the
        # bone - Godot uses the lowest contacting body of the chain, so a planted foot gives the hip
        # a lever arm of most of a leg. Applying it at the bone instead makes the lever arm nearly
        # zero for the foot and wrong for everything above it.
        self._contact_bones = contact_bones
        self._contact_ids = [index[name] for name in contact_bones if name in index]
        support = torch.zeros(n_bones, len(contact_bones), device=device)
        bone_row = {bone: row for row, bone in enumerate(self._bones)}
        for slot, contact_bone in enumerate(contact_bones):
            # Godot plants on Foot/Forearm; the contact sensor reports Hand/Foot. A hand in contact
            # plants the forearm chain, which is the same limb.
            effector = contact_bone
            if contact_bone.startswith("Hand_"):
                effector = "Forearm_" + contact_bone.split("_", 1)[1]
            if effector not in SUPPORT_END_EFFECTORS:
                continue
            for member in _support_chain(effector):
                if member in bone_row:
                    support[bone_row[member], slot] = 1.0
        self._support = support

        gravity = torch.zeros(3, device=device)
        gravity[2] = -GRAVITY_MAGNITUDE  # Isaac is Z-up; Godot is Y-up but the dump is frame-free.
        self._gravity = gravity

    @property
    def bones(self) -> list[str]:
        """Bone names, in the row order of everything this class returns."""
        return self._bones

    @property
    def body_ids(self) -> list[int]:
        return self._body_ids

    @property
    def parent_ids(self) -> list[int]:
        return self._parent_ids

    def torques(
        self,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
        contacts: torch.Tensor | None = None,
        clamp: bool = True,
    ) -> torch.Tensor:
        """`(num_envs, n_bones, 3)` world-frame compensation torques, clamped unless `clamp=False`.

        `body_pos_w` and `body_quat_w` must come from the LIVE per-body buffers - everything
        per-body is live under XPBD, everything joint-level or root-derived is frozen and reads a
        plausible zero (see `state.py`).
        """
        pivot = body_pos_w[:, self._body_ids] + _rotate(
            body_quat_w[:, self._body_ids],
            self._pivot_local.unsqueeze(0).expand(body_pos_w.shape[0], -1, -1),
        )

        # sum_i m_i r_i  =  (W @ pos) - chain_mass * pivot, so the cross product is taken once.
        weighted = torch.einsum("bj,njk->nbk", self._weights, body_pos_w)
        lever = weighted - self._chain_mass.view(1, -1, 1) * pivot
        torque = -torch.cross(lever, self._gravity.view(1, 1, 3).expand_as(lever), dim=-1)

        if contacts is not None:
            planted = (contacts > 0.5).float()                       # (n, slots)
            count = planted.sum(dim=1, keepdim=True).clamp(min=1.0)
            share = (self._total_mass / count).unsqueeze(-1)          # (n, 1, 1)

            # (n, slots, 3): the force each planted contact pushes up with, at its own position.
            contact_pos = body_pos_w[:, self._contact_ids]
            active = planted.unsqueeze(1) * self._support.unsqueeze(0)  # (n, bones, slots)

            # r from each bone's pivot to each contact point, crossed with that contact's force.
            r = contact_pos.unsqueeze(1) - pivot.unsqueeze(2)         # (n, bones, slots, 3)
            force = self._gravity.view(1, 1, 1, 3) * share.unsqueeze(-1)
            per_slot = torch.cross(r, force.expand_as(r), dim=-1)
            torque = torque + (per_slot * active.unsqueeze(-1)).sum(dim=2)

        if not clamp:
            # Compliance wants the LOAD, not Godot's capped compensation for it. The cap is a
            # property of Godot's feed-forward feature; the sag a joint shows under its own weight
            # is not capped by anything.
            return torque

        magnitude = torque.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        limit = self._max_ff.view(1, -1, 1)
        return torque * (magnitude.clamp(max=limit) / magnitude)


def _rotate(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate `vec` by `quat_wxyz`, batched over (env, bone)."""
    w = quat_wxyz[..., 0:1]
    xyz = quat_wxyz[..., 1:4]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec + w * t + torch.cross(xyz, t, dim=-1)
