"""Batched reference-pose mapping using the dummy's original hinge coordinates.

This supplies MimicKit-style body poses (xyzw quaternions), including offset pivots.
It deliberately does not interpret motion-library exponential maps as hinge angles.
Retargeting a licensed clip into these coordinates is a separate stage.
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch

from util.torch_util import axis_angle_to_quat, quat_mul, quat_rotate

from .rig import Rig


class DummyKinematics:
    def __init__(self, rig: Rig, device="cpu", dtype=torch.float32):
        model = rig.model
        if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_bodyid[0] != 1:
            raise ValueError("Expected a free pelvis followed by the original hinge skeleton")
        if not np.all(model.jnt_type[1:] == mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Only the source dummy's ordered hinge joints are supported")
        self.model = model
        self.body_names = [model.body(i).name for i in range(1, model.nbody)]
        self.joint_names = [model.joint(i).name for i in range(1, model.njnt)]
        def tensor(value):
            return torch.as_tensor(value.copy(), dtype=dtype, device=device)
        self.body_pos = tensor(model.body_pos)
        self.body_rot = tensor(model.body_quat[:, [1, 2, 3, 0]])
        self.anchors = tensor(model.jnt_pos)
        self.axes = tensor(model.jnt_axis)
        self.reference = tensor(model.qpos0[7:])

    def forward(self, root_pos, root_rot, hinge_angles):
        """Return body origins and orientations; root quaternion is xyzw.

        Shapes are (..., 3), (..., 4), (..., nq-7). Joint rotations occur around
        their original pivots in their original order, including passive wrists.
        """
        if hinge_angles.shape[-1] != len(self.joint_names):
            raise ValueError(f"Expected {len(self.joint_names)} native hinge angles")
        if root_pos.shape[:-1] != hinge_angles.shape[:-1] or root_rot.shape != (*root_pos.shape[:-1], 4):
            raise ValueError("Root and joint pose batch dimensions must match")
        positions = [torch.zeros_like(root_pos), root_pos]
        rotations = [torch.zeros_like(root_rot), root_rot]
        for body in range(2, self.model.nbody):
            parent = self.model.body_parentid[body]
            parent_rot = rotations[parent]
            pos = positions[parent] + quat_rotate(parent_rot, self.body_pos[body].expand_as(root_pos))
            rot = quat_mul(parent_rot, self.body_rot[body].expand_as(root_rot))
            first = self.model.body_jntadr[body]
            for joint in range(first, first + self.model.body_jntnum[body]):
                angle = hinge_angles[..., joint - 1] - self.reference[joint - 1]
                axis = self.axes[joint].expand_as(root_pos)
                delta = axis_angle_to_quat(axis, angle)
                anchor = self.anchors[joint].expand_as(root_pos)
                # Keep the pivot fixed while the body's origin rotates around it.
                new_rot = quat_mul(rot, delta)
                pos = pos + quat_rotate(rot, anchor) - quat_rotate(new_rot, anchor)
                rot = new_rot
            positions.append(pos)
            rotations.append(rot)
        return torch.stack(positions[1:], dim=-2), torch.stack(rotations[1:], dim=-2)

    def from_native_qpos(self, qpos):
        if qpos.shape[-1] != self.model.nq:
            raise ValueError(f"Expected {self.model.nq} native pose coordinates")
        return self.forward(qpos[..., :3], qpos[..., [4, 5, 6, 3]], qpos[..., 7:])
