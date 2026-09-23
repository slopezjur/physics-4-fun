"""Dependent stance-leg coordinates for flat-foot and toe-only reference fitting."""
import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation


def bounded_ik_step(jacobian, error, position, limits, max_step=.2):
    """Redistribute an IK correction when a hinge reaches its permitted interval."""
    lower = np.maximum(limits[:, 0] + 1e-6 - position, -max_step)
    upper = np.minimum(limits[:, 1] - 1e-6 - position, max_step)
    step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(len(error)) * 1e-9, error)
    if np.all((step >= lower) & (step <= upper)):
        return step
    matrix = np.vstack((jacobian, np.eye(len(position)) * np.sqrt(1e-9)))
    target = np.r_[error, np.zeros(len(position))]
    return lsq_linear(matrix, target, bounds=(lower, upper), method="bvls", tol=1e-10).x


class StanceProjection:
    """Keep stance feet fixed while optimizing pelvis and remaining joint coordinates.

    Six leg hinges solve each foot's world pose. Ankle yaw remains an independent
    coordinate; toe-only phases also leave toe pitch adjustable. Joint limits are always respected. If a
    target is unreachable, the remaining error is exposed to the fit/validator.
    This is an offline reference operation, never a runtime controller.
    """
    def __init__(self, rig, poses, targets, contact, toe_only=None, anchor_offsets=None, stance_rotations=None):
        self.model = rig.model
        self.data = mujoco.MjData(self.model)
        self.contact = contact.copy()
        self.anchor_offsets = np.zeros(contact.shape + (2,)) if anchor_offsets is None else np.asarray(anchor_offsets)
        if self.anchor_offsets.shape != contact.shape + (2,) or not np.isfinite(self.anchor_offsets).all():
            raise ValueError("Expected finite XY anchor offsets per frame and foot")
        self.toe_only = np.zeros_like(contact) if toe_only is None else np.asarray(toe_only, dtype=bool)
        if self.toe_only.shape != contact.shape or np.any(self.toe_only & ~contact):
            raise ValueError("Toe-only phases must be a subset of the foot support schedule")
        self.targets = targets.copy()
        self.anchors = targets.copy()
        self.rotations = np.zeros((len(poses), 2, 3, 3))
        if stance_rotations is not None:
            stance_rotations = np.asarray(stance_rotations)
            if stance_rotations.shape != self.rotations.shape or not np.isfinite(stance_rotations).all():
                raise ValueError("Expected finite world anchor rotations per frame and foot")
        self.legs = []
        for foot, side in enumerate(("L", "R")):
            names = [f"Thigh_{side}_{axis}" for axis in ("rx", "ry", "rz")]
            names += [f"Shin_{side}_rx", f"Foot_{side}_rx", f"Foot_{side}_rz"]
            joints = np.array([self.model.joint(name).id for name in names])
            qi, vi = self.model.jnt_qposadr[joints], self.model.jnt_dofadr[joints]
            body = self.model.body(f"Foot_{side}").id
            toe_body = self.model.body(f"Toe_{side}").id
            geom = self.model.geom(f"g_Foot_{side}").id
            toe = self.model.jnt_qposadr[self.model.joint(f"Toe_{side}_rx").id]
            if not np.allclose(self.model.geom_quat[geom], [1, 0, 0, 0]):
                raise ValueError("Stance projection expects the existing aligned sole geometry")
            self.legs.append((body, toe_body, qi, vi, toe, self.model.jnt_range[joints]))
            anchor_rotation = np.eye(3)
            for frame, pose in enumerate(poses):
                if not contact[frame, foot]:
                    continue
                if frame == 0 or not contact[frame - 1, foot]:
                    self.data.qpos[:] = pose
                    mujoco.mj_kinematics(self.model, self.data)
                    matrix = self.data.xmat[body].reshape(3, 3)
                    anchor_rotation = Rotation.from_euler("z", np.arctan2(matrix[1, 0], matrix[0, 0])).as_matrix()
                self.rotations[frame, foot] = anchor_rotation
                if stance_rotations is not None:
                    anchor_rotation = stance_rotations[frame, foot]
                    self.rotations[frame, foot] = anchor_rotation
                self.targets[frame, foot, 2] = self.model.geom_size[geom, 2] - self.model.geom_pos[geom, 2]
                self.anchors[frame, foot] = self.targets[frame, foot]
                if self.toe_only[frame, foot]:
                    self.anchors[frame, foot] += anchor_rotation @ self.model.body_pos[toe_body]
        self.seeds = poses.copy()
        self.seeds = self.project(self.seeds, iterations=12, use_seeds=False)

    def independent_coordinates(self, q_indices, root_width=3):
        active = np.ones((len(self.contact), root_width + len(q_indices)), dtype=bool)
        for foot, (_, _, qi, _, toe, _) in enumerate(self.legs):
            dependent = np.flatnonzero(np.isin(q_indices, qi)) + root_width
            active[np.ix_(self.contact[:, foot], dependent)] = False
            active[np.ix_(self.contact[:, foot] & ~self.toe_only[:, foot],
                          np.flatnonzero(q_indices == toe) + root_width)] = False
        return active

    def project(self, poses, iterations=4, use_seeds=True, anchor_offsets=None, active_frames=None):
        offsets = self.anchor_offsets if anchor_offsets is None else anchor_offsets
        result = poses.copy()
        for frame, pose in enumerate(result):
            if active_frames is not None and not active_frames[frame]:
                continue
            for foot, (body, toe_body, qi, vi, toe, limits) in enumerate(self.legs):
                if not self.contact[frame, foot]:
                    continue
                if use_seeds:
                    pose[qi] = self.seeds[frame, qi]
                # Avoid the straight-knee singularity when initializing the IK branch.
                if not use_seeds and pose[qi[3]] > -.1:
                    pose[qi[0]] += .15
                    pose[qi[3]] = -.3
                    pose[qi[4]] += .15
                pose[qi] = np.clip(pose[qi], limits[:, 0] + 1e-6, limits[:, 1] - 1e-6)
                if self.toe_only[frame, foot]:
                    body = toe_body
                else:
                    pose[toe] = 0
                for _ in range(iterations):
                    self.data.qpos[:] = pose
                    mujoco.mj_kinematics(self.model, self.data)
                    mujoco.mj_comPos(self.model, self.data)
                    rotation = self.data.xmat[body].reshape(3, 3)
                    target = self.anchors[frame, foot] + np.r_[offsets[frame, foot], 0.]
                    error = np.r_[target - self.data.xpos[body],
                                  Rotation.from_matrix(self.rotations[frame, foot] @ rotation.T).as_rotvec()]
                    jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
                    mujoco.mj_jacBody(self.model, self.data, jp, jr, body)
                    jacobian = np.vstack((jp[:, vi], jr[:, vi]))
                    pose[qi] += bounded_ik_step(jacobian, error, pose[qi], limits)
        return result
