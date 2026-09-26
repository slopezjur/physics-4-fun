"""Versioned COM reconstruction from the existing Stand actor observation."""
import mujoco
import numpy as np
import torch


SCHEMA = "mimic_com_from_stand_observation_v1"
SOURCE_SCHEMA = "mimic_stand_target_pd_v2"
FIELDS = ("root_position_world", "root_rotation_tangent_normal",
          "root_linear_velocity_world", "root_angular_velocity_world",
          "all_hinge_positions", "all_hinge_velocities_scaled_0.1")


class ObservationCom:
    """Decode the native rig state already represented in a support-v2 observation.

    MuJoCo supplies the rigid-body mass geometry, not hidden live simulator state.
    The resulting feedback is diagnostic until this decoder has Godot parity.
    """

    def __init__(self, rig, source_contract):
        if source_contract.get("observation_contract") != SOURCE_SCHEMA:
            raise ValueError("COM observer requires the support-v2 Stand contract")
        if source_contract.get("model_sha256") != rig.contract()["model_sha256"]:
            raise ValueError("COM observer model hash mismatch")
        layout = {item["name"]: (item["offset"], item["width"])
                  for item in source_contract["observation_layout"]}
        if any(name not in layout for name in FIELDS):
            raise ValueError("COM observer requires root and hinge pose/velocity fields")
        expected = ((0, 3), (3, 6), (9, 3), (12, 3),
                    (15, rig.model.nq - 7), (54, rig.model.nv - 6))
        if (source_contract.get("num_obs") != 362
                or tuple(layout[name] for name in FIELDS) != expected):
            raise ValueError("COM observer source field layout differs from the rig")
        self.rig, self.layout = rig, layout
        self.num_obs = source_contract["num_obs"]
        self.data = mujoco.MjData(rig.model)

    def contract(self):
        return dict(schema=SCHEMA, source_observation_contract=SOURCE_SCHEMA,
                    source_model_sha256=self.rig.contract()["model_sha256"],
                    input_fields=list(FIELDS), output_fields=["whole_body_com_x_world_m",
                                                               "whole_body_com_vx_world_m_s"],
                    state_source="Existing policy observation and fixed MuJoCo rig geometry")

    def __call__(self, observation):
        if observation.ndim != 2 or observation.shape[1] != self.num_obs:
            raise ValueError("COM observer input shape differs from the source contract")
        values = observation.detach().cpu().numpy() if isinstance(observation, torch.Tensor) else np.asarray(observation)
        if not np.isfinite(values).all():
            raise ValueError("COM observer requires finite observations")

        def field(row, name):
            offset, width = self.layout[name]
            return row[offset:offset + width]

        result = np.empty((len(values), 2), dtype=np.float64)
        for i, row in enumerate(values):
            tangent, normal = np.split(field(row, "root_rotation_tangent_normal"), 2)
            tangent = tangent / np.linalg.norm(tangent)
            normal = normal - tangent * np.dot(tangent, normal)
            normal = normal / np.linalg.norm(normal)
            sideways = np.cross(normal, tangent)
            rotation = np.column_stack((tangent, sideways, normal))
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, rotation.ravel())
            self.data.qpos[:3] = field(row, "root_position_world")
            self.data.qpos[3:7] = quaternion
            self.data.qpos[7:] = field(row, "all_hinge_positions")
            self.data.qvel[:3] = field(row, "root_linear_velocity_world")
            self.data.qvel[3:6] = rotation.T @ field(row, "root_angular_velocity_world")
            self.data.qvel[6:] = field(row, "all_hinge_velocities_scaled_0.1") * 10.
            mujoco.mj_forward(self.rig.model, self.data)
            mujoco.mj_subtreeVel(self.rig.model, self.data)
            result[i] = (self.data.subtree_com[1, 0], self.data.subtree_linvel[1, 0])
        return result
