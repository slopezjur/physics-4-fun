"""Control-rate native measurements of slip and oscillation after physical hits."""
import mujoco
import numpy as np
import torch

from .foot_motion import FootMotionCost


class RecoveryQuality:
    def __init__(self, task, actor):
        self.actor = actor
        self.action = self.previous_action = self.previous_delta = None
        count = task.get_num_envs()
        self.slip = np.zeros((count, 2))
        self.action_variation = np.zeros(count)
        self.action_second_difference = np.zeros(count)
        self.load_switches = np.zeros((count, 2), dtype=int)
        self.previous_supported = None
        self.hit_seconds = np.zeros(count)
        self.foot_motion = None
        self.foot_acceleration_sum = np.zeros((count, 2))
        self.rapid_reversals = np.zeros((count, 2), dtype=int)
        self.post_hit_samples = np.zeros(count)
        model = task.engine.rig.model
        self.foot_bodies = [{model.body(name + side).id for name in ("Foot_", "Toe_")}
                            for side in ("L", "R")]
        self.jacobian = np.zeros((3, model.nv))
        self.wrench = np.zeros(6)
        self.joint_groups = {group: [model.jnt_qposadr[model.joint(name).id] for name in names]
            for group, names in {
                "hip": [f"Thigh_{side}_{axis}" for side in ("L", "R") for axis in ("rx", "ry", "rz")],
                "knee": [f"Shin_{side}_rx" for side in ("L", "R")],
                "ankle_toe": [f"{body}_{side}_{axis}" for side in ("L", "R")
                    for body, axes in (("Foot", ("rx", "ry", "rz")), ("Toe", ("rx",))) for axis in axes]
            }.items()}
        self.joint_min = {key: np.full((count, len(ids)), np.inf) for key, ids in self.joint_groups.items()}
        self.joint_max = {key: np.full((count, len(ids)), -np.inf) for key, ids in self.joint_groups.items()}
        self.joint_path = {key: np.zeros((count, len(ids))) for key, ids in self.joint_groups.items()}
        self.previous_qpos = None

    def __call__(self, observations):
        self.action = self.actor(observations).detach().cpu().numpy()
        return torch.as_tensor(self.action, device=observations.device)

    def record(self, task, step, active, done):
        scored = (active & task.engine.hit).cpu().numpy()
        qpos = task.engine.native_qpos().cpu().numpy()
        # Restrict articulation measurements to the first five seconds after contact.
        moving = scored & (self.hit_seconds < 5.)
        for key, ids in self.joint_groups.items():
            values = qpos[:, ids]
            self.joint_min[key] = np.minimum(self.joint_min[key], np.where(moving[:, None], values, np.inf))
            self.joint_max[key] = np.maximum(self.joint_max[key], np.where(moving[:, None], values, -np.inf))
            if self.previous_qpos is not None:
                self.joint_path[key] += np.abs(values - self.previous_qpos[:, ids]) * moving[:, None]
        self.previous_qpos = qpos.copy()
        self.hit_seconds += scored * task.dt
        if self.previous_action is not None:
            delta = self.action - self.previous_action
            self.action_variation += np.abs(delta).mean(-1) * scored
            if self.previous_delta is not None:
                self.action_second_difference += np.abs(delta - self.previous_delta).mean(-1) * scored
            self.previous_delta = delta.copy()
        self.previous_action = self.action.copy()
        loads = task.engine.get_ground_contact_forces(0)[:, task.allowed_ground, 2].abs()
        supported = torch.stack((loads[:, 0] + loads[:, 2], loads[:, 1] + loads[:, 3]), -1).cpu().numpy() >= 30.
        foot_loads = torch.stack((loads[:, 0] + loads[:, 2], loads[:, 1] + loads[:, 3]), -1)
        positions = task.engine.get_body_pos(0)[:, task.feet]
        if self.foot_motion is None:
            self.foot_motion = FootMotionCost(positions, foot_loads, task.dt)
        else:
            _, acceleration, reversal = self.foot_motion.update(positions, foot_loads)
            self.foot_acceleration_sum += acceleration.cpu().numpy() * scored[:, None]
            self.rapid_reversals += reversal.cpu().numpy() & scored[:, None]
            self.post_hit_samples += scored
        if self.previous_supported is not None:
            self.load_switches += (supported != self.previous_supported) & scored[:, None]
        self.previous_supported = supported.copy()
        model = task.engine.rig.model
        for world in np.flatnonzero(scored):
            data = task.engine.datas[world]
            speeds, forces = np.zeros(2), np.zeros(2)
            for contact_id, contact in enumerate(data.contact[:data.ncon]):
                body_ids = model.geom_bodyid[contact.geom]
                if 0 not in body_ids:
                    continue
                body = int(body_ids[0] if body_ids[1] == 0 else body_ids[1])
                for foot, bodies in enumerate(self.foot_bodies):
                    if body not in bodies:
                        continue
                    mujoco.mj_contactForce(model, data, contact_id, self.wrench)
                    force = max(0., self.wrench[0])
                    mujoco.mj_jac(model, data, self.jacobian, None, contact.pos, body)
                    velocity = self.jacobian @ data.qvel
                    normal = contact.frame[:3]
                    tangent_speed = np.linalg.norm(velocity - normal * np.dot(velocity, normal))
                    speeds[foot] += force * tangent_speed
                    forces[foot] += force
            # Load-weighted contact-point speed avoids labeling an ankle pivot as slip.
            self.slip[world] += np.divide(speeds, forces, out=np.zeros(2), where=forces >= 5.) * task.dt

    def report(self):
        seconds = self.hit_seconds.clip(min=1e-9)
        return dict(schema="native_recovery_quality_v1", sample_rate="one sample per control interval after confirmed hit",
                    scope="Contact slip and motor variation are diagnostics, not a validated naturalness score",
                    mean_contact_slip_distance_m=float(self.slip.sum(-1).mean()),
                    mean_action_variation_per_second=float((self.action_variation / seconds).mean()),
                    mean_action_second_difference_per_second=float((self.action_second_difference / seconds).mean()),
                    mean_foot_load_switches=float(self.load_switches.sum(-1).mean()),
                    mean_foot_acceleration_m_s2=float((self.foot_acceleration_sum /
                        self.post_hit_samples.clip(min=1)[:, None]).mean()),
                    mean_rapid_contact_reversals=float(self.rapid_reversals.sum(-1).mean()),
                    cases=[dict(case=i, post_hit_seconds=float(self.hit_seconds[i]),
                        contact_slip_distance_by_foot_m=self.slip[i].tolist(),
                        action_variation_per_second=float(self.action_variation[i] / seconds[i]),
                        action_second_difference_per_second=float(self.action_second_difference[i] / seconds[i]),
                        load_switches_by_foot=self.load_switches[i].tolist(),
                        mean_foot_acceleration_by_foot_m_s2=(self.foot_acceleration_sum[i] /
                            max(1, self.post_hit_samples[i])).tolist(),
                        rapid_contact_reversals_by_foot=self.rapid_reversals[i].tolist(),
                        articulation_first_5s={key: dict(
                            range_rad=np.where(np.isfinite(self.joint_min[key][i]),
                                self.joint_max[key][i] - self.joint_min[key][i], 0.).tolist(),
                            travel_rad=self.joint_path[key][i].tolist()) for key in self.joint_groups})
                        for i in range(len(seconds))])
