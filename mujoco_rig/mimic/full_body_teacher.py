"""Native whole-body IK candidate; virtual root targets never modify the plant."""
import mujoco
import numpy as np
import torch

from .capture_step_teacher import bridge_source_action
from .recovery_motor import authority_gate


def bounded_vector(vector, maximum):
    return vector * min(1., maximum / max(np.linalg.norm(vector), 1e-12))


def upright_rotation(rotation):
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def rotation_error(current, desired):
    return .5 * np.cross(current.T, desired.T).sum(axis=0)


class FullBodyTeacher:
    """Transfer, lift, place and land from measured support/clearance, with either leg."""
    def __init__(self, task, actor, *, correction_gain=.5, composition='blend', release_settled=False,
                 rearm_quiet=False):
        if task.device != 'cpu' or task.recovery_motor_radians is None:
            raise ValueError('Teacher requires native experimental recovery motors')
        if not np.isfinite(correction_gain) or not 0. < correction_gain <= 1.:
            raise ValueError('Correction gain must be in (0,1]')
        if composition not in ('blend', 'source_offset'):
            raise ValueError('Unknown teacher target composition')
        self.task, self.actor, self.correction_gain = task, actor, correction_gain
        self.composition = composition
        self.release_settled = release_settled
        self.rearm_quiet = rearm_quiet
        self.model = task.rig.model
        count = task.get_num_envs()
        self.phase = np.zeros(count, dtype=int)
        self.armed = np.ones(count, dtype=bool)
        self.foot = np.zeros(count, dtype=int)
        self.anchors = np.zeros((count, 2, 3))
        self.rotations = np.zeros((count, 2, 3, 3))
        self.goal = np.zeros((count, 3))
        self.root_height = np.zeros(count)
        self.triggers = np.zeros((count, 2), dtype=int)
        self.transitions = np.zeros((count, 5), dtype=int)
        self.phase_frames = np.zeros((count, 5), dtype=int)
        self.support_error_peak = np.zeros(count)
        self.settled_releases = np.zeros(count, dtype=int)
        self.body = [self.model.body('Foot_' + s).id for s in ('L', 'R')]
        self.pelvis = self.model.body('Pelvis').id
        self.chest = self.model.body('Chest').id
        names = task.observation_contract()['actuator_names']
        self.motors = np.array([i for i, name in enumerate(names)
                               if name.startswith(('Thigh_', 'Shin_', 'Foot_', 'Spine_', 'Chest_'))])
        self.joints = self.model.actuator_trnid[task.rig.actuators[self.motors], 0]
        self.indices = self.model.jnt_qposadr[self.joints]
        self.dofs = self.model.jnt_dofadr[self.joints]
        self.variables = np.concatenate((np.arange(6), self.dofs))
        self.data = mujoco.MjData(self.model)
        self.jac = np.zeros((3, self.model.nv))
        self.jacr = np.zeros_like(self.jac)

    def contract(self):
        return dict(schema='native_whole_body_teacher_v1', correction_gain=self.correction_gain,
                    target_composition=self.composition,
                    release_settled=self.release_settled,
                    rearm_quiet=self.rearm_quiet,
                    rearm_condition='Both loads >= 200 N; COM speed <= 0.05 m/s; tilt <= 5 degrees; '
                                    'capture error <= 0.01 m; leg/trunk speed <= 0.1 rad/s; '
                                    'root angular speed <= 0.25 rad/s',
                    release_condition='Authority zero; both loads >= 30 N; root >= 0.75 m; '
                                      'tilt <= 15 degrees; COM horizontal speed <= 0.15 m/s',
                    phases=['transfer', 'lift', 'place', 'land', 'settle'],
                    trigger='Observable authority > 0.2; minimum foot load < 150 N; opposite > 100 N',
                    foot_choice='Less loaded foot', maximum_placement_m=.10, clearance_m=.035,
                    support_constraints='Both anchors during transfer; stance anchor during swing',
                    coordination='Virtual pelvis translation/height, COM braking and upright chest',
                    solver='Floating-root whole-body damped least squares; joint commands only',
                    external_forces=False, plant_root_writes=False, step_clock=False)

    def diagnostics(self):
        return dict(triggers=self.triggers.tolist(), transitions=self.transitions.tolist(),
                    phase_frames=self.phase_frames.tolist(),
                    settled_releases=self.settled_releases.tolist(),
                    maximum_support_anchor_error_m=self.support_error_peak.tolist())

    def solve(self, pose, reference, source_targets, radius, goals, rotations, support, com_goal, root_goal):
        data = self.data
        data.qpos[:] = pose
        lower = np.maximum(self.model.jnt_range[self.joints, 0], reference[self.indices] - radius)
        upper = np.minimum(self.model.jnt_range[self.joints, 1], reference[self.indices] + radius)
        variables = self.variables
        root_rotation = upright_rotation(rotations[support])
        for _ in range(6):
            mujoco.mj_kinematics(self.model, data)
            mujoco.mj_comPos(self.model, data)
            errors, rows = [], []
            for foot in range(2):
                mujoco.mj_jacBody(self.model, data, self.jac, self.jacr, self.body[foot])
                weight = 3. if foot == support else 1.5
                errors.extend((weight * (goals[foot] - data.xpos[self.body[foot]]),
                               .20 * rotation_error(data.xmat[self.body[foot]].reshape(3, 3), rotations[foot])))
                rows.extend((weight * self.jac[:, variables], .20 * self.jacr[:, variables]))
            mujoco.mj_jacSubtreeCom(self.model, data, self.jac, self.pelvis)
            errors.append(.8 * (com_goal - data.subtree_com[self.pelvis]))
            rows.append(.8 * self.jac[:, variables])
            mujoco.mj_jacBody(self.model, data, self.jac, self.jacr, self.pelvis)
            errors.extend((.7 * (root_goal - data.xpos[self.pelvis]),
                           .20 * rotation_error(data.xmat[self.pelvis].reshape(3, 3), root_rotation)))
            rows.extend((.7 * self.jac[:, variables], .20 * self.jacr[:, variables]))
            mujoco.mj_jacBody(self.model, data, self.jac, self.jacr, self.chest)
            errors.append(.35 * rotation_error(data.xmat[self.chest].reshape(3, 3), root_rotation))
            rows.append(.35 * self.jacr[:, variables])
            posture_jac = np.zeros((len(self.motors), len(variables)))
            posture_jac[:, 6:] = np.eye(len(self.motors)) * .025
            rows.append(posture_jac)
            errors.append(.025 * (source_targets - data.qpos[self.indices]))
            jac = np.vstack(rows)
            error = np.concatenate(errors)
            delta = np.linalg.solve(jac.T @ jac + np.eye(len(variables)) * .002, jac.T @ error)
            velocity = np.zeros(self.model.nv)
            velocity[variables] = np.clip(delta, -.08, .08)
            velocity[:3] = np.clip(velocity[:3], -.025, .025)
            mujoco.mj_integratePos(self.model, data.qpos, velocity, 1.)
            data.qpos[self.indices] = np.clip(data.qpos[self.indices], lower, upper)
        return data.qpos[self.indices].copy()

    def command_targets(self, world, pose, reference, source_targets, radius,
                        goals, rotations, support, com_goal, root_goal):
        solved = self.solve(pose, reference, source_targets, radius,
                            goals, rotations, support, com_goal, root_goal)
        origin = source_targets if self.composition == 'blend' else pose[self.indices]
        return source_targets + self.correction_gain * (solved - origin)

    @torch.no_grad()
    def __call__(self, observation):
        task, control = self.task, self.task.engine.policy_control
        action = bridge_source_action(task, self.actor(observation), observation)
        reset = task.steps.numpy() == 0
        self.phase[reset] = 0
        self.armed[reset] = True
        gate = authority_gate(observation).numpy()
        positions = task.engine.get_body_pos(0)[:, task.feet].numpy()
        loads = (observation[:, 93:95] * 1000).numpy()
        com = task.center_of_mass().numpy()
        velocity = task.engine.get_center_of_mass_velocity().numpy()
        reference, _ = task.sample_reference(task.offset)
        reference = reference.numpy()
        poses = task.engine.native_qpos().numpy()
        speeds = task.engine._array('qvel').numpy() if self.rearm_quiet else None
        for world in range(len(action)):
            upright = 1. - 2. * (poses[world, 4] ** 2 + poses[world, 5] ** 2)
            settled = (gate[world] == 0 and loads[world].min() >= 30 and poses[world, 2] >= .75
                       and upright >= np.cos(np.deg2rad(15.)) and np.linalg.norm(velocity[world, :2]) <= .15)
            if self.rearm_quiet and self.phase[world] == 0:
                quiet = (settled and loads[world].min() >= 200
                         and np.linalg.norm(velocity[world, :2]) <= .05
                         and upright >= np.cos(np.deg2rad(5.)) and observation[world, 362] <= .01
                         and abs(speeds[world, self.dofs]).max() <= .1
                         and np.linalg.norm(speeds[world, 3:6]) <= .25)
                self.armed[world] |= bool(quiet)
            if self.release_settled and self.phase[world] != 0 and settled:
                self.phase[world] = 0
                self.settled_releases[world] += 1
                continue
            if self.phase[world] == 0:
                if self.rearm_quiet and not self.armed[world]:
                    continue
                foot = int(np.argmin(loads[world])); support = 1 - foot
                if gate[world] <= .2 or loads[world, foot] >= 150 or loads[world, support] < 100:
                    continue
                height = max(.05, com[world, 2] - positions[world, :, 2].min())
                capture = com[world, :2] + velocity[world, :2] * np.sqrt(height / 9.81)
                delta = capture - positions[world, foot, :2]
                if np.linalg.norm(delta) < .04:
                    continue
                self.foot[world] = foot
                self.anchors[world] = positions[world]
                self.goal[world] = positions[world, foot]
                self.goal[world, :2] += bounded_vector(delta, .10)
                for i in range(2):
                    self.rotations[world, i] = upright_rotation(task.engine.datas[world].xmat[self.body[i]].reshape(3, 3))
                self.root_height[world] = poses[world, 2]
                self.phase[world] = 1
                self.armed[world] = False
                self.triggers[world, foot] += 1
            foot = self.foot[world]; support = 1 - foot
            phase = self.phase[world]
            if phase == 1 and loads[world, foot] < 5 and loads[world, support] >= 50:
                phase = 2
            elif phase == 2 and positions[world, foot, 2] >= self.anchors[world, foot, 2] + .025:
                phase = 3
            elif phase == 3 and np.linalg.norm(positions[world, foot, :2] - self.goal[world, :2]) < .025:
                phase = 4
            elif phase == 4 and loads[world, foot] >= 30:
                phase = 5
                self.anchors[world, foot] = positions[world, foot]
            elif phase == 5 and loads[world].min() >= 30 and gate[world] == 0:
                self.phase[world] = 0
                continue
            if phase != self.phase[world]:
                self.transitions[world, phase - 1] += 1
            self.phase[world] = phase
            self.phase_frames[world, phase - 1] += 1
            # Release unsuccessful transfer attempts once the source has restored standing.
            if phase == 1 and gate[world] == 0 and loads[world].min() >= 30:
                self.phase[world] = 0
                continue
            goals = self.anchors[world].copy()
            if phase in (2, 3, 4):
                if phase != 2:
                    goals[foot, :2] = self.goal[world, :2]
                goals[foot, 2] += .035 if phase in (2, 3) else 0.
            target_center = goals.mean(axis=0) if phase == 5 else goals[support]
            com_goal = com[world].copy()
            com_goal[:2] += bounded_vector(target_center[:2] - com[world, :2] - .12 * velocity[world, :2], .10)
            root_goal = poses[world, :3].copy()
            root_goal[:2] += com_goal[:2] - com[world, :2]
            root_goal[2] = self.root_height[world] - .02
            com_goal[2] += root_goal[2] - poses[world, 2]
            radius = np.where(control.leg_mask[self.motors].numpy(),
                              .25 + (control.maximum_radians - .25) * gate[world], .25)
            targets = reference[world, self.indices] + action[world, self.motors].numpy() * radius
            targets = self.command_targets(world, poses[world], reference[world], targets, radius,
                                           goals, self.rotations[world], support, com_goal, root_goal)
            action[world, self.motors] = torch.from_numpy((targets - reference[world, self.indices]) / radius).float()
            self.support_error_peak[world] = max(self.support_error_peak[world],
                np.linalg.norm(positions[world, support] - self.anchors[world, support]))
        return action.clamp(-1, 1)
