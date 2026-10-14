"""Spring-free standing: learned joint torques, with no pose servo or step clock."""
import json
from pathlib import Path

import gymnasium.spaces as spaces
import numpy as np
import torch

from envs.base_env import EnvMode, DoneFlags
from envs.deepmimic_env import compute_reward
from util.torch_util import quat_to_tan_norm, quat_diff_angle
from .amp_recovery import AMPRecoveryMixin, MotionHistory, HISTORY_FRAMES
from .amp_step import load_demonstrations
from .baseline import sha256
from .capture_reward import capture_distance, rotate_xyzw
from .contact_contract import SUPPORT_V2
from .continuous_stand import SEMANTICS
from .stand_task import StandTask
from .torque_stand_curriculum import CURRICULUM, STAGES, UPRIGHT_CURRICULUM, UPRIGHT_HORIZONS

CONTRACT = 'mimic_stationary_stand_torque_v2'
POSTURE = dict(schema='torque_stand_whole_body_reward_v2',
    bodies=['Spine', 'Chest', 'Head'], tilt_scale_degrees=[25., 25., 35.],
    clearance_scale_m=[.08, .10, .12], upper_joint_scale_rad=.35,
    posture_weights=dict(orientation=.4, clearance=.35, upper_joint_pose=.25),
    settled_tilt_degrees=[20., 20., 30.], settled_clearance_deficit_m=[.05, .06, .10],
    scope='Soft upper-body reference preferences; no hip, knee or foot-position target; no motor pose servo')
BALANCE_GATE = dict(min_pelvis_height_m=.75, max_pelvis_tilt_degrees=15.,
    max_horizontal_speed_m_s=.2, max_angular_speed_rad_s=1., min_each_foot_load_n=5.,
    max_non_foot_ground_contact_n=1., final_stable_seconds=2.)
STAGED_POSTURE = dict(POSTURE, schema='torque_stand_staged_posture_reward_v3',
    score_kernel='Mean per-coordinate 1/(1+squared normalized error)',
    composition='balance*((1-posture_weight)+posture_weight*posture)-action_cost',
    balance_gate=BALANCE_GATE, curriculum=CURRICULUM)
WORST_POSTURE = dict(STAGED_POSTURE, schema='torque_stand_worst_posture_reward_v4',
    score_kernel='1/(1+maximum squared normalized absolute body tilt or clearance deficit)',
    posture_weights=dict(worst_physical_alignment=.75, upper_joint_pose=.25),
    orientation_target='World upright; same axes as the fixed upright gate',
    clearance_target='At least reference height above pelvis; excess extension is not penalized')
STAGED_MODES = ('staged_v3', 'worst_v4')
UPRIGHT_IMITATION = dict(WORST_POSTURE, schema='torque_stand_short_upright_reward_v5',
    composition='balance*posture-action_cost; no posture-independent balance floor',
    curriculum=UPRIGHT_CURRICULUM,
    training_failure_tilt_degrees=[35., 35., 45.],
    training_failure_clearance_deficit_m=[.07, .09, .13],
    training_failure_scope='Training episodes only; native admission and Arena physics remain unchanged')
WHOLE_POSE_IMITATION = dict(UPRIGHT_IMITATION, schema='torque_stand_whole_pose_reward_v6',
    composition='balance*whole_body_pose*worst_alignment-action_cost-joint_velocity_cost',
    posture_weights=dict(whole_body_pose=1., worst_physical_alignment=1.),
    pose_kernel='exp(-weighted mean squared angular error beyond each region tolerance)',
    pose_regions={
        'Spine': dict(weight=1., tolerance_degrees=3., scale_degrees=20.),
        'Chest': dict(weight=1., tolerance_degrees=3., scale_degrees=20.),
        'Head': dict(weight=.8, tolerance_degrees=5., scale_degrees=25.),
        'UpperArm': dict(weight=.3, tolerance_degrees=10., scale_degrees=35.),
        'Forearm': dict(weight=.3, tolerance_degrees=10., scale_degrees=35.),
        'Thigh': dict(weight=.6, tolerance_degrees=8., scale_degrees=30.),
        'Shin': dict(weight=.6, tolerance_degrees=10., scale_degrees=30.),
        'Foot': dict(weight=.4, tolerance_degrees=8., scale_degrees=20.),
        'Toe': dict(weight=.15, tolerance_degrees=10., scale_degrees=25.)},
    velocity_cost=dict(weight=.005, scale_rad_s=5., maximum_normalized_squared_speed=25.),
    scope='Quiet Stand only; soft local rotations for all actuated bodies, with no world-space foot anchors or motor pose servo')
LIMIT_AWARE_IMITATION = dict(WHOLE_POSE_IMITATION, schema='torque_stand_limit_aware_reward_v7',
    composition='0.5*balance*whole_body_pose*worst_alignment+0.5*dense_upper_pose-action_cost-joint_velocity_cost-bracing_cost',
    posture_weights=dict(balanced_whole_body_pose=.5, dense_upper_pose=.5),
    dense_upper_pose=dict(bodies=['Spine', 'Chest', 'Head'],
        kernel='Per-body 1/(1+squared normalized angular error beyond tolerance)',
        local_rotation_weight=.75, physical_alignment_weight=.25,
        physical_kernel='Per-body 1/(1+maximum squared normalized upright tilt or clearance deficit)',
        aggregation='Equal mean of the three independent body scores'),
    bracing=dict(bodies=['Spine', 'Chest', 'Head'], weight=.5, margin_degrees=5.,
        kernel='Mean of lower proximity*negative delayed normalized torque + upper proximity*positive delayed normalized torque',
        proximity='Linear ramp in the final five degrees before each limit; capped at one outside the range',
        control_sampling='Held delayed torque control for the completed control interval, paired with its ending joint positions',
        scope='Nine upper-body motors only; inward torques and proximity without outward torque have zero cost'),
    scope='Quiet Stand reward only; physical limits, zero passive springs, observations and direct torque control are unchanged')
LIMIT_OCCUPANCY_IMITATION = dict(LIMIT_AWARE_IMITATION, schema='torque_stand_limit_occupancy_reward_v8',
    composition=LIMIT_AWARE_IMITATION['composition']+'-limit_occupancy_cost',
    limit_occupancy=dict(bodies=['Spine', 'Chest', 'Head'], weight=.25,
        kernel='Mean squared upper-joint limit proximity, using the existing five-degree bracing ramp',
        sampling='Ending joint positions at each completed control interval, independent of commanded torque',
        duration='Cost at every occupied control interval; cumulative episode return penalizes sustained occupancy',
        scope='Nine upper-body hinges only; no dwell clock, forced position, torque modification or new termination gate'))
REFERENCE_IMITATION = dict(schema='torque_stand_deepmimic_reference_reward_v9',
    implementation='MimicKit envs.deepmimic_env.compute_reward',
    composition='0.5*pose+0.1*joint_velocity+0.15*root_pose+0.1*root_velocity+0.15*key_positions',
    weights=dict(pose=.5, joint_velocity=.1, root_pose=.15, root_velocity=.1, key_positions=.15),
    scales=dict(pose=.25, joint_velocity=.01, root_pose=5., root_velocity=1., key_positions=10.),
    coefficient_source='Pinned MimicKit data/envs/deepmimic_humanoid_env.yaml',
    joint_weighting='One per actuated body, including head; zero for unactuated wrist rotations',
    velocity_weighting='One per actuated hinge; zero for unactuated hinges',
    key_bodies=['Head', 'Hand_L', 'Hand_R', 'Foot_L', 'Foot_R'],
    track_root_height=True, track_absolute_root=False,
    position_tracking='Root-relative keypoints in local heading; no world-space foot anchors',
    training_termination='Physical fall/contact or nonfinite state only; pose mismatch is a soft reward',
    default_episode_seconds=60.,
    scope='Additive reference imitation; no bracing, occupancy, capture, action-cost or posture-reset terms; direct learned torques')
BRACING_MODES = ('limit_aware_v7', 'limit_occupancy_v8')
POSE_MODES = ('whole_pose_v6', *BRACING_MODES)
UPRIGHT_MODES = ('short_upright_v5', *POSE_MODES)
REFERENCE_MODES = ('reference_v9',)
TRACKED_POSE_MODES = (*POSE_MODES, *REFERENCE_MODES)


def reward_contract(mode):
    if mode == 'whole_body_v2':
        return POSTURE
    if mode == 'staged_v3':
        return STAGED_POSTURE
    if mode == 'worst_v4':
        return WORST_POSTURE
    if mode == 'short_upright_v5':
        return UPRIGHT_IMITATION
    if mode == 'whole_pose_v6':
        return WHOLE_POSE_IMITATION
    if mode == 'limit_aware_v7':
        return LIMIT_AWARE_IMITATION
    if mode == 'limit_occupancy_v8':
        return LIMIT_OCCUPANCY_IMITATION
    if mode == 'reference_v9':
        return REFERENCE_IMITATION
    raise ValueError(f'Unknown torque Stand reward mode: {mode}')


class TorqueStandingReference:
    """Existing licensed poses supply goals/reset states, never motor targets."""
    def __init__(self, path, rig, device):
        self.path, self.sha256 = path, sha256(path)
        metadata = json.loads(path.with_suffix('.json').read_text())
        if metadata['reference_sha256'] != self.sha256 or metadata['source']['license'] != 'CC-BY-4.0':
            raise ValueError('Standing pose provenance mismatch')
        with np.load(path, allow_pickle=False) as data:
            q, v = data['qpos'], data['qvel']
        if q.shape != (8, rig.model.nq) or v.shape != (8, rig.model.nv) or np.any(v) or not np.isfinite(q).all():
            raise ValueError('Expected eight finite stationary reset poses')
        self.qpos = torch.tensor(q, device=device, dtype=torch.float32)
        self.qvel = torch.tensor(v, device=device, dtype=torch.float32)
        self.duration = None

    def sample(self, time, ids):
        indices = ids % 8
        while indices.ndim < time.ndim:
            indices = indices.unsqueeze(-1)
        indices = indices.expand(time.shape)
        return self.qpos[indices].clone(), self.qvel[indices].clone()


class TorqueStandTask(StandTask):
    def __init__(self, rig, reference, num_envs=8, device='cpu', *, reward_mode='whole_body_v2',
                 episode_seconds=None, **kwargs):
        self.reward_mode = reward_mode
        self.posture_reward = reward_contract(reward_mode)
        self.posture_stage = 0
        self.upright_horizon_stage = 0
        self.previous_com = None
        self._stationary_cache = None
        if len(rig.actuators) != 33:
            raise ValueError('Whole-body torque Stand requires all 33 motors, including the neck')
        self.previous_action = torch.zeros((num_envs, len(rig.actuators)), device=device)
        self.action_cost = torch.zeros(num_envs, device=device)
        self.mass = torch.tensor(rig.model.body_mass[1:], device=device, dtype=torch.float32)
        self.total_mass = self.mass.sum()
        self.ipos = torch.tensor(rig.model.body_ipos[1:], device=device, dtype=torch.float32)
        self.z_axis = torch.tensor([0., 0., 1.], device=device)
        self.tilt_scale = torch.deg2rad(torch.tensor(POSTURE['tilt_scale_degrees'], device=device))
        self.clearance_scale = torch.tensor(POSTURE['clearance_scale_m'], device=device)
        self.settled_tilt_limit = torch.deg2rad(torch.tensor(POSTURE['settled_tilt_degrees'], device=device))
        self.settled_clearance_limit = torch.tensor(POSTURE['settled_clearance_deficit_m'], device=device)
        self.training_tilt_limit = torch.deg2rad(torch.tensor(UPRIGHT_IMITATION['training_failure_tilt_degrees'], device=device))
        self.training_clearance_limit = torch.tensor(UPRIGHT_IMITATION['training_failure_clearance_deficit_m'], device=device)
        self.posture_bodies = torch.tensor([rig.model.body(n).id-1 for n in POSTURE['bodies']], device=device)
        self.upper_joints = torch.tensor([rig.model.jnt_qposadr[j] for j in range(1, rig.model.njnt)
            if rig.model.joint(j).name.split('_')[0] in POSTURE['bodies']], device=device)
        if reward_mode in TRACKED_POSE_MODES:
            joints = [int(rig.model.actuator_trnid[a, 0]) for a in rig.actuators]
            bodies = sorted({int(rig.model.jnt_bodyid[j]) for j in joints})
            self.imitation_body_names = [rig.model.body(b).name for b in bodies]
            # joint_rotations excludes world and pelvis; cover only actuated bodies.
            self.imitation_bodies = torch.tensor([b-2 for b in bodies], device=device, dtype=torch.long)
            self.imitation_dofs = torch.tensor([rig.model.jnt_dofadr[j]-6 for j in joints], device=device, dtype=torch.long)
        if reward_mode in POSE_MODES:
            regions = [WHOLE_POSE_IMITATION['pose_regions'][name.split('_')[0]] for name in self.imitation_body_names]
            self.imitation_weights = torch.tensor([r['weight'] for r in regions], device=device)
            self.imitation_weight_sum = self.imitation_weights.sum()
            self.imitation_tolerance = torch.deg2rad(torch.tensor([r['tolerance_degrees'] for r in regions], device=device))
            self.imitation_scale = torch.deg2rad(torch.tensor([r['scale_degrees'] for r in regions], device=device))
        if reward_mode in BRACING_MODES:
            upper_actuators = [int(a) for a, joint in zip(rig.actuators, joints)
                if rig.model.body(int(rig.model.jnt_bodyid[joint])).name in POSTURE['bodies']]
            upper_joint_ids = [int(rig.model.actuator_trnid[a, 0]) for a in upper_actuators]
            ranges = rig.model.jnt_range[upper_joint_ids]
            if len(upper_actuators) != 9 or not rig.model.jnt_limited[upper_joint_ids].all() or np.any(ranges[:, 1] <= ranges[:, 0]):
                raise ValueError('Limit-aware Stand requires nine limited upper-body hinge motors')
            self.bracing_actuators = torch.tensor(upper_actuators, device=device, dtype=torch.long)
            self.bracing_qpos = torch.tensor(rig.model.jnt_qposadr[upper_joint_ids], device=device, dtype=torch.long)
            self.bracing_joint_names = [rig.model.joint(j).name for j in upper_joint_ids]
            self.bracing_lower = torch.tensor(ranges[:, 0], device=device, dtype=torch.float32)
            self.bracing_upper = torch.tensor(ranges[:, 1], device=device, dtype=torch.float32)
            authority = dict(zip(rig.actuators, rig.action_scale))
            self.bracing_scale = torch.tensor([authority[a] for a in upper_actuators], device=device, dtype=torch.float32)
            self.bracing_margin = np.deg2rad(self.posture_reward['bracing']['margin_degrees'])
            self.dense_upper_indices = torch.tensor([self.imitation_body_names.index(name)
                for name in POSTURE['bodies']], device=device, dtype=torch.long)
        super().__init__(rig, reference, num_envs, device, control_mode='torque',
                         contact_mode=SUPPORT_V2, reference_factory=TorqueStandingReference, **kwargs)
        self.episode_seconds = 30.
        if reward_mode in UPRIGHT_MODES:
            self.set_upright_horizon_stage(0)
        if reward_mode in REFERENCE_MODES:
            self.joint_weights.zero_()
            self.joint_weights[self.imitation_bodies] = 1.
            self.episode_seconds = REFERENCE_IMITATION['default_episode_seconds']
        if episode_seconds is not None:
            if reward_mode in UPRIGHT_MODES or not np.isfinite(episode_seconds) or episode_seconds <= 0:
                raise ValueError('Fixed episode duration must be positive and cannot override a horizon curriculum')
            self.episode_seconds = episode_seconds

    def sample_reference(self, time, env_ids=None):
        return self.reference.sample(time, self.ids if env_ids is None else env_ids)

    def reference_phase(self, time):
        return torch.zeros_like(time)

    def _ensure_stationary_cache(self):
        """Map the eight immutable reference poses once; reset does not change case assignment."""
        if self._stationary_cache is not None:
            return
        poses = self.reference.qpos
        positions, rotations = self.mapper.from_native_qpos(poses)
        cases = self.ids % len(poses)
        target_tilt = torch.acos(rotate_xyzw(rotations[:, self.posture_bodies], self.z_axis)[..., 2].clamp(-1, 1))
        target_clearance = positions[:, self.posture_bodies, 2]-positions[:, :1, 2]
        pose_features = torch.cat((poses[:, :3], quat_to_tan_norm(poses[:, [4, 5, 6, 3]]), poses[:, 7:]), -1)
        self._stationary_cache = dict(height=poses[cases, 2], tilt=target_tilt[cases],
            clearance=target_clearance[cases], upper_joints=poses[cases][:, self.upper_joints],
            future=pose_features[cases].repeat(1, 3), phase=torch.zeros((len(self.ids), 1), device=self.device))
        if self.reward_mode in TRACKED_POSE_MODES:
            self._stationary_cache['local_joint_rotations'] = self.joint_rotations(rotations)[cases][:, self.imitation_bodies]
        if self.reward_mode in REFERENCE_MODES:
            self._stationary_cache.update(reference_qpos=poses[cases], reference_qvel=self.reference.qvel[cases],
                reference_joint_rotations=self.joint_rotations(rotations)[cases],
                reference_key_positions=positions[cases][:, self.key_bodies])

    def observations(self):
        self._ensure_stationary_cache()
        q = self.engine.native_qpos()
        control = self.engine.policy_control
        pending = torch.cat([control.pending[(control.cursor+i) % len(control.pending)]
                             for i in range(len(control.pending))], dim=-1)
        # Keep the original arithmetic/order for exported foot-and-toe observations.
        forces = self.engine.get_ground_contact_forces(0)
        loads = forces[:, self.feet, 2].abs()*.001 + forces[:, self.allowed_ground[2:], 2].abs()*.001
        return torch.cat((q[:, :3], quat_to_tan_norm(self.engine.get_root_rot(0)),
            self.engine.get_root_vel(0), self.engine.get_root_ang_vel(0), q[:, 7:],
            self.engine.get_dof_vel(0)*.1, loads, pending,
            self._stationary_cache['future'], self._stationary_cache['phase']), dim=-1)

    def center_of_mass(self, body_pos=None, body_rot=None):
        body_pos = self.engine.get_body_pos(0) if body_pos is None else body_pos
        body_rot = self.engine.get_body_rot(0) if body_rot is None else body_rot
        centers = body_pos + rotate_xyzw(body_rot, self.ipos)
        return (centers * self.mass[None, :, None]).sum(1) / self.total_mass

    def reset(self, env_ids=None):
        ids = self.ids if env_ids is None else env_ids
        if len(ids):
            self.steps[ids], self.offset[ids] = 0, 0
            self.engine.reset_envs(ids, *self.sample_reference(self.offset[ids], ids))
            self.previous_action[ids], self.action_cost[ids] = 0, 0
            com = self.center_of_mass()
            if self.previous_com is None:
                self.previous_com = com.clone()
            else:
                self.previous_com[ids] = com[ids]
        return self.observations(), {}

    def foot_loads(self):
        loads = self.engine.get_ground_contact_forces(0)[:, self.allowed_ground, 2].abs()
        return torch.stack((loads[:, 0] + loads[:, 2], loads[:, 1] + loads[:, 3]), -1)

    def set_posture_stage(self, stage):
        if self.reward_mode not in STAGED_MODES or not 0 <= stage < len(STAGES):
            raise ValueError('Invalid stage for this torque Stand reward')
        self.posture_stage = stage

    def set_upright_horizon_stage(self, stage):
        if self.reward_mode not in UPRIGHT_MODES or not 0 <= stage < len(UPRIGHT_HORIZONS):
            raise ValueError('Invalid upright training horizon')
        self.upright_horizon_stage = stage
        self.episode_seconds = UPRIGHT_HORIZONS[stage]

    def posture_metrics(self, q=None, body_pos=None, body_rot=None):
        """Use measured body rotations and height above pelvis, independently of foot placement."""
        self._ensure_stationary_cache()
        q = self.engine.native_qpos() if q is None else q
        body_pos = self.engine.get_body_pos(0) if body_pos is None else body_pos
        body_rot = self.engine.get_body_rot(0) if body_rot is None else body_rot
        target = self._stationary_cache
        tilt = torch.acos(rotate_xyzw(body_rot[:, self.posture_bodies], self.z_axis)[..., 2].clamp(-1, 1))
        clearance = body_pos[:, self.posture_bodies, 2]-body_pos[:, :1, 2]
        return tilt, target['tilt'], clearance, target['clearance'], q[:, self.upper_joints]-target['upper_joints']

    def reference_pose_angles(self, body_rot):
        local_rotations = self.joint_rotations(body_rot)[:, self.imitation_bodies]
        return quat_diff_angle(local_rotations, self._stationary_cache['local_joint_rotations']).abs()

    def upper_joint_bracing(self, q):
        """Reward-only cost for delayed torque driving upper hinges into their stops."""
        position = q[:, self.bracing_qpos]
        lower = ((self.bracing_lower+self.bracing_margin-position)/self.bracing_margin).clamp(0, 1)
        upper = ((position-self.bracing_upper+self.bracing_margin)/self.bracing_margin).clamp(0, 1)
        delayed = (self.engine.policy_control.ctrl[:, self.bracing_actuators]/self.bracing_scale).clamp(-1, 1)
        outward = lower*(-delayed).clamp_min(0)+upper*delayed.clamp_min(0)
        return torch.maximum(lower, upper), outward

    def reward(self):
        if self.reward_mode in REFERENCE_MODES:
            return self.stationary_imitation_reward()
        q = self.engine.native_qpos()
        self._ensure_stationary_cache()
        body_pos, body_rot = self.engine.get_body_pos(0), self.engine.get_body_rot(0)
        loads = self.foot_loads()
        up = 1 - 2 * (q[:, 4].square() + q[:, 5].square())
        upright = torch.exp(-((1-up)/.3).square())
        height = torch.exp(-((q[:, 2]-self._stationary_cache['height'])/.2).square())
        com = self.center_of_mass(body_pos, body_rot)
        velocity = (com-self.previous_com)/self.dt
        error = capture_distance(com, velocity, body_pos[:, self.feet], loads)
        capture = torch.exp(-(error/.1).square())
        settle = torch.exp(-(self.engine.get_root_vel(0)[:, :2]/.5).square().sum(-1)
                           -(self.engine.get_root_ang_vel(0)/2.).square().sum(-1))
        support = (loads.amin(-1)/30.).clamp(0, 1)
        tilt, target_tilt, clearance, target_clearance, joint_error = self.posture_metrics(q, body_pos, body_rot)
        orientation_error = ((tilt-target_tilt)/self.tilt_scale).square()
        extension_error = ((clearance-target_clearance)/self.clearance_scale).square()
        joint_error = (joint_error/POSTURE['upper_joint_scale_rad']).square()
        if self.reward_mode in (*STAGED_MODES, *UPRIGHT_MODES):
            # Keep posture differences visible even for heavily folded bodies.
            orientation = (1+orientation_error).reciprocal().mean(-1)
            extension = (1+extension_error).reciprocal().mean(-1)
            upper_pose = (1+joint_error).reciprocal().mean(-1)
        else:
            orientation = torch.exp(-orientation_error.mean(-1))
            extension = torch.exp(-extension_error.mean(-1))
            upper_pose = torch.exp(-joint_error.mean(-1))
        weights = POSTURE['posture_weights']
        posture = weights['orientation']*orientation+weights['clearance']*extension+weights['upper_joint_pose']*upper_pose
        if self.reward_mode in ('worst_v4', *UPRIGHT_MODES):
            physical_body_error = torch.maximum((tilt/self.tilt_scale).square(),
                ((target_clearance-clearance).clamp_min(0)/self.clearance_scale).square())
            physical_error = physical_body_error.amax(-1)
            worst_alignment = (1+physical_error).reciprocal()
            posture = .75*worst_alignment+.25*upper_pose
            self._diagnostics['worst_upper_body_alignment_reward'] = worst_alignment.mean()
        joint_velocity_cost = 0.
        bracing_cost = 0.
        limit_occupancy_cost = 0.
        if self.reward_mode in POSE_MODES:
            pose_angles = self.reference_pose_angles(body_rot)
            regional_pose_error = ((pose_angles-self.imitation_tolerance).clamp_min(0)/self.imitation_scale).square()
            pose_error = (regional_pose_error*self.imitation_weights).sum(-1)/self.imitation_weight_sum
            pose_reward = torch.exp(-pose_error)
            posture = pose_reward*worst_alignment
            velocity_contract = WHOLE_POSE_IMITATION['velocity_cost']
            speeds = self.engine.get_dof_vel(0)[:, self.imitation_dofs]/velocity_contract['scale_rad_s']
            joint_velocity_cost = velocity_contract['weight']*speeds.square().mean(-1).clamp_max(
                velocity_contract['maximum_normalized_squared_speed'])
            self._diagnostics['whole_body_pose_reward'] = pose_reward.mean()
            self._diagnostics['joint_velocity_cost'] = joint_velocity_cost.mean()
            mean_pose_angles = torch.rad2deg(pose_angles).mean(0)
            for index, name in enumerate(self.imitation_body_names):
                self._diagnostics[f'{name.lower()}_reference_error_degrees'] = mean_pose_angles[index]
            if self.reward_mode in BRACING_MODES:
                dense_contract = self.posture_reward['dense_upper_pose']
                local_scores = (1+regional_pose_error[:, self.dense_upper_indices]).reciprocal()
                physical_scores = (1+physical_body_error).reciprocal()
                dense_scores = (dense_contract['local_rotation_weight']*local_scores
                                +dense_contract['physical_alignment_weight']*physical_scores)
                dense_upper_pose = dense_scores.mean(-1)
                proximity, outward = self.upper_joint_bracing(q)
                bracing_cost = self.posture_reward['bracing']['weight']*outward.mean(-1)
                self._diagnostics['upper_joint_bracing_cost'] = bracing_cost.mean()
                self._diagnostics['upper_joint_limit_proximity'] = proximity.mean()
                self._diagnostics['dense_upper_pose_reward'] = dense_upper_pose.mean()
                for index, name in enumerate(POSTURE['bodies']):
                    self._diagnostics[f'{name.lower()}_dense_local_pose_reward'] = local_scores[:, index].mean()
                    self._diagnostics[f'{name.lower()}_dense_physical_reward'] = physical_scores[:, index].mean()
                if self.reward_mode == 'limit_occupancy_v8':
                    limit_occupancy_cost = self.posture_reward['limit_occupancy']['weight']*proximity.square().mean(-1)
                    self._diagnostics['upper_joint_limit_occupancy_cost'] = limit_occupancy_cost.mean()
                    self._diagnostics['upper_joint_near_limit_fraction'] = (proximity>0).float().mean()
        self.previous_com.copy_(com)
        self._diagnostics['capture_error_m'] = error.mean()
        self._diagnostics['upper_body_posture_reward'] = posture.mean()
        balance = upright*height*(.35+.3*capture+.25*settle+.1*support)
        self._diagnostics['balance_reward'] = balance.mean()
        for index, name in enumerate(POSTURE['bodies']):
            self._diagnostics[f'{name.lower()}_tilt_degrees'] = torch.rad2deg(tilt[:, index]).mean()
            self._diagnostics[f'{name.lower()}_clearance_deficit_m'] = (target_clearance[:, index]-clearance[:, index]).clamp_min(0).mean()
        # Posture is a learned preference, never a motor target or a leg-placement constraint.
        if self.reward_mode in UPRIGHT_MODES:
            self._training_posture_failed = (tilt > self.training_tilt_limit).any(-1) | (
                (target_clearance-clearance) > self.training_clearance_limit).any(-1)
            self._diagnostics['training_horizon_seconds'] = self.episode_seconds
            self._diagnostics['posture_weight'] = 1.
            if self.reward_mode in BRACING_MODES:
                weights = self.posture_reward['posture_weights']
                return (weights['balanced_whole_body_pose']*balance*posture
                        +weights['dense_upper_pose']*dense_upper_pose
                        -self.action_cost-joint_velocity_cost-bracing_cost-limit_occupancy_cost)
            return balance*posture-self.action_cost-joint_velocity_cost
        if self.reward_mode in STAGED_MODES:
            weight = STAGES[self.posture_stage]['posture_weight']
            self._diagnostics['posture_stage'] = self.posture_stage
            self._diagnostics['posture_weight'] = weight
            return balance*((1-weight)+weight*posture)-self.action_cost
        return upright*height*posture*(.35+.3*capture+.25*settle+.1*support)-self.action_cost

    def stationary_imitation_reward(self):
        """Use upstream additive tracking rewards; the reference never commands a motor."""
        self._ensure_stationary_cache()
        target = self._stationary_cache
        q, v = target['reference_qpos'], target['reference_qvel']
        positions, rotations = self.engine.get_body_pos(0), self.engine.get_body_rot(0)
        weights, scales = REFERENCE_IMITATION['weights'], REFERENCE_IMITATION['scales']
        result = compute_reward(
            self.engine.get_root_pos(0), self.engine.get_root_rot(0), self.engine.get_root_vel(0),
            self.engine.get_root_ang_vel(0), self.joint_rotations(rotations), self.engine.get_dof_vel(0),
            positions[:, self.key_bodies], q[:, :3], q[:, [4, 5, 6, 3]], v[:, :3], v[:, 3:6],
            target['reference_joint_rotations'], v[:, 6:], target['reference_key_positions'],
            self.joint_weights, self.dof_weights,
            REFERENCE_IMITATION['track_root_height'], REFERENCE_IMITATION['track_absolute_root'],
            weights['pose'], weights['joint_velocity'], weights['root_pose'], weights['root_velocity'],
            weights['key_positions'], scales['pose'], scales['joint_velocity'], scales['root_pose'],
            scales['root_velocity'], scales['key_positions'])
        self._diagnostics['reference_imitation_reward'] = result.mean()
        self._diagnostics['training_horizon_seconds'] = self.episode_seconds
        angles = torch.rad2deg(self.reference_pose_angles(rotations)).mean(0)
        for index, name in enumerate(self.imitation_body_names):
            self._diagnostics[f'{name.lower()}_reference_error_degrees'] = angles[index]
        return result

    def standing_measurements(self):
        """Read live physics for fixed admission and per-body failure reporting."""
        q = self.engine.native_qpos()
        positions, rotations = self.engine.get_body_pos(0), self.engine.get_body_rot(0)
        linear, angular = self.engine.get_root_vel(0), self.engine.get_root_ang_vel(0)
        forces = self.engine.get_ground_contact_forces(0)
        support = forces[:, self.allowed_ground, 2].abs()
        loads = torch.stack((support[:, 0]+support[:, 2], support[:, 1]+support[:, 3]), -1)
        non_foot_contacts = forces.abs().amax(-1).clone()
        non_foot_contacts[:, self.allowed_ground] = 0
        tilt, _, clearance, target_clearance, _ = self.posture_metrics(q, positions, rotations)
        up = 1-2*(q[:, 4].square()+q[:, 5].square())
        finite = torch.ones(len(q), dtype=torch.bool, device=self.device)
        for value in (q, positions, rotations, linear, angular, forces):
            finite &= torch.isfinite(value).flatten(1).all(-1)
        state = dict(pelvis_height_m=q[:, 2], pelvis_up=up,
            pelvis_tilt_degrees=torch.rad2deg(torch.acos(up.clamp(-1, 1))),
            horizontal_speed_m_s=torch.linalg.vector_norm(linear[:, :2], dim=-1),
            angular_speed_rad_s=torch.linalg.vector_norm(angular, dim=-1), foot_loads_n=loads,
            upper_body_tilt_degrees=torch.rad2deg(tilt),
            upper_body_clearance_deficit_m=(target_clearance-clearance).clamp_min(0),
            non_foot_ground_contacts_n=non_foot_contacts, finite_state=finite)
        if self.reward_mode in TRACKED_POSE_MODES:
            state['whole_body_reference_error_degrees'] = torch.rad2deg(self.reference_pose_angles(rotations))
        if self.reward_mode in BRACING_MODES:
            proximity, outward = self.upper_joint_bracing(q)
            state['upper_joint_limit_proximity'] = proximity
            state['upper_joint_outward_drive'] = outward
            if self.reward_mode == 'limit_occupancy_v8':
                state['upper_joint_near_limit'] = proximity>0
                state['upper_joint_occupancy_cost'] = self.posture_reward['limit_occupancy']['weight']*proximity.square()
        return state

    def balance_criteria(self, state):
        gate = BALANCE_GATE
        return dict(pelvis_height=state['pelvis_height_m']>=gate['min_pelvis_height_m'],
            pelvis_upright=state['pelvis_up']>=np.cos(np.deg2rad(gate['max_pelvis_tilt_degrees'])),
            horizontal_speed=state['horizontal_speed_m_s']<=gate['max_horizontal_speed_m_s'],
            angular_speed=state['angular_speed_rad_s']<=gate['max_angular_speed_rad_s'],
            two_foot_support=state['foot_loads_n'].amin(-1)>=gate['min_each_foot_load_n'],
            no_non_foot_ground_contact=state['non_foot_ground_contacts_n'].amax(-1)<=gate['max_non_foot_ground_contact_n'],
            finite_state=state['finite_state'])

    def posture_criteria(self, state, *, stage=False):
        if stage and self.reward_mode in STAGED_MODES:
            gate = STAGES[self.posture_stage]
            tilts, deficits = gate['tilt_degrees'], gate['clearance_deficit_m']
        else:
            tilts, deficits = POSTURE['settled_tilt_degrees'], POSTURE['settled_clearance_deficit_m']
        if tilts is None:
            return {}
        flags = {}
        for index, name in enumerate(POSTURE['bodies']):
            flags[f'{name.lower()}_tilt'] = state['upper_body_tilt_degrees'][:, index]<=tilts[index]
            flags[f'{name.lower()}_clearance'] = state['upper_body_clearance_deficit_m'][:, index]<=deficits[index]
        return flags

    def step(self, action):
        bounded = action.clamp(-1, 1)
        self.action_cost = .002*bounded.square().mean(-1)+.005*(bounded-self.previous_action).square().mean(-1)
        self.previous_action.copy_(bounded)
        obs, reward, done, info = super().step(action)
        if self.reward_mode in UPRIGHT_MODES and self._mode == EnvMode.TRAIN:
            folded = self._training_posture_failed
            done[folded] = DoneFlags.FAIL.value
            reward[folded] = 0.
            self._diagnostics['training_posture_failure_fraction'] = folded.float().mean()
        return obs, reward, done, info

    def observation_contract(self):
        c = super().observation_contract()
        c.update(observation_contract=CONTRACT, reference_semantics=SEMANTICS,
                 non_policy_control='none; all 33 motors including neck are controlled by the policy',
                 passive_joint_stiffness=0., standing_pose_servo=False,
                 posture_reward=self.posture_reward,
                 reference_role='Reset states, observational goals and soft imitation rewards; never motor targets')
        c['observation_layout'][-1]['name'] = 'stationary_reference_phase_zero'
        return c


class AMPTorqueStandTask(TorqueStandTask):
    _live_disc_state = AMPRecoveryMixin._live_disc_state
    fetch_disc_obs_demo = AMPRecoveryMixin.fetch_disc_obs_demo

    def __init__(self, *args, demonstrations, demonstration_rig, **kwargs):
        self._history = None
        super().__init__(*args, **kwargs)
        # Coordinates/topology are unchanged. Source admission is not new-plant admission.
        self.demonstrations = load_demonstrations(demonstrations, demonstration_rig, self.device)
        self.demo_weights = torch.full((len(demonstrations),), 1/len(demonstrations), device=self.device)
        self.mirror_demos = True
        self._demo_rng = torch.Generator(device=self.device).manual_seed(kwargs.get('seed', 0)+1907)
        self._disc_obs_space = spaces.Box(-np.inf, np.inf, (HISTORY_FRAMES*(self.rig.model.nq+self.rig.model.nv),), dtype=np.float32)
        self._history = MotionHistory(self._live_disc_state())
        self._demo_times = {'train': [torch.arange((HISTORY_FRAMES-1)*self.dt, r.duration, self.dt, device=self.device) for r in self.demonstrations]}

    def get_disc_obs_space(self):
        return self._disc_obs_space

    def reset(self, env_ids=None):
        result = super().reset(env_ids)
        if self._history is not None:
            self._history.reset(self.ids if env_ids is None else env_ids, self._live_disc_state())
        return result

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info['disc_obs'] = self._history.append(self._live_disc_state())
        return obs, reward, done, info


def finite_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


@torch.no_grad()
def evaluate_standing(task, actor, seconds=30.):
    """Fixed whole-body admission, stage readiness and live per-case failure diagnostics."""
    task.set_mode(EnvMode.TEST)
    task.episode_seconds = seconds
    task.reset()
    active = torch.ones(task.get_num_envs(), dtype=torch.bool, device=task.device)
    count = torch.zeros_like(task.steps)
    stable = {name: torch.zeros_like(count) for name in ('balance', 'stage', 'whole_body')}
    survived = torch.zeros_like(active)
    last_state = task.standing_measurements()
    violations = {name: torch.zeros_like(count) for name in
                  {**task.balance_criteria(last_state), **task.posture_criteria(last_state)}}
    tilt_sum = torch.zeros_like(last_state['upper_body_tilt_degrees'])
    deficit_sum = torch.zeros_like(tilt_sum)
    finite_count = torch.zeros_like(count)
    if task.reward_mode in BRACING_MODES:
        bracing_sum = torch.zeros_like(last_state['upper_joint_outward_drive'])
        limit_proximity_sum = torch.zeros_like(bracing_sum)
        if task.reward_mode == 'limit_occupancy_v8':
            near_limit_sum = torch.zeros_like(bracing_sum)
            occupancy_cost_sum = torch.zeros_like(bracing_sum)
    for _ in range(int(np.ceil(seconds/task.dt))):
        _, _, done, _ = task.step(actor(task.observations()))
        state = task.standing_measurements()
        balance_flags = task.balance_criteria(state)
        posture_flags = task.posture_criteria(state)
        stage_flags = task.posture_criteria(state, stage=True)
        balanced = torch.stack(list(balance_flags.values())).all(0) & (done != DoneFlags.FAIL.value)
        stage_ready = balanced.clone()
        for flag in stage_flags.values():
            stage_ready &= flag
        whole_body = balanced & torch.stack(list(posture_flags.values())).all(0)
        for name, flag in dict(balance=balanced, stage=stage_ready, whole_body=whole_body).items():
            stable[name] = torch.where(active, torch.where(flag, stable[name]+1, 0), stable[name])
        for name, flag in {**balance_flags, **posture_flags}.items():
            violations[name] += active & ~flag
        valid = active & state['finite_state']
        tilt_sum += torch.where(valid[:, None], state['upper_body_tilt_degrees'], 0.)
        deficit_sum += torch.where(valid[:, None], state['upper_body_clearance_deficit_m'], 0.)
        finite_count += valid
        if task.reward_mode in BRACING_MODES:
            bracing_sum += torch.where(valid[:, None], state['upper_joint_outward_drive'], 0.)
            limit_proximity_sum += torch.where(valid[:, None], state['upper_joint_limit_proximity'], 0.)
            if task.reward_mode == 'limit_occupancy_v8':
                near_limit_sum += valid[:, None] & state['upper_joint_near_limit']
                occupancy_cost_sum += torch.where(valid[:, None], state['upper_joint_occupancy_cost'], 0.)
        for name, value in state.items():
            mask = active.reshape((-1,)+(1,)*(value.ndim-1))
            last_state[name] = torch.where(mask, value, last_state[name])
        count += active
        survived |= active & (done == DoneFlags.TIME.value)
        active &= done == DoneFlags.NULL.value
        if not active.any():
            break
    balance_flags = task.balance_criteria(last_state)
    final_flags = {**balance_flags, **task.posture_criteria(last_state)}
    cases = []
    for i in range(task.get_num_envs()):
        contacts = [{'body': task.rig.model.body(index+1).name, 'force_n': finite_number(force)}
                    for index, force in enumerate(last_state['non_foot_ground_contacts_n'][i]) if force>1.]
        termination = []
        if not survived[i]:
            if last_state['pelvis_height_m'][i]<.65:
                termination.append('pelvis_below_failure_height')
            if contacts:
                termination.append('non_foot_ground_contact')
            if not last_state['finite_state'][i]:
                termination.append('nonfinite_physics_state')
            if not termination:
                termination.append('task_failure')
        posture = {}
        for index, name in enumerate(POSTURE['bodies']):
            denominator = int(finite_count[i])
            posture[name] = dict(final_tilt_degrees=finite_number(last_state['upper_body_tilt_degrees'][i, index]),
                final_clearance_deficit_m=finite_number(last_state['upper_body_clearance_deficit_m'][i, index]),
                mean_tilt_degrees=finite_number(tilt_sum[i, index]/denominator) if denominator else None,
                mean_clearance_deficit_m=finite_number(deficit_sum[i, index]/denominator) if denominator else None)
        cases.append(dict(case=i, survived=bool(survived[i]), survival_seconds=float(count[i]*task.dt),
            final_stable_seconds=float(stable['whole_body'][i]*task.dt),
            final_balance_stable_seconds=float(stable['balance'][i]*task.dt),
            final_stage_stable_seconds=float(stable['stage'][i]*task.dt),
            final_pelvis_height_m=finite_number(last_state['pelvis_height_m'][i]),
            final_pelvis_tilt_degrees=finite_number(last_state['pelvis_tilt_degrees'][i]),
            final_horizontal_speed_m_s=finite_number(last_state['horizontal_speed_m_s'][i]),
            final_angular_speed_rad_s=finite_number(last_state['angular_speed_rad_s'][i]),
            final_foot_loads_n=[finite_number(value) for value in last_state['foot_loads_n'][i]],
            posture=posture, termination_reasons=termination, non_foot_ground_contacts=contacts,
            failed_criteria_at_end=[name for name, flags in final_flags.items() if not flags[i]],
            criterion_violation_seconds={name: float(values[i]*task.dt) for name, values in violations.items()}))
        if task.reward_mode in TRACKED_POSE_MODES:
            cases[-1]['whole_body_reference_error_degrees'] = {
                name: finite_number(last_state['whole_body_reference_error_degrees'][i, index])
                for index, name in enumerate(task.imitation_body_names)}
        if task.reward_mode in BRACING_MODES:
            denominator = int(finite_count[i])
            cases[-1]['upper_joint_bracing'] = {
                name: dict(mean_limit_proximity=finite_number(limit_proximity_sum[i, index]/denominator) if denominator else None,
                    mean_outward_drive=finite_number(bracing_sum[i, index]/denominator) if denominator else None,
                    final_outward_drive=finite_number(last_state['upper_joint_outward_drive'][i, index]))
                for index, name in enumerate(task.bracing_joint_names)}
            if task.reward_mode == 'limit_occupancy_v8':
                for index, name in enumerate(task.bracing_joint_names):
                    cases[-1]['upper_joint_bracing'][name].update(
                        mean_near_limit_fraction=finite_number(near_limit_sum[i, index]/denominator) if denominator else None,
                        mean_occupancy_cost=finite_number(occupancy_cost_sum[i, index]/denominator) if denominator else None,
                        final_occupancy_cost=finite_number(last_state['upper_joint_occupancy_cost'][i, index]))
    def passes(case, field):
        return case['survived'] and case[field]>=BALANCE_GATE['final_stable_seconds']
    return dict(schema='mimic_torque_standing_assessment_v3', seconds=seconds, cases=cases,
        passed=all(passes(case, 'final_stable_seconds') for case in cases),
        settled_cases=sum(passes(case, 'final_stable_seconds') for case in cases),
        balance_passed=all(passes(case, 'final_balance_stable_seconds') for case in cases),
        balance_settled_cases=sum(passes(case, 'final_balance_stable_seconds') for case in cases),
        stage_passed=all(passes(case, 'final_stage_stable_seconds') for case in cases),
        stage_settled_cases=sum(passes(case, 'final_stage_stable_seconds') for case in cases),
        assessed_stage=task.posture_stage if task.reward_mode in STAGED_MODES else None,
        balance_gate=BALANCE_GATE, posture_gate=POSTURE,
        scope='Fixed whole-body gate remains independent of curriculum; live termination and per-body diagnostics; no naturalness admission')
