"""Joint-only goal skills added to an immutable 33-motor Stand policy."""
from dataclasses import dataclass
from functools import partial

import gymnasium.spaces as spaces
import mujoco
import numpy as np
import torch

from .target_control33 import DelayedTargetControl33
from .representation_engine import RepresentationGPU, RepresentationNative
from .representation_task import RepresentationMixin
from .torque_stand_task import TorqueStandTask
from .capture_reward import rotate_xyzw


class AddedTargetControl(DelayedTargetControl33):
    """Zero addition exactly preserves the source's clipped physical targets."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.addition = torch.zeros((len(self.ctrl), 33), device=self.ctrl.device)

    def target_positions(self, action):
        return (super().target_positions(action)+self.addition).clamp(self.lower, self.upper)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.addition[slice(None) if env_ids is None else env_ids] = 0

    def contract(self):
        return dict(super().contract(), schema='anchored_joint_target_addition_v1',
                    composition='clip(clip(reference+.25*frozen_actor,joint_limits)+addition,joint_limits)',
                    source_actor_frozen=True, extra_root_actuation=False)


class PhysicalSkillTask(RepresentationMixin, TorqueStandTask):
    def __init__(self, rig, reference, count, device, seconds):
        self.representation = 'target_pd'
        engine = RepresentationNative if device == 'cpu' else RepresentationGPU
        TorqueStandTask.__init__(self, rig, reference, count, device, reward_mode='reference_v9',
            episode_seconds=seconds, engine_factory=partial(engine, control_factory=AddedTargetControl, trace=False))


def fit_goal(model, pose, *, shift, lift, distance, swing, lean_radians=0.):
    """Private floating-root IK; only resulting motor offsets can reach the plant."""
    data = mujoco.MjData(model)
    data.qpos[:] = pose
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    feet = [model.body('Foot_'+side).id for side in ('L', 'R')]
    goals, rotations = data.xpos[feet].copy(), data.xmat[feet].reshape(2, 3, 3).copy()
    goals[swing] += [distance, 0, lift]
    com = data.subtree_com[1].copy()
    com[1] += shift
    if lean_radians:
        # Preserve the original foot goals; rotate only the private IK seed.
        angle = lean_radians*(-1 if swing == 0 else 1)
        rotation = np.array([np.cos(angle/2), np.sin(angle/2), 0., 0.])
        orientation = np.empty(4)
        mujoco.mju_mulQuat(orientation, rotation, data.qpos[3:7])
        data.qpos[3:7] = orientation
    joints = [j for j in range(1, model.njnt) if model.joint(j).name.startswith(('Thigh_', 'Shin_', 'Foot_', 'Toe_'))]
    qi, vi = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    variables = np.r_[np.arange(3), vi]
    jac, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(100):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        rows, errors = [], []
        for side, body in enumerate(feet):
            mujoco.mj_jacBody(model, data, jac, jacr, body)
            rows.extend((3*jac[:, variables], .3*jacr[:, variables]))
            errors.extend((3*(goals[side]-data.xpos[body]),
                .3*.5*np.cross(data.xmat[body].reshape(3, 3).T, rotations[side].T).sum(axis=0)))
        mujoco.mj_jacSubtreeCom(model, data, jac, 1)
        rows.append(2*jac[:2, variables])
        errors.append(2*(com[:2]-data.subtree_com[1, :2]))
        regularizer = np.zeros((len(qi)+1, len(variables)))
        regularizer[0, 2] = .04
        regularizer[1:, 3:] = .003*np.eye(len(qi))
        rows.append(regularizer)
        errors.append(np.r_[.04*(pose[2]-data.qpos[2]), .003*(pose[qi]-data.qpos[qi])])
        matrix, error = np.vstack(rows), np.concatenate(errors)
        delta = np.linalg.solve(matrix.T@matrix+1e-5*np.eye(len(variables)), matrix.T@error)
        velocity = np.zeros(model.nv)
        velocity[variables] = np.clip(delta, -.08, .08)
        velocity[:3] = np.clip(velocity[:3], -.015, .015)
        mujoco.mj_integratePos(model, data.qpos, velocity, 1.)
        data.qpos[qi] = np.clip(data.qpos[qi], model.jnt_range[joints, 0]+1e-5, model.jnt_range[joints, 1]-1e-5)
        if np.linalg.norm(delta) < 1e-7:
            break
    mujoco.mj_forward(model, data)
    return data.qpos.copy(), dict(foot_error_m=float(np.linalg.norm(data.xpos[feet]-goals, axis=-1).max()),
        com_error_m=float(np.linalg.norm(data.subtree_com[1, :2]-com[:2])),
        penetration_m=max([0., *[-float(c.dist) for c in data.contact[:data.ncon]]]))


def smooth(time, begin, duration):
    t = ((time-begin)/duration).clamp(0, 1)
    return t*t*t*(10+t*(-15+6*t))


@dataclass(frozen=True)
class Goal:
    side: int = 0  # Swing foot: 0 left, 1 right.
    shift: float = .10
    lift: float = 0.
    distance: float = 0.
    onset: float = 1.
    ramp: float = 1.5
    hold: float = 1.
    return_to_stand: bool = True
    enabled: bool = True


class SkillEnvironment:
    """A learned residual receives explicit goals; the source actor never does."""
    def __init__(self, rig, reference, anchor, goals, device, *, seconds=12., action_mode='basis', radius=.5):
        self.goals, self.device, self.rig = goals, device, rig
        self.task = PhysicalSkillTask(rig, reference, len(goals), device, seconds)
        self.anchor = anchor.to(device).eval().requires_grad_(False)
        self.action_mode, self.radius = action_mode, radius
        self.count, self.dt = len(goals), self.task.dt
        self._action_space = spaces.Box(-1., 1., (3 if action_mode == 'basis' else 33,), dtype=np.float32)
        self._obs_space = spaces.Box(-np.inf, np.inf, (380,), dtype=np.float32)
        self.side = torch.tensor([g.side for g in goals], device=device)
        self.enabled = torch.tensor([g.enabled for g in goals], device=device)
        self.settings = {key:torch.tensor([getattr(g, key) for g in goals], device=device, dtype=torch.float32)
                         for key in ('shift','lift','distance','onset','ramp','hold','return_to_stand')}
        q = self.task.reference.qpos.detach().cpu().numpy()
        motor_qi = rig.model.jnt_qposadr[rig.model.actuator_trnid[rig.actuators, 0]]
        cache, basis, fits = {}, [], []
        for world, goal in enumerate(goals):
            key = (world % 8, goal.side, goal.shift, goal.lift, goal.distance)
            if key not in cache:
                pose = q[world % 8]
                shift = goal.shift*(1 if goal.side == 0 else -1)
                poses, checks = [], []
                for lift, distance in ((0.,0.), (max(goal.lift,.02),0.), (max(goal.lift,.02),goal.distance)):
                    target, fit = fit_goal(rig.model, pose, shift=shift, lift=lift, distance=distance, swing=goal.side)
                    poses.append(target); checks.append(fit)
                cache[key] = (np.stack((poses[0][motor_qi]-pose[motor_qi],
                                       poses[1][motor_qi]-poses[0][motor_qi],
                                       poses[2][motor_qi]-poses[1][motor_qi])), checks)
            values, checks = cache[key]
            basis.append(values); fits.append(checks)
        self.basis = torch.tensor(np.asarray(basis), device=device, dtype=torch.float32)
        self.fits = fits
        self.foot_bodies = [rig.model.body(f'{body}_{side}').id-1 for side in ('L','R') for body in ('Foot','Toe')]
        gids = [rig.model.geom(f'g_{body}_{side}').id for side in ('L','R') for body in ('Foot','Toe')]
        self.geom_centers = torch.tensor(rig.model.geom_pos[gids], device=device, dtype=torch.float32)
        self.geom_sizes = torch.tensor(rig.model.geom_size[gids], device=device, dtype=torch.float32)
        self.initial_xy = self.task.engine.get_body_pos(0)[:, self.task.feet, :2].clone()
        self.last = None

    def __getattr__(self, name):
        return getattr(self.task, name)

    def get_action_space(self): return self._action_space
    def get_obs_space(self): return self._obs_space

    def coefficients(self):
        s, time = self.settings, self.task.steps*self.dt
        rise_start = s['onset']+s['ramp']+.25
        has_lift = (s['lift'] > 0).float()
        move_start = rise_start+.7
        land_start = move_start+.5+s['hold']
        return_start = torch.where(has_lift.bool(), land_start+.75, s['onset']+s['ramp']+s['hold'])
        transfer = smooth(time,s['onset'],s['ramp'])*(1-s['return_to_stand']*smooth(time,return_start,1.5))
        lift = has_lift*smooth(time,rise_start,.5)*(1-s['return_to_stand']*smooth(time,land_start,.5))
        move = (s['distance'] != 0).float()*smooth(time,move_start,.5)
        return torch.stack((transfer,lift,move),dim=-1)*self.enabled[:,None]

    def goal_features(self):
        c = self.coefficients()
        sign = torch.where(self.side == 0, -1., 1.)
        share = (2*sign*self.settings['shift']/.31*c[:,0]).clamp(-1,1)
        height = c[:,1]*self.settings['lift']/.05
        distance = c[:,2]*self.settings['distance']/.1
        return torch.stack((share, height*(self.side==0), height*(self.side==1),
            distance*(self.side==0), distance*(self.side==1), self.enabled.float()),dim=-1)

    def observations(self):
        return torch.cat((self.task.observations(),self.goal_features()),dim=-1)

    def reset(self, env_ids=None):
        self.task.reset(env_ids)
        self.last = None
        return self.observations(), {}

    def measurements(self):
        state = self.task.standing_measurements()
        positions, rotations = self.task.engine.get_body_pos(0), self.task.engine.get_body_rot(0)
        rot = rotations[:,self.foot_bodies]
        centers = positions[:,self.foot_bodies]+rotate_xyzw(rot,self.geom_centers)
        x,y,z,w = rot.unbind(-1)
        row = torch.stack((2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),dim=-1)
        clearance = centers[...,2]-(row.abs()*self.geom_sizes).sum(-1)
        state['clearance_m'] = clearance.reshape(-1,2,2).amin(-1)
        state['foot_xy_m'] = positions[:,self.task.feet,:2]
        state['com_m'] = self.task.center_of_mass(positions,rotations)
        return state

    def skill_reward(self, state):
        goal = self.goal_features()
        loads = state['foot_loads_n']
        share = loads[:,0]/loads.sum(-1).clamp_min(1.)
        load_score = torch.exp(-((share-(.5+.5*goal[:,0]))/.15).square())
        target_height = goal[:,1:3]*.05
        lift_score = torch.exp(-((state['clearance_m']-target_height)/.025).square().sum(-1))
        target_xy = self.initial_xy.clone()
        target_xy[:,:,0] += goal[:,3:5]*.1
        placement = torch.exp(-((state['foot_xy_m']-target_xy)/.05).square().sum((1,2)))
        posture = torch.exp(-(state['pelvis_tilt_degrees']/15.).square())
        height = torch.exp(-((state['pelvis_height_m']-self.task.reference.qpos[self.task.ids%8,2])/.15).square())
        support = (loads.sum(-1)/(self.task.total_mass*9.81*.5)).clamp(0,1)
        reward = posture*height*support*(.1+.55*load_score+.25*lift_score+.1*placement)
        return reward

    def step(self, action):
        with torch.no_grad():
            source = self.anchor(self.task.observations())
        action = action.clamp(-1,1)
        offset = torch.einsum('ni,nij->nj',action,self.basis) if self.action_mode == 'basis' else self.radius*action
        offset = offset.clamp(-self.radius,self.radius)*self.enabled[:,None]
        self.task.engine.policy_control.addition.copy_(offset)
        _, _, done, info = self.task.step(source)
        self.last = self.measurements()
        reward = self.skill_reward(self.last)
        return self.observations(), reward, done, info

    def teacher_action(self):
        c = self.coefficients()
        if self.action_mode == 'basis':
            return c
        return (torch.einsum('ni,nij->nj',c,self.basis)/self.radius).clamp(-1,1)

    def contract(self):
        return dict(schema='stand_anchored_goal_skill_v1', action_mode=self.action_mode,
            radius_rad=self.radius, observation_width=380, frozen_anchor=True,
            original_observation_width=374, goals_are_commands=True, external_support=False,
            training='New PPO residual, no stationary AMP discriminator; original AMP-trained anchor remains frozen',
            zero_residual='Original clipped source targets exactly; native/GPU physical verification is separate')
