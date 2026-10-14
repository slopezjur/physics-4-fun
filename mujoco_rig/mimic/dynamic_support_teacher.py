"""Native inverse-dynamics teacher; contact forces are planned, never applied."""
import itertools

import mujoco
import numpy as np

from .full_body_teacher import FullBodyTeacher, rotation_error, upright_rotation
from .inverse_contact_control import inverse_contact_map, refine_forward_torque, solve_box


def solve_force_qp(objective, target, dynamics, required, free_dofs, motor_dofs,
                   motor_scale, torque_lower, torque_upper, contact_count, initial=None):
    """Eliminate accelerations exactly; solve bounded motor/contact coefficients with MuJoCo."""
    nv = dynamics.shape[0]
    motors = len(motor_dofs)
    width = motors + contact_count
    actuation = np.zeros((nv, motors))
    actuation[motor_dofs, np.arange(motors)] = motor_scale
    acceleration_map = np.linalg.solve(dynamics[:, :nv], np.column_stack((actuation, -dynamics[:, nv:])))
    acceleration_offset = np.linalg.solve(dynamics[:, :nv], -required)
    transform = np.zeros((nv + contact_count, width))
    transform[:nv] = acceleration_map
    transform[nv:, motors:] = np.eye(contact_count)
    offset = np.r_[acceleration_offset, np.zeros(contact_count)]
    reduced = objective @ transform
    reduced_target = target - objective @ offset
    hessian = reduced.T @ reduced + 1e-5 * (transform.T @ transform) + np.eye(width) * 1e-8
    linear = -reduced.T @ reduced_target + 1e-5 * transform.T @ offset
    lower = np.r_[torque_lower / motor_scale, np.zeros(contact_count)]
    upper = np.r_[torque_upper / motor_scale, np.full(contact_count, 3.)]
    solution = np.zeros(width)
    if initial is not None and len(initial) == nv + contact_count:
        solution[:motors] = (dynamics[motor_dofs] @ initial + required[motor_dofs]) / motor_scale
        solution[motors:] = initial[nv:]
    solution[:] = np.clip(solution, lower, upper)
    solution, rank = solve_box(hessian, linear, lower, upper, solution)
    full = transform @ solution + offset
    torque = solution[:motors] * motor_scale
    residual = np.max(abs((dynamics @ full + required)[free_dofs]) / 700., initial=0.)
    violation = max(0., np.max(torque_lower - torque), np.max(torque - torque_upper))
    feasible = (rank >= 0 and np.isfinite(full).all() and residual <= 1e-5 and violation <= .002
                and np.min(full[nv:], initial=0.) >= -1e-8)
    return full, torque, dict(feasible=bool(feasible), optimizer_success=bool(rank >= 0),
        free_variables=int(rank), unactuated_residual_fraction=float(residual), torque_violation_nm=float(violation))


class DynamicSupportTeacher(FullBodyTeacher):
    """Keep state-based stepping phases, replacing geometric IK commands with feasible torques."""
    def __init__(self, task, actor, *, contact_damping=18., contact_tracking_weight=0., all_motors=False,
                 force_model='independent_rays', balance_tracking_weight=1.):
        super().__init__(task, actor, composition='source_offset', release_settled=True, rearm_quiet=True)
        if not np.isfinite(contact_damping) or contact_damping <= 0:
            raise ValueError('Contact damping must be finite and positive')
        if not np.isfinite(contact_tracking_weight) or contact_tracking_weight < 0:
            raise ValueError('Contact tracking weight must be finite and nonnegative')
        self.contact_damping = contact_damping
        self.contact_tracking_weight = contact_tracking_weight
        if not np.isfinite(balance_tracking_weight) or balance_tracking_weight <= 0:
            raise ValueError('Balance tracking weight must be finite and positive')
        self.balance_tracking_weight = balance_tracking_weight
        self.all_motors = all_motors
        if force_model not in ('independent_rays','mujoco_inverse_contacts','mujoco_forward_refined'):
            raise ValueError('Unknown contact force model')
        self.force_model = force_model
        if all_motors:
            self.motors = np.arange(len(task.rig.actuators))
            self.joints = self.model.actuator_trnid[task.rig.actuators, 0]
            self.indices = self.model.jnt_qposadr[self.joints]
            self.dofs = self.model.jnt_dofadr[self.joints]
            self.variables = np.r_[np.arange(6), self.dofs]
        self.predicted = mujoco.MjData(self.model)
        self.mass = np.zeros((self.model.nv, self.model.nv))
        self.weight = self.model.body_mass.sum() * 9.81
        self.scale = task.rig.action_scale[self.motors]
        self.free_dofs = np.setdiff1d(np.arange(self.model.nv), self.dofs)
        self.warm = [None] * task.get_num_envs()
        self.solves = np.zeros(task.get_num_envs(), dtype=int)
        self.failures = np.zeros_like(self.solves)
        self.max_root_residual = np.zeros(task.get_num_envs())
        self.max_tracking_error = np.zeros(task.get_num_envs())
        self.last_solution = None
        self.last_by_world = [None] * task.get_num_envs()
        self.max_inverse_residual = np.zeros(task.get_num_envs())

    def contract(self):
        result = super().contract()
        result.update(schema='native_contact_dynamics_teacher_v1',
            solver='MuJoCo box QP after exact acceleration elimination; full mass matrix and bias/passive forces',
            correction_gain=None,target_composition='Inverse PD of planned feasible torque',
            all_policy_motors=self.all_motors,
            force_model=self.force_model,
            force_model_description='Only actual floor contacts; nonnegative inner-pyramid friction rays at 80% friction',
            unactuated_dynamics='Exact non-policy coordinates and any fixed source motors',
            motor_bounds='Existing torque limits, joint target envelope and conservative extension slew bounds',
            target_conversion='Invert existing PD at current q/v; original two-command delay remains',
            contact_damping=self.contact_damping, contact_tracking_weight=self.contact_tracking_weight,
            balance_tracking_weight=self.balance_tracking_weight,
            contact_tracking='Weighted acceleration feedback at actual foot/toe contact points',acceleration_scale=20.,
            approximation='Control-rate planning; actual contact compliance and command delay are not optimized',
            failure_action='Source physical targets; no external force or root-state correction')
        if self.force_model != 'independent_rays':
            result['approximation'] = 'Local contact linearization; actual command delay is not optimized'
            result['force_model_description'] = 'Local derivative of MuJoCo inverse compliant/constrained dynamics; no independent support-force variables'
            result['solver'] = 'MuJoCo motor box QP with inverse-contact acceleration map'
            if self.force_model == 'mujoco_forward_refined':
                result['solver'] = 'Bounded inverse-contact refinements with native forward-dynamics backtracking'
                result['approximation'] = 'Eight refinements, normalized torque trust radius 0.1; command delay remains'
        return result

    def diagnostics(self):
        result = super().diagnostics()
        result.update(qp_solves=self.solves.tolist(), infeasible_solves=self.failures.tolist(),
            maximum_unactuated_residual_fraction=self.max_root_residual.tolist(),
            maximum_nonlinear_inverse_residual_fraction=self.max_inverse_residual.tolist(),
            maximum_support_acceleration_tracking_error=self.max_tracking_error.tolist())
        return result

    def task_jacobians(self, data):
        rows = []
        mujoco.mj_jacSubtreeCom(self.model, data, self.jac, self.pelvis)
        rows.append(self.jac.copy())
        mujoco.mj_jacBody(self.model, data, self.jac, self.jacr, self.pelvis)
        rows.extend((self.jac.copy(), self.jacr.copy()))
        mujoco.mj_jacBody(self.model, data, self.jac, self.jacr, self.chest)
        rows.append(self.jacr.copy())
        for body in self.body:
            mujoco.mj_jacBody(self.model, data, self.jac, self.jacr, body)
            rows.extend((self.jac.copy(), self.jacr.copy()))
        return np.vstack(rows)

    def floor_contacts(self, data, world):
        selected = self.foot[world]
        for contact in data.contact[:data.ncon]:
            bodies = self.model.geom_bodyid[contact.geom]
            if 0 not in bodies:
                continue
            body = int(bodies[0] if bodies[1] == 0 else bodies[1])
            name = self.model.body(body).name
            if not name or not name.startswith(('Foot_', 'Toe_')):
                continue
            foot = 0 if name.endswith('_L') else 1
            if self.phase[world] in (2, 3, 4) and foot == selected:
                continue
            yield contact, body, foot

    def contact_columns(self, data, world):
        columns, feet = [], []
        for contact, body, foot in self.floor_contacts(data, world):
            # An inner pyramid leaves friction margin; coefficients are in body-weight units.
            mu = float(contact.friction[0]) * .8 / np.sqrt(2.)
            mujoco.mj_jac(self.model, data, self.jac, None, contact.pos, body)
            for x, y in itertools.product((-1., 1.), repeat=2):
                columns.append(self.jac.T @ np.array([x * mu, y * mu, 1.]))
                feet.append(foot)
        return (np.column_stack(columns) if columns else np.empty((self.model.nv, 0))), np.asarray(feet)

    def contact_tracking(self, data, world, goals, rotations):
        jacobians, accelerations = [], []
        for contact, body, foot in self.floor_contacts(data, world):
            mujoco.mj_jac(self.model, data, self.jac, None, contact.pos, body)
            jacobian = self.jac.copy()
            local = data.xmat[body].reshape(3, 3).T @ (contact.pos - data.xpos[body])
            predicted_point = self.predicted.xpos[body] + self.predicted.xmat[body].reshape(3, 3) @ local
            mujoco.mj_jac(self.model, self.predicted, self.jac, None, predicted_point, body)
            bias = ((self.jac - jacobian) / 1e-4) @ data.qvel
            foot_rotation = data.xmat[self.body[foot]].reshape(3, 3)
            goal = goals[foot] + rotations[foot] @ foot_rotation.T @ (contact.pos - data.xpos[self.body[foot]])
            goal[2] = 0.
            acceleration = 120. * (goal - contact.pos) - self.contact_damping * (jacobian @ data.qvel)
            jacobians.append(jacobian)
            accelerations.append(np.clip(acceleration, -40., 40.) - bias)
        return (np.vstack(jacobians), np.concatenate(accelerations)) if jacobians else (np.empty((0, self.model.nv)), np.empty(0))

    def torque_bounds(self, world, pose, velocity, reference, radius):
        control = self.task.engine.policy_control
        lower = np.maximum(self.model.jnt_range[self.joints, 0], reference[self.indices] - radius)
        upper = np.minimum(self.model.jnt_range[self.joints, 1], reference[self.indices] + radius)
        queue = control.pending[(control.cursor - 1) % len(control.pending), world].numpy()
        origin = queue[self.motors] if queue[60] else reference[self.indices]
        limit = control.target_speed_rad_s * self.task.dt
        expanded = (radius > .25) & control.leg_mask[self.motors].numpy()
        lower[expanded] = np.maximum(lower[expanded], origin[expanded] - limit)
        upper[expanded] = np.minimum(upper[expanded], origin[expanded] + limit)
        # The measured gate can shrink between commands. The legacy interval remains
        # reachable immediately; fall back to it if the conservative slew interval vanishes.
        empty = lower > upper
        lower[empty] = np.maximum(self.model.jnt_range[self.joints[empty], 0], reference[self.indices[empty]] - .25)
        upper[empty] = np.minimum(self.model.jnt_range[self.joints[empty], 1], reference[self.indices[empty]] + .25)
        damping = control.gain * control.damping_time * velocity[self.dofs]
        lo = np.clip(self.scale * (control.gain * (lower - pose[self.indices]) - damping), -self.scale, self.scale)
        hi = np.clip(self.scale * (control.gain * (upper - pose[self.indices]) - damping), -self.scale, self.scale)
        return lo, hi, lower, upper

    def command_targets(self, world, pose, reference, source_targets, radius,
                        goals, rotations, support, com_goal, root_goal):
        self.last_solution = None
        self.last_by_world[world] = None
        data = self.data
        live = self.task.engine.datas[world]
        data.qpos[:] = pose
        data.qvel[:] = live.qvel[:self.model.nv]
        data.ctrl[:] = live.ctrl[:self.model.nu]
        mujoco.mj_forward(self.model, data)
        mujoco.mj_subtreeVel(self.model, data)
        jac = self.task_jacobians(data)
        self.predicted.qpos[:] = data.qpos
        self.predicted.qvel[:] = data.qvel
        epsilon = 1e-4
        mujoco.mj_integratePos(self.model, self.predicted.qpos, data.qvel, epsilon)
        mujoco.mj_forward(self.model, self.predicted)
        jdot_v = ((self.task_jacobians(self.predicted) - jac) / epsilon) @ data.qvel
        task_velocity = jac @ data.qvel
        rotation = upright_rotation(rotations[support])
        desired = np.zeros(len(jac))
        desired[:3] = 15. * (com_goal - data.subtree_com[self.pelvis]) - 6. * task_velocity[:3]
        desired[3:6] = 25. * (root_goal - data.xpos[self.pelvis]) - 8. * task_velocity[3:6]
        desired[6:9] = 25. * rotation_error(data.xmat[self.pelvis].reshape(3, 3), rotation) - 8. * task_velocity[6:9]
        desired[9:12] = 25. * rotation_error(data.xmat[self.chest].reshape(3, 3), rotation) - 8. * task_velocity[9:12]
        weights = np.r_[np.ones(3), np.full(3, .5), np.full(6, .4), np.ones(12)]
        weights[:12] *= self.balance_tracking_weight
        for foot in range(2):
            index = 12 + foot * 6
            desired[index:index + 3] = 80. * (goals[foot] - data.xpos[self.body[foot]]) - self.contact_damping * task_velocity[index:index + 3]
            desired[index + 3:index + 6] = 35. * rotation_error(data.xmat[self.body[foot]].reshape(3, 3), rotations[foot]) - 10. * task_velocity[index + 3:index + 6]
            if foot == support or self.phase[world] in (0, 1, 5):
                weights[index:index + 3] = 4.
                weights[index + 3:index + 6] = 1.
        desired = np.clip(desired, -40., 40.)
        columns, feet = self.contact_columns(data, world)
        mujoco.mj_fullM(self.model, self.mass, data.qM)
        fixed = data.qfrc_actuator.copy()
        fixed[self.dofs] = 0.
        required = data.qfrc_bias - data.qfrc_passive - fixed
        dynamics = np.column_stack((self.mass * 20., -columns * self.weight))
        width = dynamics.shape[1]
        objective = np.zeros((len(jac), width))
        objective[:, :self.model.nv] = jac * (weights[:, None] * 2.)
        target = weights * (desired - jdot_v) / 10.
        if self.contact_tracking_weight:
            contact_jac, contact_accel = self.contact_tracking(data, world, goals, rotations)
            extra = np.zeros((len(contact_jac), width))
            extra[:, :self.model.nv] = contact_jac * (2. * self.contact_tracking_weight)
            objective = np.vstack((objective, extra))
            target = np.r_[target, contact_accel * self.contact_tracking_weight / 10.]
        if self.force_model != 'independent_rays':
            targets = self.inverse_targets(world,pose,reference,source_targets,radius,objective[:,:self.model.nv]/20.,target)
            if self.last_solution is not None:
                predicted_accel = jac @ self.last_solution['acceleration'] + jdot_v
                self.max_tracking_error[world] = max(self.max_tracking_error[world],
                    np.linalg.norm(predicted_accel[12 + support * 6:15 + support * 6] - desired[12 + support * 6:15 + support * 6]))
            return targets
        # Torque regularization preserves useful source support without imposing its pose.
        motor_map = dynamics[self.dofs] / self.scale[:, None]
        source_torque = self.scale * np.clip(4. * (source_targets - pose[self.indices]) - .08 * data.qvel[self.dofs], -1., 1.)
        objective = np.vstack((objective, motor_map * .15))
        target = np.r_[target, .15 * (source_torque - required[self.dofs]) / self.scale]
        lo, hi, target_lower, target_upper = self.torque_bounds(world, pose, data.qvel, reference, radius)
        self.solves[world] += 1
        if np.any(lo > hi):
            self.failures[world] += 1
            return source_targets
        solution, torque, diagnostics = solve_force_qp(objective, target, dynamics, required,
            self.free_dofs, self.dofs, self.scale, lo, hi, len(feet), self.warm[world])
        self.max_root_residual[world] = max(self.max_root_residual[world], diagnostics['unactuated_residual_fraction'])
        if not diagnostics['feasible']:
            self.failures[world] += 1
            return source_targets
        self.warm[world] = solution
        predicted_accel = jac @ (solution[:self.model.nv] * 20.) + jdot_v
        self.max_tracking_error[world] = max(self.max_tracking_error[world],
            np.linalg.norm(predicted_accel[12 + support * 6:15 + support * 6] - desired[12 + support * 6:15 + support * 6]))
        self.last_solution = dict(torque=torque.copy(), coefficients=solution[self.model.nv:].copy(),
            diagnostics=diagnostics, acceleration=solution[:self.model.nv] * 20.,
            contact_feet=feet.copy(), dynamics=dynamics, required=required.copy())
        self.last_by_world[world] = self.last_solution
        return np.clip(pose[self.indices] + (torque / self.scale + .08 * data.qvel[self.dofs]) / 4.,
                       target_lower, target_upper)

    def inverse_targets(self,world,pose,reference,source_targets,radius,tracking,target):
        data = self.data
        mapping,offset,derivative,acceleration0,force0,fixed = inverse_contact_map(self.model,data,self.dofs,self.scale)
        reduced = tracking @ mapping
        desired = target - tracking @ offset
        source = np.clip(4.*(source_targets-pose[self.indices])-.08*data.qvel[self.dofs],-1.,1.)
        hessian = reduced.T @ reduced + np.eye(len(self.motors))*(.15**2+1e-7)
        linear = -reduced.T @ desired - .15**2 * source
        lo,hi,lower,upper = self.torque_bounds(world,pose,data.qvel,reference,radius)
        refinement = {}
        if self.force_model == 'mujoco_forward_refined':
            solution, acceleration, refinement = refine_forward_torque(self.model,data,
                self.task.rig.actuators[self.motors],self.dofs,self.scale,tracking,target,source,lo/self.scale,hi/self.scale)
            rank = 0
        else:
            solution,rank = solve_box(hessian,linear,lo/self.scale,hi/self.scale)
            acceleration = mapping@solution+offset
        self.solves[world] += 1
        if rank < 0 or not np.isfinite(solution).all():
            self.failures[world] += 1
            return source_targets
        torque = solution*self.scale
        linear_force = force0+derivative@(acceleration-acceleration0)
        residual = np.max(abs((linear_force-fixed)[self.free_dofs]))/self.weight
        data.qacc[:]=acceleration
        mujoco.mj_inverse(self.model,data)
        nonlinear = np.max(abs((data.qfrc_inverse-fixed)[self.free_dofs]))/self.weight
        if self.force_model == 'mujoco_forward_refined':
            residual = nonlinear
        self.max_root_residual[world] = max(self.max_root_residual[world],residual)
        self.max_inverse_residual[world] = max(self.max_inverse_residual[world],nonlinear)
        loads = np.zeros(2)
        wrench = np.zeros(6)
        included = {contact.efc_address: foot for contact,body,foot in self.floor_contacts(data,world)
                    if contact.efc_address >= 0}
        for index, contact in enumerate(data.contact[:data.ncon]):
            foot = included.get(contact.efc_address)
            if foot is None:
                continue
            mujoco.mj_contactForce(self.model,data,index,wrench)
            loads[foot] += max(0.,wrench[0])
        self.last_solution = dict(torque=torque.copy(),coefficients=loads/self.weight,contact_feet=np.array([0,1]),
            acceleration=acceleration,diagnostics=dict(feasible=bool(nonlinear <= 1e-5),
                motor_box_feasible=True,nonlinear_inverse_residual_fraction=nonlinear,**refinement))
        self.last_by_world[world] = self.last_solution
        return np.clip(pose[self.indices]+(solution+.08*data.qvel[self.dofs])/4.,lower,upper)
