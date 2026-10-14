"""Private delayed-target prediction and bounded whole-body support feedback."""
import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .inverse_contact_control import inverse_contact_map, refine_forward_torque, solve_box
from .stand_foundation_followup import PredictiveStandController
from .dynamic_support_teacher import DynamicSupportTeacher
from .full_body_teacher import rotation_error


class PredictiveTargetSupport(PredictiveStandController):
    """Reuse the admitted acceleration objective with the current target-PD queue."""

    def __init__(self, task, objective='pose'):
        super().__init__(task)
        if objective not in ('pose','task','com'):
            raise ValueError('Unknown support objective')
        self.objective = objective
        self.jac = np.zeros((3,self.model.nv))
        self.jacr = np.zeros_like(self.jac)
        self.pelvis,self.chest = self.model.body('Pelvis').id,self.model.body('Chest').id
        self.body = [self.model.body('Foot_'+side).id for side in ('L','R')]
        self.kinematic_probe = mujoco.MjData(self.model)
        self.goal_data = mujoco.MjData(self.model)
        control = task.engine.policy_control
        if control.count != 33 or control.pending.shape[-1] != 67:
            raise ValueError('Expected the 33-motor position/velocity/valid queue')
        self.qi = control.q_indices.numpy()
        self.vi = control.v_indices.numpy()
        self.scale32 = control.scale.numpy()
        self.private_steps = 0

    def predict(self, world):
        """Replay committed targets; never write the live state or its queue."""
        control = self.task.engine.policy_control
        data = self.data[world]
        mujoco.mj_copyData(data, self.model, self.task.engine.datas[world])
        for lag in range(len(control.pending)):
            slot = control.pending[(control.cursor + lag) % len(control.pending), world].numpy()
            for _ in range(self.task.rig.decimation):
                # Match the live controller's float32 state and arithmetic.
                q = data.qpos[self.qi].astype(np.float32)
                v = data.qvel[self.vi].astype(np.float32)
                value = control.gain * (slot[:33] - q)
                value += control.gain * control.damping_time * (slot[33:66] - v)
                data.ctrl[:] = 0
                data.ctrl[self.actuators] = np.clip(value, -1, 1) * self.scale32 * slot[66]
                mujoco.mj_step(self.model, data)
                self.private_steps += 1
            mujoco.mj_forward(self.model, data)
            data.xfrc_applied[:] = 0
        return data

    def tracking_problem(self, data, goal):
        error = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(self.model, error, 1., data.qpos, goal)
        frequency = 2*np.pi
        desired = np.clip(frequency**2*error-2*frequency*data.qvel,-20.,20.)
        tracking,target = self.tracking,self.tracking@desired
        if self.objective == 'pose':
            return tracking,target
        # Reuse the old teacher's kinematics, not its obsolete motor/queue adapter.
        jac = DynamicSupportTeacher.task_jacobians(self,data)
        probe = self.kinematic_probe
        mujoco.mj_copyData(probe,self.model,data)
        epsilon = 1e-4
        mujoco.mj_integratePos(self.model,probe.qpos,data.qvel,epsilon)
        mujoco.mj_kinematics(self.model,probe);mujoco.mj_comPos(self.model,probe)
        bias = ((DynamicSupportTeacher.task_jacobians(self,probe)-jac)/epsilon)@data.qvel
        goal_data = self.goal_data
        goal_data.qpos[:] = goal
        mujoco.mj_kinematics(self.model,goal_data);mujoco.mj_comPos(self.model,goal_data)
        if self.objective == 'com':
            error = goal_data.subtree_com[self.pelvis]-data.subtree_com[self.pelvis]
            acceleration = np.clip(frequency**2*error-2*frequency*(jac[:3]@data.qvel),-20.,20.)
            return np.vstack((tracking,10.*jac[:3])),np.r_[target,10.*(acceleration-bias[:3])]
        indices = np.r_[np.arange(3),np.arange(12,24)]
        errors = [goal_data.subtree_com[self.pelvis]-data.subtree_com[self.pelvis]]
        for body in self.body:
            errors.extend((goal_data.xpos[body]-data.xpos[body],
                rotation_error(data.xmat[body].reshape(3,3),goal_data.xmat[body].reshape(3,3))))
        errors = np.concatenate(errors)
        frequencies = np.r_[np.full(3,frequency),np.full(12,2*frequency)]
        acceleration = np.clip(frequencies**2*errors-2*frequencies*(jac[indices]@data.qvel),-20.,20.)
        weights = np.r_[np.full(3,10.),np.full(3,8.),np.ones(3),np.full(3,8.),np.ones(3)]
        return np.vstack((tracking,weights[:,None]*jac[indices])),np.r_[target,weights*(acceleration-bias[indices])]

    def target(self, world, goal, source, radius=1.):
        data = self.predict(world)
        control = self.task.engine.policy_control
        lower = np.maximum(control.lower.numpy(), source - radius)
        upper = np.minimum(control.upper.numpy(), source + radius)
        q, v = data.qpos[self.qi].copy(), data.qvel[self.vi].copy()
        damping = control.gain * control.damping_time * v
        lo = np.clip(control.gain * (lower - q) - damping, -1, 1)
        hi = np.clip(control.gain * (upper - q) - damping, -1, 1)
        tracking,target_acceleration = self.tracking_problem(data,goal)
        mapping, offset, *_ = inverse_contact_map(self.model, data, self.dofs, self.scale)
        reduced = tracking @ mapping
        baseline = np.clip(control.gain * (source - q) - damping, lo, hi)
        regularizer = .15**2
        hessian = reduced.T @ reduced + np.eye(33) * (regularizer + 1e-7)
        linear = -reduced.T @ (target_acceleration-tracking@offset) - regularizer * baseline
        proposed, rank = solve_box(hessian, linear, lo, hi, baseline)
        if rank < 0 or not np.isfinite(proposed).all():
            raise RuntimeError('Predictive target QP failed')
        torque, acceleration, diagnostics = refine_forward_torque(
            self.model, data, self.actuators, self.dofs, self.scale, tracking,
            target_acceleration, proposed, lo, hi, iterations=2)
        target = np.clip(q + (torque + damping) / control.gain, lower, upper)
        realized = np.clip(control.gain * (target - q) - damping, -1, 1)
        error = float(np.max(abs(realized - torque)))
        if error > 1e-6:
            raise RuntimeError('Torque-to-target conversion mismatch')
        return target, dict(tracking_residual=float(np.linalg.norm(tracking@acceleration-target_acceleration)),
            conversion_error=error, saturation=float(np.mean(abs(torque) >= .98)), **diagnostics)


def blend(start, end, elapsed, duration):
    """Smooth interpolation of scalars or arrays."""
    t = np.clip(elapsed / duration, 0., 1.)
    return start + (end - start) * (t**3 * (10 + t * (-15 + 6*t)))


def blend_pose(start, end, elapsed, duration):
    """Interpolate a free-root pose while retaining a unit root quaternion."""
    target = end.copy()
    if np.dot(start[3:7], target[3:7]) < 0:
        target[3:7] *= -1
    result = blend(start, target, elapsed, duration)
    result[3:7] /= np.linalg.norm(result[3:7])
    return result


def capture_margin(model, data, support_side):
    """Signed margin inside actual stance-foot floor contacts; read-only geometry."""
    side = ('L', 'R')[support_side]
    bodies = {model.body('Foot_'+side).id, model.body('Toe_'+side).id}
    points = [contact.pos.copy() for contact in data.contact
        if contact.dist <= .001 and 0 in model.geom_bodyid[contact.geom]
        and any(body in bodies for body in model.geom_bodyid[contact.geom])]
    if len(points) < 3:
        return -1.
    points = np.asarray(points)
    try:
        hull = ConvexHull(points[:, :2])
    except QhullError:
        return -1.
    jacobian = np.zeros((3, model.nv))
    mujoco.mj_jacSubtreeCom(model, data, jacobian, 1)
    height = max(.05, data.subtree_com[1, 2]-points[:, 2].min())
    capture = data.subtree_com[1, :2] + (jacobian @ data.qvel)[:2]*np.sqrt(height/9.81)
    return float(-np.max(hull.equations[:, :2] @ capture + hull.equations[:, 2]))


class SupportSequence:
    """Measured unloading and hold gates between bounded commanded transitions."""

    def __init__(self, reference, cleared, transfer, lifted, dt, release_gate='load'):
        self.reference, self.cleared, self.transfer, self.lifted = reference, cleared, transfer, lifted
        self.dt = dt
        if release_gate not in ('load', 'capture'):
            raise ValueError('Unknown release gate')
        self.release_gate = release_gate
        self.phase = 'clear'
        self.started = 1.
        self.unloaded_seconds = self.hold_seconds = 0.
        self.reached_hold = False
        self.finished = False
        self.reason = None

    def update(self, time, support_share, com_speed, supported, strict_lift, reloaded, margin=-1.):
        elapsed = time - self.started
        if self.phase == 'clear' and elapsed >= 1.5:
            self.phase, self.started = 'transfer', time
        elif self.phase == 'transfer':
            release = support_share >= .90 if self.release_gate == 'load' else support_share >= 2/3 and margin >= .005
            unloaded = elapsed >= 4 and supported and release and com_speed <= .08
            self.unloaded_seconds = self.unloaded_seconds + self.dt if unloaded else 0.
            if self.unloaded_seconds >= .1:
                self.phase, self.started = 'lift', time
            elif elapsed >= 6:
                self.finished, self.reason = True, 'unloading_timeout'
        elif self.phase == 'lift' and elapsed >= 1:
            self.phase, self.started = 'hold', time
        elif self.phase == 'hold':
            self.hold_seconds = self.hold_seconds + self.dt if strict_lift else 0.
            if self.hold_seconds >= 1:
                self.reached_hold = True
                self.phase, self.started = 'land', time
            elif elapsed >= 3:
                self.finished, self.reason = True, 'support_hold_timeout'
        elif self.phase == 'land':
            if elapsed >= 1 and reloaded:
                self.phase, self.started = 'return', time
            elif elapsed >= 3:
                self.finished, self.reason = True, 'replant_timeout'
        elif self.phase == 'return' and elapsed >= 4:
            self.phase, self.started = 'settle', time

    def goal(self, time):
        elapsed = time - self.started
        if self.phase == 'clear':
            return blend_pose(self.reference, self.cleared, elapsed, 1.5)
        if self.phase == 'transfer':
            return blend_pose(self.cleared, self.transfer, elapsed, 4.)
        if self.phase == 'lift':
            return blend_pose(self.transfer, self.lifted, elapsed, 1.)
        if self.phase == 'hold':
            return self.lifted
        if self.phase == 'land':
            return blend_pose(self.lifted, self.transfer, elapsed, 1.)
        if self.phase == 'return':
            return blend_pose(self.transfer, self.reference, elapsed, 4.)
        return self.reference
