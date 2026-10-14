"""Private transfer goals and a teacher constrained to the stepping motor path."""
import mujoco
import numpy as np

from .full_body_teacher import rotation_error
from .interval_target_support import IntervalTargetSupport


SLEW = .02
LATENT_MARGIN = 1e-6


def fit_transfer_goal(model, reference, target_xy, *, lean_radians=0.):
    """Fit exact world-XY COM and fixed feet using root translation and leg joints.

    Only private kinematics are evaluated. The original upper-body joint angles
    are preserved; an optional root lean follows the requested COM direction.
    The returned goal is a geometric candidate, not a physical feasibility claim.
    """
    reference = np.asarray(reference, dtype=np.float64).copy()
    target_xy = np.asarray(target_xy, dtype=np.float64).copy()
    if reference.shape != (model.nq,) or target_xy.shape != (2,):
        raise ValueError('Expected a complete reference pose and a two-dimensional COM goal')
    if not np.isfinite(reference).all() or not np.isfinite(target_xy).all() or not np.isfinite(lean_radians):
        raise ValueError('Transfer geometry must be finite')
    if abs(np.linalg.norm(reference[3:7]) - 1.) > 1e-5:
        raise ValueError('Reference root quaternion must have unit norm')
    data = mujoco.MjData(model)
    data.qpos[:] = reference
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    pelvis = model.body('Pelvis').id
    feet = [model.body(f'Foot_{side}').id for side in ('L', 'R')]
    foot_positions = data.xpos[feet].copy()
    foot_rotations = data.xmat[feet].reshape(2, 3, 3).copy()
    initial_com = data.subtree_com[pelvis].copy()
    shift = target_xy - initial_com[:2]
    if lean_radians:
        distance = np.linalg.norm(shift)
        if distance <= 1e-9:
            raise ValueError('A directional root lean needs a nonzero horizontal COM shift')
        direction = shift / distance
        axis = np.array([-direction[1], direction[0], 0.])
        rotation = np.r_[np.cos(lean_radians / 2), axis * np.sin(lean_radians / 2)]
        orientation = np.empty(4)
        mujoco.mju_mulQuat(orientation, rotation, reference[3:7])
        data.qpos[3:7] = orientation / np.linalg.norm(orientation)
    joints = [joint for joint in range(1, model.njnt)
              if model.joint(joint).name.startswith(('Thigh_', 'Shin_', 'Foot_', 'Toe_'))]
    if len(joints) != 16 or not np.all(model.jnt_limited[joints]):
        raise ValueError('Expected the sixteen original bounded leg hinges')
    qi, vi = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    variables = np.r_[np.arange(3), vi]
    jacobian = np.zeros((3, model.nv))
    rotation_jacobian = np.zeros_like(jacobian)
    converged = False
    delta = np.full(len(variables), np.inf)
    for iteration in range(100):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        rows, errors = [], []
        for index, body in enumerate(feet):
            mujoco.mj_jacBody(model, data, jacobian, rotation_jacobian, body)
            rows.extend((3 * jacobian[:, variables], .3 * rotation_jacobian[:, variables]))
            errors.extend((3 * (foot_positions[index] - data.xpos[body]),
                .3 * rotation_error(data.xmat[body].reshape(3, 3), foot_rotations[index])))
        mujoco.mj_jacSubtreeCom(model, data, jacobian, pelvis)
        rows.append(2 * jacobian[:2, variables])
        errors.append(2 * (target_xy - data.subtree_com[pelvis, :2]))
        regularizer = np.zeros((len(qi) + 1, len(variables)))
        regularizer[0, 2] = .04
        regularizer[1:, 3:] = .003 * np.eye(len(qi))
        rows.append(regularizer)
        errors.append(np.r_[.04 * (reference[2] - data.qpos[2]), .003 * (reference[qi] - data.qpos[qi])])
        matrix, error = np.vstack(rows), np.concatenate(errors)
        delta = np.linalg.solve(matrix.T @ matrix + 1e-5 * np.eye(len(variables)), matrix.T @ error)
        if not np.isfinite(delta).all():
            raise RuntimeError('Nonfinite private transfer IK correction')
        velocity = np.zeros(model.nv)
        velocity[variables] = np.clip(delta, -.08, .08)
        velocity[:3] = np.clip(velocity[:3], -.015, .015)
        mujoco.mj_integratePos(model, data.qpos, velocity, 1.)
        data.qpos[qi] = np.clip(data.qpos[qi], model.jnt_range[joints, 0] + 1e-5,
                                model.jnt_range[joints, 1] - 1e-5)
        if np.linalg.norm(delta) < 1e-7:
            converged = True
            break
    mujoco.mj_forward(model, data)
    untouched = np.setdiff1d(np.arange(7, model.nq), qi)
    if not np.array_equal(data.qpos[untouched], reference[untouched]):
        raise RuntimeError('Transfer goal changed a non-leg joint')
    foot_error = np.linalg.norm(data.xpos[feet] - foot_positions, axis=-1)
    orientation_error = [np.linalg.norm(rotation_error(data.xmat[body].reshape(3, 3), foot_rotations[index]))
                         for index, body in enumerate(feet)]
    report = dict(target_xy_m=target_xy.tolist(), initial_com_m=initial_com.tolist(),
        fitted_com_m=data.subtree_com[pelvis].tolist(), shift_xy_m=shift.tolist(),
        foot_error_m=float(foot_error.max()), foot_errors_m=foot_error.tolist(),
        foot_orientation_error_norm=float(max(orientation_error)),
        com_error_m=float(np.linalg.norm(data.subtree_com[pelvis, :2] - target_xy)),
        penetration_m=max([0., *[-float(contact.dist) for contact in data.contact[:data.ncon]]]),
        non_leg_joints_exact=True, lean_radians=float(lean_radians), iterations=iteration + 1,
        convergence_delta_reached=converged, final_delta_norm=float(np.linalg.norm(delta)),
        live_state_modified=False, physical_feasibility_established=False)
    return data.qpos.copy(), report


class SteppingTransferTeacher(IntervalTargetSupport):
    """Reuse interval prediction inside the exact reachable residual-target box."""

    def __init__(self, task, objective='interval_com'):
        super().__init__(task, objective=objective)

    @staticmethod
    def _vector(value, name, *, scalar=False):
        value = np.asarray(value, dtype=np.float64)
        if scalar and value.ndim == 0:
            value = np.full(33, float(value))
        if value.shape != (33,) or not np.isfinite(value).all():
            raise ValueError(f'{name} must contain thirty-three finite values')
        return value.copy()

    def command(self, world, goal, nominal, previous, radius):
        """Return a finite latent command; only the caller's environment applies it."""
        if not isinstance(world, (int, np.integer)) or not 0 <= world < self.task.get_num_envs():
            raise ValueError('Invalid native world index')
        goal = np.asarray(goal, dtype=np.float64)
        if goal.shape != (self.model.nq,) or not np.isfinite(goal).all():
            raise ValueError('Expected a finite complete goal pose')
        nominal = self._vector(nominal, 'nominal')
        previous = self._vector(previous, 'previous')
        radius = self._vector(radius, 'radius', scalar=True)
        if (radius <= 0).any() or (np.abs(previous) > radius + 1e-7).any():
            raise ValueError('Invalid residual radius or previous command')
        control = self.task.engine.policy_control
        joint_lower, joint_upper = control.lower.numpy(), control.upper.numpy()
        if (nominal < joint_lower - 1e-7).any() or (nominal > joint_upper + 1e-7).any():
            raise ValueError('Nominal source targets must already respect physical joint limits')
        residual_lower = np.maximum(-radius * (1 - LATENT_MARGIN), previous - SLEW)
        residual_upper = np.minimum(radius * (1 - LATENT_MARGIN), previous + SLEW)
        if (residual_lower > residual_upper).any():
            raise ValueError('Empty reachable residual interval')
        # Clipping both endpoints also represents a joint already pinned at its limit.
        lower = np.clip(nominal + residual_lower, joint_lower, joint_upper)
        upper = np.clip(nominal + residual_upper, joint_lower, joint_upper)
        midpoint, halfwidth = .5 * (lower + upper), .5 * (upper - lower)
        live = self.task.engine.datas[world]
        before_q, before_v = live.qpos.copy(), live.qvel.copy()
        before_pending = control.pending.clone()
        before_active, before_addition = control.active.clone(), control.addition.clone()
        before_cursor = control.cursor
        try:
            target, diagnostics = super().target(world, goal, midpoint, halfwidth)
        finally:
            unchanged = (np.array_equal(live.qpos, before_q) and np.array_equal(live.qvel, before_v)
                and np.array_equal(control.pending.numpy(), before_pending.numpy())
                and np.array_equal(control.active.numpy(), before_active.numpy())
                and np.array_equal(control.addition.numpy(), before_addition.numpy())
                and control.cursor == before_cursor)
            if not unchanged:
                raise RuntimeError('Private target solver modified the live plant or command queue')
        target = self._vector(target, 'planned target')
        if (target < lower - 1e-7).any() or (target > upper + 1e-7).any():
            raise RuntimeError('Teacher target escaped the reachable issue box')
        residual = np.clip(target - nominal, residual_lower, residual_upper)
        latent = np.arctanh(np.clip(residual / radius, -1 + LATENT_MARGIN, 1 - LATENT_MARGIN)).astype(np.float32)
        desired = radius.astype(np.float32) * np.tanh(latent)
        old = previous.astype(np.float32)
        applied = old + np.clip(desired - old, -SLEW, SLEW)
        reconstructed = np.clip(nominal.astype(np.float32) + applied, joint_lower, joint_upper)
        reconstruction_error = float(np.max(np.abs(reconstructed - target)))
        if not np.isfinite(latent).all() or reconstruction_error > 2e-6:
            raise RuntimeError('Finite latent does not reconstruct the planned target')
        if (np.abs(applied) > radius + 1e-7).any() or (np.abs(applied - old) > SLEW + 1e-7).any():
            raise RuntimeError('Teacher latent escaped the radius or slew contract')
        return latent, dict(diagnostics, planned_target=target.tolist(),
            reconstructed_target=reconstructed.tolist(), requested_residual=residual.tolist(),
            predicted_applied_addition=applied.tolist(), residual_lower=residual_lower.tolist(),
            residual_upper=residual_upper.tolist(), target_lower=lower.tolist(), target_upper=upper.tolist(),
            target_roundtrip_error=reconstruction_error, live_state_and_queue_unchanged=True,
            collapsed_target_dimensions=int((halfwidth <= 1e-10).sum()),
            interval_slew_max_rad=float(np.abs(applied - old).max()),
            residual_envelope_fraction_max=float(np.max(np.abs(applied) / radius)),
            optimizer_baseline='Midpoint of the reachable issued-target interval',
            label='Scripted diagnostic teacher; no learned policy or physical success claim')
