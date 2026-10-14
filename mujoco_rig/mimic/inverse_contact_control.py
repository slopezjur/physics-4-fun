"""Local torque-to-acceleration map using MuJoCo's compliant inverse contacts."""
import mujoco
import numpy as np


def solve_box(hessian, linear, lower, upper, initial=None):
    """Solve an SPD bounded quadratic, eliminating saturated/fixed coordinates."""
    if np.any(lower > upper):
        raise ValueError('Empty control interval')
    solution = np.clip(np.zeros(len(lower)) if initial is None else initial.copy(), lower, upper)
    movable = upper - lower > 1e-10
    fixed = ~movable
    solution[fixed] = lower[fixed]
    count = int(movable.sum())
    rank = 0
    if count:
        h = np.ascontiguousarray(hessian[np.ix_(movable, movable)])
        g = linear[movable] + hessian[np.ix_(movable, fixed)] @ solution[fixed]
        free = solution[movable].copy()
        rank = mujoco.mju_boxQP(free, np.zeros((count, count + 7)), None, h, g, lower[movable], upper[movable])
        solution[movable] = free
    return solution, rank


def inverse_contact_map(model, data, dofs, scale):
    """Linearize generalized inverse force at the current physical acceleration.

    DfDa is transposed in MuJoCo's API. The resulting map includes the contact
    compliance/cone response instead of independent, freely assignable wrenches.
    Call only on the controller's private data object.
    """
    acceleration = data.qacc.copy()
    mujoco.mj_inverse(model, data)
    force = data.qfrc_inverse.copy()
    derivative = np.zeros((model.nv, model.nv))
    mujoco.mjd_inverseFD(model, data, 1e-5, 0, None, None, derivative, None, None, None, None)
    derivative = derivative.T.copy()
    fixed = data.qfrc_actuator.copy()
    fixed[dofs] = 0.
    actuation = np.zeros((model.nv, len(dofs)))
    actuation[dofs, np.arange(len(dofs))] = scale
    mapping = np.linalg.solve(derivative, actuation)
    offset = acceleration + np.linalg.solve(derivative, fixed - force)
    return mapping, offset, derivative, acceleration, force, fixed


def refine_forward_torque(model, data, actuators, dofs, scale, tracking, target,
                          source, lower, upper, iterations=8):
    """Accept torque updates only when native forward dynamics reduces the objective.

    Contact mode changes invalidate a large inverse-linearization step. Small
    bounded steps and backtracking evaluate the real compliant contact response.
    The caller supplies private data; this does not advance time or alter q/v.
    """
    def evaluate(control):
        data.ctrl[actuators] = control * scale
        mujoco.mj_forward(model, data)
        error = tracking @ data.qacc - target
        return float(error @ error + .15**2 * np.sum((control - source)**2))

    control = np.clip(source, lower, upper)
    initial_cost = cost = evaluate(control)
    accepted = 0
    for _ in range(iterations):
        mapping, offset, *_ = inverse_contact_map(model, data, dofs, scale)
        reduced = tracking @ mapping
        hessian = reduced.T @ reduced + np.eye(len(control)) * (.15**2 + 1e-7)
        linear = -reduced.T @ (target - tracking @ offset) - .15**2 * source
        proposal, rank = solve_box(hessian, linear, np.maximum(lower, control - .1),
                                  np.minimum(upper, control + .1), control)
        if rank < 0:
            break
        direction = proposal - control
        if np.max(abs(direction)) < 1e-7:
            break
        improved = False
        for fraction in (1., .5, .25, .125):
            trial = control + fraction * direction
            trial_cost = evaluate(trial)
            if np.isfinite(trial_cost) and trial_cost < cost - 1e-9:
                control, cost = trial, trial_cost
                accepted += 1
                improved = True
                break
        if not improved:
            break
    evaluate(control)
    return control, data.qacc.copy(), dict(initial_tracking_cost=initial_cost,
        final_tracking_cost=cost, accepted_refinements=accepted)
