"""Bounded least-squares objectives with explicit nonlinear inequalities."""
import numpy as np
from scipy.optimize import minimize, least_squares, OptimizeResult
from scipy.sparse import csr_matrix


def constrained_least_squares(evaluate, initial, lower, upper, max_iterations,
                              numerical_columns=None, analytic_tail=None, restore_feasibility=False):
    """Evaluate returns (residual, margins); every margin must be nonnegative.

    Share finite differences between objective and constraint Jacobians. This
    avoids running the expensive dynamics twice per coordinate and uses only
    public SciPy interfaces. Infeasible termination is reported, never admitted.
    An optional analytic tail supplies Jacobian columns after numerical_columns.
    Optional restoration first minimizes inequality violations. Its evaluations
    and subsequent SLSQP iterations share max_iterations; infeasible restoration
    never enters objective polishing. Feasible incumbents survive budget expiry.
    """
    numerical_columns = len(initial) if numerical_columns is None else numerical_columns
    if not 0 <= numerical_columns <= len(initial) or (numerical_columns < len(initial) and analytic_tail is None):
        raise ValueError("Every non-numerical column requires an analytic Jacobian")
    point = residual = margins = derivatives = None
    evaluations = 0
    incumbent, incumbent_cost = None, np.inf

    def feasible(c):
        return bool(np.isfinite(c).all() and c.min(initial=0.) >= -1e-8)

    def values(x):
        nonlocal point, residual, margins, derivatives, evaluations, incumbent, incumbent_cost
        if point is None or not np.array_equal(x, point):
            residual, margins = evaluate(x)
            point, derivatives = x.copy(), None
            evaluations += 1
            cost = float(residual @ residual / 2)
            if feasible(margins) and np.isfinite(cost) and cost < incumbent_cost:
                incumbent, incumbent_cost = x.copy(), cost
        return residual, margins

    def jacobians(x):
        nonlocal derivatives, evaluations
        r, c = values(x)
        if derivatives is None:
            jr, jc = np.empty((len(r), len(x))), np.empty((len(c), len(x)))
            for column in range(numerical_columns):
                step = np.sqrt(np.finfo(float).eps) * max(1., abs(x[column]))
                if x[column] + step > upper[column]:
                    step = -step
                step = np.clip(x[column] + step, lower[column], upper[column]) - x[column]
                if step == 0:
                    raise ValueError("Eliminate fixed coordinates before constrained fitting")
                trial = x.copy()
                trial[column] += step
                rr, cc = evaluate(trial)
                evaluations += 1
                jr[:, column], jc[:, column] = (rr - r) / step, (cc - c) / step
            if numerical_columns < len(x):
                jr[:, numerical_columns:], jc[:, numerical_columns:] = analytic_tail(x)
                evaluations += 1
            derivatives = jr, jc
        return derivatives

    # Force residuals are weighted by 200 in the fitting objective. Rescaling the
    # entire objective leaves its minimizer unchanged and improves SLSQP scaling.
    scale = 200. ** 2
    history = []
    phase = "optimization"

    def progress(x):
        r, c = values(x)
        violations = np.minimum(c, 0.)
        row = dict(callback=len(history) + 1, phase=phase, cost=float(r @ r / 2),
                   feasibility_cost=float(violations @ violations / 2),
                   max_constraint_violation=float(max(0., -c.min(initial=0.))))
        history.append(row)
        print(row, flush=True)

    restoration = None
    remaining = max_iterations
    if restore_feasibility and not feasible(values(initial)[1]):
        phase = "restoration"
        restored = least_squares(lambda x: np.minimum(values(x)[1], 0.), initial,
                                 jac=lambda x: csr_matrix(jacobians(x)[1] * (values(x)[1] < 0)[:, None]),
                                 bounds=(lower, upper), max_nfev=max_iterations, x_scale="jac",
                                 tr_solver="lsmr", tr_options={"atol": 1e-8, "btol": 1e-8, "maxiter": 1000},
                                 ftol=1e-9, xtol=1e-9, gtol=1e-9,
                                 callback=lambda intermediate_result: progress(intermediate_result.x))
        restoration = dict(evaluations=int(restored.nfev), cost=float(restored.cost),
                           message=restored.message, constraints_satisfied=feasible(values(restored.x)[1]))
        initial = restored.x
        remaining -= restored.nfev

    if restoration is not None and (not restoration["constraints_satisfied"] or remaining <= 0):
        result = OptimizeResult(x=initial, success=False, nit=0, nfev=0,
                                message="Feasibility restoration: " + restoration["message"])
    else:
        phase = "optimization"
        result = minimize(lambda x: float(values(x)[0] @ values(x)[0] / (2 * scale)), initial,
                          jac=lambda x: jacobians(x)[0].T @ values(x)[0] / scale,
                          bounds=list(zip(lower, upper)), method="SLSQP",
                          constraints={"type": "ineq", "fun": lambda x: values(x)[1],
                                       "jac": lambda x: jacobians(x)[1]},
                          options={"maxiter": remaining, "ftol": 1e-9}, callback=progress)
    r, c = values(result.x)
    result.final_iterate_constraint_violation = float(max(0., -c.min(initial=0.)))
    result.used_feasible_incumbent = bool(incumbent is not None and
                                        (not feasible(c) or incumbent_cost < float(r @ r / 2)))
    if result.used_feasible_incumbent:
        result.x = incumbent.copy()
        r, c = values(result.x)
        result.fun = float(r @ r / (2 * scale))
        result.success = False
        result.message += "; returned best feasible incumbent"
    result.cost = float(r @ r / 2)
    result.fun = result.cost / scale
    result.max_constraint_violation = float(max(0., -c.min(initial=0.)))
    result.constraints_satisfied = feasible(c)
    result.residual_evaluations = evaluations
    result.restoration = restoration
    result.history = history
    # The objective gradient is not constrained optimality; don't label it so.
    result.optimality = None
    return result
