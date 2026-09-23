"""Explicit nonnegative contact forces with root balance and delayed-PD bounds."""
import numpy as np


def force_balance(columns, required, controlled, torque_scale, weight, coefficients, bounds):
    """Return residuals, feasible margins and their exact force Jacobians.

    Coefficients multiply the existing friction rays in body-weight units and
    must be nonnegative. Motor commands are uniquely determined by the remaining
    generalized demand; no inner optimization or clipping enters the constraints.
    bounds=None means startup root balance only, matching the native contract.
    """
    scale = np.r_[np.full(6, weight), torque_scale]
    matrix = columns[controlled] * weight / scale[:, None]
    demand = required[controlled] / scale - matrix @ coefficients
    error = demand.copy()
    error_jac = -matrix.copy()
    margins = [1e-8 - demand[:6], 1e-8 + demand[:6]]
    margin_jac = [matrix[:6], -matrix[:6]]
    if bounds is None:
        error[6:], error_jac[6:] = 0., 0.
    else:
        error[6:] -= np.clip(demand[6:], bounds[:, 0], bounds[:, 1])
        outside = (demand[6:] < bounds[:, 0]) | (demand[6:] > bounds[:, 1])
        error_jac[6:] *= outside[:, None]
        margins.extend((demand[6:] - bounds[:, 0], bounds[:, 1] - demand[6:]))
        margin_jac.extend((-matrix[6:], matrix[6:]))
    return error, np.concatenate(margins), error_jac, np.vstack(margin_jac)
