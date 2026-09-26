"""Transactional actor updates, including optimizer momentum on rejection."""
import copy
import math

import torch


class UpdateGuard:
    def __init__(self, parameters, optimizer, max_kl, max_backtracks=0):
        if not math.isfinite(max_kl) or max_kl <= 0:
            raise ValueError("The KL limit must be finite and positive")
        self.parameters = list(parameters)
        self.optimizer = optimizer
        self.max_kl = max_kl
        if type(max_backtracks) is not int or not 0 <= max_backtracks <= 12:
            raise ValueError("Backtracking must be an integer between zero and twelve")
        self.max_backtracks = max_backtracks
        self.last_search = {}

    def step(self, loss, measure_kl, *, accept=None, measure_loss=None, project_direction=None):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite actor loss")
        if self.max_backtracks and measure_loss is None:
            raise ValueError("Backtracking requires a same-batch loss measurement")
        if project_direction is not None and not self.max_backtracks:
            raise ValueError("Direction projection requires loss-checked backtracking")
        weights = [p.detach().clone() for p in self.parameters]
        state = copy.deepcopy(self.optimizer.state_dict())
        before = float(loss.detach())
        largest_kl = 0.
        self.last_search = dict(trials=0, scale=0., loss_decrease=0., constraint_rejections=0,
                                loss_rejections=0, projected=False, projected_trials=0)
        try:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, 1.0, error_if_nonfinite=True)
            gradients = [None if p.grad is None else p.grad.detach().clone() for p in self.parameters]
            direction = None
            for projected in range(1 + (project_direction is not None)):
                if projected:
                    self._restore(weights, state)
                    direction = project_direction(direction)
                    if direction is None:
                        break
                    if len(direction) != len(weights) or any(d.shape != w.shape for d,w in zip(direction,weights)):
                        raise ValueError("Projected direction disagrees with actor parameters")
                for attempt in range(self.max_backtracks + 1):
                    self._restore(weights, state)
                    for parameter, gradient in zip(self.parameters, gradients):
                        parameter.grad = None if gradient is None else gradient.clone()
                    scale = .5 ** attempt
                    for group, original in zip(self.optimizer.param_groups, state["param_groups"]):
                        group["lr"] = original["lr"] * scale
                    self.optimizer.step()
                    if projected:
                        # Project the parameter displacement; the optimizer state
                        # still records exactly one update using the original gradient.
                        with torch.no_grad():
                            for parameter, weight, delta in zip(self.parameters, weights, direction):
                                parameter.copy_(weight + scale * delta)
                    elif attempt == 0:
                        direction = [p.detach().clone() - w for p,w in zip(self.parameters,weights)]
                    self.last_search["trials"] += 1
                    self.last_search["projected_trials"] += projected
                    if not all(torch.isfinite(p).all() for p in self.parameters):
                        raise FloatingPointError("Nonfinite actor weights")
                    kl = float(measure_kl())
                    if not math.isfinite(kl):
                        raise FloatingPointError("Nonfinite actor KL")
                    largest_kl = max(largest_kl, kl)
                    feasible = kl <= self.max_kl and (accept is None or accept())
                    if not feasible:
                        self.last_search["constraint_rejections"] += 1
                        continue
                    after = before
                    if self.max_backtracks:
                        with torch.no_grad():
                            after = float(measure_loss())
                        if not math.isfinite(after):
                            raise FloatingPointError("Nonfinite candidate actor loss")
                        if before - after <= 1e-7 * max(1., abs(before)):
                            self.last_search["loss_rejections"] += 1
                            continue
                    self.last_search.update(scale=scale, loss_decrease=before-after, projected=bool(projected))
                    for group, original in zip(self.optimizer.param_groups, state["param_groups"]):
                        group["lr"] = original["lr"]
                    return True, largest_kl
        except Exception:
            self._restore(weights, state)
            raise
        self._restore(weights, state)
        return False, largest_kl

    def _restore(self, weights, state):
        with torch.no_grad():
            for parameter, value in zip(self.parameters, weights):
                parameter.copy_(value)
        # load_state_dict may alias tensors on the same device. Each trial needs
        # an independent momentum/Adam state, not a snapshot mutated by a retry.
        self.optimizer.load_state_dict(copy.deepcopy(state))
        self.optimizer.zero_grad(set_to_none=True)
