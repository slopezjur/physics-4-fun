"""Transactional actor updates, including optimizer momentum on rejection."""
import copy
import math

import torch


class UpdateGuard:
    def __init__(self, parameters, optimizer, max_kl):
        if not math.isfinite(max_kl) or max_kl <= 0:
            raise ValueError("The KL limit must be finite and positive")
        self.parameters = list(parameters)
        self.optimizer = optimizer
        self.max_kl = max_kl

    def step(self, loss, measure_kl):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite actor loss")
        weights = [p.detach().clone() for p in self.parameters]
        state = copy.deepcopy(self.optimizer.state_dict())
        try:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, 1.0, error_if_nonfinite=True)
            self.optimizer.step()
            if not all(torch.isfinite(p).all() for p in self.parameters):
                raise FloatingPointError("Nonfinite actor weights")
            kl = float(measure_kl())
            if not math.isfinite(kl):
                raise FloatingPointError("Nonfinite actor KL")
            if kl <= self.max_kl:
                return True, kl
        except Exception:
            self._restore(weights, state)
            raise
        self._restore(weights, state)
        return False, kl

    def _restore(self, weights, state):
        with torch.no_grad():
            for parameter, value in zip(self.parameters, weights):
                parameter.copy_(value)
        self.optimizer.load_state_dict(state)
        self.optimizer.zero_grad(set_to_none=True)
