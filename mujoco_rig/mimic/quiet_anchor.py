"""Frozen starting-policy distributions on valid quiet-standing states."""
import math
import torch


def valid_observations(traces, metrics, dt, stride=4):
    observations = torch.as_tensor(traces["observations"])
    counts = torch.tensor([round(s / dt) for s in metrics["survival_seconds"]])
    valid = torch.arange(len(observations))[:, None] < counts[None, :]
    valid &= (torch.arange(len(observations)) % stride == 0)[:, None]
    return observations[valid].clone()


class QuietAnchor:
    def __init__(self, agent, observations, max_kl):
        if not math.isfinite(max_kl) or max_kl <= 0:
            raise ValueError("Quiet anchor KL limit must be finite and positive")
        observations = observations.to(agent._device)
        if observations.ndim != 2 or not len(observations) or not torch.isfinite(observations).all():
            raise ValueError("Quiet anchor requires finite, nonempty observation rows")
        from learning.distribution_gaussian_diag import DistributionGaussianDiag
        self.max_kl = max_kl
        with torch.no_grad():
            self.observations = agent._obs_norm.normalize(observations).detach().clone()
            distribution = agent._model.eval_actor(self.observations)
            self.teacher = DistributionGaussianDiag(distribution.mean.detach().clone(), distribution.logstd.detach().clone())

    @torch.no_grad()
    def measure(self, agent):
        kl = self.teacher.kl(agent._model.eval_actor(self.observations)).clamp_min(0)
        if not torch.isfinite(kl).all():
            raise FloatingPointError("Nonfinite quiet-anchor KL")
        return float(kl.max())

    def accepts(self, agent):
        return self.measure(agent) <= self.max_kl

    def project_direction(self, agent, parameters, direction):
        """Remove outward displacement at the current worst state's tangent plane.

        A 0.1% inward margin avoids an exactly tangent proposal at the boundary.
        The guard must still check every state and actual loss after backtracking.
        """
        value = self.teacher.kl(agent._model.eval_actor(self.observations)).max()
        gradients = torch.autograd.grad(value, parameters, allow_unused=True)
        gradients = [torch.zeros_like(p) if g is None else g.detach() for p,g in zip(parameters,gradients)]
        norm = sum(g.square().sum() for g in gradients)
        dot = sum((g*d).sum() for g,d in zip(gradients,direction))
        if not all(torch.isfinite(x) for x in (value, norm, dot)):
            raise FloatingPointError("Nonfinite quiet-anchor direction")
        excess = dot - (self.max_kl * .999 - value.detach())
        if norm <= 1e-20 or excess <= 0:
            return None
        return [d - (excess / norm) * g for d,g in zip(direction,gradients)]
