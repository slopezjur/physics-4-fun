"""Stable observation moments for narrowly spaced stationary targets."""
import torch

from learning.normalizer import Normalizer


class StableNormalizer(Normalizer):
    """Keep the existing serialized mean/std/count and inference mapping."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_count = 0
        self._pending_mean = None
        self._pending_m2 = None
        self._running_mean = None
        self._running_m2 = None

    def record(self, x):
        values = x.detach().reshape(-1, *self.get_shape()).double()
        variance, mean = torch.var_mean(values, dim=0, correction=0)
        count = len(values)
        m2 = variance * count
        if self._pending_count:
            delta = mean - self._pending_mean
            total = self._pending_count + count
            self._pending_m2 += m2 + delta.square() * (self._pending_count * count / total)
            self._pending_mean += delta * (count / total)
            self._pending_count = total
        else:
            self._pending_count = count
            self._pending_mean, self._pending_m2 = mean, m2

    def update(self):
        if not self._pending_count:
            return
        old_count = int(self._count.item())
        if self._running_mean is None:
            self._running_mean = self._mean.detach().double().clone()
            self._running_m2 = self._std.detach().double().square() * old_count
        total = old_count + self._pending_count
        delta = self._pending_mean - self._running_mean
        self._running_m2 += self._pending_m2 + delta.square() * (old_count * self._pending_count / total)
        self._running_mean += delta * (self._pending_count / total)
        self._mean.data.copy_(self._running_mean)
        self._std.data.copy_((self._running_m2 / total).clamp_min(self._min_var).sqrt())
        self._count.data.fill_(total)
        self._pending_count = 0
        self._pending_mean = self._pending_m2 = None


def install(agent):
    """Install before loading a checkpoint; inference remains standard normalization."""
    original = agent._obs_norm
    agent._obs_norm = StableNormalizer(original.get_shape(), device=agent._device,
                                       clip=10., dtype=original.dtype)
