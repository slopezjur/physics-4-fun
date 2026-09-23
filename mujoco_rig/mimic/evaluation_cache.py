"""Bounded, run-local reuse of deterministic evaluation for identical inference state."""
from collections import OrderedDict
import copy
import hashlib
import json

import torch


def actor_fingerprint(actor):
    digest = hashlib.sha256()
    for name, tensor in sorted(actor.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def evaluation_key(actor, context):
    payload = json.dumps(context, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(payload + actor_fingerprint(actor).encode()).hexdigest()


class EvaluationCache:
    """Own immutable snapshots; pin the selected best while evicting older candidates."""
    def __init__(self, capacity=3):
        if capacity < 2:
            raise ValueError("Cache needs room for best and current evaluations")
        self.capacity = capacity
        self._entries = OrderedDict()
        self._pinned = None
        self.hits = self.misses = 0

    def get_or_compute(self, key, compute):
        if key in self._entries:
            self.hits += 1
            self._entries.move_to_end(key)
        else:
            result = compute()  # A failed evaluation must never populate the cache.
            self._entries[key] = copy.deepcopy(result)
            self.misses += 1
            while len(self._entries) > self.capacity:
                victim = next(k for k in self._entries if k != self._pinned)
                del self._entries[victim]
        return copy.deepcopy(self._entries[key])

    def pin(self, key):
        if key not in self._entries:
            raise KeyError("Cannot pin an unevaluated policy")
        self._pinned = key

    def statistics(self):
        return dict(hits=self.hits, misses=self.misses, entries=len(self._entries), capacity=self.capacity)


def evaluation_due(completed_updates, interval):
    return completed_updates > 0 and completed_updates % interval == 0
