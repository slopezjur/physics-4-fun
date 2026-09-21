"""Validated, non-looping motion reference in the dummy's native coordinates."""
import json
from pathlib import Path
import numpy as np
import torch

from .baseline import sha256


class StandReference:
    def __init__(self, path: Path, rig, device):
        self.path = path
        metadata = json.loads(path.with_suffix(".json").read_text())
        if not metadata["passed"] or metadata["model_sha256"] != sha256(rig.path):
            raise ValueError("Motion validation failed or motion targets a different rig")
        if metadata["reference_sha256"] != sha256(path):
            raise ValueError("Motion content does not match its validated manifest")
        if metadata["source"]["license"] != "CC-BY-4.0":
            raise ValueError("This source license has not been reviewed for the experiment")
        self.sha256 = metadata["reference_sha256"]
        with np.load(path, allow_pickle=False) as data:
            self.qpos = torch.tensor(data["qpos"], dtype=torch.float32, device=device)
            self.qvel = torch.tensor(data["qvel"], dtype=torch.float32, device=device)
            self.dt = float(data["dt"])
        self.duration = (len(self.qpos) - 1) * self.dt
        if (self.qpos.shape != (len(self.qpos), rig.model.nq)
                or self.qvel.shape != (len(self.qpos), rig.model.nv)
                or self.duration < 3.0 or not np.isfinite(self.dt)
                or not torch.isfinite(self.qpos).all() or not torch.isfinite(self.qvel).all()):
            raise ValueError("Invalid reference dimensions, duration or state")
        if (torch.sum(self.qpos[:-1, 3:7] * self.qpos[1:, 3:7], dim=-1) <= 0).any():
            raise ValueError("Reference quaternions must use a continuous hemisphere")

    def sample(self, time):
        frame = (time / self.dt).clamp(0, len(self.qpos) - 1)
        i = frame.long().clamp(max=len(self.qpos) - 2)
        alpha = (frame - i).unsqueeze(-1)
        q = torch.lerp(self.qpos[i], self.qpos[i + 1], alpha)
        q[..., 3:7] = torch.nn.functional.normalize(q[..., 3:7], dim=-1)
        v = torch.lerp(self.qvel[i], self.qvel[i + 1], alpha)
        return q, v
