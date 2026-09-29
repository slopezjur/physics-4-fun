"""Quiet-standing foot anchoring; no prescribed recovery step or motion clock."""
import numpy as np
import torch

from envs.base_env import EnvMode
from .baseline import sha256

from .continuous_stand_task import ContinuousStandTask


def foot_anchor_multiplier(feet, targets, sigma):
    squared_error = (feet - targets).square().sum(dim=-1).mean(dim=-1)
    return torch.exp(-squared_error / sigma ** 2)


class FootAnchoredStandTask(ContinuousStandTask):
    def __init__(self, *args, foot_anchor_sigma_m, **kwargs):
        if not 0 < foot_anchor_sigma_m < float("inf"):
            raise ValueError("Foot anchoring requires a positive finite position scale")
        self.foot_anchor_sigma_m = foot_anchor_sigma_m
        self.late_states = None
        super().__init__(*args, **kwargs)
        q, _ = self.sample_reference(self.offset)
        positions, _ = self.mapper.from_native_qpos(q)
        self._fixed_feet = positions[:, self.feet].clone()

    def load_late_states(self, path):
        """Mix real settled rollout states with ordinary starts, retaining posture IDs."""
        bank = np.load(path, allow_pickle=False)
        if str(bank["model_sha256"]) != sha256(self.rig.path):
            raise ValueError("Late-state bank uses a different physical model")
        if str(bank["reference_sha256"]) != sha256(self.reference.path):
            raise ValueError("Late-state bank uses a different stationary reference")
        fields = ("qpos", "qvel", "ground", "pending", "active")
        if any(not np.isfinite(bank[key]).all() for key in fields):
            raise ValueError("Late-state bank contains nonfinite state")
        self.late_states = {key: torch.tensor(bank[key], dtype=torch.float32, device=self.device)
                            for key in fields}
        self.late_groups = [torch.as_tensor(np.flatnonzero(bank["posture"] == i), device=self.device)
                            for i in range(8)]
        if any(len(group) == 0 for group in self.late_groups):
            raise ValueError("Late-state bank must cover all eight postures")

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = self.ids if env_ids is None else env_ids
        if self.late_states is not None and self._mode == EnvMode.TRAIN and len(ids):
            selected = ids[torch.rand(len(ids), device=self.device, generator=self.rng) < .5]
            for posture, group in enumerate(self.late_groups):
                worlds = selected[selected % 8 == posture]
                if not len(worlds):
                    continue
                rows = group[torch.randint(len(group), (len(worlds),), device=self.device, generator=self.rng)]
                bank = self.late_states
                self.engine.reset_envs(worlds, bank["qpos"][rows], bank["qvel"][rows])
                control = self.engine.policy_control
                control.active[worlds] = bank["active"][rows]
                for slot in range(len(control.pending)):
                    control.pending[(control.cursor + slot) % len(control.pending), worlds] = bank["pending"][rows, slot]
                if hasattr(self.engine, "_ground_force_cache"):
                    self.engine._ground_force_cache[worlds.cpu().numpy()] = bank["ground"][rows].cpu().numpy()
                else:
                    self.engine._ground_contact_forces[0][worlds] = bank["ground"][rows]
        return self.observations(), {}

    def reward(self):
        reward = super().reward()
        multiplier = foot_anchor_multiplier(self.engine.get_body_pos(0)[:, self.feet],
                                             self._fixed_feet, self.foot_anchor_sigma_m)
        self._diagnostics["foot_anchor_multiplier"] = multiplier.mean()
        return reward * multiplier
