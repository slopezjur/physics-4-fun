"""Fixed quiet/entry world mix with long terminal-pose practice."""
import math

import torch
from envs.base_env import EnvMode

from .reference import StandReference
from .stand_task import StandTask
from .support_transition import (SupportTransitionReference, TRACKING_REWARD_SCHEMA,
                                 support_tracking_reward)


class SupportRehearsalTask(StandTask):
    def __init__(self, rig, reference, quiet_reference, num_envs, device, *,
                 quiet_fraction=.25, support_seconds=12., quiet_seconds=3.,
                 reward_mode="reference", **kwargs):
        if (not all(math.isfinite(value) for value in (quiet_fraction, support_seconds, quiet_seconds))
                or not 0 < quiet_fraction < 1 or support_seconds <= 6 or quiet_seconds <= 0
                or reward_mode not in ("reference", TRACKING_REWARD_SCHEMA)):
            raise ValueError("Require valid quiet/entry rehearsal settings")
        self.quiet_reference = StandReference(quiet_reference, rig, device)
        if quiet_seconds > self.quiet_reference.duration:
            raise ValueError("Quiet episode exceeds the quiet reference")
        self.quiet_worlds = round(num_envs * quiet_fraction)
        if not 0 < self.quiet_worlds < num_envs:
            raise ValueError("Both quiet and entry worlds are required")
        self.support_seconds = support_seconds
        self.quiet_seconds = quiet_seconds
        self.reward_mode = reward_mode
        super().__init__(rig, reference, num_envs, device, control_mode="target_pd",
                         contact_mode="support_v2", reference_factory=lambda p, r, d:
                         SupportTransitionReference(p, r, d, mode="transition"), **kwargs)
        self.episode_seconds = support_seconds

    def quiet_mask(self, ids=None):
        ids = self.ids if ids is None else ids
        return (ids < self.quiet_worlds) & (self._mode == EnvMode.TRAIN)

    def episode_duration(self, env_ids=None):
        return torch.where(self.quiet_mask(env_ids), self.quiet_seconds, self.support_seconds)

    def reference_duration(self, env_ids=None):
        return torch.where(self.quiet_mask(env_ids), self.quiet_reference.duration,
                           self.reference.duration)

    def reference_phase(self, time):
        support = (time / self.reference.duration).clamp(max=1.)
        quiet = time / self.quiet_reference.duration
        return torch.where(self.quiet_mask(), quiet, support)

    def sample_reference(self, time, env_ids=None):
        support_q, support_v = self.reference.sample(time)
        quiet_q, quiet_v = self.quiet_reference.sample(time)
        mask = self.quiet_mask(env_ids).reshape((-1,) + (1,) * time.ndim)
        return torch.where(mask, quiet_q, support_q), torch.where(mask, quiet_v, support_v)

    def reset(self, env_ids=None):
        ids = self.ids if env_ids is None else env_ids
        if len(ids):
            self.steps[ids] = 0
            self.offset[ids] = 0
            quiet_ids = ids[self.quiet_mask(ids)]
            if len(quiet_ids):
                self.offset[quiet_ids] = torch.rand(len(quiet_ids), device=self.device,
                                                    generator=self.rng) * (
                    self.quiet_reference.duration - self.quiet_seconds)
            q, v = self.sample_reference(self.offset[ids], ids)
            self.engine.reset_envs(ids, q, v)
        return self.observations(), {}

    def reward(self):
        base = self.reference_reward()
        if self.reward_mode == "reference":
            return base
        q, v = self.sample_reference(self.offset + self.steps * self.dt)
        tracked = support_tracking_reward(base, self.engine.get_root_pos(0)[:, :2],
                                          q[:, :2], self.engine.get_root_vel(0)[:, :2], v[:, :2])
        return torch.where(self.quiet_mask(), base, tracked)

    def experiment_contract(self):
        return dict(schema="mimic_support_rehearsal_v1",
                    quiet_worlds=self.quiet_worlds,
                    support_worlds=self.get_num_envs() - self.quiet_worlds,
                    quiet_seconds=self.quiet_seconds,
                    support_seconds=self.support_seconds,
                    quiet_reference_sha256=self.quiet_reference.sha256,
                    support_reference_sha256=self.reference.sha256,
                    reward_mode=self.reward_mode,
                    terminal_pose_and_phase_held_after_seconds=self.reference.duration)
