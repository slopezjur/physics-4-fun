"""Fixed world allocation for quiet-standing rehearsal and longer motion episodes."""
import math
import torch

from envs.base_env import EnvMode
from .stand_task import StandTask
from .reference import StandReference
from . import shift_reward


class RehearsalTask(StandTask):
    def __init__(self, rig, reference, quiet_reference, num_envs=128, device="cuda:0", seed=210921,
                 *, quiet_fraction=.5, motion_seconds=8., quiet_seconds=5., reward_mode="reference", **kwargs):
        self.reward_contract = shift_reward.contract(reward_mode)
        self.reward_mode = reward_mode
        if (not all(math.isfinite(x) for x in (quiet_fraction, motion_seconds, quiet_seconds))
                or not 0 <= quiet_fraction < 1 or motion_seconds <= 3 or quiet_seconds <= 0):
            raise ValueError("Require a finite quiet fraction below one and motion episodes longer than three seconds")
        self.quiet_reference = StandReference(quiet_reference, rig, device)
        motion = StandReference(reference, rig, device)
        if motion_seconds > motion.duration or quiet_seconds > self.quiet_reference.duration:
            raise ValueError("Training episodes must fit inside their reference durations")
        self.quiet_worlds = round(num_envs * quiet_fraction)
        if self.quiet_worlds >= num_envs or quiet_fraction > 0 and self.quiet_worlds == 0:
            raise ValueError("World allocation must retain motion worlds and the requested quiet worlds")
        self.motion_seconds, self.quiet_seconds = motion_seconds, quiet_seconds
        super().__init__(rig, reference, num_envs, device, seed, **kwargs)

    def quiet_mask(self, env_ids=None):
        ids = self.ids if env_ids is None else env_ids
        return (ids < self.quiet_worlds) & (self._mode == EnvMode.TRAIN)

    def reference_duration(self, env_ids=None):
        return torch.where(self.quiet_mask(env_ids), self.quiet_reference.duration, self.reference.duration)

    def episode_duration(self, env_ids=None):
        if self._mode != EnvMode.TRAIN:
            return self.episode_seconds
        return torch.where(self.quiet_mask(env_ids), self.quiet_seconds, self.motion_seconds)

    def sample_reference(self, time, env_ids=None):
        motion_q, motion_v = self.reference.sample(time)
        quiet_q, quiet_v = self.quiet_reference.sample(time)
        mask = self.quiet_mask(env_ids).reshape((-1,) + (1,) * time.ndim)
        return torch.where(mask, quiet_q, motion_q), torch.where(mask, quiet_v, motion_v)

    def experiment_contract(self):
        return dict(schema="mimic_standing_rehearsal_v1", task="weight_shift",
                    quiet_worlds=self.quiet_worlds, motion_worlds=self.get_num_envs() - self.quiet_worlds,
                    quiet_fraction=self.quiet_worlds / self.get_num_envs(),
                    motion_seconds=self.motion_seconds, quiet_seconds=self.quiet_seconds,
                    motion_reference_sha256=self.reference.sha256,
                    quiet_reference_sha256=self.quiet_reference.sha256,
                    reward=self.reward_contract,
                    allocation="Fixed world groups in training; stochastic phases on reset. TEST mode uses motion only.")

    def reward(self):
        base = super().reward()
        if self.reward_mode == "reference":
            return base
        q, v = self.sample_reference(self.offset + self.steps * self.dt)
        precise = shift_reward.precision_reward(base, self.engine.get_root_pos(0), q[:, :3],
                                                self.engine.get_root_vel(0), v[:, :3])
        return torch.where(self.quiet_mask(), base, precise)
