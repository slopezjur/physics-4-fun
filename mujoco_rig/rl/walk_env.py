"""Velocity-commanded walking on MuJoCo's C engine - the SCORER's copy of `walk_env_warp.py`.

Training happens on the GPU; every checkpoint is scored here, on the float64 engine that Godot
actually drives through P/Invoke. `test_walk_parity.py` asserts the two define the same task.

Builds on `BodyEnv` - the body, observation, reset and step - and adds only what the task changes:
three command channels with a heading hold, and a reward that pays for tracking them.
"""
from __future__ import annotations

import numpy as np

from body_env import BodyEnv
from env_config import CMD_FLOOR, PENALTY_CAP, TRACK_REL, WALK_FOOT_CLEAR
from walk_config import COMMAND_SECONDS, HEADING_GAIN, STAND_PLANTED, TARGET_AIR_TIME, WalkCommandConfig


class WalkEnv(BodyEnv):
    """Commanded locomotion on the CPU engine. Mirrors WalkEnvWarp exactly."""

    def __init__(self, num_envs=8, device="cpu", episode_seconds=20.0, seed=0, model="dummy.xml",
                 command_config=None):
        self.command_config = command_config or WalkCommandConfig()
        super().__init__(num_envs=num_envs, device=device, episode_seconds=episode_seconds,
                         seed=seed, model=model)

    # ---------------------------------------------------------------- task state
    def _extra_init(self):
        n = self.num_envs
        # `self.commands` already exists on the base as reserved slots.
        self.command_age = np.zeros(n)
        self.air_time = np.zeros((n, 2))
        self.was_airborne = np.zeros((n, 2), dtype=bool)
        self.last_landed = np.ones(n, dtype=int)     # see walk_env_warp: 0 = left, 1 = right
        self.step_dt = self.dt * self.decimation
        self.cmd_yaw = np.zeros(n)                   # see walk_env_warp: heading hold
        self.heading_target = np.zeros(n)

    def _resample_commands(self, idx):
        """The four shapes a stick produces. Mirrors WalkEnvWarp._resample_commands exactly."""
        cfg = self.command_config
        stand_share, straight_share, turn_share, _ = cfg.mix
        for i in np.atleast_1d(idx):
            fwd = self.rng.uniform(*cfg.forward)
            lat = self.rng.uniform(*cfg.lateral)
            turn = self.rng.uniform(*cfg.turn)
            roll = self.rng.random()
            if roll < stand_share:
                self.commands[i] = 0.0
            elif roll < stand_share + straight_share:
                self.commands[i] = (abs(fwd), 0.0, 0.0)
            elif roll < stand_share + straight_share + turn_share:
                self.commands[i] = (0.0, 0.0, turn)
            else:
                speed_frac = (fwd - cfg.forward[0]) / (cfg.forward[1] - cfg.forward[0])
                self.commands[i] = (fwd, lat, turn * speed_frac)
            self.command_age[i] = 0.0
            self.cmd_yaw[i] = self.commands[i, 2]
            self.heading_target[i] = self.heading_of(i)

    def heading_of(self, i):
        r = self.datas[i].xmat[self.pelvis].reshape(3, 3)
        return float(np.arctan2(r[1, 0], r[0, 0]))

    def apply_heading_hold(self):
        """Mirrors WalkEnvWarp.apply_heading_hold."""
        for i in range(self.num_envs):
            if self.cmd_yaw[i] == 0.0:
                err = (self.heading_target[i] - self.heading_of(i) + np.pi) % (2 * np.pi) - np.pi
                self.commands[i, 2] = float(np.clip(HEADING_GAIN * err, -1.0, 1.0))
            else:
                self.commands[i, 2] = self.cmd_yaw[i]

    def set_command(self, vx, vy=0.0, yaw=0.0):
        """Pin every world to one command. Used by the scorer to measure a specific manoeuvre."""
        self.commands[:] = (vx, vy, yaw)
        self.command_age[:] = 0.0
        self.cmd_yaw[:] = yaw
        self.heading_target[:] = [self.heading_of(i) for i in range(self.num_envs)]
        self._pinned = True

    def reset_idx(self, idx):
        idx = np.atleast_1d(np.asarray(list(idx), dtype=int))
        super().reset_idx(idx)
        if idx.size == 0:
            return
        if not getattr(self, "_pinned", False):
            self._resample_commands(idx)
        else:
            for i in idx:
                self.heading_target[i] = self.heading_of(i)
        self.air_time[idx] = 0.0
        self.was_airborne[idx] = False
        self.last_landed[idx] = 1
        self.command_age[idx] = 0.0
        self.commands[idx, 2] = self.cmd_yaw[idx]

    # ---------------------------------------------------------------- stepping
    def step(self, actions):
        obs, rew, done, extras = super().step(actions)
        self.command_age += self.step_dt
        if not getattr(self, "_pinned", False):
            expired = np.nonzero(self.command_age >= COMMAND_SECONDS)[0]
            if expired.size:
                self._resample_commands(expired)
        self.apply_heading_hold()
        return self.get_observations(), rew, done, extras

    # ---------------------------------------------------------------- reward
    def reward(self, i, action):
        d = self.datas[i]
        r = d.xmat[self.pelvis].reshape(3, 3)
        lin = r.T @ d.cvel[self.pelvis][3:6]
        ang = r.T @ d.cvel[self.pelvis][:3]
        up_z = r[2, 2]
        height = d.xpos[self.pelvis][2]

        self.vx_ema = 0.99 * getattr(self, 'vx_ema', 0.0) + 0.01 * float(lin[0])
        lin_err = float(np.square(self.commands[i, :2] - lin[:2]).sum())
        # Relative to the commanded speed - see WalkEnvWarp.reward for the measurement.
        scale = TRACK_REL * max(float(np.linalg.norm(self.commands[i, :2])), CMD_FLOOR) ** 2
        track_lin = float(np.exp(-lin_err / scale))
        yaw_err = float((self.commands[i, 2] - ang[2]) ** 2)
        track_yaw = float(np.exp(-yaw_err / 0.15))

        upright = float(np.clip(up_z, 0.0, 1.0))
        tall = float(np.exp(-40.0 * (height - self.rest_pelvis_z) ** 2))
        penalty_vz = min(float(lin[2] ** 2), PENALTY_CAP)          # clamped - see WalkEnvWarp
        penalty_rp = min(float(np.square(ang[:2]).sum()), PENALTY_CAP)

        foot_z = np.array([d.xpos[self.foot_l][2], d.xpos[self.foot_r][2]])
        airborne = foot_z > self.rest_foot_z + WALK_FOOT_CLEAR     # above rest - see WalkEnvWarp
        self.air_time[i] += airborne * self.step_dt
        landed = self.was_airborne[i] & ~airborne
        air_reward = float((np.clip(self.air_time[i], None, TARGET_AIR_TIME) * landed).sum())
        self.air_time[i] = np.where(airborne, self.air_time[i], 0.0)
        self.was_airborne[i] = airborne
        single_support = float(airborne.sum() == 1)
        flight = float(airborne.sum() == 2)          # a hop; see walk_env_warp for why it is paid for
        alternated = float((landed[0] and self.last_landed[i] == 1)
                           or (landed[1] and self.last_landed[i] == 0))
        if landed[0]:
            self.last_landed[i] = 0
        elif landed[1]:
            self.last_landed[i] = 1

        effort = float(np.mean(np.square(action)))
        jerk = float(np.mean(np.square(action - self.prev_action[i])))

        moving = float(np.linalg.norm(self.commands[i, :2]) + abs(self.commands[i, 2]) > 0.15)
        planted = float(airborne.sum() == 0)

        return (4.0 * track_lin
                + 3.0 * track_yaw
                + 2.0 * upright
                + 1.0 * tall
                + moving * (1.5 * air_reward + 0.4 * single_support + 1.0 * alternated)
                + (1.0 - moving) * STAND_PLANTED * planted
                - 1.5 * flight
                - 0.5 * penalty_vz
                - 0.05 * penalty_rp
                - 0.05 * effort
                - 0.05 * jerk)
