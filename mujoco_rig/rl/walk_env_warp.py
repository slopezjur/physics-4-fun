"""Velocity-commanded walking on the MuJoCo plant, batched on the GPU.

**Why this exists at all.** Before a policy, the Godot MuJoCo Walk scene was driven by a scripted
oscillator (now `MjScriptedController.cs`) - a lateral weight shift plus a half-cycle leg lift. It has no push-off, no
ankle control, and above all **no heading input**, so "walk in a straight line" and "turn left" are
not things it can be asked for. It travels by shifting weight and dragging, which is what makes it
look wrong. A commanded gait has to be learned.

The command is three numbers: forward speed, lateral speed, and turn rate, all in the pelvis frame.
That is what makes straight-line walking and turning the SAME policy rather than two scripts.

**Failure modes this reward is shaped against**, all of them recorded on this project:

  * the *statue* - a policy that scores well by not moving. Tracking a nonzero commanded speed is
    the only way to earn the largest reward term.
  * the *flamingo* - standing on one leg. Air time is rewarded per foot but only up to a target,
    and only while the foot is actually swinging, so parking one leg in the air earns nothing.
  * the *hop* - both feet leaving together. Rewarded air time is capped and the feet are pushed out
    of phase by rewarding single support, which is what a walk actually is.

Builds on `BodyEnvWarp` rather than re-implementing the plumbing: the model, buffers, stepping,
resets and zero-copy views are shared, and this adds only the commands, the heading hold and the
reward. There is no projectile.
"""
from __future__ import annotations

import numpy as np
import torch

from env_config import CMD_FLOOR, PENALTY_CAP, TRACK_REL, WALK_FOOT_CLEAR
from body_env_warp import BodyEnvWarp
from walk_config import COMMAND_SECONDS, HEADING_GAIN, STAND_PLANTED, TARGET_AIR_TIME, WalkCommandConfig



class WalkEnvWarp(BodyEnvWarp):
    """Commanded locomotion. Same body, same actuators, different question."""

    def __init__(self, num_envs=8192, device="cuda", episode_seconds=20.0, seed=0,
                 model="dummy.xml", command_config=None):
        self.command_config = command_config or WalkCommandConfig()
        super().__init__(num_envs=num_envs, device=device, episode_seconds=episode_seconds,
                         seed=seed, model=model)

    # ---------------------------------------------------------------- task state
    def _extra_init(self):
        dev = torch.device(self.device)
        n = self.num_envs
        # `self.commands` already exists on the base as reserved slots; walk simply starts
        # writing them. The observation width is unchanged, so a perturb brain seeds a walk one
        # directly, with no surgery on the network.
        self.command_age = torch.zeros(n, device=dev)
        self.air_time = torch.zeros(n, 2, device=dev)          # per foot, seconds airborne
        self.was_airborne = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        # 0 = left was the last foot to land, 1 = right. Starts at right so the first left
        # landing counts, rather than making the first step of every episode unpayable.
        self.last_landed = torch.ones(n, dtype=torch.long, device=dev)
        self.step_dt = self.dt * self.decimation
        # The yaw rate the command was DRAWN with, and the heading a zero-yaw command holds.
        self.cmd_yaw = torch.zeros(n, device=dev)
        self.heading_target = torch.zeros(n, device=dev)
        # Three command channels ride on the end of the base observation.

    def _resample_commands(self, mask):
        """Draw commands in the four shapes a stick actually produces. See MIX_* above."""
        n = self.num_envs
        cfg = self.command_config
        stand_share, straight_share, turn_share, _ = cfg.mix
        fwd = self._rand(n, lo=cfg.forward[0], hi=cfg.forward[1])
        lat = self._rand(n, lo=cfg.lateral[0], hi=cfg.lateral[1])
        turn = self._rand(n, lo=cfg.turn[0], hi=cfg.turn[1])
        zero = torch.zeros_like(fwd)

        roll = self._rand(n)
        stand = roll < stand_share
        straight = (roll >= stand_share) & (roll < stand_share + straight_share)
        in_place = ((roll >= stand_share + straight_share)
                    & (roll < stand_share + straight_share + turn_share))
        # everything else is an arc: forward and turn together

        # An arc's turn rate scales with how fast it is going, because that is what a stick does -
        # a hard turn at a crawl and a hard turn at full speed are different manoeuvres, and only
        # the correlated one appears in play.
        speed_frac = (fwd - cfg.forward[0]) / (cfg.forward[1] - cfg.forward[0])
        cmd = torch.stack([fwd, lat, turn * speed_frac], dim=-1)
        cmd = torch.where(straight.unsqueeze(-1),
                          torch.stack([fwd.abs(), zero, zero], dim=-1), cmd)
        cmd = torch.where(in_place.unsqueeze(-1),
                          torch.stack([zero, zero, turn], dim=-1), cmd)
        cmd = torch.where(stand.unsqueeze(-1), torch.zeros_like(cmd), cmd)
        self.commands = torch.where(mask.unsqueeze(-1), cmd, self.commands)
        self.command_age = torch.where(mask, torch.zeros_like(self.command_age), self.command_age)
        self.cmd_yaw = torch.where(mask, self.commands[:, 2], self.cmd_yaw)
        self.heading_target = torch.where(mask, self.heading(), self.heading_target)

    def heading(self):
        """Pelvis yaw, as eval_walk measures it."""
        r = self.xmat[:, self.pelvis]
        return torch.atan2(r[:, 1, 0], r[:, 0, 0])

    def apply_heading_hold(self):
        """A zero yaw command holds the heading the command began on."""
        err = torch.remainder(self.heading_target - self.heading() + np.pi, 2 * np.pi) - np.pi
        hold = self.cmd_yaw == 0.0
        self.commands[:, 2] = torch.where(hold, (HEADING_GAIN * err).clamp(-1.0, 1.0), self.cmd_yaw)

    def set_command(self, vx, vy=0.0, yaw=0.0):
        """Pin every world to one command, holding its current heading. Mirrors WalkEnv."""
        self.commands[:] = torch.tensor((vx, vy, yaw), dtype=torch.float32, device=self.device)
        self.cmd_yaw[:] = float(yaw)
        self.heading_target = self.heading()
        self.command_age.zero_()
        self._pinned = True

    def reset_idx(self, idx):
        super().reset_idx(idx)
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[idx] = True
        if getattr(self, "_pinned", False):
            self.heading_target = torch.where(mask, self.heading(), self.heading_target)
        else:
            self._resample_commands(mask)
        self.air_time[idx] = 0.0
        self.was_airborne[idx] = False
        self.last_landed[idx] = 1
        self.command_age[idx] = 0.0
        self.commands[idx, 2] = self.cmd_yaw[idx]

    # ---------------------------------------------------------------- stepping
    def step(self, actions):
        obs, rew, done, extras = super().step(actions)
        # Commands expire on their own clock, independent of episodes, so one episode contains
        # several - a policy that only ever sees one command per life learns to ignore the channel.
        self.command_age += self.step_dt
        expired = self.command_age >= COMMAND_SECONDS
        if not getattr(self, "_pinned", False) and bool(expired.any()):
            self._resample_commands(expired)
        self.apply_heading_hold()
        # The command slots are part of the observation: hand the policy the corrected one.
        return self.get_observations(), rew, done, extras

    # ---------------------------------------------------------------- reward
    def reward(self, action):
        r = self.xmat[:, self.pelvis]
        lin = torch.einsum("nji,nj->ni", r, self.cvel[:, self.pelvis, 3:6])   # pelvis frame
        ang = torch.einsum("nji,nj->ni", r, self.cvel[:, self.pelvis, :3])
        up_z = r[:, 2, 2]
        height = self.xpos[:, self.pelvis, 2]

        # --- the terms that define the task
        # **Tracking is judged RELATIVE to the commanded speed.** A fixed width was tried twice: at
        # sigma 0.25 a statue collected 24% of the tracking reward at 0.60 m/s and the first policy
        # stood still; sigma 0.15 fixed that at 0.60 m/s (9%) - and then stage 1 was slowed to
        # 0.15-0.35 m/s, where the same width paid a statue 44-86%, and the slow stage stood still
        # again (vx 0.03). Measured, a statue against a body gliding at exactly the command:
        #
        #   command                0.15   0.25   0.35   0.60 m/s
        #   margin, sigma 0.15    +0.56  +1.36  +2.23  +3.64
        #   margin, relative      +3.60  +3.60  +3.60  +3.60   (a statue keeps 10% at any speed)
        #
        # preflight's check_walk_beats_statue fails on the first row. Yaw keeps its fixed width:
        # tightening a zero-yaw kernel would pay stillness.
        # Exposed for the trainer's log. Episode length cannot tell a walk from a statue -
        # three sessions in a row raised ep_len 573 -> 736 while travelling zero metres -
        # so the thing being optimised has to be visible while the session is still running.
        self.vx_ema = 0.99 * getattr(self, 'vx_ema', 0.0) + 0.01 * float(lin[:, 0].mean())
        lin_err = (self.commands[:, :2] - lin[:, :2]).square().sum(dim=-1)
        scale = TRACK_REL * torch.clamp(self.commands[:, :2].norm(dim=-1), min=CMD_FLOOR).square()
        track_lin = torch.exp(-lin_err / scale)
        yaw_err = (self.commands[:, 2] - ang[:, 2]).square()
        track_yaw = torch.exp(-yaw_err / 0.15)

        # --- posture, so it tracks the command while upright rather than by falling forwards
        upright = torch.clamp(up_z, 0.0, 1.0)
        tall = torch.exp(-40.0 * (height - self.rest_pelvis_z) ** 2)
        # Vertical bounce and body roll/pitch are the pogo signature this project has hit before.
        # Clamped at PENALTY_CAP: a real gait never reaches it, a blow-up did, and unclamped these
        # two turned one exploding world into -3.6e15 and then NaN for the whole update.
        penalty_vz = lin[:, 2].square().clamp(max=PENALTY_CAP)
        penalty_rp = ang[:, :2].square().sum(dim=-1).clamp(max=PENALTY_CAP)

        # --- gait shaping
        foot_z = torch.stack([self.xpos[:, self.foot_l, 2], self.xpos[:, self.foot_r, 2]], dim=-1)
        # **Above where the foot RESTS, not above the floor** - as in perturb. The origin rests at
        # 4.4 cm, so an absolute 6 cm read a 1.6 cm shuffle as a step, and every gait term below
        # (and "both feet down" for a stand) was measuring shuffles.
        airborne = foot_z > (self.rest_foot_z + WALK_FOOT_CLEAR)
        self.air_time = self.air_time + airborne.float() * self.step_dt
        # Reward air time only as the foot LANDS, and only up to a target. Rewarding it while the
        # foot is up pays a flamingo to keep standing on one leg.
        landed = self.was_airborne & ~airborne
        air_reward = (torch.clamp(self.air_time, max=TARGET_AIR_TIME) * landed.float()).sum(dim=-1)
        self.air_time = torch.where(airborne, self.air_time, torch.zeros_like(self.air_time))
        self.was_airborne = airborne
        # Exactly one foot down is a walk; two is a stand, zero is a hop or a fall.
        single_support = (airborne.sum(dim=-1) == 1).float()
        # **And nothing used to punish zero.** Both feet off the ground is a hop, and a hop tracks a
        # forward command perfectly well - it was reachable, unpenalised, and it is exactly the
        # "weird movement" that makes a gait unusable. This is the term that makes walking the
        # cheapest way to satisfy the command.
        #
        # **Weighted 3.0, and every value here was measured rather than chosen.** The first
        # bracket was taken on a policy whose exploration had collapsed, so it could not move at
        # all, and it does not transfer:
        #
        #   exploration 0.10, penalty 1.0  ->  +2.34 m, 21% airborne,   6.6% upright  (lunging)
        #   exploration 0.10, penalty 2.5  ->  -0.01 m,  0% airborne, 100.0% upright  (a statue)
        #   exploration 0.40, penalty 1.5  ->  0.507 m/s, 19% airborne, 9.5% upright  (bounding)
        #
        # The third row is the informative one: once the policy CAN move, `4.0 * track_lin` plus
        # `1.0 * alternated` outbid a penalty of 1.5, and the gait becomes a bound - both feet off
        # the ground, which is the failure this term exists to prevent. The penalty has to be
        # weighed against what a hop now EARNS, not against what it earned when nothing moved.
        flight = (airborne.sum(dim=-1) == 2).float()
        # **Alternation - "use both feet".** Everything else here is satisfied by hopping on one
        # leg: air time pays on landing, single support pays for one foot down, and a one-legged
        # hop does both. This pays only when the foot that lands is the OTHER one from last time,
        # which is the difference between a gait and a limp.
        landed_l, landed_r = landed[:, 0], landed[:, 1]
        alternated = ((landed_l & (self.last_landed == 1))
                      | (landed_r & (self.last_landed == 0))).float()
        self.last_landed = torch.where(
            landed_l, torch.zeros_like(self.last_landed),
            torch.where(landed_r, torch.ones_like(self.last_landed), self.last_landed))

        # --- cost
        effort = action.square().mean(dim=-1)
        jerk = (action - self.prev_action).square().mean(dim=-1)

        # A commanded stand should not be paid for stepping, so the gait terms are gated on the
        # command actually asking for motion.
        moving = (self.commands[:, :2].norm(dim=-1) + self.commands[:, 2].abs() > 0.15).float()
        planted = (airborne.sum(dim=-1) == 0).float()

        # No flat survival bonus: it pays the body for existing, which is exactly what a statue
        # does best. Posture terms are demoted to support - they keep it upright WHILE tracking,
        # they are not the reason to be here.
        # **Posture is worth 3.0, not 0.8.** With it low the policy found a third strategy that is
        # neither statue nor walk: DIVE forward, collect the velocity-tracking reward for two
        # seconds, fall, reset. Episode length plateaued at 128 steps while return per step stayed
        # high, which is the signature. Standing now pays 3.4/step against a dive's ~5 for 2 s and
        # then -10, while walking upright pays 9 - so the ordering is dive < stand < walk.
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
