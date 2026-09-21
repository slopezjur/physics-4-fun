"""Stay standing through ball impacts, using the LEGS - on MuJoCo's C engine, the scorer's copy.

**The point of this task is what it takes away.** Every previous result on this project leaned on a
pelvis attitude controller that writes an external torque straight onto the pelvis - nothing in the
body produces it. Measured 2026-09-09: standing needs 0.5 N.m of it and is genuinely the legs' work,
but ball recovery peaks at 90.7 N.m and the body falls to 33.4% upright without it. The recovery was
the assist, not the dummy.

So here the assist is GONE. A policy that stays upright has to do it with ankles, hips and a
protective step, which is the behaviour the assist has been standing in for.

The body, observation, reset, step and divergence guard are `BodyEnv`'s; this adds the projectile
and the reward. With `dummy.xml` there is no ball, and the same task is the quiet room.
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch

from body_env import BodyEnv
from env_config import (BALL_SPAWN_DISTANCE,
                        JOINT_SOFT_LIMIT, LEG_BONES, LEG_POSE_SCALE, STANCE_SIGMA,
                        TARGET_BONES, TRUNK_BONES, TRUNK_POSE_SCALE, target_probabilities)
from foot_contacts import FootContacts, loaded_feet
from observation_contract import LEGACY, FOUNDATION, observation_size
from recovery_reward import (CONFIG, REWARD_VERSION, contact_history, reward_terms,
                             settled_state, support_state)



class PerturbEnv(BodyEnv):
    """Many independent MuJoCo dummies under fire, stepped together."""

    def __init__(self, num_envs=64, device="cpu", episode_seconds=12.0,
                 ball_every=(2.0, 4.0), ball_speed=6.0, seed=0, model="dummy_ball.xml",
                 target_weights=None, observation_version=LEGACY):
        # Set before the base constructor runs: its first reset already schedules a shot.
        self.reward_version = REWARD_VERSION
        self.observation_version = observation_version
        self.ball_every = ball_every
        self.max_shots_per_episode = None
        self.ball_speed = ball_speed
        self.target_p = target_probabilities(target_weights)
        super().__init__(num_envs=num_envs, device=device, episode_seconds=episode_seconds,
                         seed=seed, model=model)

    # ---------------------------------------------------------------- task hooks
    def _extra_init(self):
        self.num_obs = observation_size(self.num_actions, self.observation_version)
        if self.observation_version == FOUNDATION and len(self.action_queue) != 2:
            raise ValueError("foundation_v2 requires a two-step action queue")
        m = self.model
        name2id = mujoco.mj_name2id
        self.ball = name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
        if self.ball >= 0:
            self.mass[self.ball] = 0.0           # never let the projectile into the body's COM
            self.total_mass = self.mass.sum()
            self.ball_joint = name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
            self.ball_q = m.jnt_qposadr[self.ball_joint]
            self.ball_v = m.jnt_dofadr[self.ball_joint]
            # The projectile's free joint is the model's last: its spin is not the body's.
            self._body_nv = self.ball_v
        # **The projectile does not fall.** Aimed straight at a bone from 2 m, a ballistic ball
        # cannot reach it below 4.43 m/s - the range of a projectile is v^2/g, which is 0.60 m at
        # the 2.42 m/s the curriculum reached. Every shot of every perturb run landed on the floor
        # short of the dummy, so "ball speed" varied the MISS DISTANCE rather than the difficulty,
        # and the policy was never hit. Cancelling gravity on the ball alone makes the flight a
        # straight line, so speed maps linearly onto delivered impulse (m * v) and the standoff and
        # reaction time stay constant across the whole curriculum.
        self.ball_gravity_cancel = 0.0
        if self.ball >= 0:
            self.ball_gravity_cancel = float(m.body_mass[self.ball]) * abs(float(m.opt.gravity[2]))
        # **Only for the approach.** Cancelling gravity for the whole episode leaves a shot that
        # misses flying in a straight line for ever, and it comes back through the dummy - a
        # projectile that never lands is its own perturbation. Gravity is restored once the ball has
        # had time to cover the standoff, after which it falls and rolls like any other object.
        self.ball_flight = np.zeros(self.num_envs)    # sim time until which the shot is inbound
        self.targets = [name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in TARGET_BONES]
        self.next_ball = np.zeros(self.num_envs)
        self.shots_fired = np.zeros(self.num_envs, dtype=int)
        self.grounded = np.ones((self.num_envs, 2), dtype=bool)
        self.foot_contacts = FootContacts(m)
        self.since_landing = np.full((self.num_envs, 2), CONFIG.replant_interval)
        self.rapid_replants = np.zeros(self.num_envs)
        self.settle_hold = np.zeros(self.num_envs)
        # The joints the rest stance pulls back: the legs and the trunk. The arms stay free to swing.
        self.leg_q = self.qadr[[i for i, n in enumerate(self.act_names)
                                if n.split("_")[0] in LEG_BONES]]
        self.trunk_q = self.qadr[[i for i, n in enumerate(self.act_names)
                                  if n.split("_")[0] in TRUNK_BONES]]
        # Soft joint limits: JOINT_SOFT_LIMIT of each half-range from the middle of the range,
        # stretched to include the rest pose - a straight knee or elbow rests on its own stop.
        mid, half = 0.5 * (self.lo + self.hi), 0.5 * (self.hi - self.lo) * JOINT_SOFT_LIMIT
        rest = self.rest_qpos[self.qadr]
        self.soft_lo, self.soft_hi = np.minimum(mid - half, rest), np.maximum(mid + half, rest)

    def _obs_one(self, i):
        legacy = super()._obs_one(i)
        if self.observation_version == LEGACY:
            return legacy
        d = self.datas[i]
        rotation = d.xmat[self.pelvis].reshape(3, 3)
        load, _ = self.foot_contacts.read(d)
        obs = np.concatenate((legacy, rotation.T @ self._foot_velocity(d, self.pelvis),
                              load * 0.001, self.action_queue[0][i]))
        return np.clip(np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0), -100, 100)

    def _reset_world(self, i, d):
        self._park_ball(d)
        self.next_ball[i] = self.rng.uniform(*self.ball_every)
        self.ball_flight[i] = 0.0
        self.shots_fired[i] = 0
        self.grounded[i] = True
        self.since_landing[i] = CONFIG.replant_interval
        self.rapid_replants[i] = 0.0
        self.settle_hold[i] = 0.0

    def _before_physics(self, i, d, t):
        if self.ball < 0:
            return
        if t >= self.next_ball[i]:
            self.fire_ball(d, i)
            self.next_ball[i] = t + self.rng.uniform(*self.ball_every)
        # Inbound: hold it level. Past that: let it fall.
        d.xfrc_applied[self.ball, 2] = self.ball_gravity_cancel if t < self.ball_flight[i] else 0.0

    def randomize_episode_phase(self):
        """The ball schedule moves with the clock, or every world whose offset had already passed
        its first shot would be fired at on the first step."""
        seconds = super().randomize_episode_phase()
        self.next_ball += seconds
        return seconds

    # ---------------------------------------------------------------- the projectile
    def _park_ball(self, d):
        if self.ball < 0:
            return
        d.qpos[self.ball_q:self.ball_q + 3] = (40.0, 40.0, 2.0)
        d.qpos[self.ball_q + 3:self.ball_q + 7] = (1.0, 0.0, 0.0, 0.0)
        d.qvel[self.ball_v:self.ball_v + 6] = 0.0
        d.xfrc_applied[self.ball, :3] = 0.0

    def fire_ball(self, d, i=0):
        """Launch at a chosen bone, from a random heading, in world `i`."""
        if self.ball < 0:
            return
        if self.max_shots_per_episode is not None and self.shots_fired[i] >= self.max_shots_per_episode:
            return
        self.shots_fired[i] += 1
        # Uniform unless training weighted the bones; see env_config.target_probabilities for why
        # the uniform draw must stay exactly this call.
        target = int(self.rng.choice(self.targets) if self.target_p is None
                     else self.rng.choice(self.targets, p=self.target_p))
        aim = np.array(d.xpos[target], dtype=float)
        theta = self.rng.uniform(0.0, 2.0 * np.pi)
        start = aim + np.array([np.cos(theta), np.sin(theta), 0.0]) * BALL_SPAWN_DISTANCE
        start[2] = max(start[2], 0.15)
        v = aim - start
        v = v / max(np.linalg.norm(v), 1e-9) * self.ball_speed
        d.qpos[self.ball_q:self.ball_q + 3] = start
        d.qpos[self.ball_q + 3:self.ball_q + 7] = (1.0, 0.0, 0.0, 0.0)
        d.qvel[self.ball_v:self.ball_v + 3] = v
        d.qvel[self.ball_v + 3:self.ball_v + 6] = 0.0
        d.xfrc_applied[self.ball, :3] = (0.0, 0.0, self.ball_gravity_cancel)
        # 20% past the nominal flight so a shot at a moving bone still arrives level.
        flight = BALL_SPAWN_DISTANCE / max(self.ball_speed, 1e-6) * 1.2
        self.ball_flight[i] = float(self.episode_length_buf[i]) * self.dt * self.decimation + flight

    # ---------------------------------------------------------------- reward
    def stance_of(self, d):
        """Strict rest-stance similarity for historical diagnostics.

        The feet in the pelvis's own heading frame: `width` is the lateral distance between them,
        positive on the side the rest pose puts the left foot and negative when crossed; `split` is
        how far one is ahead of the other; `leg_rms` the leg joints' error from the rest pose;
        `score` 1 at the rest stance.
        """
        r = d.xmat[self.pelvis].reshape(3, 3)
        heading = np.arctan2(r[1, 0], r[0, 0])
        rel = d.xpos[self.foot_l][:2] - d.xpos[self.foot_r][:2]
        fwd = np.cos(heading) * rel[0] + np.sin(heading) * rel[1]
        left = self.rest_lateral_sign * (-np.sin(heading) * rel[0] + np.cos(heading) * rel[1])
        leg_err = float(np.mean(np.square(d.qpos[self.leg_q] - self.rest_qpos[self.leg_q])))
        trunk_err = float(np.mean(np.square(d.qpos[self.trunk_q] - self.rest_qpos[self.trunk_q])))
        feet = np.exp(-((left - self.rest_stance) ** 2 + fwd ** 2) / STANCE_SIGMA ** 2)
        q = d.qpos[self.qadr]
        return {"width": float(left), "split": float(abs(fwd)), "leg_rms": float(np.sqrt(leg_err)),
                "trunk_rms": float(np.sqrt(trunk_err)),
                "at_limit": int(np.sum((q > self.soft_hi) | (q < self.soft_lo))),
                "score": float(feet * np.exp(-leg_err / LEG_POSE_SCALE)
                               * np.exp(-trunk_err / TRUNK_POSE_SCALE))}

    def past_soft_limits(self, d):
        """Radians past the soft joint limits, summed over the policy's joints."""
        q = d.qpos[self.qadr]
        return float(np.sum(np.maximum(q - self.soft_hi, 0.0) + np.maximum(self.soft_lo - q, 0.0)))

    def _after_physics(self, i, d):
        f = self.recovery_features(i)
        ground, elapsed, rapid = contact_history(
            np, f['grounded'], self.grounded[i], self.since_landing[i], self.dt * self.decimation)
        self.grounded[i], self.since_landing[i], self.rapid_replants[i] = ground, elapsed, rapid
        error, _ = support_state(np, f['com'], f['velocity'], f['feet'], f['grounded'])
        settled = settled_state(np, f['up'], f['pelvis_z'] / self.rest_pelvis_z,
                                f['velocity'], d.cvel[self.pelvis, :3], f['foot_velocity'],
                                f['grounded'], error)
        self.settle_hold[i] = self.settle_hold[i] + self.dt * self.decimation if settled else 0.0

    def _step_extras(self):
        now = self.episode_length_buf.cpu().numpy() * self.dt * self.decimation
        exposed = ((self.shots_fired > 0) & (now >= self.ball_flight + CONFIG.settle_seconds)
                   if self.ball >= 0 else True)
        return {'recovered': torch.tensor((self.settle_hold >= CONFIG.settle_seconds) & exposed,
                                          dtype=torch.bool, device=self.device)}

    def recovery_features(self, i):
        """Physical inputs also used by the independent recovery scorer."""
        d = self.datas[i]
        feet = d.xpos[[self.foot_l, self.foot_r]]
        load, slip_speed_sq = self.foot_contacts.read(d)
        grounded = loaded_feet(np, load, self.grounded[i])
        stance = self.stance_of(d)
        return dict(up=d.xmat[self.pelvis].reshape(3, 3)[2, 2],
                    pelvis_z=d.xpos[self.pelvis, 2], rest_pelvis_z=self.rest_pelvis_z,
                    com=self.com(d), velocity=self.com_velocity(d), feet=feet,
                    foot_velocity=np.stack([self._foot_velocity(d, self.foot_l),
                                            self._foot_velocity(d, self.foot_r)]),
                    grounded=grounded, slip_speed_sq=slip_speed_sq,
                    pose_error=np.mean((d.qpos[self.qadr] - self.rest_qpos[self.qadr]) ** 2),
                    width=stance['width'], split=stance['split'], rest_width=self.rest_stance,
                    limit_excess=self.past_soft_limits(d), rapid_replants=self.rapid_replants[i])

    def reward(self, i, action):
        return sum(reward_terms(np, **self.recovery_features(i), action=action,
                                previous_action=self.prev_action[i]).values())
