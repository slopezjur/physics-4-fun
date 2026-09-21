"""Stay standing through ball impacts, on the GPU via mujoco_warp. Same task, ~24x the experience.

The body, observation, reset, step and divergence guard are `BodyEnvWarp`'s - see body_env_warp.py
for why the GPU twin exists and how `test_env_parity.py` keeps the two backends honest. This adds the
projectile and the reward, mirroring `perturb_env.py` exactly.
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch

from body_env_warp import BodyEnvWarp
from env_config import (BALL_SPAWN_DISTANCE,
                        JOINT_SOFT_LIMIT, LEG_BONES,
                        TARGET_BONES, TRUNK_BONES, target_probabilities)
from foot_contacts import loaded_feet
from foot_contacts_warp import FootContactsWarp
from observation_contract import LEGACY, FOUNDATION, observation_size
from recovery_reward import (CONFIG, REWARD_VERSION, contact_history, reward_terms,
                             settled_state, support_state)



class PerturbEnvWarp(BodyEnvWarp):
    """Many MuJoCo dummies under fire, stepped together on the GPU. Mirrors `PerturbEnv` exactly."""

    def __init__(self, num_envs=4096, device="cuda", episode_seconds=20.0,
                 ball_every=(4.0, 7.0), ball_speed=6.0, seed=0, model="dummy_ball.xml",
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
        dev = torch.device(self.device)
        name2id = mujoco.mj_name2id
        self.ball = name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
        if self.ball >= 0:
            mass = np.array(m.body_mass).copy()
            mass[self.ball] = 0.0                # never let the projectile into the body's COM
            self.mass = torch.tensor(mass, dtype=torch.float32, device=dev).unsqueeze(-1)
            self.total_mass = float(mass.sum())
            ball_joint = name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
            self.ball_q = int(m.jnt_qposadr[ball_joint])
            self.ball_v = int(m.jnt_dofadr[ball_joint])
            # The projectile's free joint is the model's last: its spin is not the body's.
            self._body_nv = self.ball_v
        # See perturb_env.py: the projectile's gravity is cancelled so a shot flies straight. Below
        # 4.43 m/s a ballistic ball cannot even reach a target 2 m away, so the whole curriculum was
        # firing at the floor.
        self.ball_gravity_cancel = (float(m.body_mass[self.ball]) * abs(float(m.opt.gravity[2]))
                                    if self.ball >= 0 else 0.0)
        # Only for the approach - see perturb_env.ball_flight. A shot that misses has to land.
        self.ball_flight = torch.zeros(self.num_envs, device=dev)
        self.targets = torch.tensor(
            [name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in TARGET_BONES], device=dev)
        if self.target_p is not None:
            self.target_p = torch.tensor(self.target_p, dtype=torch.float32, device=dev)
        # The joints the rest stance pulls back: the legs and the trunk. The arms stay free to swing.
        legs = [i for i, n in enumerate(self.act_names) if n.split("_")[0] in LEG_BONES]
        self.leg_q = self.qadr[torch.tensor(legs, device=dev)]
        trunk = [i for i, n in enumerate(self.act_names) if n.split("_")[0] in TRUNK_BONES]
        self.trunk_q = self.qadr[torch.tensor(trunk, device=dev)]
        # Soft joint limits: JOINT_SOFT_LIMIT of each half-range from the middle of the range,
        # stretched to include the rest pose - a straight knee or elbow rests on its own stop.
        rng = np.asarray(m.jnt_range)[[int(m.actuator_trnid[int(i), 0]) for i in self.act_idx]]
        mid, half = rng.mean(axis=1), 0.5 * (rng[:, 1] - rng[:, 0]) * JOINT_SOFT_LIMIT
        rest = self.rest_qpos[self.qadr].cpu().numpy()
        self.soft_lo = torch.tensor(np.minimum(mid - half, rest), dtype=torch.float32, device=dev)
        self.soft_hi = torch.tensor(np.maximum(mid + half, rest), dtype=torch.float32, device=dev)
        # Allocated once rather than rebuilt per reset.
        self._park_pos = torch.tensor([40.0, 40.0, 2.0], device=dev).expand(self.num_envs, 3)
        self._identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0],
                                           device=dev).expand(self.num_envs, 4)
        self.next_ball = torch.zeros(self.num_envs, device=dev)
        self.shots_fired = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self.grounded = torch.ones((self.num_envs, 2), dtype=torch.bool, device=dev)
        self.foot_contacts = FootContactsWarp(m, self._m, self._d, self.num_envs, self.device)
        self.since_landing = torch.full((self.num_envs, 2), CONFIG.replant_interval, device=dev)
        self.rapid_replants = torch.zeros(self.num_envs, device=dev)
        self.settle_hold = torch.zeros(self.num_envs, device=dev)

    def get_observations(self):
        legacy = super().get_observations()
        if self.observation_version == LEGACY:
            return legacy
        rotation = self.xmat[:, self.pelvis].reshape(-1, 3, 3)
        velocity = torch.bmm(rotation.transpose(1, 2), self._foot_velocity(self.pelvis).unsqueeze(-1)).squeeze(-1)
        load, _ = self.foot_contacts.read()
        obs = torch.cat((legacy, velocity, load * 0.001, self.action_queue[0]), dim=-1)
        return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).clamp(-100, 100)

    def _reset_worlds(self, idx):
        self.shots_fired[idx] = 0
        self.grounded[idx] = True
        self.since_landing[idx] = CONFIG.replant_interval
        self.rapid_replants[idx] = 0.0
        self.settle_hold[idx] = 0.0
        self._park_ball(idx)
        self.next_ball[idx] = self._rand(idx.numel(), lo=self.ball_every[0], hi=self.ball_every[1])

    def _before_physics(self, t):
        if self.ball >= 0:
            self.xfrc[:, self.ball, 2] = torch.where(t < self.ball_flight,
                                                     self.ball_gravity_cancel, 0.0)
        due = torch.nonzero(t >= self.next_ball).flatten()
        if due.numel():
            self.fire_ball(due)
            self.next_ball[due] = t[due] + self._rand(
                due.numel(), lo=self.ball_every[0], hi=self.ball_every[1])

    def randomize_episode_phase(self):
        """The ball schedule moves with the clock, or every world whose offset had already passed
        its first shot would be fired at on the first step."""
        seconds = super().randomize_episode_phase()
        self.next_ball += seconds
        return seconds

    # ---------------------------------------------------------------- the projectile
    def _park_ball(self, idx):
        if self.ball < 0:
            return
        q, n = self.ball_q, idx.numel()
        # Sliced to the reset count: these are expanded to num_envs, and a PARTIAL reset indexes
        # fewer rows than that.
        self.qpos[idx, q:q + 3] = self._park_pos[:n]
        self.qpos[idx, q + 3:q + 7] = self._identity_quat[:n]
        self.qvel[idx, self.ball_v:self.ball_v + 6] = 0.0
        self.xfrc[idx, self.ball, 2] = 0.0
        self.ball_flight[idx] = 0.0

    def fire_ball(self, idx):
        """Launch at a chosen bone (uniform unless weighted), from a random heading, for the given worlds.

        **Indexed, not masked, and that was measured.** The obvious optimisation is to compute the
        launch full-width and merge with `torch.where`, which removes the `idx.numel()` host sync.
        It was tried and it is 9-16% SLOWER across every env count, with GPU utilisation unchanged
        (75% vs 79% at 8,192) - so the sync was never stalling anything, and computing launch
        geometry for sixteen thousand worlds to use a handful of them simply costs more.
        """
        if self.ball < 0 or idx.numel() == 0:
            return
        if self.max_shots_per_episode is not None:
            idx = idx[self.shots_fired[idx] < self.max_shots_per_episode]
            if not idx.numel():
                return
        self.shots_fired[idx] += 1
        n = idx.numel()
        pick = (torch.randint(len(self.targets), (n,), generator=self.gen, device=self.device)
                if self.target_p is None else
                torch.multinomial(self.target_p, n, replacement=True, generator=self.gen))
        bodies = self.targets[pick]
        aim = self.xpos[idx, bodies]                                 # (n, 3)
        theta = self._rand(n, lo=0.0, hi=2.0 * float(np.pi))
        offset = torch.stack([torch.cos(theta), torch.sin(theta),
                              torch.zeros_like(theta)], dim=-1) * BALL_SPAWN_DISTANCE
        start = aim + offset
        start[:, 2] = torch.clamp(start[:, 2], min=0.15)
        v = aim - start
        v = v / torch.clamp(v.norm(dim=-1, keepdim=True), min=1e-9) * self.ball_speed
        q, dv = self.ball_q, self.ball_v
        self.qpos[idx, q:q + 3] = start
        self.qpos[idx, q + 3:q + 7] = self._identity_quat[:n]
        self.qvel[idx, dv:dv + 3] = v
        self.qvel[idx, dv + 3:dv + 6] = 0.0
        self.xfrc[idx, self.ball, 2] = self.ball_gravity_cancel
        now = self.episode_length_buf[idx].float() * (self.dt * self.decimation)
        self.ball_flight[idx] = now + BALL_SPAWN_DISTANCE / max(self.ball_speed, 1e-6) * 1.2

    # ---------------------------------------------------------------- reward
    def _after_physics(self):
        f = self.recovery_features()
        self.grounded, self.since_landing, self.rapid_replants = contact_history(
            torch, f['grounded'], self.grounded, self.since_landing, self.dt * self.decimation)
        error, _ = support_state(torch, f['com'], f['velocity'], f['feet'], f['grounded'])
        settled = settled_state(torch, f['up'], f['pelvis_z'] / self.rest_pelvis_z,
                                f['velocity'], self.cvel[:, self.pelvis, :3], f['foot_velocity'],
                                f['grounded'], error)
        self.settle_hold = torch.where(settled, self.settle_hold + self.dt * self.decimation, 0.0)

    def _step_extras(self):
        now = self.episode_length_buf * self.dt * self.decimation
        exposed = ((self.shots_fired > 0) & (now >= self.ball_flight + CONFIG.settle_seconds)
                   if self.ball >= 0 else True)
        return {'recovered': (self.settle_hold >= CONFIG.settle_seconds) & exposed}

    def recovery_features(self):
        r = self.xmat[:, self.pelvis]
        heading = torch.atan2(r[:, 1, 0], r[:, 0, 0])
        ch, sh = torch.cos(heading), torch.sin(heading)
        rel = self.xpos[:, self.foot_l, :2] - self.xpos[:, self.foot_r, :2]
        split = ch * rel[:, 0] + sh * rel[:, 1]
        width = self.rest_lateral_sign * (-sh * rel[:, 0] + ch * rel[:, 1])
        feet = self.xpos[:, [self.foot_l, self.foot_r]]
        load, slip_speed_sq = self.foot_contacts.read()
        grounded = loaded_feet(torch, load, self.grounded)
        q = self.qpos[:, self.qadr]
        over = ((q - self.soft_hi).clamp_min(0.0) + (self.soft_lo - q).clamp_min(0.0)).sum(-1)
        return dict(up=r[:, 2, 2], pelvis_z=self.xpos[:, self.pelvis, 2],
            rest_pelvis_z=self.rest_pelvis_z, com=self.com(), velocity=self.com_velocity(),
            feet=feet, foot_velocity=torch.stack([self._foot_velocity(self.foot_l),
                                                  self._foot_velocity(self.foot_r)], dim=1),
            grounded=grounded, slip_speed_sq=slip_speed_sq,
            pose_error=(q - self.rest_qpos[self.qadr]).square().mean(-1),
            width=width, split=split, rest_width=self.rest_stance, limit_excess=over,
            rapid_replants=self.rapid_replants)

    def reward(self, action):
        return sum(reward_terms(torch, **self.recovery_features(), action=action,
                                previous_action=self.prev_action).values())
