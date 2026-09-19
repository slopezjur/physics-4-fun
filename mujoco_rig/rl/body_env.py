"""The dummy on MuJoCo's C engine with no task attached - the SCORER's engine.

Everything a task shares lives here: the model and one `MjData` per world, the policy's actuators
and how an action becomes a control, the observation, the `rest`-keyframe reset, the step with its
action latency, and the divergence guard. A task subclass adds only what makes it that task - its
own state, its reward, and whatever it does to the world around the physics step:

    PerturbEnv(BodyEnv)   a projectile fired on a schedule, and a reward for taking the hit
    WalkEnv(BodyEnv)      commands with a heading hold, and a reward for tracking them

**Why a base, rather than walk subclassing perturb as it once did.** Walk inherited the whole gun
and switched it off with `ball_every=(1e9, 1e9)`: a walk env is not a kind of perturb env, and the
inert gun still drew from the random stream on every reset.

Training runs on the GPU twin, `body_env_warp.py`; every checkpoint is scored here, on the float64
engine Godot drives through P/Invoke. `test_env_parity.py` and `test_walk_parity.py` assert the two
backends define the same tasks.
"""
from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod

import mujoco
import numpy as np
import torch

from env_config import (ACTION_LATENCY_STEPS, FALL_FRACTION, FALL_PENALTY, POLICY_EXCLUDE,
                        POSITION_AUTHORITY, QVEL_CEILING, TORQUE_AUTHORITY)

ROOT = pathlib.Path(__file__).resolve().parent.parent


class BodyEnv(ABC):
    """Many independent MuJoCo dummies, stepped together. A task supplies `reward`."""

    # No projectile. A task that adds one sets this to the ball's body id; the trainer reads it to
    # decide whether there is an impact curriculum at all.
    ball = -1

    def __init__(self, num_envs=64, device="cpu", episode_seconds=12.0, seed=0, model="dummy.xml"):
        self.device = device
        self.num_envs = num_envs
        self.auto_reset = True
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / model))
        self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]
        self.rng = np.random.default_rng(seed)
        self.dt = self.model.opt.timestep
        self.decimation = 4                      # policy at ~60 Hz against a 240 Hz sim
        self.max_episode_length = int(episode_seconds / (self.dt * self.decimation))
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)

        m = self.model
        name2id = mujoco.mj_name2id
        self.pelvis = name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Pelvis")
        self.foot_l = name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Foot_L")
        self.foot_r = name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Foot_R")

        # The actuators the POLICY drives, and where each sits in qpos / qvel. `m.nu` is larger:
        # the neck is actuated but excluded, see env_config.POLICY_EXCLUDE.
        self.act_idx = np.array([i for i in range(m.nu)
                                 if m.actuator(i).name.split("_")[0] not in POLICY_EXCLUDE])
        self.act_names = [m.actuator(int(i)).name for i in self.act_idx]
        self.act_joint = [m.actuator_trnid[i, 0] for i in self.act_idx]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.act_joint])
        self.vadr = np.array([m.jnt_dofadr[j] for j in self.act_joint])
        self.lo = m.jnt_range[self.act_joint, 0].copy()
        self.hi = m.jnt_range[self.act_joint, 1].copy()
        # The action is a TORQUE when the model carries `motor` actuators - see
        # build_mjcf.ACTUATOR_MODE.
        self.torque_mode = bool(
            (m.actuator_gaintype[self.act_idx] == mujoco.mjtGain.mjGAIN_FIXED).all()
            and (m.actuator_biastype[self.act_idx] == mujoco.mjtBias.mjBIAS_NONE).all())
        self.force_limit = m.actuator_forcerange[self.act_idx, 1].copy()
        self.num_actions = len(self.act_idx)

        # projected gravity (3) + pelvis lin/ang vel (6) + height (1) + q + qd + contacts (2)
        # + previous action + 3 command slots. ONE layout for every task: perturb leaves the command
        # slots at zero and walk fills them, which is what lets a perturb brain seed a walk one.
        self.num_obs = 3 + 6 + 1 + 2 * self.num_actions + 2 + self.num_actions + 3
        self.mass = np.array(m.body_mass).copy()
        self.total_mass = self.mass.sum()
        # The DOFs that belong to the BODY - all of them, unless a task adds bodies of its own.
        self._body_nv = m.nv

        # How much of the actuator the policy may use - a fraction of the joint's range in
        # position mode, of its peak torque in torque mode. Exposed so the tolerated exploration
        # noise can be measured against it rather than assumed.
        self.authority = TORQUE_AUTHORITY if self.torque_mode else POSITION_AUTHORITY
        self.prev_action = np.zeros((num_envs, self.num_actions))
        self.commands = np.zeros((num_envs, 3))     # reserved slots; walk fills them
        # ~33 ms of neuromuscular delay; see env_config.ACTION_LATENCY_STEPS.
        self.action_queue = [np.zeros((num_envs, self.num_actions))
                             for _ in range(ACTION_LATENCY_STEPS)]
        rest = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, rest, 0)
        mujoco.mj_forward(m, rest)
        self.rest_qpos = rest.qpos.copy()
        # Standing height read from the model rather than hardcoded; see body_env_warp.py for what
        # the stale constant was costing.
        self.rest_pelvis_z = float(rest.xpos[self.pelvis][2])
        # "Airborne" is relative to where a foot RESTS; the sole is on the floor but the body
        # origin is a few centimetres up.
        self.rest_foot_z = np.array([rest.xpos[self.foot_l][2], rest.xpos[self.foot_r][2]])
        self.rest_stance = abs(float(rest.xpos[self.foot_l][1] - rest.xpos[self.foot_r][1]))

        # Which side of the right foot the left one lies on, in the pelvis's own heading frame - READ
        # off the rest pose rather than assumed: through the frame map it is the negative side, and
        # the first version of the stance term assumed the opposite and read the rest stance as crossed.
        r0 = rest.xmat[self.pelvis].reshape(3, 3)
        h0 = np.arctan2(r0[1, 0], r0[0, 0])
        rel0 = rest.xpos[self.foot_l][:2] - rest.xpos[self.foot_r][:2]
        self.rest_lateral_sign = float(np.sign(-np.sin(h0) * rel0[0] + np.cos(h0) * rel0[1]))
        self._extra_init()
        self.reset_all()

    # ---------------------------------------------------------------- task hooks
    def _extra_init(self):
        """Allocate a task's own state. Runs before the first reset.

        `__init__` ends with `reset_all()`, which runs the reset hook and `get_observations()`, so a
        task's state must exist by then and cannot be created after `super().__init__()` returns.
        """

    def _reset_world(self, i, d):
        """A task's part of resetting world `i`: after the pose noise, before `mj_forward`."""

    def _before_physics(self, i, d, t):
        """A task's part of each control step for world `i`, before the physics runs."""

    @abstractmethod
    def reward(self, i, action):
        """The task's reward for world `i` after this step's physics."""
        raise NotImplementedError

    # ---------------------------------------------------------------- body
    def com(self, d):
        return (self.mass[:, None] * d.xipos).sum(0) / self.total_mass

    def _foot_velocity(self, d, body):
        """World linear velocity of a body's ORIGIN - see BodyEnvWarp._foot_velocity for why raw
        cvel is the wrong quantity."""
        v = d.cvel[body]
        return v[3:6] + np.cross(v[:3], d.xpos[body] - d.subtree_com[self.model.body_rootid[body]])

    def com_velocity(self, d):
        """World velocity of the whole body's centre of mass - see BodyEnvWarp.com_velocity for
        why raw cvel is the wrong quantity."""
        arm = d.xipos - d.subtree_com[self.model.body_rootid]
        lin = d.cvel[:, 3:6] + np.cross(d.cvel[:, :3], arm)
        return (self.mass[:, None] * lin).sum(0) / self.total_mass

    def _obs_one(self, i):
        d = self.datas[i]
        R = d.xmat[self.pelvis].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        lin = R.T @ d.cvel[self.pelvis][3:6]
        ang = R.T @ d.cvel[self.pelvis][:3]
        q = d.qpos[self.qadr]
        qd = d.qvel[self.vadr]
        contacts = np.array([float(d.xpos[self.foot_l][2] < 0.05),
                             float(d.xpos[self.foot_r][2] < 0.05)])
        obs = np.concatenate([grav, lin, ang, [d.xpos[self.pelvis][2]],
                              q, qd * 0.1, contacts, self.prev_action[i], self.commands[i]])
        # Matches the GPU env exactly; on sane values this is the identity, so parity is unaffected.
        return np.clip(np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)

    # ---------------------------------------------------------------- api
    def reset_idx(self, idx):
        for i in idx:
            d = self.datas[i]
            # The `rest` keyframe, not qpos0 - at qpos0 the arms hang inside the legs.
            mujoco.mj_resetDataKeyframe(self.model, d, 0)
            # small pose noise so the policy cannot memorise one trajectory
            d.qpos[self.qadr] += self.rng.uniform(-0.03, 0.03, size=self.num_actions)
            self._reset_world(i, d)
            mujoco.mj_forward(self.model, d)
            for pending in self.action_queue:
                pending[i] = 0.0
            self.prev_action[i] = 0.0
            self.episode_length_buf[i] = 0

    def reset_all(self):
        self.reset_idx(range(self.num_envs))
        return self.get_observations()

    def randomize_episode_phase(self):
        """Put every world at a random point of its episode - see BodyEnvWarp.

        Returns each world's offset in seconds, so a task can move its own schedule with the clock.
        """
        offset = self.rng.integers(0, self.max_episode_length, size=self.num_envs)
        self.episode_length_buf[:] = torch.as_tensor(offset, device=self.episode_length_buf.device)
        return offset * (self.dt * self.decimation)

    def get_observations(self):
        obs = np.stack([self._obs_one(i) for i in range(self.num_envs)])
        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    def step(self, actions):
        a = actions.detach().cpu().numpy()
        a = np.clip(a, -1.0, 1.0)
        # **Authority is deliberately limited.** An untrained Gaussian policy commands every joint
        # to a random extreme simultaneously, and at full authority that drove MuJoCo to "Nan, Inf
        # or huge value in QACC" before any learning happened.
        self.action_queue.append(a)
        applied = self.action_queue.pop(0)
        span = (self.force_limit if self.torque_mode
                else np.where(applied >= 0.0, self.hi, -self.lo))
        target = applied * span * self.authority

        rewards = np.zeros(self.num_envs)
        for i in range(self.num_envs):
            d = self.datas[i]
            # Only the policy's actuators; the excluded ones keep their zero target, which for a
            # position actuator is "hold the rest pose" - the neck's muscle layer.
            d.ctrl[self.act_idx] = target[i]
            self._before_physics(i, d, float(self.episode_length_buf[i]) * self.dt * self.decimation)
            for _ in range(self.decimation):
                mujoco.mj_step(self.model, d)
            rewards[i] = self.reward(i, a[i])

        self.prev_action = a
        self.episode_length_buf += 1

        pelvis_z = np.array([d.xpos[self.pelvis][2] for d in self.datas])
        # Same retirement as BodyEnvWarp.step: a non-finite or runaway world ends as a fall and its
        # reward is REPLACED by the penalty, never passed on.
        diverged = np.array([not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
                             or not np.isfinite(rewards[i])
                             or np.abs(d.qvel[:self._body_nv]).max() > QVEL_CEILING
                             for i, d in enumerate(self.datas)])
        fell = (pelvis_z < FALL_FRACTION * self.rest_pelvis_z) | diverged
        timeout = self.episode_length_buf.cpu().numpy() >= self.max_episode_length
        dones = fell | timeout
        rewards[fell] -= FALL_PENALTY
        rewards[diverged] = -FALL_PENALTY

        idx = np.nonzero(dones)[0]
        if self.auto_reset and len(idx):
            self.reset_idx(idx)

        return (self.get_observations(),
                torch.tensor(rewards, dtype=torch.float32, device=self.device),
                torch.tensor(dones, dtype=torch.bool, device=self.device),
                {"time_outs": torch.tensor(timeout, dtype=torch.bool, device=self.device)})
