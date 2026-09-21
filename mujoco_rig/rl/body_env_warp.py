"""The dummy on the GPU, via mujoco_warp, with no task attached - the TRAINER's engine.

This is a second implementation of `body_env.py`, not a replacement for it. The CPU env stays,
because it is the engine Godot actually drives through P/Invoke and is therefore what every
checkpoint gets SCORED on. The rule this whole architecture rests on:

    Train on GPU. Score and ship on CPU MuJoCo. A checkpoint is never judged by the engine that
    trained it.

`mujoco_warp` is float32 where MuJoCo's C engine is float64, so trajectories do diverge - measured
by `parity_gpu.py`, with a CPU-vs-CPU control run proving the divergence is the engine rather than
chaos. That is tolerable only because scoring happens on the deployment engine.

**Two implementations of one task will drift apart unless something checks.** Rather than an
abstraction fighting a per-env NumPy loop on one side and a batched CUDA kernel on the other,
`test_env_parity.py` and `test_walk_parity.py` assert both produce identical observations and rewards
from identical states.

Everything stays on the GPU: `wp.to_torch` is zero-copy, so observations, rewards, resets and task
events are all torch ops on CUDA with no host transfer in the loop. Tasks subclass this exactly as
they subclass `BodyEnv`: `PerturbEnvWarp` adds the projectile, `WalkEnvWarp` the commands.
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

# Constraint buffers. **Two of these are PER-WORLD and one is AGGREGATE**, which is not guessable
# and is worth stating because both mistakes are silent:
#
#   njmax     per world - constraint rows. Sizing it as a total allocates a dense Jacobian of
#             (nworld, 30 * njmax) floats: njmax = num_envs * 128 asked for 15 GiB at 1,024 envs.
#   nconmax   per world - contacts.
#   naconmax  ACROSS ALL WORLDS. At 256 worlds the engine asked for 1,020 while a generous-looking
#             512 was set, so contacts in the batch were being dropped and the only symptom was a
#             stderr line printed from inside a kernel.
#
# Measured on the torque plant: 10 rollouts of 20 s under random torque peaked at nefc 117 and 27
# contacts, and the stepping walk gait at 72 rows in the worst of 1,024 worlds. These carry ~2x
# that. The first torque run was sized for the old plant and spent its whole 15 minutes printing
# `nefc overflow` before the policy went NaN. An overflow mid-run is now a SYMPTOM worth reading:
# it was an exploding world, not the gait, both times walk diverged on 2026-09-10.
NJMAX = 256                  # per world
NCONMAX = 64                 # per world
CONTACTS_PER_WORLD = 64      # multiplied by num_envs for the aggregate naconmax


class BodyEnvWarp(ABC):
    """Many MuJoCo worlds stepped together on the GPU. Mirrors `BodyEnv` exactly."""

    # No projectile; see BodyEnv.ball.
    ball = -1

    def __init__(self, num_envs=4096, device="cuda", episode_seconds=20.0, seed=0,
                 model="dummy.xml"):
        import mujoco_warp as mjw
        import warp as wp

        self._mjw, self._wp = mjw, wp
        self.device = device
        self.num_envs = num_envs
        self.auto_reset = True
        # `model` matches BodyEnv's attribute name deliberately. train.py is backend-agnostic and
        # reads env.model.body_mass to report the curriculum impulse; naming this `mjm` once crashed
        # a 55-minute run at its FIRST promotion, three minutes in. test_env_parity.py asserts the
        # interface, not just the numbers.
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / model))
        mjd = mujoco.MjData(self.model)
        # The `rest` keyframe, not qpos0: at qpos0 the arms hang INSIDE the legs (forearm 5.5 cm
        # into the thigh), which cost 331.8 N.m per shoulder to hold and started every episode in
        # contact. build_mjcf.add_rest_keyframe emits the cleared pose and verifies it.
        mujoco.mj_resetDataKeyframe(self.model, mjd, 0)
        mujoco.mj_forward(self.model, mjd)

        self.dt = self.model.opt.timestep
        self.decimation = 4                      # policy at ~60 Hz against a 240 Hz sim
        self.max_episode_length = int(episode_seconds / (self.dt * self.decimation))

        self._m = mjw.put_model(self.model)
        self._d = mjw.put_data(self.model, mjd, nworld=num_envs,
                               njmax=NJMAX, nconmax=NCONMAX,
                               naconmax=num_envs * CONTACTS_PER_WORLD)

        # Zero-copy views. Writing these writes the simulation state directly.
        self.qpos = wp.to_torch(self._d.qpos)
        self.qvel = wp.to_torch(self._d.qvel)
        self.ctrl = wp.to_torch(self._d.ctrl)
        self.xpos = wp.to_torch(self._d.xpos)
        self.xmat = wp.to_torch(self._d.xmat)
        self.xipos = wp.to_torch(self._d.xipos)
        self.cvel = wp.to_torch(self._d.cvel)
        self.subtree_com = wp.to_torch(self._d.subtree_com)
        self.xfrc = wp.to_torch(self._d.xfrc_applied)      # (nworld, nbody, 6)

        m = self.model
        name2id = mujoco.mj_name2id
        self.pelvis = name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Pelvis")
        self.foot_l = name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Foot_L")
        self.foot_r = name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Foot_R")

        # The actuators the POLICY drives - `m.nu` minus POLICY_EXCLUDE. Everything downstream
        # (observation width, action width, the contract Godot reads) is derived from this list, so
        # excluding a bone here removes it from all of them consistently.
        self.act_idx = np.array([i for i in range(m.nu)
                                 if m.actuator(i).name.split("_")[0] not in POLICY_EXCLUDE])
        self.act_names = [m.actuator(int(i)).name for i in self.act_idx]
        act_joint = [m.actuator_trnid[i, 0] for i in self.act_idx]
        dev = torch.device(device)
        self.ctrl_idx = torch.tensor(self.act_idx, dtype=torch.long, device=dev)
        self.qadr = torch.tensor([m.jnt_qposadr[j] for j in act_joint], device=dev)
        self.vadr = torch.tensor([m.jnt_dofadr[j] for j in act_joint], device=dev)
        self.lo = torch.tensor(m.jnt_range[act_joint, 0].copy(), dtype=torch.float32, device=dev)
        self.hi = torch.tensor(m.jnt_range[act_joint, 1].copy(), dtype=torch.float32, device=dev)
        # **The action is a TORQUE, not a target pose.** A position actuator's zero command means
        # "hold the rest pose" and holds it at kp 533 N.m/rad, which is why the body was a plank
        # with a brain and a rag without one. With `motor` actuators zero output is zero torque, so
        # an unpowered body falls exactly like dummy_limp.xml - measured, mean joint bend 24.3 deg
        # against the ragdoll's 23.3. See build_mjcf.ACTUATOR_MODE.
        self.torque_mode = bool(
            (m.actuator_gaintype[self.act_idx] == mujoco.mjtGain.mjGAIN_FIXED).all()
            and (m.actuator_biastype[self.act_idx] == mujoco.mjtBias.mjBIAS_NONE).all())
        self.force_limit = torch.tensor(m.actuator_forcerange[self.act_idx, 1].copy(),
                                        dtype=torch.float32, device=dev)
        self.authority = TORQUE_AUTHORITY if self.torque_mode else POSITION_AUTHORITY
        self.num_actions = len(self.act_idx)
        # **Every task shares ONE observation layout**, ending in three RESERVED command slots that
        # perturb leaves at zero and walk fills with (vx, vy, yaw). Any other choice makes
        # checkpoints non-interchangeable: a 120-wide perturb brain could not seed a 123-wide walk
        # one without surgery on its first layer.
        self.num_obs = 3 + 6 + 1 + 2 * self.num_actions + 2 + self.num_actions + 3

        mass = np.array(m.body_mass).copy()
        self.mass = torch.tensor(mass, dtype=torch.float32, device=dev).unsqueeze(-1)
        self.total_mass = float(mass.sum())
        # The DOFs that belong to the BODY - all of them, unless a task adds bodies of its own.
        self._body_nv = m.nv

        self.rest_qpos = torch.tensor(mjd.qpos.copy(), dtype=torch.float32, device=dev)
        # **Read the standing height from the MODEL.** It was hardcoded at 0.82 m, from a body that
        # no longer exists; this one rests at 0.917 m, so the height reward paid 0.685 of its
        # maximum for standing correctly and its gradient pointed 10 cm DOWNWARD - the policy was
        # being paid to crouch. The fall threshold was hardcoded the same way.
        self.rest_pelvis_z = float(mjd.xpos[self.pelvis][2])
        # Both feet rest a few centimetres up - the sole, not the body origin, is on the floor -
        # so "airborne" is measured against this rather than against zero.
        self.rest_foot_z = torch.tensor(
            [float(mjd.xpos[self.foot_l][2]), float(mjd.xpos[self.foot_r][2])],
            dtype=torch.float32, device=dev)
        self.rest_stance = abs(float(mjd.xpos[self.foot_l][1] - mjd.xpos[self.foot_r][1]))

        # Which side of the right foot the left one lies on, in the pelvis's own heading frame - READ
        # off the rest pose rather than assumed: through the frame map it is the negative side, and
        # the first version of the stance term assumed the opposite and read the rest stance as crossed.
        r0 = mjd.xmat[self.pelvis].reshape(3, 3)
        h0 = np.arctan2(r0[1, 0], r0[0, 0])
        rel0 = mjd.xpos[self.foot_l][:2] - mjd.xpos[self.foot_r][:2]
        self.rest_lateral_sign = float(np.sign(-np.sin(h0) * rel0[0] + np.cos(h0) * rel0[1]))
        self.gen = torch.Generator(device=dev).manual_seed(seed)
        self.prev_action = torch.zeros(num_envs, self.num_actions, device=dev)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=dev)
        # Reserved command slots. Perturb never writes them; walk does.
        self.commands = torch.zeros(num_envs, 3, device=dev)
        # Ring of pending actions, so what reaches the actuators is what was decided ~33 ms ago.
        self.action_queue = [torch.zeros(num_envs, self.num_actions, device=dev)
                             for _ in range(ACTION_LATENCY_STEPS)]
        self._extra_init()
        self.reset_all()

    # ---------------------------------------------------------------- task hooks
    def _extra_init(self):
        """Allocate a task's own state. Runs before the first reset - see BodyEnv._extra_init."""

    def _reset_worlds(self, idx):
        """A task's part of resetting the worlds `idx`: after the pose noise, before forward."""

    def _before_physics(self, t):
        """A task's part of each control step, before the physics runs. `t` is each world's time."""

    def _after_physics(self):
        """Update task history once per control step, before reading its reward."""

    def _step_extras(self):
        """Task metrics captured before automatic resets clear episode history."""
        return {}

    @abstractmethod
    def reward(self, action):
        """The task's reward for every world after this step's physics."""
        raise NotImplementedError

    # ---------------------------------------------------------------- helpers
    def _rand(self, *shape, lo=0.0, hi=1.0):
        return torch.rand(*shape, generator=self.gen, device=self.device) * (hi - lo) + lo

    def assert_buffers_ok(self):
        """Fail loudly if the constraint budget was exceeded.

        Overflow is reported only as a stderr line from inside a kernel and otherwise just drops
        constraints, so a run can look healthy while its contacts are wrong. Call this after a few
        hundred steps of a representative rollout.
        """
        # nefc is per world, so take the worst world; nacon is the batch total.
        nefc = int(self._d.nefc.numpy().max())
        nacon = int(self._d.nacon.numpy().sum())
        njmax, naconmax = self._d.njmax, self._d.naconmax
        if nefc >= njmax or nacon >= naconmax:
            raise RuntimeError(
                f"constraint buffer overflow: worst-world nefc {nefc}/{njmax}, "
                f"batch contacts {nacon}/{naconmax}. Raise NJMAX / CONTACTS_PER_WORLD "
                f"in body_env_warp.py.")
        return {"nefc": nefc, "njmax": njmax, "contacts": nacon, "naconmax": naconmax}

    # ---------------------------------------------------------------- body
    def com(self):
        return (self.mass * self.xipos).sum(1) / self.total_mass          # (n, 3)

    def com_velocity(self):
        """World velocity of the whole body's centre of mass, (n, 3).

        **Shifted, not raw cvel.** Each body's cvel is referenced to its subtree's centre of mass
        (see _foot_velocity), so summed raw it is not the COM velocity at all: measured on the
        shipped perturb brain under fire, raw was off by a median 0.32 m/s whenever the body moved
        faster than 0.2 m/s - as large as the signal - where the shifted sum matches a finite
        difference to 0.017 m/s.
        """
        if not hasattr(self, "_rootid"):
            self._rootid = torch.tensor([int(r) for r in self.model.body_rootid],
                                        dtype=torch.long, device=self.xipos.device)
        v = self.cvel
        arm = self.xipos - self.subtree_com[:, self._rootid]
        lin = v[:, :, 3:6] + torch.cross(v[:, :, :3], arm, dim=-1)
        return (self.mass * lin).sum(1) / self.total_mass

    def _foot_velocity(self, body):
        """World linear velocity of a body's ORIGIN.

        **Not `cvel[:, body, 3:6]`.** MuJoCo references cvel to the subtree centre of mass, so read
        raw it is the velocity of the point on the foot that sits at the body's COM: -0.14 m/s for a
        foot actually moving at +0.70, the wrong sign. Shifted to the foot's own origin it matches a
        finite difference of the foot's position to 1e-5.
        """
        v = self.cvel[:, body]
        arm = self.xpos[:, body] - self.subtree_com[:, int(self.model.body_rootid[body])]
        return v[:, 3:6] + torch.cross(v[:, :3], arm, dim=-1)

    # ---------------------------------------------------------------- api
    def reset_idx(self, idx):
        if idx.numel() == 0:
            return
        self.qpos[idx] = self.rest_qpos
        # small pose noise so the policy cannot memorise one trajectory
        noise = self._rand(idx.numel(), self.num_actions, lo=-0.03, hi=0.03)
        self.qpos[idx.unsqueeze(-1), self.qadr.unsqueeze(0)] += noise
        self.qvel[idx] = 0.0
        self._reset_worlds(idx)
        for pending in self.action_queue:
            pending[idx] = 0.0
        self.prev_action[idx] = 0.0
        self.episode_length_buf[idx] = 0
        # Derived quantities (xpos, xmat, cvel) are stale until this runs, and get_observations()
        # is called immediately afterwards - a reset world would otherwise report the pose it had
        # before it was reset.
        self._mjw.forward(self._m, self._d)

    def randomize_episode_phase(self):
        """Put every world at a random point of its episode. Call once, after `reset_all`.

        **Worlds that reset together time out together.** Every survivor reaches the episode limit
        in the same iteration (1199 steps at 16 per iteration = 75) and floods the trainer's episode
        statistics with maximal episodes, so a gate reading the most recent ones sees only
        survivors, however many fell. All four perturb sessions on 2026-09-10 promoted at exactly
        iterations 75 and 150, one of them from an episode length of 736. This is legged_gym's
        `init_at_random_ep_len`.

        Returns each world's offset in seconds, so a task can move its own schedule with the clock.
        """
        offset = torch.randint(0, self.max_episode_length, (self.num_envs,),
                               generator=self.gen, device=self.device)
        self.episode_length_buf[:] = offset
        return offset.float() * (self.dt * self.decimation)

    def reset_all(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.get_observations()

    def get_observations(self):
        r = self.xmat[:, self.pelvis]                                     # (n, 3, 3)
        down = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        grav = torch.einsum("nji,j->ni", r, down)                         # R.T @ down
        lin = torch.einsum("nji,nj->ni", r, self.cvel[:, self.pelvis, 3:6])
        ang = torch.einsum("nji,nj->ni", r, self.cvel[:, self.pelvis, :3])
        q = self.qpos[:, self.qadr]
        qd = self.qvel[:, self.vadr]
        contacts = torch.stack([(self.xpos[:, self.foot_l, 2] < 0.05).float(),
                                (self.xpos[:, self.foot_r, 2] < 0.05).float()], dim=-1)
        height = self.xpos[:, self.pelvis, 2:3]
        obs = torch.cat([grav, lin, ang, height, q, qd * 0.1, contacts, self.prev_action,
                         self.commands], dim=-1)
        # **Sanitise.** MuJoCo can diverge under an aggressive policy, and ONE non-finite value in
        # one of 8,192 worlds poisons the entire batch through the network - the actor's mean goes
        # NaN and PPO dies with "Expected parameter loc ... to satisfy the constraint Real()".
        # Clamping as well as replacing matters: a merely enormous value trains just as badly.
        return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).clamp(-100.0, 100.0)

    def step(self, actions):
        a = torch.clamp(actions.to(self.device), -1.0, 1.0)
        # Authority is deliberately limited; see body_env.py and probe_noise.py for the measurement
        # that fixes it, which is a property of the plant and not of the backend. The policy's
        # decision joins the queue; the actuators get the oldest one.
        self.action_queue.append(a)
        applied = self.action_queue.pop(0)
        if self.torque_mode:
            span = self.force_limit                      # newton-metres
        else:
            span = torch.where(applied >= 0.0, self.hi, -self.lo)
        # Only the policy's own actuators. The excluded ones keep their zero command - for the
        # neck's position actuator that is "hold the head straight", its muscle-tone layer.
        self.ctrl[:, self.ctrl_idx] = applied * span * self.authority

        self._before_physics(self.episode_length_buf.float() * (self.dt * self.decimation))
        for _ in range(self.decimation):
            self._mjw.step(self._m, self._d)

        self._after_physics()
        rewards = self.reward(a)
        self.prev_action = a
        self.episode_length_buf += 1

        pelvis_z = self.xpos[:, self.pelvis, 2]
        # A world whose state has gone non-finite is unrecoverable and must be retired, not left to
        # contaminate the batch. Treated as a fall so it carries the same penalty.
        #
        # **And its reward is REPLACED, not decremented.** This used to retire the world and then
        # apply `rewards - 10`, and NaN - 10 is NaN: the world was reset while its poisoned reward
        # still went into the batch. Nor did it catch a state that is huge but still finite - one
        # injected blow-up gave a walk reward of -3.6e15 with qpos and qvel finite, and walk training
        # diverged to NaN twice on 2026-09-10 exactly this way. The body's speed is the early
        # signal; a task's own bodies (the projectile) are excluded through `_body_nv`.
        diverged = (~(torch.isfinite(self.qpos).all(dim=-1) & torch.isfinite(self.qvel).all(dim=-1))
                    | ~torch.isfinite(rewards)
                    | (self.qvel[:, :self._body_nv].abs().amax(dim=-1) > QVEL_CEILING))
        fell = (pelvis_z < FALL_FRACTION * self.rest_pelvis_z) | diverged
        timeout = self.episode_length_buf >= self.max_episode_length
        dones = fell | timeout
        rewards = torch.where(fell, rewards - FALL_PENALTY, rewards)
        rewards = torch.where(diverged, torch.full_like(rewards, -FALL_PENALTY), rewards)

        extras = self._step_extras()
        terminal_observation = self.get_observations()
        idx = torch.nonzero(dones).flatten()
        if self.auto_reset and idx.numel():
            self.reset_idx(idx)

        obs = self.get_observations() if self.auto_reset and idx.numel() else terminal_observation
        return obs, rewards, dones, {"time_outs": timeout & ~fell,
                                    "terminal_observation": terminal_observation, **extras}
