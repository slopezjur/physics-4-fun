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
from env_config import (BALL_SPAWN_DISTANCE, CAPTURE_PLACE_SIGMA, CAPTURE_V, FOOT_CLEAR,
                        JOINT_SOFT_LIMIT, LEG_BONES, LEG_POSE_SCALE, OFF_BALANCE, STANCE_SIGMA,
                        TARGET_BONES, TRUNK_BONES, TRUNK_POSE_SCALE, target_probabilities)


class PerturbEnvWarp(BodyEnvWarp):
    """Many MuJoCo dummies under fire, stepped together on the GPU. Mirrors `PerturbEnv` exactly."""

    def __init__(self, num_envs=4096, device="cuda", episode_seconds=20.0,
                 ball_every=(4.0, 7.0), ball_speed=6.0, seed=0, model="dummy_ball.xml",
                 target_weights=None):
        # Set before the base constructor runs: its first reset already schedules a shot.
        self.ball_every = ball_every
        self.max_shots_per_episode = None
        self.ball_speed = ball_speed
        self.target_p = target_probabilities(target_weights)
        super().__init__(num_envs=num_envs, device=device, episode_seconds=episode_seconds,
                         seed=seed, model=model)

    # ---------------------------------------------------------------- task hooks
    def _extra_init(self):
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

    def _reset_worlds(self, idx):
        self.shots_fired[idx] = 0
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
    def reward(self, action):
        """Take the hit, keep the feet under the body, and go back to standing.

        **Rewritten after a night of measurement.** The previous version paid 2.0 upright + 1.0
        height + 0.2 still + 3.0 support + a flat 0.5, and produced a body that braces: stance
        widened by 13 cm under fire, feet shuffled 7-12 cm, and single support measured 0.0%. Every
        one of those terms is satisfiable without ever picking a foot up, so "recover" was never
        actually the cheapest thing to do. Four changes, each aimed at one of them:

          * `pose` is what "get back to the original stance" means numerically. Nothing previously
            distinguished standing upright in the authored pose from standing upright in whatever
            tangle the last impact left.
          * `recover_step` pays for the swing foot moving TOWARD the escaping COM, while the COM is
            outside the feet. A step is only a recovery when the body needs one; paying for it
            unconditionally buys a flamingo, and paying for the lift alone bought a stomp.
          * `still` is gated on being balanced. Paying for a motionless centre of mass while the
            body is toppling is a headwind on the one behaviour the task is named after.
          * the flat survival bonus is 0.2, not 0.5. It pays the body for existing, which is what a
            statue does best; the walk reward removed its own for the same reason.
        """
        r = self.xmat[:, self.pelvis]
        up_z = r[:, 2, 2]                                    # (R @ [0,0,1])[2]
        com = self.com()
        com_v = self.com_velocity()
        feet_mid = 0.5 * (self.xpos[:, self.foot_l, :2] + self.xpos[:, self.foot_r, :2])
        off = (com[:, :2] - feet_mid).norm(dim=-1)

        upright = torch.clamp(up_z, 0.0, 1.0)
        height = torch.exp(-40.0 * (self.xpos[:, self.pelvis, 2] - self.rest_pelvis_z) ** 2)
        support = torch.exp(-12.0 * off)
        balanced = (off < OFF_BALANCE).float()
        still = torch.exp(-2.0 * com_v.norm(dim=-1)) * balanced

        # Back to the authored stance, joint by joint.
        q_err = (self.qpos[:, self.qadr] - self.rest_qpos[self.qadr]).square().mean(dim=-1)
        pose = torch.exp(-2.0 * q_err)

        # **Stance width.** Measured on the shipped brain under fire, the feet splay to 1.02 m
        # against a 0.30 m rest stance - a 72 cm brace. Nothing in the reward objected: `support`
        # measures the COM against the MIDPOINT between the feet, and widening the base moves the
        # midpoint hardly at all while making the body far harder to topple. It is the cheapest
        # way to survive a hit and it looks nothing like a person.
        #
        # **Gated on being balanced, and that gate is not optional.** Ungated, this term penalises
        # the one behaviour the task exists to produce: a protective step MOVES a foot, so it
        # necessarily changes the separation, and paying only for the rest stance makes stepping
        # cost reward. Measured - one session with it ungated took `steps taken` from 0.62 to 0.34.
        #
        # **In the pelvis's own heading frame, signed, and the legs too** (2026-09-11): measured in
        # world axes with abs(), crossed feet read as a normal width - see env_config.STANCE_SIGMA.
        heading = torch.atan2(r[:, 1, 0], r[:, 0, 0])
        ch, sh = torch.cos(heading), torch.sin(heading)
        rel = self.xpos[:, self.foot_l, :2] - self.xpos[:, self.foot_r, :2]
        fwd = ch * rel[:, 0] + sh * rel[:, 1]
        left = self.rest_lateral_sign * (-sh * rel[:, 0] + ch * rel[:, 1])
        leg_err = (self.qpos[:, self.leg_q] - self.rest_qpos[self.leg_q]).square().mean(dim=-1)
        trunk_err = (self.qpos[:, self.trunk_q] - self.rest_qpos[self.trunk_q]).square().mean(dim=-1)
        stance = (torch.exp(-((left - self.rest_stance).square() + fwd.square()) / STANCE_SIGMA ** 2)
                  * torch.exp(-leg_err / LEG_POSE_SCALE) * torch.exp(-trunk_err / TRUNK_POSE_SCALE)
                  * balanced)

        # **Off the joint stops** - see env_config.JOINT_SOFT_LIMIT. Always on: a stop holds a pose
        # for no torque, and that free ride is the thing being taken away.
        q = self.qpos[:, self.qadr]
        over = ((q - self.soft_hi).clamp_min(0.0) + (self.soft_lo - q).clamp_min(0.0)).sum(dim=-1)

        # A foot is airborne relative to where it rests, not to the floor: the sole sits at a
        # non-zero height and a fixed threshold would read one foot as permanently up.
        foot_z = torch.stack([self.xpos[:, self.foot_l, 2], self.xpos[:, self.foot_r, 2]], dim=-1)
        airborne = foot_z > (self.rest_foot_z + FOOT_CLEAR)
        single = (airborne.sum(dim=-1) == 1).float()

        # **A step is a recovery only when the foot GOES somewhere useful.** This paid for single
        # support alone, and a stomp satisfies that completely: measured at 6 m/s, 78% of foot
        # lifts travelled under 5 cm and the rest were directionally random (mean cos -0.069 to
        # the COM escape), across four sessions that learned nothing. Pay for the swing foot's
        # velocity along the direction the COM has escaped to - a stomp earns nothing, and so does
        # a step the wrong way.
        want = com[:, :2] - feet_mid
        want_hat = want / want.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        swing = torch.where(airborne[:, :1], self._foot_velocity(self.foot_l),
                            self._foot_velocity(self.foot_r))
        progress = (swing[:, :2] * want_hat).sum(dim=-1).clamp(0.0, CAPTURE_V) / CAPTURE_V
        recover_step = single * progress * (1.0 - balanced)

        # **Where the step lands, not only how fast it swings.** With CAPTURE_V at 1.0 the steps
        # grew, but on an 80.9% brain the first step after a high hit still landed 0.16-0.23 m
        # SHORT of the capture point - the point a foot must reach for the body to come to rest
        # over it (com + v / omega0, the linear inverted pendulum). Paid while off balance on one
        # foot, for the swing foot's closeness to it.
        omega0 = torch.sqrt(9.81 / com[:, 2].clamp_min(0.3))
        capture = com[:, :2] + com_v[:, :2] / omega0.unsqueeze(-1)
        swing_xy = torch.where(airborne[:, :1], self.xpos[:, self.foot_l, :2],
                               self.xpos[:, self.foot_r, :2])
        gap = (swing_xy - capture).square().sum(dim=-1)
        place = single * (1.0 - balanced) * torch.exp(-gap / CAPTURE_PLACE_SIGMA ** 2)

        effort = action.square().mean(dim=-1)
        jerk = (action - self.prev_action).square().mean(dim=-1)
        return (2.0 * upright + 1.0 * height + 3.0 * support
                + 1.5 * pose + 2.5 * recover_step + 2.0 * place + 2.0 * stance + 0.2 * still
                + 0.2 - 0.05 * effort - 0.05 * jerk - 2.0 * over)
