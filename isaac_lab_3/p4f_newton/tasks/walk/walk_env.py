"""Walk under Newton/XPBD: follow a velocity command while staying upright.

Inherits `StandEnv` rather than restating it. The observation layout, action mapping, contact
handling and reset are shared verbatim, which is the precondition for bootstrapping this policy
from a Stand checkpoint (`train.py --init_from`).

Three things change, and each is forced by something measured rather than chosen:

1. **The reward is a product, not a sum.** Posture gates the task terms instead of being paid for
   separately - see `WalkEnvCfg` for the measurement that made this a project rule.
2. **Air time is derived, not sensed.** Newton has no working `ContactSensor` here, so
   `compute_first_contact` and `last_air_time` do not exist. Both are reconstructed from the same
   height proxy the observation's contact flags already use.
3. **Body-frame velocity is derived, not read.** `root_lin_vel_b` and `root_ang_vel_b` are frozen
   under XPBD and read a clean zero, so a reward built on them would pay a perfect tracking score
   to a body lying motionless on the floor. Everything comes through `NewtonRigState`.
"""

from __future__ import annotations

import torch

from p4f_newton.assets import CONTACT_BONES
from p4f_newton.state import quat_rotate_inverse, yaw_only
from p4f_newton.tasks.stand.stand_env import StandEnv

from .walk_env_cfg import WalkEnvCfg


class WalkEnv(StandEnv):
    cfg: WalkEnvCfg

    def __init__(self, cfg: WalkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Positions of the two feet within the contact-flag vector, resolved by name rather than
        # assumed to be the trailing pair. The order comes from the rig JSON, which is the
        # authoritative contract - prose about it has been wrong before.
        self._feet_slots = torch.tensor(
            [CONTACT_BONES.index("Foot_L"), CONTACT_BONES.index("Foot_R")],
            device=self.device,
            dtype=torch.long,
        )

        # Air-time bookkeeping, standing in for the ContactSensor that does not report here.
        self._feet_air_time = torch.zeros(self.num_envs, 2, device=self.device)
        self._feet_grounded = torch.ones(self.num_envs, 2, dtype=torch.bool, device=self.device)

        # --- command curriculum state ---
        _start = float(getattr(self.cfg, "cmd_speed_start", -1.0))
        self._cmd_speed_ceiling = _start if _start > 0.0 else float(self.cfg.cmd_speed_min)
        self._cmd_window_n = 0
        self._cmd_window_ok = 0
        self._cmd_window_speed = 0.0
        self._cmd_window_drive = 0.0
        self._cmd_window_fell = 0
        # Per-episode drive accumulation. The gate needs "did it actually travel", which no single
        # step can answer - a policy lurching forward once looks identical to one holding a pace.
        # Signed forward speed along the command, in the HEADING frame and UNCLAMPED - the same
        # quantity evaluate_walk reports, so the two are directly falsifiable against each other.
        # Net displacement projected on the START heading was the previous attempt and it misreads
        # any episode where the dummy turns: it scored -0.13 m/s for a policy the evaluator, once
        # its own frame error was fixed, measured walking forward at +0.44 m/s.
        # Flight time actually achieved at each touchdown. **The air-time term has paid ~0 for
        # every leg of this session**, and twice now I have moved its threshold by guessing where
        # the distribution sits. Measuring it is the only way to set that threshold honestly: the
        # gait has the right cadence (0.72 Hz per foot) and the right stride (0.61 m), so the feet
        # are cycling correctly and simply not clearing the ground.
        self._air_at_touchdown = 0.0
        self._touchdown_count = 0
        self._along_sum = torch.zeros(self.num_envs, device=self.device)
        self._drive_sum = torch.zeros(self.num_envs, device=self.device)
        self._drive_steps = torch.zeros(self.num_envs, device=self.device)

        # Where the episode started, so the gate can ask "did it get anywhere" rather than "did it
        # move forward often". **`_drive` clamps its ratio at zero** - correct for the reward, since
        # a negative factor would flip the sign of the posture gate it multiplies - but it means an
        # average over `_drive` counts the forward half of an oscillation and discards the backward
        # half. A dummy rocking on the spot scores 0.56 there while its net displacement is ~0,
        # which is precisely the marching-in-place behaviour this curriculum was raising its ceiling
        # on. Net displacement cannot be faked that way.
        self._episode_start_xy = torch.zeros(self.num_envs, 2, device=self.device)

        # Smoothed forward progress, for the same reason the gate needed net displacement: the
        # REWARD has the identical loophole. `_drive` clamps at zero, so an oscillation is paid for
        # its forward half and charged nothing for its backward half. Measured over 130,000
        # near-ceiling episodes: mean drive rose to 0.67 while net speed stayed at +-0.004 m/s.
        # Averaging `along` over roughly a stride before the clamp makes rocking on the spot
        # average to zero, while a real gait - which is forward on net over that window - is
        # unaffected.
        self._along_ema = torch.zeros(self.num_envs, device=self.device)

        # Heading at the start of the episode, as a world-frame unit vector. **The command is
        # expressed in the HEADING frame while `root_pos_w` is world**, so projecting one onto the
        # other directly is a frame error - it silently reads correct only while the dummy happens
        # to face its spawn direction. It reported +0.069 m/s of forward progress for a policy the
        # evaluator measured travelling backwards at 0.437 m/s.
        self._episode_start_fwd = torch.zeros(self.num_envs, 2, device=self.device)

    # ------------------------------------------------------------------ commands

    def _resample_commands(self, env_ids) -> None:
        count = len(env_ids)
        if count == 0:
            return

        def uniform(bounds: tuple[float, float]) -> torch.Tensor:
            return torch.empty(count, device=self.device).uniform_(*bounds)

        # The forward range is scaled by the live ceiling rather than taken from the config, so
        # the curriculum widens what is asked for instead of the policy choosing which commands to
        # satisfy. Lateral and yaw scale with it: a fast sidestep is not easier than a fast walk.
        if self.cfg.cmd_curriculum:
            span = self._cmd_speed_ceiling / max(1e-6, self.cfg.cmd_speed_max)
            lo_x, hi_x = self.cfg.cmd_lin_vel_x
            x_bounds = (lo_x * span, hi_x * span)
            y_bounds = tuple(v * span for v in self.cfg.cmd_lin_vel_y)
            z_bounds = tuple(v * span for v in self.cfg.cmd_ang_vel_z)
        else:
            x_bounds = self.cfg.cmd_lin_vel_x
            y_bounds = self.cfg.cmd_lin_vel_y
            z_bounds = self.cfg.cmd_ang_vel_z

        command = torch.stack(
            [
                uniform(x_bounds),
                uniform(y_bounds),
                uniform(z_bounds),
            ],
            dim=-1,
        )
        # A slice of episodes commanded to hold still, so standing stays in the policy rather than
        # being trained out of it by a reward that only ever pays for moving.
        stand_still = torch.rand(count, device=self.device) < self.cfg.cmd_zero_probability
        command[stand_still] = 0.0
        self._command[env_ids] = command

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # Before super(), which resamples the command and clears the episode's history.
        self._observe_commands(env_ids)

        super()._reset_idx(env_ids)
        # A fresh body is standing on both feet. Leaving the clock running would pay a spurious
        # first "step" on the first touchdown after the reset.
        self._feet_air_time[env_ids] = 0.0
        self._feet_grounded[env_ids] = True
        self._along_sum[env_ids] = 0.0
        self._drive_sum[env_ids] = 0.0
        self._drive_steps[env_ids] = 0.0
        self._episode_start_xy[env_ids] = self.rig.root_pos_w[env_ids, :2]
        self._along_ema[env_ids] = 0.0
        _yaw = self._root_yaw()[env_ids]
        self._episode_start_fwd[env_ids] = torch.stack([torch.cos(_yaw), torch.sin(_yaw)], dim=-1)

        if isinstance(self.extras.get("log"), dict):
            self.extras["log"]["Curriculum/cmd_speed_ceiling"] = self._cmd_speed_ceiling

    def _root_yaw(self) -> torch.Tensor:
        """Heading angle in world, from the root quaternion."""
        q = self.rig.root_quat_w
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # ------------------------------------------------------------------ command curriculum

    def _observe_commands(self, env_ids) -> None:
        """Widen the commanded speed range only once the policy is tracking near the top of it.

        **Success is tracking, not survival.** Gating on uprightness alone is what the fixed range
        already rewarded: a policy that stands still through a command it cannot satisfy stays
        upright for the whole episode and learns nothing. So an episode counts only if it did not
        fall AND averaged at least `cmd_success_drive` of its commanded speed.
        """
        if not self.cfg.cmd_curriculum:
            return
        # Never while measuring - `playback` marks the scoring entry points, and evaluate_walk
        # additionally commands every env the same speed, which would drag the ceiling around
        # mid-measurement and make the number non-comparable with any other run.
        if getattr(self.cfg, "playback", False):
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.arange(self.num_envs, device=self.device)

        steps = self._drive_steps[env_ids]
        cmd_speed = torch.linalg.vector_norm(self._command[env_ids, :2], dim=1)
        near = (
            (cmd_speed > self.cfg.cmd_deadband)
            & (cmd_speed >= self.cfg.cmd_curriculum_band * self._cmd_speed_ceiling)
            & (steps > 0)
        )
        n = int(near.sum().item())
        if n == 0:
            return

        mean_drive = self._drive_sum[env_ids] / steps.clamp(min=1.0)

        # Mean signed forward speed along the command, heading frame, per episode. Signed: going
        # backwards under a forward command is a negative result, not a zero one.
        net_speed = self._along_sum[env_ids] / steps.clamp(min=1.0)

        ok = near & ~self._fell[env_ids] & (net_speed >= self.cfg.cmd_success_drive * cmd_speed)

        # **The achieved SPEED, in m/s, so this gate can be checked against evaluate_walk.**
        # `drive` is a ratio, and a ratio cannot be compared with the evaluator's tracked speed -
        # which is how a gate can report 74% success while the evaluator measures no forward motion
        # at all. Recording the same quantity in the same units is what makes the two falsifiable
        # against each other; this project has been burnt repeatedly by a metric that reads healthy
        # because nothing it produces is comparable to anything else.
        self._cmd_window_speed += float(net_speed[near].sum().item())
        self._cmd_window_drive += float(mean_drive[near].sum().item())
        self._cmd_window_fell += int((near & self._fell[env_ids]).sum().item())

        self._cmd_window_n += n
        self._cmd_window_ok += int(ok.sum().item())
        if self._cmd_window_n < self.cfg.cmd_curriculum_window:
            return

        rate = self._cmd_window_ok / self._cmd_window_n
        before = self._cmd_speed_ceiling
        if rate > self.cfg.cmd_curriculum_raise_above:
            self._cmd_speed_ceiling = min(
                self._cmd_speed_ceiling * self.cfg.cmd_curriculum_raise_factor,
                self.cfg.cmd_speed_max,
            )
        elif rate < self.cfg.cmd_curriculum_lower_below:
            self._cmd_speed_ceiling = max(
                self._cmd_speed_ceiling * self.cfg.cmd_curriculum_lower_factor,
                self.cfg.cmd_speed_min,
            )

        if abs(self._cmd_speed_ceiling - before) > 1e-9:
            arrow = "up" if self._cmd_speed_ceiling > before else "DOWN"
            move = f"ceiling {arrow} {before:.3f} -> {self._cmd_speed_ceiling:.3f} m/s"
        else:
            move = f"ceiling HOLD {self._cmd_speed_ceiling:.3f} m/s"
        _n = max(1, self._cmd_window_n)
        print(f"[cmd-curriculum] tracked {rate:.0%} of {self._cmd_window_n} near-ceiling episodes "
              f"(mean drive {self._cmd_window_drive / _n:.2f}, "
              f"net speed {self._cmd_window_speed / _n:+.3f} m/s, "
              f"fell {self._cmd_window_fell / _n:.0%}, "
              f"flight {self._air_at_touchdown / max(1, self._touchdown_count):.3f} s); {move}",
              flush=True)

        self._cmd_window_n = 0
        self._cmd_window_ok = 0
        self._cmd_window_speed = 0.0
        self._cmd_window_drive = 0.0
        self._cmd_window_fell = 0
        self._air_at_touchdown = 0.0
        self._touchdown_count = 0

    # ------------------------------------------------------------------ reward

    def _reward_terms(self) -> dict[str, torch.Tensor]:
        """Posture-gated task terms plus additive costs.

        Extends `StandEnv._reward_terms` by key rather than rebuilding the dictionary. The 2.3.2
        Walk rebuilt it and silently lost `action_clip` - the barrier that stops the policy mean
        drifting outside the clip range - which every quality metric reported as fine.
        """
        terms = super()._reward_terms()
        dt = self.step_dt
        rig = self.rig

        # Stand's objective was to be motionless and upright. Both are now either the wrong
        # objective or the wrong shape, so they are deleted by key, visibly, rather than left to be
        # outweighed. `action_clip`, `joint_vel`, `effort` and `action_rate` survive untouched.
        del terms["alive"]  # paying for existence is what made the additive form stand still
        del terms["upright"]  # becomes a gate below, not a payment
        del terms["head_height"]  # likewise
        del terms["lin_vel"]  # planar motion is the objective now, not a cost
        del terms["ang_vel"]  # replaced below with a planar-only version; yaw is commanded

        quat = rig.root_quat_w
        # Heading-relative, matching observation slice [3:6]: the policy reads and is scored on the
        # same "forward", so a command of +x means the direction the dummy faces.
        lin_vel_b = quat_rotate_inverse(yaw_only(quat), rig.root_lin_vel_w)
        ang_vel_b = quat_rotate_inverse(quat, rig.root_ang_vel_w)

        posture = rig.upright().clamp(min=0.0) * self._height_reward()

        # One product, not a sum of two. See `WalkEnvCfg.rew_track`.
        drive = self._drive(lin_vel_b)
        self._drive_sum += drive
        self._drive_steps += 1.0
        obedience = drive * self._yaw_factor(ang_vel_b)
        terms["track"] = self.cfg.rew_track * dt * posture * obedience
        terms["feet_air_time"] = self.cfg.rew_feet_air_time * posture * self._air_time_reward()

        terms["lin_vel_z"] = self.cfg.rew_lin_vel_z * dt * lin_vel_b[:, 2] ** 2
        terms["ang_vel_xy"] = self.cfg.rew_ang_vel_xy * dt * torch.sum(ang_vel_b[:, :2] ** 2, dim=1)
        terms["termination"] = self.cfg.rew_termination * self._fell.float()
        return terms

    def _drive(self, lin_vel_b: torch.Tensor) -> torch.Tensor:
        """How much of the commanded motion is actually being produced, in [0, 1].

        Speed along the commanded direction over the commanded speed, saturating at 1 and clamped
        at 0. Saturating matters because the first thing a policy finds is that diving forward
        produces speed; clamping at 0 matters because a negative factor would flip the sign of the
        posture gate it multiplies, making a backwards topple outscore a backwards step.

        A commanded stand-still is scored by a stillness kernel instead of by division: zero is a
        real objective for 15% of episodes, not a degenerate case.
        """
        cmd = self._command[:, :2]
        cmd_speed = torch.linalg.vector_norm(cmd, dim=1)
        moving = cmd_speed > self.cfg.cmd_deadband

        direction = cmd / cmd_speed.clamp(min=1e-6).unsqueeze(-1)
        along = torch.sum(lin_vel_b[:, :2] * direction, dim=1)

        # ~0.83 s time constant at 60 Hz, a little under one gait cycle. Applied BEFORE the clamp,
        # which is the whole point: clamping first discards the backward half of an oscillation and
        # makes rocking indistinguishable from walking.
        self._along_sum = self._along_sum + torch.where(moving, along, torch.zeros_like(along))
        self._along_ema = torch.lerp(self._along_ema, along, self.cfg.drive_smoothing)
        tracking = (self._along_ema / cmd_speed.clamp(min=1e-6)).clamp(0.0, 1.0)

        speed = torch.sum(lin_vel_b[:, :2] ** 2, dim=1)
        stillness = torch.exp(-speed / self.cfg.still_velocity_scale)
        return torch.where(moving, tracking, stillness)

    def _yaw_factor(self, ang_vel_b: torch.Tensor) -> torch.Tensor:
        """Yaw-rate tracking, in [0, 1].

        An exponential kernel rather than a negative squared error: error terms are unbounded below,
        so a policy that has fallen and can track nothing keeps accruing large negatives and the
        gradient ends up dominated by the worst episodes rather than the informative ones. Bounded
        in [0, 1] is also what lets this be multiplied by the posture gate.
        """
        error = (self._command[:, 2] - ang_vel_b[:, 2]) ** 2
        return torch.exp(-error / 0.25)

    # ------------------------------------------------------------------ air time

    def _air_time_reward(self) -> torch.Tensor:
        """Paid once per touchdown, for how much longer than the threshold that foot was airborne.

        **The sign is easy to get backwards, and it was, in 2.3.2.** Written as
        `(min(air_time, target) - target)` the term is <= 0 everywhere, so a foot that never leaves
        the ground registers no touchdown and scores 0 while a foot that actually steps scores
        negative - it penalises walking. Measured at -3.08 per episode against a +18.1 tracking
        term, quietly pushing the policy toward sliding.

        As written here it is positive for any step longer than the threshold and capped, so a long
        step is rewarded and hanging in the air is not worth progressively more.
        """
        grounded = self._contacts()[:, self._feet_slots] > 0.5
        touchdown = grounded & ~self._feet_grounded

        # Read before the update: at a touchdown `_feet_air_time` still holds the flight just ended.
        # **`min=0.0` is not decoration.** `clamp(max=cap)` alone caps only the top, so any flight
        # shorter than the threshold scores NEGATIVE and a foot that never leaves the ground scores
        # zero - the term then pays for sliding, which is the precise inversion described above.
        # Measured in the smoke run at -0.0197 per episode before this clamp was corrected.
        flight = (self._feet_air_time - self.cfg.feet_air_time_threshold).clamp(
            min=0.0, max=self.cfg.feet_air_time_cap
        )
        reward = torch.sum(flight * touchdown.float(), dim=1)

        _td = touchdown.sum().item()
        if _td:
            self._air_at_touchdown += float((self._feet_air_time * touchdown.float()).sum().item())
            self._touchdown_count += int(_td)

        self._feet_air_time = torch.where(
            grounded, torch.zeros_like(self._feet_air_time), self._feet_air_time + self.step_dt
        )
        self._feet_grounded = grounded

        # Only while a motion is commanded - otherwise the cheapest way to collect this is to march
        # on the spot when asked to stand still.
        commanded = torch.linalg.vector_norm(self._command[:, :2], dim=1) > self.cfg.cmd_deadband
        return reward * commanded.float()
