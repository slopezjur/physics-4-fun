"""Perturbation under Newton/XPBD: stay standing through an external impulse.

Inherits `StandEnv` and adds one thing - something pushes back. The observation, action mapping and
reward are Stand's unchanged, which is both what makes a bootstrap from a Stand checkpoint valid and
the honest expression of the task: the same objective with a disturbance, not a new objective.

**The impulse is not in the observation, and that is deliberate.** `BallGun` states it plainly: the
agent has to react to what it feels through its own proprioception. Telling it about an incoming hit
turns "learn to recover" into "learn to anticipate a scripted event", and dodging is a different
task that would need the disturbance in the observation vector.

The one thing that had to be rebuilt for this backend is how the wrench is delivered.
`robot.set_external_force_and_torque` goes through the PhysX view, which does not exist under
Newton; `NewtonRigState.add_body_wrench` writes `State.body_f` instead, which is the buffer this
solver integrates. It is zeroed every step, which is exactly what an impulse wants.
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from p4f_newton.tasks.stand.stand_env import StandEnv

from .perturb_env_cfg import PerturbEnvCfg


class PerturbEnv(StandEnv):
    cfg: PerturbEnvCfg

    def __init__(self, cfg: PerturbEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Candidate targets, resolved once. `preserve_order` keeps them aligned with the config list
        # so a per-body breakdown in an evaluator means what it says.
        target_ids, self._target_names = self.robot.find_bodies(
            list(self.cfg.push_target_bones), preserve_order=True
        )
        self._target_ids = torch.tensor(target_ids, device=self.device, dtype=torch.long)

        # Which body each environment's shot is aimed at, as an index into ALL bodies.
        self._push_body_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Step at which the NEXT hit lands, and the wrench it will carry.
        self._push_step = torch.zeros(self.num_envs, device=self.device)
        self._push_impulse = torch.zeros(self.num_envs, 3, device=self.device)
        self._push_torque = torch.zeros(self.num_envs, 3, device=self.device)
        # True once the body has been hit this episode; an evaluator reads it to know when a
        # measurement is meaningful.
        self._was_hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # The shot that ALREADY landed, retained only so the marker can rebound off it. Load-bearing
        # for the visual: `_draw_shot` overwrites `_push_impulse` and `_push_body_idx` the instant a
        # shot lands, so without this the rebound would be drawn along the *following* shot's
        # direction, away from the following shot's target.
        self._last_hit_step = torch.full((self.num_envs,), -1.0e9, device=self.device)
        self._last_hit_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_hit_dir = torch.zeros(self.num_envs, 3, device=self.device)

        self._episode_sums["push_magnitude"] = torch.zeros(self.num_envs, device=self.device)

        # Cosmetic incoming ball, built only when something is actually rendering - during headless
        # training this stays None and nothing in the step loop touches it.
        #
        # Gated on `visualizer_cfgs` rather than `sim.has_gui`, which is what the 2.3.2 version used.
        # Two reasons that would not work here: in Isaac Lab 3 `has_gui` is a PROPERTY, so calling it
        # raises `TypeError: 'bool' object is not callable`, and read correctly it is False under
        # Newton regardless, because this backend never starts Kit. `visualizer_cfgs` is what
        # `play.py:attach_viewer` actually sets, and needs no private API.
        self._ball = None
        if self.cfg.show_impact_ball and getattr(self.cfg.sim, "visualizer_cfgs", None):
            self._ball = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/PerturbBall",
                    markers={
                        "ball": sim_utils.SphereCfg(
                            radius=self.cfg.ball_radius,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.15, 0.05)),
                        )
                    },
                )
            )
            self._ball_lead_s = self.cfg.ball_spawn_distance / self.cfg.ball_speed

    # ------------------------------------------------------------------ scheduling

    def _resample_commands(self, env_ids) -> None:
        """Per-episode reset hook. The velocity command stays zero, like Stand's."""
        super()._resample_commands(env_ids)
        self._draw_shot(env_ids)

    def _draw_shot(self, env_ids) -> None:
        """Choose WHAT the next shot is: magnitude, direction, target bone, off-centre torque.

        Separate from `_resample_commands` because it is called from two places with different
        meanings - once per episode on reset, and again after every hit to arm the next one. It
        deliberately does NOT decide *when* the shot lands; that is `_reset_idx` and the interval
        advance. Conflating the two made the schedule reset itself on every shot in the 2.3.2 port.
        """
        count = len(env_ids)
        if count == 0:
            return

        low, high = self.cfg.push_impulse_range
        magnitude = torch.empty(count, device=self.device).uniform_(low, high)

        if self.cfg.push_randomize_direction:
            angle = torch.empty(count, device=self.device).uniform_(0.0, 2.0 * math.pi)
        else:
            angle = torch.zeros(count, device=self.device)
        direction = torch.stack([torch.cos(angle), torch.sin(angle), torch.zeros_like(angle)], dim=-1)

        impulse = direction * magnitude.unsqueeze(-1)

        # An off-centre hit adds r x F. r is vertical (the aim jitter) and F horizontal, so the
        # torque is horizontal and perpendicular to the push - the component that actually rotates
        # the torso rather than translating it.
        offset = torch.empty(count, 1, device=self.device).uniform_(
            -self.cfg.push_height_jitter, self.cfg.push_height_jitter
        )
        r = torch.cat([torch.zeros(count, 2, device=self.device), offset], dim=-1)

        choice = torch.randint(0, len(self._target_ids), (count,), device=self.device)
        self._push_body_idx[env_ids] = self._target_ids[choice]
        self._push_impulse[env_ids] = impulse
        self._push_torque[env_ids] = torch.cross(r, impulse, dim=-1)
        self._was_hit[env_ids] = False
        self._episode_sums["push_magnitude"][env_ids] = magnitude

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # Schedule the first shot. `_draw_shot` decides WHAT; this decides WHEN, and must not run on
        # the per-shot redraw or the interval would never advance.
        self._push_step[env_ids] = self.cfg.push_first_s / self.step_dt
        # Forget the previous episode's impact. `_last_hit_step` is an episode-relative step count
        # and `episode_length_buf` restarts at 0, so a retained value does not merely go stale - the
        # new episode walks back onto it and draws a phantom rebound at the OLD hit position.
        self._last_hit_step[env_ids] = -1.0e9

    # ------------------------------------------------------------------ delivery

    def _apply_action(self) -> None:
        super()._apply_action()
        self._deliver_shot()
        if self._ball is not None:
            self._update_ball()

    def _deliver_shot(self) -> None:
        """Apply any hit that is due, as a one-step wrench.

        Applied from `_apply_action` rather than `_pre_physics_step` because `State.body_f` is
        cleared by the solver every step - so the write has to happen inside the substep loop, next
        to where the joint targets are written, or it is zeroed before it does anything.

        `J = F * dt`, so a one-step force of `J / dt` delivers exactly the sampled impulse. The
        substep dt is `sim.dt`, not `step_dt`: writing the policy-step value would over-deliver by
        the decimation factor.
        """
        due = self.episode_length_buf >= self._push_step
        if not bool(due.any()):
            return

        dt = self.cfg.sim.dt
        mask = due.float().unsqueeze(-1)
        self.rig.add_body_wrench(
            self._push_body_idx,
            force=self._push_impulse * mask / dt,
            torque=self._push_torque * mask / dt,
        )

        self._was_hit |= due
        hit = due.nonzero(as_tuple=False).squeeze(-1)

        # Retain WHERE this shot landed and along WHICH direction, before `_draw_shot` below
        # replaces both with the next shot's. Only the marker reads these.
        if self._ball is not None and hit.numel() > 0:
            impact = self.robot.data.body_com_pos_w.torch[hit, self._push_body_idx[hit]]
            magnitude = self._push_impulse[hit].norm(dim=-1, keepdim=True).clamp(min=1e-6)
            self._last_hit_step[hit] = self.episode_length_buf[hit].float()
            self._last_hit_pos[hit] = impact
            self._last_hit_dir[hit] = self._push_impulse[hit] / magnitude

        if self.cfg.push_interval_s > 0.0:
            self._push_step[hit] += self.cfg.push_interval_s / self.step_dt
            self._draw_shot(hit)
        else:
            # One shot per episode: park the next one past the end of the window.
            self._push_step[hit] = float(self.max_episode_length) + 1.0

    # ------------------------------------------------------------------ visualisation

    def _update_ball(self) -> None:
        """Fly the cosmetic ball in along the impulse direction, then rebound it off the impact.

        The rebound is the point. The ball has no collider - the policy was trained against a direct
        impulse, and a real projectile would deliver a different, contact-dependent one, so a
        physical ball would stop the arena showing what the policy actually learned. But a marker
        that sails straight through 80 kg of dummy reads as a ghost, which is worse than drawing
        nothing. Reflecting at the moment of impact costs nothing and shows the correct event.

        Positions are derived from time-to-impact rather than integrated, so the ball is exactly on
        the target on the frame the wrench is applied and cannot drift out of sync with it.
        """
        now = self.episode_length_buf.float()
        env_idx = torch.arange(self.num_envs, device=self.device)

        # --- incoming: approaching the next scheduled hit ---
        # `.torch` is required on this backend - the Newton buffer is a wrapper, not a tensor, and
        # reading the attribute directly (as 2.3.2 does) yields a plausible zero rather than raising.
        target = self.robot.data.body_com_pos_w.torch[env_idx, self._push_body_idx]
        magnitude = self._push_impulse.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = self._push_impulse / magnitude
        to_impact = (now - self._push_step) * self.step_dt          # negative before the hit
        incoming_pos = target + direction * (to_impact.unsqueeze(-1) * self.cfg.ball_speed)
        # The impulse is sampled from [0, max], so near-zero shots exist; drawing a ball for one
        # would show an impact that never visibly happens.
        incoming = (to_impact >= -self._ball_lead_s) & (to_impact < 0.0) & (magnitude.squeeze(-1) > 0.1)

        # --- rebound: leaving the hit that already landed ---
        since = (now - self._last_hit_step) * self.step_dt
        rebound_pos = self._last_hit_pos - self._last_hit_dir * (
            since.unsqueeze(-1) * self.cfg.ball_speed * self.cfg.ball_rebound_speed_factor
        )
        rebounding = (since >= 0.0) & (since <= self.cfg.ball_trail_s)

        position = torch.where(rebounding.unsqueeze(-1), rebound_pos, incoming_pos)
        visible = incoming | rebounding
        # Scale rather than culling: VisualizationMarkers wants an entry per marker every call, so a
        # hidden ball is a zero-scale one.
        scales = visible.float().unsqueeze(-1).repeat(1, 3)
        self._ball.visualize(translations=position, scales=scales)
