"""Perturbation: stay standing through an external impulse.

Inherits `StandEnv` and adds one thing - something pushes back. The observation, action mapping and
reward are Stand's, unchanged, which is both what makes a bootstrap from a Stand checkpoint valid
and the honest expression of the task: this is the same objective with a disturbance, not a new one.

**The impulse is not in the observation, and that is the point.** `BallGun` states it plainly: the
agent has to react to what it feels through its own proprioception, and telling it about the
incoming hit turns "learn to recover" into "learn to anticipate a scripted event". Dodging is a
different task and would need the disturbance in the observation vector.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from p4f_isaac.tasks.stand.stand_env import StandEnv

from .perturb_env_cfg import PerturbEnvCfg


class PerturbEnv(StandEnv):
    cfg: PerturbEnvCfg

    def __init__(self, cfg: PerturbEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Candidate targets, resolved once. preserve_order keeps them aligned with the config list
        # so a per-body breakdown in the evaluator means what it says.
        target_ids, self._target_names = self.robot.find_bodies(
            list(self.cfg.push_target_bones), preserve_order=True
        )
        self._target_ids = torch.tensor(target_ids, device=self.device, dtype=torch.long)
        # Which body each env's shot is aimed at this episode, as an index into ALL bodies.
        self._push_body_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Step at which the NEXT hit lands, and the impulse it will carry.
        self._push_step = torch.zeros(self.num_envs, device=self.device)
        self._push_impulse = torch.zeros(self.num_envs, 3, device=self.device)
        self._push_torque = torch.zeros(self.num_envs, 3, device=self.device)
        # True once the body has been hit at least once this episode. Kept because the evaluator
        # reads it to know when a measurement is meaningful.
        self._was_hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # The shot that already landed, retained only so the marker can rebound off it. The next
        # shot's parameters are resampled the instant one lands, so without this the rebound would
        # be drawn using the *following* shot's direction and target.
        self._last_hit_step = torch.full((self.num_envs,), -1.0e9, device=self.device)
        self._last_hit_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_hit_dir = torch.zeros(self.num_envs, 3, device=self.device)

        self._episode_sums["push_magnitude"] = torch.zeros(self.num_envs, device=self.device)

        # Cosmetic incoming ball. Only built when something is actually rendering - during headless
        # training this stays None and nothing in the step loop touches it.
        self._ball = None
        if self.cfg.show_impact_ball and self.sim.has_gui():
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

    def _resample_commands(self, env_ids) -> None:
        """Per-episode reset hook. Perturbation's velocity command stays zero, like Stand's."""
        super()._resample_commands(env_ids)
        self._draw_shot(env_ids)

    def _draw_shot(self, env_ids) -> None:
        """Choose WHAT the next shot is: magnitude, direction, target bone, off-centre torque.

        Separate from _resample_commands because it is called from two places with different
        meanings - once per episode on reset, and again after every hit to arm the next one. It
        deliberately does NOT decide *when* the shot lands; that is _reset_idx and the interval
        advance, and conflating the two made the schedule reset itself on every shot.
        """
        count = len(env_ids)
        if count == 0:
            return

        low, high = self.cfg.push_impulse_range
        magnitude = torch.empty(count, device=self.device).uniform_(low, high)

        if self.cfg.push_randomize_direction:
            angle = torch.empty(count, device=self.device).uniform_(0.0, 2.0 * torch.pi)
        else:
            angle = torch.zeros(count, device=self.device)
        direction = torch.stack([torch.cos(angle), torch.sin(angle), torch.zeros_like(angle)], dim=-1)

        impulse = direction * magnitude.unsqueeze(-1)
        # An off-centre hit adds r x F. r is vertical (the aim jitter), F horizontal, so the torque
        # is horizontal and perpendicular to the push - the component that actually rotates the
        # torso rather than translating it.
        offset = torch.empty(count, 1, device=self.device).uniform_(
            -self.cfg.push_height_jitter, self.cfg.push_height_jitter
        )
        r = torch.cat([torch.zeros(count, 2, device=self.device), offset], dim=-1)

        # Uniform over the target list, redrawn per episode like BallGun does per shot.
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
        # Schedule the first shot. _resample_commands draws WHAT the shot is; this decides WHEN,
        # and must not run on the per-shot redraw or the interval would never advance.
        self._push_step[env_ids] = self.cfg.push_first_s / self.step_dt
        self._last_hit_step[env_ids] = -1.0e9

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)

        due = self.episode_length_buf >= self._push_step
        # Spanning every body rather than a fixed slice, because the target now varies per env.
        # Cleared every step, so the force applies for exactly one env step and behaves as an
        # impulse rather than a sustained wind - Isaac holds external wrenches until overwritten.
        forces = torch.zeros(self.num_envs, self.robot.num_bodies, 3, device=self.device)
        torques = torch.zeros_like(forces)
        if due.any():
            hit = due.nonzero(as_tuple=False).squeeze(-1)
            body = self._push_body_idx[hit]
            # J = F * dt, so a one-step force of J / step_dt delivers exactly the sampled impulse.
            forces[hit, body] = self._push_impulse[hit] / self.step_dt
            torques[hit, body] = self._push_torque[hit] / self.step_dt
            self._was_hit |= due

            # Remember this impact for the rebound, then schedule and draw the next shot.
            mag = self._push_impulse[hit].norm(dim=-1, keepdim=True).clamp(min=1e-6)
            self._last_hit_step[hit] = self.episode_length_buf[hit].float()
            self._last_hit_pos[hit] = self.robot.data.body_com_pos_w[hit, body]
            self._last_hit_dir[hit] = self._push_impulse[hit] / mag
            if self.cfg.push_interval_s > 0.0:
                self._push_step[hit] += self.cfg.push_interval_s / self.step_dt
                self._draw_shot(hit)
            else:
                # One shot per episode: park the next one past the end of the window.
                self._push_step[hit] = float(self.max_episode_length) + 1.0

        self.robot.set_external_force_and_torque(forces, torques)

        if self._ball is not None:
            self._update_ball()

    def _update_ball(self) -> None:
        """Fly the cosmetic ball in along the impulse direction, then rebound off the impact.

        The rebound is the point. The ball has no collider - the policy was trained against a
        direct impulse, and a real projectile would deliver a different, contact-dependent one, so
        the arena would stop showing what the policy actually learned. But a marker that continues
        straight through the body reads as the ball being a ghost, which is worse than not drawing
        it. Reflecting at the moment of impact costs nothing and shows the correct event.

        Positions are derived from time-to-impact rather than integrated, so the ball is exactly on
        the target on the frame the wrench is applied and cannot drift out of sync with it.
        """
        data = self.robot.data
        now = self.episode_length_buf.float()
        env_idx = torch.arange(self.num_envs, device=self.device)

        # --- incoming: approaching the next scheduled hit ---
        target = data.body_com_pos_w[env_idx, self._push_body_idx]
        magnitude = self._push_impulse.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = self._push_impulse / magnitude
        to_impact = (now - self._push_step) * self.step_dt          # negative before the hit
        incoming_pos = target + direction * (to_impact.unsqueeze(-1) * self.cfg.ball_speed)
        incoming = (to_impact >= -self._ball_lead_s) & (to_impact < 0.0) & (magnitude.squeeze(-1) > 0.1)

        # --- rebound: leaving the hit that already landed ---
        since = (now - self._last_hit_step) * self.step_dt
        # Reflected, and slower - a ball that bounces off 80 kg of dummy does not keep its speed.
        rebound_pos = self._last_hit_pos - self._last_hit_dir * (
            since.unsqueeze(-1) * self.cfg.ball_speed * self.cfg.ball_rebound_speed_factor
        )
        rebounding = (since >= 0.0) & (since <= self.cfg.ball_trail_s)

        position = torch.where(rebounding.unsqueeze(-1), rebound_pos, incoming_pos)
        visible = incoming | rebounding
        # Scale rather than culling: VisualizationMarkers wants an entry per marker every call, so
        # a hidden ball is a zero-scale one.
        scales = visible.float().unsqueeze(-1).repeat(1, 3)
        self._ball.visualize(translations=position, scales=scales)
