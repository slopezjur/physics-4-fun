"""Independent joint corrections for explicitly commanded stepping practice."""
from __future__ import annotations

import torch
from torch import nn

from envs.base_env import DoneFlags

from .capture_reward import capture_distance
from .run_direct_recovery import Recovery
from .stepping_curriculum import AUTHORITY_INDEX, GOAL_WIDTH, StepConfig, StepCurriculum
from .target_control33 import DelayedTargetControl33


COUNT = 16
ACTION_WIDTH = 33
HISTORY_WIDTH = 17
HISTORY_LENGTH = 3
PREVIOUS_LATENT_START = 374 + HISTORY_LENGTH * HISTORY_WIDTH + ACTION_WIDTH
GOAL_START = PREVIOUS_LATENT_START + ACTION_WIDTH
OBSERVATION_WIDTH = GOAL_START + GOAL_WIDTH
RHO = .9
SLEW = .02
LEG_RADIUS = .5
OTHER_RADIUS = .1


class SteppingHead(nn.Module):
    """The 33 conditional means retain independent sampled action memory."""

    def __init__(self, input_width=OBSERVATION_WIDTH, outputs=ACTION_WIDTH):
        super().__init__()
        if input_width != OBSERVATION_WIDTH or outputs not in (1, ACTION_WIDTH):
            raise ValueError('Expected the declared stepping observation and 1 or 33 outputs')
        self.net = nn.Sequential(nn.Linear(input_width, 128), nn.ELU(),
                                 nn.Linear(128, 128), nn.ELU(), nn.Linear(128, outputs))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.outputs = outputs
        self.input_width = input_width

    def forward(self, features):
        value = self.net(features)
        if self.outputs == 1:
            return value.squeeze(-1)
        previous = features[..., PREVIOUS_LATENT_START:GOAL_START]
        authority = features[..., GOAL_START + AUTHORITY_INDEX:GOAL_START + AUTHORITY_INDEX + 1]
        return authority * (RHO * previous + (1 - RHO) * value)


class StepSkillEnvironment(Recovery):
    """Commands select tasks; the actor alone produces joint target corrections."""

    def __init__(self, rig, anchor, *, seconds=12.):
        self.seconds = float(seconds)
        self.history = torch.zeros(COUNT, HISTORY_LENGTH, HISTORY_WIDTH)
        self.last_latent = torch.zeros(COUNT, ACTION_WIDTH)
        self.reset_records = []
        self.curriculum = None
        self.motor_names = [rig.model.joint(int(j)).name
                            for j in rig.model.actuator_trnid[rig.actuators, 0]]
        self.leg_indices = [i for i, name in enumerate(self.motor_names)
                            if name.startswith(('Thigh_', 'Shin_', 'Foot_', 'Toe_'))]
        if len(self.motor_names) != ACTION_WIDTH or len(self.leg_indices) != 16:
            raise ValueError('Expected the original 33 motors, including 16 leg motors')
        self.radius = torch.full((ACTION_WIDTH,), OTHER_RADIUS)
        self.radius[self.leg_indices] = LEG_RADIUS
        anchor.eval().requires_grad_(False)
        super().__init__(rig, anchor, seconds=seconds)
        self.curriculum = StepCurriculum(COUNT, self.dt, 'cpu',
                                         config=StepConfig(episode_seconds=self.seconds))
        self.reset()

    @staticmethod
    def _selected(value, ids, *, dtype):
        result = torch.as_tensor(value, dtype=dtype, device='cpu')
        if result.ndim == 0:
            return result.expand(len(ids)).clone()
        if result.shape == (COUNT,):
            return result[ids].clone()
        if result.shape == (len(ids),):
            return result.clone()
        raise ValueError('Reset parameters must be scalar, all-world, or selected-world vectors')

    @torch.no_grad()
    def reset(self, ids=None, rng=None, level=0, side=None, distance=None, onset=None, enabled=None):
        ids = self.task.ids if ids is None else torch.as_tensor(ids, dtype=torch.long)
        Recovery.reset(self, ids)
        self.force[ids] = 0
        self.onset[ids] = 0
        self.last_latent[ids] = 0
        self.history[ids] = self.balance_features()[ids, None, :].expand(-1, HISTORY_LENGTH, -1)
        if self.curriculum is None:
            return
        count = len(ids)
        if side is None:
            side = (ids // 8 if rng is None else torch.randint(2, (count,), generator=rng))
        if distance is None:
            signs = (2 * (ids % 2) - 1 if rng is None else 2 * torch.randint(2, (count,), generator=rng) - 1)
            distance = .04 * signs
        if onset is None:
            onset = (torch.ones(count) if rng is None else .8 + .4 * torch.randint(2, (count,), generator=rng))
        side = self._selected(side, ids, dtype=torch.long)
        distance = self._selected(distance, ids, dtype=torch.float32)
        onset = self._selected(onset, ids, dtype=torch.float32)
        level = self._selected(level, ids, dtype=torch.long)
        enabled = self._selected(True if enabled is None else enabled, ids, dtype=torch.bool)
        distance = torch.where(enabled & (level == 2), distance, torch.zeros_like(distance))
        self.curriculum.reset(ids, self.measurement(), side=side, distance=distance,
                              onset_seconds=onset, enabled=enabled, level=level)
        self.reset_records.extend(dict(world=int(world), pose=int(world % 8),
            level=int(level[i]), side=int(side[i]), distance_m=float(distance[i]),
            onset_seconds=float(onset[i]), enabled=bool(enabled[i])) for i, world in enumerate(ids))
        return self.features()

    def measurement(self, state=None):
        state = self.state() if state is None else state
        return dict(com=state['com'], velocity=state['velocity'], feet=state['feet'],
            clearance=state['clearance'], loads=state['foot_loads_n'],
            pelvis_height=state['pelvis_height_m'], pelvis_up=state['pelvis_up'],
            angular=state['angular'], finite=state['finite_state'], stand=state['stand'])

    @torch.no_grad()
    def balance_features(self, state=None):
        state = self.state() if state is None else state
        center = state['feet'][:, :, :2].mean(1)
        height = (state['com'][:, 2] - state['feet'][:, :, 2].amin(-1)).clamp_min(.05)
        capture = state['com'][:, :2] + state['velocity'][:, :2] * (height / 9.81).sqrt()[:, None]
        return torch.cat(((state['com'][:, :2] - center) / .1, state['velocity'][:, :2] / .5,
            state['angular'] / 2, (capture - center) / .1,
            state['foot_loads_n'] / (self.task.total_mass * 9.81),
            ((state['pelvis_height_m'] - self.task.reference.qpos[self.task.ids % 8, 2]) / .15)[:, None],
            state['pelvis_up'][:, None],
            ((state['feet'][:, :, :2] - state['com'][:, None, :2]) / .2).flatten(1)), -1)

    @torch.no_grad()
    def features(self):
        value = torch.cat((self.anchor.obs_norm.normalize(self.task.observations()).clamp(-20, 20),
            self.history.flatten(1).clamp(-20, 20), self.previous / self.radius,
            self.last_latent, self.curriculum.observations()), -1)
        if value.shape != (COUNT, OBSERVATION_WIDTH) or not torch.isfinite(value).all():
            raise ValueError('Invalid stepping observations')
        return value

    @torch.no_grad()
    def step(self, latent, active):
        if latent.shape != (COUNT, ACTION_WIDTH) or not torch.isfinite(latent).all():
            raise ValueError('Expected finite [16, 33] latent joint commands')
        if active.shape != (COUNT,) or active.dtype != torch.bool:
            raise ValueError('Expected a boolean active-world mask')
        self.engine.active[:] = active.numpy()
        pre = self.state()
        before = self.previous.clone()
        before_history = self.history.clone()
        before_latent = self.last_latent.clone()
        command_before = self.curriculum.observations().clone()
        authority = command_before[:, AUTHORITY_INDEX].bool()
        realized = torch.where(authority[:, None], latent, torch.zeros_like(latent))
        realized[~active] = before_latent[~active]
        if before[active & ~authority].any():
            raise RuntimeError('Inactive task authority must begin with zero residual after reset')
        desired = self.radius * realized.tanh()
        addition = before + (desired - before).clamp(-SLEW, SLEW)
        addition[~active] = before[~active]
        source = self.anchor(self.task.observations())
        self.engine.policy_control.addition.copy_(addition)
        applied = torch.zeros_like(self.force)
        self.force.zero_()
        self.engine.set_external_force(self.chest, applied)
        _, _, done, _ = self.task.step(source)
        self.previous.copy_(addition)
        state = self.state()
        control = self.engine.policy_control
        nominal = DelayedTargetControl33.target_positions(control, source)
        issued = control.target_positions(source)
        expected = (nominal + addition).clamp(control.lower, control.upper)
        if not torch.equal(issued[active], expected[active]):
            raise RuntimeError('Residual target mapping changed')
        if (addition.abs() > self.radius + 1e-7).any() or ((addition - before).abs() > SLEW + 1e-7).any():
            raise RuntimeError('Joint correction radius or slew exceeded')
        warning = torch.from_numpy(self.engine.solver_warning.copy())
        physical_failed = (done == DoneFlags.FAIL.value) | warning | ~state['finite_state']
        native_timeout = done == DoneFlags.TIME.value
        reward, task_terminal, task_info = self.curriculum.advance(self.measurement(state), physical_failed, active)
        terminal = ((done != DoneFlags.NULL.value) | physical_failed | task_terminal) & active
        reward_before_action_cost = reward.clone()
        action_cost = .005 * (addition / self.radius).square().mean(-1) * active
        reward = reward - action_cost
        reward[~active] = 0.
        if not torch.isfinite(reward).all():
            raise RuntimeError('Nonfinite commanded skill reward')
        live = self.balance_features(state)
        self.history[active] = torch.cat((live[active, None], before_history[active, :HISTORY_LENGTH - 1]), 1)
        self.last_latent[active] = realized[active]
        capture = capture_distance(state['com'], state['velocity'], state['feet'], state['foot_loads_n'])
        task_success = task_info.get('task_success', task_info.get('success'))
        if task_success is None:
            raise RuntimeError('Curriculum must explicitly report task success')
        failed = (physical_failed | task_info['failed']) & active
        info = dict(qpos=self.engine.native_qpos().clone(), qvel=self.engine._array('qvel').clone(),
            issued=issued.clone(), active_target=control.active.clone(), addition=addition.clone(), previous=before,
            desired_addition=desired.clone(), nominal=nominal.clone(), source_action=source.clone(), force=applied,
            active=active.clone(), failed=failed, physical_failed=physical_failed.clone(),
            task_success=task_success.clone(), reward_before_action_cost=reward_before_action_cost,
            action_cost=action_cost, native_done=done.clone(),
            native_timeout=native_timeout.clone(), terminal=terminal.clone(),
            stand=state['stand'] & pre['stand'], settled=state['settled'] & pre['settled'],
            measurement_stand=state['stand'].clone(), pre_stand=pre['stand'].clone(),
            loads=state['foot_loads_n'].clone(), clearance=state['clearance'].clone(),
            xy=state['feet'][:, :, :2].clone(), feet=state['feet'].clone(), com=state['com'].clone(),
            velocity=state['velocity'].clone(), angular=state['angular'].clone(),
            pelvis_height=state['pelvis_height_m'].clone(), pelvis_up=state['pelvis_up'].clone(),
            non_foot_ground_contacts_n=state['non_foot_ground_contacts_n'].clone(),
            upper_body_tilt_degrees=state['upper_body_tilt_degrees'].clone(),
            upper_body_clearance_deficit_m=state['upper_body_clearance_deficit_m'].clone(),
            horizontal_speed_m_s=state['horizontal_speed_m_s'].clone(),
            angular_speed_rad_s=state['angular_speed_rad_s'].clone(),
            finite=state['finite_state'].clone(), capture=capture, warning=warning,
            history_before=before_history, balance_features=live, previous_latent=before_latent,
            latent=realized.clone(), sampled_latent=latent.clone(), command_before=command_before,
            command_after=self.curriculum.observations().clone(), authority=authority.clone())
        for key, value in task_info.items():
            if key == 'task_success':
                continue
            target = 'task_' + key if key in info else key
            info[target] = value.clone() if isinstance(value, torch.Tensor) else value
        return reward, terminal, info

    def contract(self):
        return dict(schema='commanded_independent_joint_stepping_v1', worlds=COUNT,
            observation_width=OBSERVATION_WIDTH, source_observation_width=374,
            measured_history=[HISTORY_LENGTH, HISTORY_WIDTH], previous_addition_width=ACTION_WIDTH,
            previous_latent_width=ACTION_WIDTH, goal_width=GOAL_WIDTH, goal_start=GOAL_START,
            authority_goal_index=AUTHORITY_INDEX, previous_latent_start=PREVIOUS_LATENT_START,
            motor_names=self.motor_names, leg_indices=self.leg_indices, radius_rad=self.radius.tolist(),
            slew_rad_per_interval=SLEW, dt=self.dt, persistent_latent_rho=RHO,
            conditional_mean='authority*(rho*previous_latent+(1-rho)*network(features))',
            source_actor_frozen=True, zero_initialized_mean_layer=True,
            task_commands=True, autonomous_hit_policy=False, external_force=False,
            disabled_and_wait_residual='exact zero from reset; no learned quiet guarantee',
            reset='original native fixed poses and native controller reset; no prepared IK states',
            reset_sampling=dict(side=[0, 1], distance_m=[-.04, .04], onset_seconds=[.8, 1.2],
                                enabled_default=True, draws='independent when unspecified and rng supplied'),
            terminal_bootstrap=0, episode_seconds=self.seconds,
            action_cost='0.005*mean((applied_addition/per_joint_radius)^2), active worlds only',
            authority_claim='Prototype .5-rad leg envelope; physical sufficiency not established',
            controller=self.engine.policy_control.contract())


SteppingEnvironment = StepSkillEnvironment
