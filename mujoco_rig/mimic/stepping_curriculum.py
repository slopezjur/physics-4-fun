"""Measured, goal-conditioned stepping phases; this module never issues motor commands."""
from dataclasses import asdict, dataclass
from enum import IntEnum
import math

import torch


class Phase(IntEnum):
    QUIET = 0
    WAIT = 1
    TRANSFER = 2
    LIFT = 3
    PLACE = 4
    LAND = 5
    SETTLE = 6


GOAL_WIDTH = 37
AUTHORITY_INDEX = 13
GOAL_LAYOUT = {
    'phase_one_hot': (0, 7), 'level_one_hot': (7, 10), 'side_one_hot': (10, 12),
    'enabled': (12, 13), 'authority': (13, 14), 'distance': (14, 15),
    'desired_clearance': (15, 16), 'clearance_error': (16, 17),
    'swing_xy_error': (17, 19), 'stance_xy_error': (19, 21), 'com_xy_error': (21, 23),
    'stance_share_target': (23, 24), 'episode_age': (24, 25), 'wait_remaining': (25, 26),
    'time_remaining': (26, 27), 'phase_age': (27, 28), 'dwell_fraction': (28, 29),
    'previous_quality': (29, 30), 'current_quality': (30, 31), 'success_latched': (31, 32),
    'transfer_completed': (32, 33), 'lift_completed': (33, 34),
    'place_completed': (34, 35), 'land_completed': (35, 36), 'maximum_stance_drift': (36, 37),
}


@dataclass(frozen=True)
class StepConfig:
    episode_seconds: float = 12.
    gamma: float = .995
    transfer_m: float = .04
    lift_m: float = .03
    step_m: float = .04
    transfer_tolerance_m: float = .015
    transfer_min_progress_m: float = .03
    support_com_tolerance_m: float = .035
    placement_tolerance_m: float = .01
    minimum_placement_m: float = .03
    stance_drift_m: float = .02
    settling_foot_tolerance_m: float = .02
    settling_com_tolerance_m: float = .025
    clearance_m: float = .015
    landing_clearance_m: float = .008
    support_n: float = 30.
    unload_n: float = 5.
    transfer_stance_share: float = .65
    min_pelvis_height_m: float = .65
    max_pelvis_tilt_degrees: float = 25.
    transfer_dwell_seconds: float = .1
    lift_dwell_seconds: float = .05
    place_dwell_seconds: float = .05
    landing_dwell_seconds: float = .1
    settling_dwell_seconds: float = 2.
    transition_bonus: float = 1.
    success_bonus: float = 5.
    active_step_cost: float = .01
    physical_failure_reward: float = -5.
    timeout_reward: float = -2.

    def __post_init__(self):
        assert all(math.isfinite(value) for value in asdict(self).values())
        assert self.settling_dwell_seconds < self.episode_seconds <= 120.
        assert 0 < self.gamma <= 1 and 0 < self.unload_n < self.support_n
        assert .5 < self.transfer_stance_share < 1 and 0 < self.max_pelvis_tilt_degrees < 90
        assert 0 < self.transfer_min_progress_m <= self.transfer_m
        assert 0 < self.clearance_m <= self.lift_m
        assert 0 < self.minimum_placement_m <= self.step_m
        assert 0 < self.landing_clearance_m < self.clearance_m
        for name, value in asdict(self).items():
            if name not in ('physical_failure_reward', 'timeout_reward', 'landing_clearance_m'):
                assert value > 0, name
        assert self.physical_failure_reward < 0 and self.timeout_reward < 0


class StepCurriculum:
    """Finite-horizon curriculum with observable command, phase and progress state.

    Levels request transfer/return, lift/replant, or a signed short step. Phase
    rewards use gamma*quality_after-quality_before, a time cost, and one-time
    measured completion bonuses. Retargeting starts a new potential baseline;
    the target change itself earns no reward. This shaping does not guarantee
    that a particular learned policy will prefer or complete the skill.

    Success is latched for diagnostics but never ends an episode early. Clean
    completion also requires two final seconds of valid settling. Every terminal
    transition has zero bootstrap, including a clean finite-horizon deadline.
    """

    def __init__(self, count, dt, device, config=StepConfig()):
        assert isinstance(count, int) and count > 0 and math.isfinite(dt) and 0 < dt <= .05
        self.count, self.dt, self.device, self.config = count, float(dt), torch.device(device), config
        self.episode_intervals = math.ceil(config.episode_seconds/self.dt)
        def zeros(*shape, dtype=torch.float32):
            return torch.zeros((count, *shape), dtype=dtype, device=self.device)
        self.phase = zeros(dtype=torch.long)
        self.level = zeros(dtype=torch.long)
        self.side = zeros(dtype=torch.long)
        self.enabled = zeros(dtype=torch.bool)
        self.distance = zeros()
        self.onset_seconds = zeros()
        self.intervals = zeros(dtype=torch.long)
        self.phase_intervals = zeros(dtype=torch.long)
        self.dwell_intervals = zeros(5, dtype=torch.long)
        self.age = zeros()
        self.phase_age = zeros()
        self.dwell = zeros(5)
        self.completed = zeros(4, dtype=torch.bool)
        self.success_latched = zeros(dtype=torch.bool)
        self.failed_latched = zeros(dtype=torch.bool)
        self.physical_failed_latched = zeros(dtype=torch.bool)
        self.timeout_failed_latched = zeros(dtype=torch.bool)
        self.terminal = zeros(dtype=torch.bool)
        self.initial_com = zeros(3)
        self.initial_feet = zeros(2, 3)
        self.initial_pelvis_height = zeros()
        self.transfer_direction = zeros(2)
        self.transfer_target = zeros(2)
        self.maximum_stance_drift = zeros()
        self.previous_quality = zeros()
        self.last_quality = zeros()
        self.initialized = zeros(dtype=torch.bool)
        self.world = torch.arange(count, device=self.device)
        required_dwell = (config.transfer_dwell_seconds, config.lift_dwell_seconds,
            config.place_dwell_seconds, config.landing_dwell_seconds, config.settling_dwell_seconds)
        self.required_dwell = torch.tensor(required_dwell, device=self.device)
        self.required_dwell_intervals = torch.tensor([math.ceil(value/self.dt) for value in required_dwell],
            dtype=torch.long, device=self.device)
        self.measurement = {key: zeros(*shape) for key, shape in self._float_shapes().items()}
        self.measurement.update(finite=zeros(dtype=torch.bool), stand=zeros(dtype=torch.bool))

    @staticmethod
    def _float_shapes():
        return dict(com=(3,), velocity=(3,), feet=(2, 3), clearance=(2,), loads=(2,),
                    pelvis_height=(), pelvis_up=(), angular=(3,))

    def _measure(self, measurement):
        result = {}
        finite = torch.as_tensor(measurement['finite'], device=self.device, dtype=torch.bool).clone()
        assert finite.shape == (self.count,)
        for key, shape in self._float_shapes().items():
            value = torch.as_tensor(measurement[key], device=self.device, dtype=torch.float32)
            assert value.shape == (self.count, *shape), key
            finite &= torch.isfinite(value).reshape(self.count, -1).all(-1)
            result[key] = torch.nan_to_num(value, nan=0., posinf=0., neginf=0.)
        result['finite'] = finite
        result['stand'] = torch.as_tensor(measurement['stand'], device=self.device, dtype=torch.bool)
        assert result['stand'].shape == (self.count,)
        return result

    def _command(self, value, ids, dtype):
        raw = torch.as_tensor(value, device=self.device)
        assert torch.isfinite(raw).all()
        if dtype == torch.long:
            assert torch.equal(raw, raw.round()), 'Command must be integral'
        if raw.ndim == 0:
            raw = raw.expand(len(ids))
        elif raw.shape == (self.count,):
            raw = raw[ids]
        else:
            assert raw.shape == (len(ids),)
        return raw.to(dtype=dtype)

    @torch.no_grad()
    def reset(self, ids, measurement, side, distance, onset_seconds, enabled, level):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        assert ids.ndim == 1 and len(ids) and len(ids.unique()) == len(ids)
        assert ((ids >= 0) & (ids < self.count)).all()
        measured = self._measure(measurement)
        assert measured['finite'][ids].all()
        self.side[ids] = self._command(side, ids, torch.long)
        self.level[ids] = self._command(level, ids, torch.long)
        self.enabled[ids] = self._command(enabled, ids, torch.bool)
        self.distance[ids] = self._command(distance, ids, torch.float32)
        self.onset_seconds[ids] = self._command(onset_seconds, ids, torch.float32)
        assert ((self.side[ids] >= 0) & (self.side[ids] <= 1)).all()
        assert ((self.level[ids] >= 0) & (self.level[ids] <= 2)).all()
        assert ((self.onset_seconds[ids] >= 0) & (self.onset_seconds[ids] < self.config.episode_seconds)).all()
        step = self.enabled[ids] & (self.level[ids] == 2)
        assert torch.allclose(self.distance[ids][step].abs(), torch.full_like(self.distance[ids][step], self.config.step_m), atol=1e-6, rtol=0)
        assert (self.distance[ids][~step].abs() <= 1e-6).all()
        for key, value in measured.items():
            self.measurement[key][ids] = value[ids]
        self.initial_com[ids] = measured['com'][ids]
        self.initial_feet[ids] = measured['feet'][ids]
        self.initial_pelvis_height[ids] = measured['pelvis_height'][ids]
        direction = measured['feet'][ids, 1-self.side[ids], :2]-measured['com'][ids, :2]
        length = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        assert (length[self.enabled[ids]] > 1e-6).all(), 'Transfer direction must be defined'
        self.transfer_direction[ids] = direction/length.clamp_min(1e-6)
        self.transfer_target[ids] = measured['com'][ids, :2]+self.config.transfer_m*self.transfer_direction[ids]
        for value in (self.intervals, self.phase_intervals, self.dwell_intervals,
                      self.age, self.phase_age, self.dwell, self.completed, self.success_latched,
                      self.failed_latched, self.physical_failed_latched, self.timeout_failed_latched,
                      self.terminal, self.maximum_stance_drift, self.previous_quality, self.last_quality):
            value[ids] = 0
        self.phase[ids] = torch.where(self.enabled[ids],
            torch.where(self.onset_seconds[ids] > 0, int(Phase.WAIT), int(Phase.TRANSFER)), int(Phase.QUIET))
        self.initialized[ids] = True
        quality = self._metrics(self.measurement)['quality']
        self.previous_quality[ids] = quality[ids]
        self.last_quality[ids] = quality[ids]

    def _metrics(self, m):
        c, w, side = self.config, self.world, self.side
        opposite = 1-side
        swing_xy, stance_xy = m['feet'][w, side, :2], m['feet'][w, opposite, :2]
        initial_swing, initial_stance = self.initial_feet[w, side, :2], self.initial_feet[w, opposite, :2]
        end_xy = initial_swing.clone()
        end_xy[:, 0] += self.distance
        swing_target = torch.where((self.phase >= int(Phase.PLACE))[:, None], end_xy, initial_swing)
        lifted_phase = (self.phase == int(Phase.LIFT)) | (self.phase == int(Phase.PLACE))
        desired_clearance = lifted_phase.float()*c.lift_m
        settle_com = self.initial_com[:, :2].clone()
        settle_com[:, 0] += .5*self.distance
        com_target = torch.where((self.phase == int(Phase.TRANSFER))[:, None], self.transfer_target,
            torch.where(lifted_phase[:, None], initial_stance, settle_com))
        swing_load, stance_load = m['loads'][w, side], m['loads'][w, opposite]
        clearance = m['clearance'][w, side]
        stance_share = stance_load/m['loads'].sum(-1).clamp_min(1.)
        stance_drift = torch.linalg.vector_norm(stance_xy-initial_stance, dim=-1)
        transfer_error = torch.linalg.vector_norm(m['com'][:, :2]-self.transfer_target, dim=-1)
        transfer_progress = ((m['com'][:, :2]-self.initial_com[:, :2])*self.transfer_direction).sum(-1)
        support_com_error = torch.linalg.vector_norm(m['com'][:, :2]-initial_stance, dim=-1)
        placement_error = torch.linalg.vector_norm(swing_xy-end_xy, dim=-1)
        world_placement = torch.linalg.vector_norm(swing_xy-initial_swing, dim=-1)
        relative_placement = torch.linalg.vector_norm((swing_xy-stance_xy)-(initial_swing-initial_stance), dim=-1)
        settle_com_error = torch.linalg.vector_norm(m['com'][:, :2]-settle_com, dim=-1)
        safe = m['finite'] & (m['pelvis_height'] >= c.min_pelvis_height_m) \
            & (m['pelvis_up'] >= math.cos(math.radians(c.max_pelvis_tilt_degrees)))
        supported = safe & (stance_load >= c.support_n) & (self.maximum_stance_drift <= c.stance_drift_m) \
            & (stance_drift <= c.stance_drift_m)
        lifted = supported & (swing_load < c.unload_n) & (clearance >= c.clearance_m) \
            & (support_com_error <= c.support_com_tolerance_m)
        translated = (world_placement >= c.minimum_placement_m) & (relative_placement >= c.minimum_placement_m)
        transfer_gate = supported & (stance_share >= c.transfer_stance_share) \
            & (transfer_error <= c.transfer_tolerance_m) & (transfer_progress >= c.transfer_min_progress_m)
        place_gate = lifted & self.completed[:, 1] & translated & (placement_error <= c.placement_tolerance_m)
        land_gate = supported & self.completed[:, 1] & (swing_load >= c.support_n) \
            & (clearance <= c.landing_clearance_m) & (placement_error <= c.placement_tolerance_m) \
            & ((self.level < 2) | (self.completed[:, 2] & translated))
        settle_gate = supported & m['stand'] & (swing_load >= c.support_n) \
            & (placement_error <= c.settling_foot_tolerance_m) & (settle_com_error <= c.settling_com_tolerance_m) \
            & self.completed[:, 0] & ((self.level == 0) | self.completed[:, 3]) \
            & ((self.level < 2) | translated)
        gates = torch.stack((transfer_gate, lifted, place_gate, land_gate, settle_gate), -1)
        posture = torch.exp(-((m['pelvis_height']-self.initial_pelvis_height)/.15).square()
                            -((1-m['pelvis_up'].clamp(-1, 1))/.2).square())
        support = (stance_load/c.support_n).clamp(0, 1)*safe.float()
        stable = (1-(stance_drift/c.stance_drift_m).square()).clamp(0, 1)
        share_quality = ((stance_share-.5)/(c.transfer_stance_share-.5)).clamp(0, 1)
        unload = (1-swing_load/c.unload_n).clamp(0, 1)
        lift_quality = (clearance/c.lift_m).clamp(0, 1)*torch.exp(-((clearance-c.lift_m).clamp_min(0)/c.lift_m).square())
        lifted_quality = unload*lift_quality
        com_quality = torch.exp(-(support_com_error/c.support_com_tolerance_m).square())
        placement_quality = torch.exp(-(placement_error/c.placement_tolerance_m).square())
        bilateral = (m['loads'].amin(-1)/c.support_n).clamp(0, 1)
        qualities = torch.stack((
            .6*torch.exp(-(transfer_error/.025).square())+.4*share_quality,
            .4*com_quality+.6*lifted_quality,
            .3*com_quality+.25*lifted_quality+.45*lifted_quality*placement_quality,
            self.completed[:, 1].float()*(.4*bilateral+.3*placement_quality
                +.3*torch.exp(-(clearance/c.landing_clearance_m).square())),
            .4*m['stand'].float()+.3*torch.exp(-(settle_com_error/c.settling_com_tolerance_m).square())
                +.3*placement_quality,
        ), -1)*posture[:, None]*support[:, None]*stable[:, None]
        index = (self.phase-2).clamp(0, 4)
        quality = qualities[w, index].clamp(0, 1)
        quality = torch.where(self.phase >= int(Phase.TRANSFER), quality, 0.)
        return dict(gates=gates, quality=quality, desired_clearance=desired_clearance,
            swing_target=swing_target, stance_target=initial_stance, com_target=com_target, end_target=end_xy,
            clearance=clearance, stance_share=stance_share, stance_drift=stance_drift,
            transfer_error=transfer_error, transfer_progress=transfer_progress,
            support_com_error=support_com_error, placement_error=placement_error,
            world_placement=world_placement, relative_placement=relative_placement, settle_com_error=settle_com_error)

    @torch.no_grad()
    def observations(self):
        m, c = self.measurement, self.config
        metrics = self._metrics(m)
        goal = torch.zeros((self.count, GOAL_WIDTH), device=self.device)
        goal[:, :7] = torch.nn.functional.one_hot(self.phase, 7)
        goal[:, 7:10] = torch.nn.functional.one_hot(self.level, 3)
        goal[:, 10:12] = torch.nn.functional.one_hot(self.side, 2)
        authority = self.enabled & (self.phase >= int(Phase.TRANSFER))
        goal[:, 12], goal[:, 13] = self.enabled, authority
        goal[:, 14] = self.distance/c.step_m
        goal[:, 15] = metrics['desired_clearance']/c.lift_m
        goal[:, 16] = (metrics['desired_clearance']-metrics['clearance'])/c.lift_m
        goal[:, 17:19] = (metrics['swing_target']-m['feet'][self.world, self.side, :2])/c.step_m
        goal[:, 19:21] = (metrics['stance_target']-m['feet'][self.world, 1-self.side, :2])/c.stance_drift_m
        goal[:, 21:23] = (metrics['com_target']-m['com'][:, :2])/c.transfer_m
        goal[:, 23] = torch.where(self.phase == int(Phase.TRANSFER), c.transfer_stance_share,
            torch.where((self.phase == int(Phase.LIFT)) | (self.phase == int(Phase.PLACE)), 1., .5))
        goal[:, 24] = self.age/c.episode_seconds
        goal[:, 25] = (self.onset_seconds-self.age).clamp_min(0)/c.episode_seconds
        goal[:, 26] = (c.episode_seconds-self.age).clamp_min(0)/c.episode_seconds
        goal[:, 27] = self.phase_age/c.episode_seconds
        index = (self.phase-2).clamp(0, 4)
        goal[:, 28] = torch.where(self.phase >= int(Phase.TRANSFER),
            self.dwell[self.world, index]/self.required_dwell[index], 0.).clamp(0, 1)
        goal[:, 29], goal[:, 30], goal[:, 31] = self.last_quality, self.previous_quality, self.success_latched
        goal[:, 32:36] = self.completed
        goal[:, 36] = self.maximum_stance_drift/c.stance_drift_m
        assert torch.isfinite(goal).all()
        return goal

    @torch.no_grad()
    def advance(self, measurement, failed, active):
        assert self.initialized.all()
        failed = torch.as_tensor(failed, device=self.device, dtype=torch.bool)
        active = torch.as_tensor(active, device=self.device, dtype=torch.bool)
        assert failed.shape == active.shape == (self.count,)
        live = active & ~self.terminal
        goals_before = self.observations()
        phase_before, dwell_before = self.phase.clone(), self.dwell.clone()
        previous_quality = self.previous_quality.clone()
        measured = self._measure(measurement)
        for key, value in measured.items():
            self.measurement[key][live] = value[live]
        self.intervals[live] += 1
        self.age[live] = self.intervals[live]*self.dt
        self.phase_intervals[live] += 1
        self.phase_age[live] = self.phase_intervals[live]*self.dt
        physical_failed = live & (failed | ~measured['finite'])
        working = live & ~physical_failed & (phase_before >= int(Phase.TRANSFER))
        current_drift = torch.linalg.vector_norm(self.measurement['feet'][self.world, 1-self.side, :2]
            -self.initial_feet[self.world, 1-self.side, :2], dim=-1)
        self.maximum_stance_drift[working] = torch.maximum(self.maximum_stance_drift[working], current_drift[working])
        metrics = self._metrics(self.measurement)
        measured_quality = metrics['quality']
        potential_reward = torch.where(working, self.config.gamma*measured_quality-previous_quality, 0.)
        time_reward = -self.config.active_step_cost*working.float()
        completion_reward = torch.zeros(self.count, device=self.device)
        transition = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        for index, phase in enumerate((Phase.TRANSFER, Phase.LIFT, Phase.PLACE, Phase.LAND, Phase.SETTLE)):
            selected = working & (phase_before == int(phase))
            self.dwell_intervals[selected, index] = torch.where(metrics['gates'][selected, index],
                self.dwell_intervals[selected, index]+1, 0)
            self.dwell[selected, index] = self.dwell_intervals[selected, index]*self.dt
            complete = selected & (self.dwell_intervals[:, index] >= self.required_dwell_intervals[index])
            if phase == Phase.SETTLE:
                completion_reward[complete & ~self.success_latched] = self.config.success_bonus
                self.success_latched |= complete
            else:
                self.completed[complete, index] = True
                completion_reward[complete] = self.config.transition_bonus
                transition |= complete
                if phase == Phase.TRANSFER:
                    self.phase[complete] = torch.where(self.level[complete] == 0, int(Phase.SETTLE), int(Phase.LIFT))
                elif phase == Phase.LIFT:
                    self.phase[complete] = torch.where(self.level[complete] == 1, int(Phase.LAND), int(Phase.PLACE))
                else:
                    self.phase[complete] = int(phase)+1
        begin = live & ~physical_failed & (phase_before == int(Phase.WAIT)) & (self.age >= self.onset_seconds)
        self.phase[begin] = int(Phase.TRANSFER)
        self.phase_intervals[transition | begin] = 0
        self.phase_age[transition | begin] = 0.
        after_metrics = self._metrics(self.measurement)
        self.last_quality[live] = previous_quality[live]
        self.previous_quality[live] = after_metrics['quality'][live]
        # The phase-entry score is the next potential baseline, without a retargeting bonus.
        settled_now = self.success_latched & (self.phase == int(Phase.SETTLE)) \
            & after_metrics['gates'][:, 4] & (self.dwell_intervals[:, 4] >= self.required_dwell_intervals[4])
        deadline = live & (self.intervals >= self.episode_intervals)
        task_timeout = deadline & self.enabled & ~settled_now & ~physical_failed
        self.physical_failed_latched |= physical_failed
        self.timeout_failed_latched |= task_timeout
        self.failed_latched |= physical_failed | task_timeout
        self.terminal |= physical_failed | deadline
        reward = potential_reward+time_reward+completion_reward
        failure_reward = torch.zeros_like(reward)
        failure_reward[physical_failed] = self.config.physical_failure_reward
        failure_reward[task_timeout] = self.config.timeout_reward
        reward = torch.where(physical_failed | task_timeout, failure_reward, reward)
        task_success = self.enabled & settled_now & ~self.failed_latched
        assert torch.isfinite(reward).all()
        info = dict(phase_before=phase_before, phase_after=self.phase.clone(), phase=self.phase.clone(),
            dwell_before=dwell_before, dwell=self.dwell.clone(), phase_age=self.phase_age.clone(), age=self.age.clone(),
            intervals=self.intervals.clone(), phase_intervals=self.phase_intervals.clone(),
            dwell_intervals=self.dwell_intervals.clone(), level=self.level.clone(), side=self.side.clone(),
            enabled=self.enabled.clone(), distance=self.distance.clone(), onset_seconds=self.onset_seconds.clone(),
            task_success=task_success, success=task_success.clone(), success_latched=self.success_latched.clone(),
            failed=self.failed_latched.clone(), physical_failed=self.physical_failed_latched.clone(),
            task_timeout=self.timeout_failed_latched.clone(), terminal=self.terminal.clone(),
            bootstrap=(~self.terminal).float(), active=live, transfer_gate=metrics['gates'][:, 0],
            lift_gate=metrics['gates'][:, 1], place_gate=metrics['gates'][:, 2], land_gate=metrics['gates'][:, 3],
            settle_gate=metrics['gates'][:, 4], transfer_completed=self.completed[:, 0].clone(),
            lift_completed=self.completed[:, 1].clone(), place_completed=self.completed[:, 2].clone(),
            land_completed=self.completed[:, 3].clone(), initial_feet=self.initial_feet.clone(),
            initial_com=self.initial_com.clone(), initial_pelvis_height=self.initial_pelvis_height.clone(),
            transfer_target=self.transfer_target.clone(),
            end_target=after_metrics['end_target'], com_target=after_metrics['com_target'],
            maximum_stance_drift=self.maximum_stance_drift.clone(),
            world_placement=metrics['world_placement'], relative_placement=metrics['relative_placement'],
            transfer_progress=metrics['transfer_progress'], transfer_error=metrics['transfer_error'],
            support_com_error=metrics['support_com_error'], placement_error=metrics['placement_error'],
            settle_com_error=metrics['settle_com_error'], quality_before=previous_quality,
            quality_measured=measured_quality, quality_next=self.previous_quality.clone(),
            reward_potential=potential_reward, reward_time=time_reward, reward_completion=completion_reward,
            reward_failure=failure_reward, reward_overridden=physical_failed | task_timeout,
            goalfeatures_before=goals_before, goalfeatures_after=self.observations())
        return reward, self.terminal.clone(), info

    def contract(self):
        return dict(schema='measured_independent_step_curriculum_v1', config=asdict(self.config),
            phases={phase.name: int(phase) for phase in Phase}, goal_width=GOAL_WIDTH,
            goal_layout=GOAL_LAYOUT, authority_index=AUTHORITY_INDEX,
            episode_intervals=self.episode_intervals, episode_clock='active integer intervals times dt',
            required_dwell_intervals=self.required_dwell_intervals.tolist(),
            quality_observations='Previous decision potential baseline, then current phase potential baseline',
            levels=['transfer-return', 'lift-replant', 'short-step'], motor_commands=False,
            all_terminals_zero_bootstrap=True, early_success_termination=False,
            final_settling_required=True, one_phase_transition_per_interval=True,
            phase_target_change_reward=False, inactive_state_frozen=True,
            target_frame='World XY relative to reset feet and COM; distance is signed world X',
            reward='Per-phase gamma*quality_after-quality_before minus active-step cost; measured '
                   'transition/final bonuses; finite failure overrides. No optimality guarantee.')
