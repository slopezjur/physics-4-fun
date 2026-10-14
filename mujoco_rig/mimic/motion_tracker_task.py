"""Bilateral motion tracking rehearsal, independent of projectile recovery tasks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from envs.base_env import EnvMode

from .balance_observation import CaptureObservationMixin
from .continuous_stand import CONTACT_SEATING_M, StationaryReference
from .motion_tracker import LOOKAHEAD_SECONDS, OBS_WIDTH, SCHEMA, motion_condition
from .reference import StandReference
from .stand_task import StandTask


def root_accuracy_reward(error, mode, sigma_m, accuracy_scale_m):
    if mode == 'exponential':
        return torch.exp(-(error / sigma_m).square().sum(-1))
    if mode == 'inverse_huber':
        scaled = error / accuracy_scale_m
        penalty = torch.nn.functional.smooth_l1_loss(scaled, torch.zeros_like(scaled),
                                                     reduction='none').sum(-1)
        # Positive, with a gradient outside the accuracy band; no fall incentive
        # from replacing the living reward with an unbounded negative cost.
        return 1. / (1. + penalty)
    raise ValueError('Unknown root accuracy reward')


class MotionBank:
    def __init__(self, paths, rig, device):
        from .track_transfer import CalibrationReference
        from .retarget_steps import validate_motion
        from .weight_shift import PHYSICAL_CHECKS
        self.references = []
        self.paths = [Path(p) for p in paths]
        for path in self.paths:
            ref = (CalibrationReference(path, rig, device) if path.is_dir()
                   else StandReference(path, rig, device))
            if not path.is_dir():
                metadata = json.loads(path.with_suffix('.json').read_text())
                if metadata.get('schema') != 'mimic_weight_shift_reference_v1':
                    raise ValueError('Tracker foundation requires a physically checked weight-shift reference')
                with np.load(path) as arrays:
                    checks, _ = validate_motion(rig, arrays['qpos'], arrays['qvel'], arrays['foot_targets'],
                        arrays['contact'], float(arrays['dt']), arrays['sole_contact'])
                if not all(checks[k] for k in PHYSICAL_CHECKS):
                    raise ValueError('Weight-shift physical admission failed')
            self.references.append(ref)
        if not self.references:
            raise ValueError('Require motion references')
        self.duration = max(r.duration for r in self.references)
        self.sha256 = hashlib.sha256(''.join(r.sha256 for r in self.references).encode()).hexdigest()


class MotionTrackingTask(CaptureObservationMixin, StandTask):
    def __init__(self, rig, stationary, motions, num_envs=8, device='cpu', *, quiet_fraction=.25,
                 training_seconds=30., root_tracking_sigma_m=.03, root_tracking_weight=.25,
                 fixed_fraction=0., fixed_pose_eval=False, root_reward_mode='exponential',
                 root_accuracy_scale_m=.005, **kwargs):
        if not np.isfinite(root_tracking_sigma_m) or root_tracking_sigma_m <= 0 or not 0 < root_tracking_weight < .8:
            raise ValueError('Require a positive root tolerance and valid reward mixture')
        self.root_tracking_sigma_m = root_tracking_sigma_m
        self.root_tracking_weight = root_tracking_weight
        if (not 0 <= fixed_fraction <= 1. - quiet_fraction or root_reward_mode not in
                ('exponential', 'inverse_huber') or not np.isfinite(root_accuracy_scale_m)
                or root_accuracy_scale_m <= 0.):
            raise ValueError('Invalid fixed-pose curriculum or root accuracy reward')
        self.fixed_worlds = round(num_envs * fixed_fraction)
        self.fixed_pose_eval = fixed_pose_eval
        self.held_phase = torch.zeros(num_envs, device=device)
        self.root_reward_mode = root_reward_mode
        self.root_accuracy_scale_m = root_accuracy_scale_m
        self.stationary = StationaryReference(stationary, rig, device)
        self.bank = MotionBank(motions, rig, device)
        self.quiet_worlds = round(num_envs * quiet_fraction)
        self.training_seconds = training_seconds
        self.motion_ids = torch.arange(num_envs, device=device) % len(self.bank.references)
        super().__init__(rig, motions[0], num_envs, device, control_mode='target_pd',
            contact_mode='support_v2', reference_factory=lambda *_: self.bank, **kwargs)
        self.episode_seconds = 30.

    def quiet_mask(self, ids=None):
        ids = self.ids if ids is None else ids
        return (ids < self.quiet_worlds) & (self._mode == EnvMode.TRAIN)

    def stationary_pose(self, ids):
        cases = torch.where(self.quiet_mask(ids), ids % 8, 0)
        return self.stationary.sample(torch.zeros_like(ids, dtype=torch.float32), cases)

    def fixed_mask(self, ids=None):
        ids = self.ids if ids is None else ids
        if self._mode == EnvMode.TRAIN:
            return (ids >= self.quiet_worlds) & (ids < self.quiet_worlds + self.fixed_worlds)
        return torch.full_like(ids, self.fixed_pose_eval, dtype=torch.bool)

    def sample_reference(self, time, env_ids=None):
        ids = self.ids if env_ids is None else env_ids
        q0, v0 = self.stationary_pose(ids)
        shape = (len(ids),) + (1,) * (time.ndim - 1)
        q = q0.reshape(*shape, -1).expand(*time.shape, q0.shape[-1]).clone()
        v = v0.reshape(*shape, -1).expand(*time.shape, v0.shape[-1]).clone()
        for index, ref in enumerate(self.bank.references):
            selected = (self.motion_ids[ids] == index) & ~self.quiet_mask(ids)
            fixed = self.fixed_mask(ids).reshape(*shape)
            phase = self.held_phase[ids].reshape(*shape).expand_as(time)
            sample_time = torch.where(fixed, phase, time)
            target, velocity = ref.sample(sample_time)
            # Hold the last pose indefinitely, with zero target velocity.
            velocity = torch.where((fixed | (sample_time >= ref.duration))[..., None], 0., velocity)
            mask = selected.reshape(*shape, 1)
            q, v = torch.where(mask, target, q), torch.where(mask, velocity, v)
        return q, v

    def reference_phase(self, time):
        return torch.zeros_like(time)

    def episode_duration(self, env_ids=None):
        ids = self.ids if env_ids is None else env_ids
        if self._mode == EnvMode.TRAIN:
            return torch.full((len(ids),), self.training_seconds, device=self.device)
        return self.episode_seconds

    def reset(self, env_ids=None):
        ids = self.ids if env_ids is None else env_ids
        if len(ids):
            self.steps[ids] = 0
            if self._mode == EnvMode.TRAIN:
                self.motion_ids[ids] = torch.randint(len(self.bank.references), (len(ids),),
                    generator=self.rng, device=self.device)
                durations = torch.tensor([r.duration for r in self.bank.references], device=self.device)
                # Retain phase coverage while long episodes rehearse terminal holds.
                extent = (durations[self.motion_ids[ids]] - min(4., self.training_seconds)).clamp_min(0.)
                self.offset[ids] = torch.rand(len(ids), generator=self.rng, device=self.device) * extent
                beginning = torch.rand(len(ids), generator=self.rng, device=self.device) < .5
                self.offset[ids] = torch.where(beginning | self.quiet_mask(ids), 0., self.offset[ids])
                if self.fixed_worlds:
                    held = self.fixed_mask(ids)
                    held_ids = ids[held]
                    self.held_phase[held_ids] = torch.rand(len(held_ids), generator=self.rng,
                        device=self.device) * durations[self.motion_ids[held_ids]]
                    self.offset[held_ids] = 0.
            else:
                self.motion_ids[ids] = ids % len(self.bank.references)
                durations = torch.tensor([r.duration for r in self.bank.references], device=self.device)
                self.offset[ids] = (ids / max(1, self.get_num_envs() - 1)) * (durations[self.motion_ids[ids]] - 4.).clamp_min(0.)
                if self.fixed_pose_eval:
                    self.held_phase[ids] = ids / max(1, self.get_num_envs() - 1) * durations[self.motion_ids[ids]]
                    self.offset[ids] = 0.
            q, v = self.sample_reference(self.offset[ids], ids)
            # Tangent float32 soles can reset with zero contacts and false capture risk.
            # Use the standing foundation's admitted contact seating only at reset.
            q[~self.quiet_mask(ids), 2] -= CONTACT_SEATING_M
            self.engine.reset_envs(ids, q, v)
        return self.observations(), {}

    def observations(self):
        core = super().observations()
        stationary, _ = self.stationary_pose(self.ids)
        core[:, 361] = 0.
        time = self.offset + self.steps * self.dt
        q, v = self.sample_reference(time[:, None] + time.new_tensor(LOOKAHEAD_SECONDS))
        command = motion_condition(q, v, stationary, ~self.quiet_mask())
        return torch.cat((core, command), -1)

    def reward(self):
        base = self.reference_reward()
        target, _ = self.sample_reference(self.offset + self.steps * self.dt)
        target_pos, _ = self.mapper.from_native_qpos(target)
        positions = self.engine.get_body_pos(0)
        root_score = root_accuracy_reward(positions[:, 0] - target[:, :3], self.root_reward_mode,
                                         self.root_tracking_sigma_m, self.root_accuracy_scale_m)
        foot_error = ((positions[:, self.feet] - target_pos[:, self.feet]) / .04).square().sum((-1, -2))
        tracking = ((.8 - self.root_tracking_weight) * base
            + self.root_tracking_weight * root_score + .2 * torch.exp(-foot_error))
        return torch.where(self.quiet_mask(), base, tracking)

    def observation_contract(self):
        result = super().observation_contract()
        result.update(observation_contract=SCHEMA, num_obs=OBS_WIDTH,
            stationary_reference_sha256=self.stationary.sha256,
            motion_reference_hashes=[r.sha256 for r in self.bank.references],
            motion_reset_contact_seating_m=CONTACT_SEATING_M,
            conditioning=dict(lookahead_seconds=LOOKAHEAD_SECONDS, phase=False, selected_foot=False,
                source_future_reference='Same moving goals used by the physical target controller; stationary when quiet',
                explicit_motion_request=True, terminal_reference='Fixed last pose, zero velocity, unlimited duration'),
            deployment_status='Offline tracking pilot; requires a new Godot adapter before export')
        result['observation_layout'].append(dict(name='future_motion_pose_deltas_velocities_request',
            offset=363, width=OBS_WIDTH - 363))
        return result
