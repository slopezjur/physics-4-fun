"""Scoped 33-motor recovery interventions; sealed baseline modules stay unchanged."""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import numpy as np
import torch

from .target_control33 import DelayedTargetControl33


def verify_control_trace(trace, contract):
    from .replicate_stand_representations import verify_control_trace as baseline_verify
    ranges = np.asarray(contract['corrective_step_target_ranges_rad'], dtype=np.float32)
    lower, upper = np.asarray(contract['target_lower_radians']), np.asarray(contract['target_upper_radians'])
    action = np.clip(trace['action'], -1, 1)
    target = np.clip(trace['reference_motor_q'] + ranges*action, lower, upper)
    if not np.allclose(trace['issued_entry'][..., :33], target, rtol=0, atol=2e-7):
        raise AssertionError('Issued targets violate the prospective per-joint range')
    # The baseline verifier checks delays, queues and every physics-substep torque.
    # Its fixed .25-rad target calculation receives an algebraic reference shift;
    # the real reference-to-target relation was independently checked above.
    shifted = dict(trace, reference_motor_q=trace['reference_motor_q']+(ranges-.25)*action)
    return baseline_verify(shifted, contract)


@contextmanager
def control_variant(leg_range):
    if leg_range not in (.25, .5):
        raise ValueError('Only preregistered leg ranges are supported')
    original_init, original_contract = DelayedTargetControl33.__init__, DelayedTargetControl33.contract

    def initialize(self, rig, num_envs, device):
        original_init(self, rig, num_envs, device)
        names = [rig.model.joint(int(j)).name for j in rig.model.actuator_trnid[rig.actuators, 0]]
        self.step_ranges = torch.tensor([leg_range if n.startswith(('Thigh_', 'Shin_', 'Foot_', 'Toe_'))
                                       else .25 for n in names], device=device, dtype=torch.float32)

    def targets(self, action):
        return (self.reference_q[:, self.q_indices] + self.step_ranges*action.clamp(-1, 1)).clamp(self.lower, self.upper)

    def contract(self):
        value = original_contract(self)
        value.update(corrective_step_target_ranges_rad=self.step_ranges.cpu().tolist(),
                     residual_radians=None, action_mode='reference_relative_target_pd_33_per_joint_range_v1')
        return value

    from . import evaluate_stand_ball_boundary as native, evaluate_stand_ball_gpu as gpu
    from . import evaluate_stand_sustained as sustained
    with ExitStack() as stack:
        stack.enter_context(patch.object(DelayedTargetControl33, '__init__', initialize))
        stack.enter_context(patch.object(DelayedTargetControl33, 'target_positions', targets))
        stack.enter_context(patch.object(DelayedTargetControl33, 'contract', contract))
        for module in (native, gpu, sustained):
            stack.enter_context(patch.object(module, 'verify_control_trace', verify_control_trace))
        yield


@contextmanager
def learning_variant(settings, recovery_reward):
    """Same onset curriculum in every arm, with a post-contact reward intervention."""
    from .run_chest_stronger_onset_learning import onset_curriculum
    from .capture_reward import capture_distance, rotate_xyzw
    from .step_reward import TransferRecoveryReward, swing_quality
    from . import stand_ball_task
    with onset_curriculum(settings, settings['varied_onsets']):
        base = stand_ball_task.StandBallAMPTask

        class StepTask(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.direction.zero_()
                self.direction[:, 0] = 1.
                self.step_hit = torch.zeros(64, dtype=torch.bool, device=self.device)
                self.step_reward = TransferRecoveryReward(imitation_weight=.05)
                model = self.rig.model
                self.sole_bodies = [model.body(f'{body}_{side}').id-1 for side in ('L', 'R') for body in ('Foot', 'Toe')]
                geoms = [model.geom(f'g_{body}_{side}').id for side in ('L', 'R') for body in ('Foot', 'Toe')]
                self.sole_centers = torch.tensor(model.geom_pos[geoms], dtype=torch.float32, device=self.device)
                self.sole_sizes = torch.tensor(model.geom_size[geoms], dtype=torch.float32, device=self.device)
                if not np.all(model.geom_quat[geoms] == [1, 0, 0, 0]):
                    raise ValueError('Reward geometry requires body-aligned foot boxes')

            def reset(self, env_ids=None):
                result = super().reset(env_ids)
                if hasattr(self, 'step_hit'):
                    self.step_hit[self.ids if env_ids is None else env_ids] = False
                return result

            def step(self, action):
                self.step_previous_com = self.center_of_mass(self.engine.get_body_pos(0), self.engine.get_body_rot(0)).clone()
                return super().step(action)

            def reward(self):
                imitation = super().reward()
                if not recovery_reward or not hasattr(self, 'step_hit'):
                    return imitation
                self.step_hit |= self.engine.normal_impulse > 0
                positions, rotations = self.engine.get_body_pos(0), self.engine.get_body_rot(0)
                com = self.center_of_mass(positions, rotations)
                velocity = (com-self.step_previous_com)/self.dt
                q = self.engine.native_qpos()
                loads = self.standing_measurements()['foot_loads_n']
                error = capture_distance(com, velocity, positions[:, self.feet], loads)
                rotation = rotations[:, self.sole_bodies]
                centers = positions[:, self.sole_bodies] + rotate_xyzw(rotation, self.sole_centers)
                extent = torch.zeros_like(centers[..., 2])
                for axis in range(3):
                    local = torch.zeros_like(self.sole_sizes)
                    local[:, axis] = self.sole_sizes[:, axis]
                    extent += rotate_xyzw(rotation, local)[..., 2].abs()
                soles = centers.clone()
                soles[..., 2] -= extent
                swing = swing_quality(com, velocity, soles.reshape(-1, 2, 2, 3), 0., loads,
                                      preparation_weight=.3)
                value = self.step_reward(imitation, q[:, 2], self._stationary_cache['height'],
                    1-2*(q[:, 4].square()+q[:, 5].square()), self.engine.get_root_vel(0),
                    self.engine.get_root_ang_vel(0), error, swing, loads)
                self._diagnostics['corrective_post_hit_fraction'] = self.step_hit.float().mean()
                self._diagnostics['corrective_swing_quality'] = swing.mean()
                return torch.where(self.step_hit, value, imitation)

            def observation_contract(self):
                value = super().observation_contract()
                value.update(corrective_recovery_reward=recovery_reward,
                    corrective_reward_contract=self.step_reward.contract() if recovery_reward else None,
                    training_impact_direction=[1, 0, 0])
                return value

        with patch.object(stand_ball_task, 'StandBallAMPTask', StepTask):
            yield
