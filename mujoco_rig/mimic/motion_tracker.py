"""Explicit motion conditioning around an immutable stationary recovery actor."""
import copy

import torch

CORE_WIDTH = 363
LOOKAHEAD_SECONDS = (.1, .3, .6)
CONDITION_WIDTH = 3 * (48 + 45) + 1
OBS_WIDTH = CORE_WIDTH + CONDITION_WIDTH
SCHEMA = 'mimic_motion_conditioned_tracker_v1'
ACTOR_NAME = 'mimic_motion_feedback_tracker_v2'
FEEDBACK_WIDTH = 122 + 48


def encode_pose(q):
    from util.torch_util import quat_to_tan_norm
    return torch.cat((q[..., :3], quat_to_tan_norm(q[..., [4, 5, 6, 3]]), q[..., 7:]), -1)


def motion_condition(q, v, stationary, requested):
    """Physical future targets, with fixed units; no reference phase or chosen foot."""
    scale = q.new_tensor([.1] * 3 + [1.] * 6 + [.25] * 39)
    velocity_scale = v.new_tensor([.5] * 3 + [2.] * 42)
    delta = (encode_pose(q) - encode_pose(stationary)[:, None]) / scale
    command = torch.cat((delta, v / velocity_scale), -1).flatten(1)
    command = command * requested[:, None]
    return torch.cat((command, requested[:, None].to(command.dtype)), -1)


class MotionTracker(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = copy.deepcopy(source).eval().requires_grad_(False)
        self.register_buffer('state_mean', source.obs_norm._mean[:95].detach().clone())
        self.register_buffer('state_std', source.obs_norm._std[:95].detach().clone().clamp_min(.05))
        self.correction = torch.nn.Sequential(torch.nn.Linear(95 + CONDITION_WIDTH, 128),
            torch.nn.SiLU(), torch.nn.Linear(128, 128), torch.nn.SiLU(), torch.nn.Linear(128, 30))
        torch.nn.init.zeros_(self.correction[-1].weight)
        torch.nn.init.zeros_(self.correction[-1].bias)
        self.correction.requires_grad_(False)
        self.register_buffer('queue_mean', source.obs_norm._mean[95:217].detach().clone())
        self.register_buffer('queue_std', source.obs_norm._std[95:217].detach().clone().clamp_min(.05))
        self.register_buffer('pose_error_scale', torch.tensor([.03] * 3 + [1.] * 6 + [.25] * 39))
        self.feedback = torch.nn.Sequential(torch.nn.Linear(FEEDBACK_WIDTH, 128),
            torch.nn.SiLU(), torch.nn.Linear(128, 30))
        torch.nn.init.zeros_(self.feedback[-1].weight)
        torch.nn.init.zeros_(self.feedback[-1].bias)

    def feedback_features(self, core):
        pending = core[..., 95:217].reshape(*core.shape[:-1], 2, 61)
        normalized = ((core[..., 95:217] - self.queue_mean) / self.queue_std).reshape_as(pending)
        valid = pending[..., 60:61]
        queue = torch.cat((normalized[..., :60].clamp(-10., 10.) * valid, valid), -1).flatten(-2)
        pose = torch.cat((core[..., :9], core[..., 15:54]), -1)
        # The nearest saved goal is one control step ahead; no additional clock input.
        error = ((core[..., 217:265] - pose) / self.pose_error_scale).clamp(-10., 10.)
        return torch.cat((queue, error), -1)

    def forward(self, observation):
        core = observation[..., :CORE_WIDTH]
        state = ((core[..., :95] - self.state_mean) / self.state_std).clamp(-10., 10.)
        command = observation[..., CORE_WIDTH:].clamp(-10., 10.)
        logits = self.correction(torch.cat((state, command), -1))
        residual = (logits + self.feedback(self.feedback_features(core))).tanh()
        return self.source(core) + observation[..., -1:] * residual


def migrate_feedback_checkpoint(saved, initialized):
    """Add only zero-output feedback state to a complete v1 tracker checkpoint."""
    prefix = '_model._actor_layers.'
    added = {prefix + name for name in ('queue_mean', 'queue_std', 'pose_error_scale',
        'feedback.0.weight', 'feedback.0.bias', 'feedback.2.weight', 'feedback.2.bias')}
    missing = initialized.keys() - saved.keys()
    if saved.keys() - initialized.keys() or missing not in (set(), added):
        raise ValueError('Unexpected tracker checkpoint keys during feedback migration')
    if any(saved[k].shape != initialized[k].shape or saved[k].dtype != initialized[k].dtype for k in saved):
        raise ValueError('Tracker checkpoint shape/dtype mismatch')
    result = {**initialized, **saved}
    if missing:
        for name in ('feedback.2.weight', 'feedback.2.bias'):
            if torch.count_nonzero(result[prefix + name]):
                raise ValueError('Feedback migration requires zero output initialization')
    return result, bool(missing)


class RawNormalizer(torch.nn.Module):
    """The source owns its frozen mapping; appended commands have explicit units."""
    def __init__(self):
        super().__init__()
        self.register_parameter('_count', torch.nn.Parameter(torch.tensor(1.), requires_grad=False))
        self.register_buffer('_mean', torch.zeros(OBS_WIDTH))
        self.register_buffer('_std', torch.ones(OBS_WIDTH))

    def normalize(self, observation):
        return observation

    def get_count(self):
        return self._count

    def get_mean(self):
        return self._mean

    def get_std(self):
        return self._std


def quiet_input(observation):
    return torch.cat((observation, observation.new_zeros(len(observation), CONDITION_WIDTH)), -1)
