"""Dense balance objective using measured COM and a conservative support proxy."""
from dataclasses import asdict, dataclass

import torch


def rotate_xyzw(rotation, vector):
    xyz = rotation[..., :3]
    cross = 2 * torch.linalg.cross(xyz, vector.expand_as(xyz), dim=-1)
    return vector + rotation[..., 3:] * cross + torch.linalg.cross(xyz, cross, dim=-1)


def capture_distance(com, velocity, feet, loads, gravity=9.81, sole_radius=.06):
    """Distance outside a capsule joining loaded feet; airborne feet add no support.

    This linear inverted-pendulum approximation shapes rewards, not admission.
    Native survival and uninterrupted settled contact remain the measured gates.
    """
    support = loads >= 5.
    a, b = feet[:, 0, :2], feet[:, 1, :2]
    a = torch.where(support[:, :1], a, b)
    b = torch.where(support[:, 1:], b, a)
    ground = feet[:, :, 2].amin(-1)
    omega_inverse = ((com[:, 2] - ground).clamp_min(.05) / gravity).sqrt()
    capture = com[:, :2] + velocity[:, :2] * omega_inverse[:, None]
    segment = b - a
    along = ((capture - a) * segment).sum(-1) / segment.square().sum(-1).clamp_min(1e-8)
    closest = a + along.clamp(0, 1)[:, None] * segment
    distance = (torch.linalg.vector_norm(capture - closest, dim=-1) - sole_radius).clamp_min(0)
    return torch.where(support.any(-1), distance, torch.ones_like(distance))


@dataclass(frozen=True)
class CaptureRecoveryReward:
    imitation_weight: float = .05
    capture_scale_m: float = .10

    def contract(self):
        return dict(schema="ball_capture_recovery_v3", **asdict(self),
                    activation="confirmed contact; quiet and pre-contact reward unchanged",
                    objective="upright posture, COM capture inside loaded support, low velocity",
                    support="0.06m capsule between loaded foot origins; either foot may step",
                    temporal_bonus="none; two-second sustained recovery is an evaluation gate")

    def __call__(self, imitation, height, target_height, up_z, velocity, angular_velocity,
                 capture_error):
        posture = torch.exp(-((height - target_height) / .15).square()
                            - ((1 - up_z.clamp(-1, 1)) / .2).square())
        capture = torch.exp(-(capture_error / self.capture_scale_m).square())
        settling = torch.exp(-(velocity / .5).square().sum(-1)
                             - (angular_velocity / 2.).square().sum(-1))
        recovery = posture * (.2 + .5 * capture + .3 * settling)
        return self.imitation_weight * imitation + (1 - self.imitation_weight) * recovery
