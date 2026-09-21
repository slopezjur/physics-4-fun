"""Perturb recovery objective shared by NumPy scoring and batched Torch training.

Reward the result of a step: grounded support and settling. Swing speed or an
airborne foot near the capture point never earns a placement bonus. Support is
a conservative foot-centre capsule proxy, not a measured contact-force polygon.
"""
from dataclasses import dataclass


REWARD_VERSION = "loaded_contact_recovery_v3"
TEMPORAL_SMOOTHNESS = 0.1


@dataclass(frozen=True)
class RecoveryConfig:
    contact_on: float = 0.012
    contact_off: float = 0.025
    support_radius: float = 0.06
    support_scale: float = 0.12
    replant_interval: float = 0.25
    settle_seconds: float = 0.5
    settle_speed: float = 0.15
    settle_angular_speed: float = 0.5
    settle_foot_speed: float = 0.10
    action_rate_weight: float = 0.10


CONFIG = RecoveryConfig()


def contact_transition(xp, height, was_grounded, since_landing, dt):
    """Frozen height-proxy transition for historical evaluation comparisons only."""
    grounded = height < xp.where(was_grounded, CONFIG.contact_off, CONFIG.contact_on)
    return contact_history(xp, grounded, was_grounded, since_landing, dt)


def contact_history(xp, grounded, was_grounded, since_landing, dt):
    """Only repeated touchdowns incur a cost, independent of the contact sensor."""
    landed = grounded & ~was_grounded
    elapsed = since_landing + dt
    rapid = landed * (1.0 - elapsed / CONFIG.replant_interval).clip(0.0, 1.0)
    return grounded, xp.where(landed, 0.0, elapsed), rapid.sum(-1)


def support_state(xp, com, velocity, feet, grounded):
    """Capture-point distance to the support capsule of grounded feet only."""
    omega = xp.sqrt(9.81 / com[..., 2].clip(0.3, None))
    capture = com[..., :2] + velocity[..., :2] / omega[..., None]
    left, right = feet[..., 0, :2], feet[..., 1, :2]
    segment = right - left
    fraction = (((capture - left) * segment).sum(-1)
                / (segment ** 2).sum(-1).clip(1e-9, None)).clip(0.0, 1.0)
    closest = left + fraction[..., None] * segment
    both = grounded[..., 0] & grounded[..., 1]
    one = xp.where(grounded[..., :1], left, right)
    closest = xp.where(both[..., None], closest, one)
    distance = xp.sqrt(((capture - closest) ** 2).sum(-1))
    error = (distance - CONFIG.support_radius).clip(0.0, None)
    supported = grounded.sum(-1) > 0
    quality = xp.exp(-error / CONFIG.support_scale) * supported
    return error, quality


def settled_state(xp, up, height_ratio, velocity, angular_velocity, foot_velocity,
                  grounded, support_error):
    """A stable hold, not simply a pelvis that has not reached the fall threshold."""
    return ((up > 0.9) & (height_ratio > 0.9) & (support_error < 0.02)
            & (grounded.sum(-1) == 2)
            & ((velocity ** 2).sum(-1) < CONFIG.settle_speed ** 2)
            & ((angular_velocity ** 2).sum(-1) < CONFIG.settle_angular_speed ** 2)
            & ((((foot_velocity ** 2).sum(-1) < CONFIG.settle_foot_speed ** 2).sum(-1)) == 2))


def reward_terms(xp, *, up, pelvis_z, rest_pelvis_z, com, velocity, feet,
                 foot_velocity, slip_speed_sq, grounded, pose_error, width, split, rest_width,
                 limit_excess, action, previous_action, rapid_replants):
    """All operations broadcast over leading world dimensions on either backend."""
    _, support = support_state(xp, com, velocity, feet, grounded)
    upright = up.clip(0.0, 1.0)
    height = xp.exp(-40.0 * (pelvis_z - rest_pelvis_z) ** 2)
    # A recovered staggered stance is valid. Deadbands avoid forcing a second,
    # cosmetic step back to an exact pose immediately after a useful landing.
    width_error = xp.clip(abs(width - rest_width) - 0.10, 0.0, None)
    split_error = xp.clip(abs(split) - 0.20, 0.0, None)
    stance = xp.exp(-(width_error ** 2 + split_error ** 2) / 0.15 ** 2)
    calm = xp.exp(-((velocity ** 2).sum(-1)) / 0.15 ** 2)
    slip = (slip_speed_sq * grounded).sum(-1).clip(0.0, 10.0)
    landing = ((-foot_velocity[..., 2] - 0.30).clip(0.0, None) ** 2
               * grounded).sum(-1).clip(0.0, 10.0)
    # Sum across joints: averaging made a single ankle's full-scale reversal cheap.
    rate = ((action - previous_action) ** 2).sum(-1)
    shape = upright * height
    return {
        "upright": 2.0 * upright,
        "height": height,
        "support": 3.0 * support * shape,
        "pose": 0.5 * xp.exp(-2.0 * pose_error) * shape,
        "settling": 2.0 * calm * support * shape * (grounded.sum(-1) == 2),
        "stance": stance * calm * support * shape,
        "sliding": -0.8 * slip,
        "landing": -0.2 * landing,
        "rapid_replants": -2.0 * rapid_replants,
        "action_rate": -CONFIG.action_rate_weight * rate,
        "effort": -0.05 * (action ** 2).mean(-1),
        "limits": -2.0 * limit_excess,
        "crossed_feet": -2.0 * (width < 0.0),
    }
