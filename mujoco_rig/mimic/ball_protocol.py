"""Versioned validation cases. These are used for selection, not held-out testing."""
TARGET_BODIES = (
    "Head", "Chest", "Spine", "Pelvis",
    "UpperArm_L", "Forearm_L", "UpperArm_R", "Forearm_R",
    "Thigh_L", "Shin_L", "Thigh_R", "Shin_R",
)
VALIDATION_SCHEMA = "mimic_ball_validation_v2"
VALIDATION_EPISODES = len(TARGET_BODIES) * 4 * 2


def validation_cases(speed_min=1., speed_max=2.5):
    return [dict(name=f"{body}-direction-{direction}-speed-{speed:g}",
                 target_body=body, direction=direction, speed=speed,
                 phase=0. if level == 0 else .8, launch_step=60, trial_steps=300)
            for level, speed in enumerate((speed_min, speed_max))
            for direction in range(4) for body in TARGET_BODIES]
