"""Versioned sensor semantics shared by training, checkpoint readers and export."""

LEGACY = "legacy_v1"
FOUNDATION = "foundation_v2"
VERSIONS = (LEGACY, FOUNDATION)


def layout(num_actions, version=LEGACY):
    if version not in VERSIONS:
        raise ValueError(f"Unsupported observation version: {version}")
    channels = [("projected_gravity", 3), ("pelvis_linear_velocity", 3),
                ("pelvis_angular_velocity", 3), ("pelvis_height", 1),
                ("joint_position", num_actions), ("joint_velocity_scaled_0.1", num_actions),
                ("foot_contact_L_R", 2), ("previous_action", num_actions),
                ("command_vx_vy_yaw", 3)]
    if version == FOUNDATION:
        # Append instead of reinterpreting old weights: zero new input columns
        # preserve the seed actor while the critic learns the richer state.
        channels += [("pelvis_origin_linear_velocity", 3),
                     ("foot_normal_load_kN_L_R", 2), ("oldest_pending_action", num_actions)]
    result, offset = [], 0
    for name, width in channels:
        result.append(dict(name=name, offset=offset, width=width))
        offset += width
    return result


def observation_size(num_actions, version=LEGACY):
    return sum(c["width"] for c in layout(num_actions, version))


def checkpoint_version(checkpoint):
    version = checkpoint.get("observation_version", LEGACY)
    expected = observation_size(checkpoint["num_actions"], version)
    if checkpoint["num_obs"] != expected:
        raise ValueError(f"Checkpoint observation width {checkpoint['num_obs']} != {expected} ({version})")
    if version == FOUNDATION and checkpoint.get("task", "perturb") != "perturb":
        raise ValueError("foundation_v2 is currently supported only for Perturb")
    return version
