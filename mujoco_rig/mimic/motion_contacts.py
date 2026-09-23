"""Explicit foot/toe support phases for offline walking-reference fitting."""
import numpy as np

SOLE_NAMES = ("Foot_L", "Foot_R", "Toe_L", "Toe_R")
TOE_REFERENCE_SCHEMA = "mimic_step_reference_v6"


def stance_phase_ids(contact):
    """One landing identity per uninterrupted support interval, including toe-off."""
    ids = np.full(np.asarray(contact).shape, -1, dtype=int)
    count = 0
    for foot in range(2):
        for frame in range(len(contact)):
            if contact[frame, foot]:
                if frame == 0 or not contact[frame - 1, foot]:
                    count += 1
                ids[frame, foot] = count - 1
    return ids, count


def expand_landing_offsets(phase_ids, offsets):
    """Swing frames have no anchor adjustment; planted frames share one offset."""
    result = np.zeros(phase_ids.shape + (2,))
    supported = phase_ids >= 0
    result[supported] = offsets[phase_ids[supported]]
    return result


def landing_seed_for_window(contact, offsets, start, stop):
    """Preserve absolute landing corrections when a window cuts a stance interval."""
    if offsets is None:
        return None
    source_ids, count = stance_phase_ids(contact)
    offsets = np.asarray(offsets, dtype=float)
    if offsets.shape != (count, 2):
        raise ValueError("Saved landing offsets do not match the source stance intervals")
    local_ids, local_count = stance_phase_ids(contact[start:stop])
    seed = np.zeros((local_count, 2))
    for phase in range(local_count):
        frame, foot = np.argwhere(local_ids == phase)[0]
        seed[phase] = offsets[source_ids[start + frame, foot]]
    return seed


def sole_schedule(contact, toe_only):
    contact, toe_only = np.asarray(contact), np.asarray(toe_only)
    if contact.dtype != bool or toe_only.dtype != bool or contact.ndim != 2 or contact.shape[1] != 2 \
            or toe_only.shape != contact.shape or np.any(toe_only & ~contact):
        raise ValueError("Expected boolean foot support and a toe-only subset")
    return np.column_stack((contact & ~toe_only, contact))


def validate_sole_schedule(contact, soles):
    contact = np.asarray(contact)
    soles = np.asarray(soles)
    if contact.dtype != bool or contact.ndim != 2 or contact.shape[1] != 2 \
            or soles.dtype != bool or soles.shape != (len(contact), 4) \
            or not np.array_equal(soles[:, 2:], contact) or np.any(soles[:, :2] & ~soles[:, 2:]):
        raise ValueError("Expected flat-foot or toe-only support consistent with foot contact labels")
    return ~soles[:, :2] & soles[:, 2:]


def infer_toe_off(original_contact, completed_contact, foot_rotations):
    """Toe support only at inferred gaps with heel-up pitch, including two lead-in frames.

    This is a candidate walking hypothesis, not a measured motion-capture label.
    Backward/heel-leading gaps are deliberately not relabeled as toe support.
    """
    toe_only = np.zeros_like(completed_contact)
    for frame, foot in np.argwhere(completed_contact & ~original_contact):
        if foot_rotations[frame, foot, 2, 0] >= -np.sin(np.deg2rad(3)):
            continue
        start = frame
        for previous in range(frame - 1, max(-1, frame - 3), -1):
            if not completed_contact[previous, foot]:
                break
            start = previous
        toe_only[start:frame + 1, foot] = completed_contact[start:frame + 1, foot]
    return toe_only
