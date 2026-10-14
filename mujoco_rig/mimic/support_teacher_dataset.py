"""Versioned, goal-conditioned samples from successful physical teacher trials."""
import json
from pathlib import Path

import numpy as np

SCHEMA = 'support_pose_teacher_v1'
OBSERVATION_SIZE = 421
ACTION_SIZE = 33


def policy_input(stand_observation, desired_pose, authority):
    """Exact issue-time state and application-time goal available to a learner."""
    state = np.asarray(stand_observation, dtype=np.float32)
    pose = np.asarray(desired_pose, dtype=np.float32)
    amount = np.asarray(authority, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != 374 or pose.shape != (len(state), 46) or amount.shape != (len(state),):
        raise ValueError('Expected [N,374] Stand observations, [N,46] goals and [N] authority')
    result = np.concatenate((state, pose, amount[:, None]), axis=1)
    if not np.isfinite(result).all() or (amount < 0).any() or (amount > 1).any():
        raise ValueError('Invalid teacher input')
    if not np.allclose(np.linalg.norm(pose[:, 3:7], axis=1), 1., atol=2e-6):
        raise ValueError('Goal root quaternions must have unit length')
    return result


def export(out, rows, alive, cases, worlds, lower, upper):
    """Write only complete successful trials; no partial-case training labels."""
    if len(cases) != 16 or not all(c['entry_hold_return'] for c in cases):
        return dict(exported=False, reason='All sixteen complete sequences are required')
    arrays = {name: np.stack([r[name] for r in rows]) for name in rows[0]}
    if alive.shape != arrays['action'].shape[:2]:
        raise ValueError('Teacher sample/physical trace shape mismatch')
    state = arrays['observation'][alive]
    action = arrays['action'][alive]
    source = arrays['source_targets'][alive]
    issued = arrays['issued_targets'][alive]
    if state.shape[1] != OBSERVATION_SIZE or action.shape[1] != ACTION_SIZE:
        raise ValueError('Unexpected learner dimensions')
    if not all(np.isfinite(v).all() for v in (state, action, source, issued)) or abs(action).max() > 1:
        raise ValueError('Invalid teacher samples')
    reconstructed = np.clip(source+action, lower, upper)
    maximum_error = float(abs(reconstructed-issued).max())
    if maximum_error >= 2e-6:
        raise ValueError('Teacher action does not reconstruct the issued target')
    frames, columns = np.nonzero(alive)
    world = np.asarray(worlds, dtype=np.int16)[columns]
    split = (world % 8 >= 6).astype(np.uint8)
    out = Path(out)
    np.savez_compressed(out/'teacher-samples.npz', observation=state, action=action,
                        source_targets=source, issued_targets=issued,
                        case_id=world, frame=frames.astype(np.int32), split=split)
    contract = dict(schema=SCHEMA, observation_size=OBSERVATION_SIZE, action_size=ACTION_SIZE,
                    observation_layout=dict(stand=[0,374], desired_pose=[374,420], authority=[420,421]),
                    goal='Desired MuJoCo pose at current issue time plus two control intervals; phase scheduler supplies it, no live state writes',
                    authority='Teacher blending fraction is an input; action labels already include blending, do not multiply them again',
                    action='clip(frozen_source_target + 1 rad * clip(action,-1,1), physical_joint_limits)',
                    radius_rad=1., delayed_command_slots=2,
                    compatibility='Requires a 421-input goal-conditioned adapter; incompatible with the old 380-input clock-goal trainer',
                    split=dict(train_poses=list(range(6)), validation_poses=[6,7], unit='entire pose, both feet'),
                    examples=len(state), train_examples=int((split==0).sum()), validation_examples=int((split==1).sum()),
                    maximum_target_reconstruction_error=maximum_error, training_updates=0)
    (out/'teacher-contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    return dict(exported=True, **contract)
