"""Capture training-only quiet anchors from one frozen, archived-parity prefix."""

import numpy as np
import torch

from . import run_direct_recovery as prior
from .train_coordinated_ankle import AnkleRecovery, ONSET_FRAMES


ONSETS = (40, 44, 48, 54, 66, 74, 82, 90)
POSES = tuple(range(6))
PREFIX_INTERVALS = 91


def _array(value):
    return value.detach().cpu().numpy().copy()


@torch.no_grad()
def _state(env):
    observation = env.task.observations().clone()
    features = env.features().clone()
    reconstructed = torch.cat((
        env.anchor.obs_norm.normalize(observation).clamp(-20, 20),
        env.history.flatten(1).clamp(-20, 20),
        env.previous[:, env.ankles[:1]] / prior.RADIUS,
        env.last_latent,
    ), -1)
    assert torch.equal(features, reconstructed)
    return {key: _array(value) for key, value in dict(
        features=features,
        observation=observation,
        history=env.history,
        steps=env.task.steps,
        previous=env.previous,
        latent=env.last_latent,
        qpos=env.engine.native_qpos(),
        qvel=env.engine._array('qvel'),
        balance=env.balance_features(),
    ).items()}


@torch.no_grad()
def build(rig, anchor, initial, old_bank, reference_trace):
    """Return bank, prefix evidence and report without writing or updating models.

    State index ``issue = onset + 1`` is after control interval ``onset`` and
    before the first continuation decision. All 16 native worlds run solely to
    reproduce the archived quiet trace; the bank selects only worlds 0..5.
    """
    assert not anchor.training and not initial.training
    assert set(ONSETS).issubset(set(ONSET_FRAMES.tolist()))
    assert not ({60, 102} & set(ONSETS))
    assert np.array_equal(old_bank['onset'], np.repeat([44, 74], 6))
    assert np.array_equal(old_bank['pose'], np.tile(np.arange(6), 2))
    assert old_bank['features'].shape == (12, 427)
    assert old_bank['features'].dtype == np.float32
    assert old_bank['target'].shape == (12, 1)
    assert not old_bank['features'][:, -2:].any() and not old_bank['target'].any()
    frozen = [
        {key: value.detach().clone() for key, value in model.state_dict().items()}
        for model in (anchor, initial)
    ]

    env = AnkleRecovery(rig, anchor, seconds=60.)
    assert env.ankles == [21, 29]
    prior.configure(env, 0, quiet=True)
    active = torch.ones(16, dtype=torch.bool)
    rows, states = [], [_state(env)]
    assert not env.force.any() and not env.onset.any()
    assert not states[0]['steps'].any()
    for frame in range(PREFIX_INTERVALS):
        latent = initial(torch.from_numpy(states[-1]['features']))
        assert latent.shape == (16, 1) and not latent.any()
        before = active.clone()
        _, terminal, info = env.step(latent, active)
        assert not terminal.any() and not info['failed'].any() and not info['warning'].any()
        rows.append(dict(time=np.array((frame + 1) * env.dt), alive=_array(before),
                         **{key: _array(value) for key, value in info.items()}))
        active &= ~terminal
        states.append(_state(env))

    trace = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    assert set(trace) == set(reference_trace), 'Archived trace schema changed'
    for key, value in trace.items():
        assert np.array_equal(value, reference_trace[key][:PREFIX_INTERVALS]), key
    assert trace['alive'].all()
    assert not any(trace[key].any() for key in ('force', 'addition', 'previous', 'latent', 'previous_latent'))

    prefix = {**trace, **{
        'state_' + key: np.stack([state[key] for state in states]) for key in states[0]
    }}
    normalizer = anchor.obs_norm
    prefix.update(normalizer_mean=_array(normalizer.get_mean()),
                  normalizer_std=_array(normalizer.get_std()),
                  normalizer_clip=np.array(float(normalizer._clip)))
    assert prefix['normalizer_mean'].shape == prefix['normalizer_std'].shape == (374,)
    assert (prefix['normalizer_std'] > 0).all()
    assert np.array_equal(prefix['state_steps'],
                          np.repeat(np.arange(PREFIX_INTERVALS + 1)[:, None], 16, axis=1))
    assert np.array_equal(prefix['state_history'][:-1], trace['history_before'])
    assert np.array_equal(prefix['state_history'][1:, :, 0], trace['balance_features'])
    assert np.array_equal(prefix['state_history'][1:, :, 1:], prefix['state_history'][:-1, :, :2])
    assert np.array_equal(prefix['state_qpos'][1:], trace['qpos'])
    assert np.array_equal(prefix['state_qvel'][1:], trace['qvel'])
    assert np.array_equal(prefix['state_previous'][1:], trace['addition'])
    assert np.array_equal(prefix['state_latent'][1:], trace['latent'])

    onset = np.repeat(np.asarray(ONSETS, dtype=np.int64), len(POSES))
    pose = np.tile(np.asarray(POSES, dtype=np.int64), len(ONSETS))
    issue = onset + 1
    features = prefix['state_features'][issue, pose].copy()
    target = _array(initial(torch.from_numpy(features)))
    original = np.flatnonzero(np.isin(onset, [44, 74])).astype(np.int64)
    assert np.array_equal(original, np.r_[6:12, 30:36])
    assert features.shape == (48, 427) and features.dtype == np.float32
    assert not features[:, -2:].any() and not target.any()
    assert np.array_equal(features[original], old_bank['features']), 'Original anchors differ'
    assert np.array_equal(target[original], old_bank['target'])
    bank = dict(features=features, target=target, onset=onset, pose=pose,
                world=pose.copy(), issue=issue, original_indices=original)
    for model, before in zip((anchor, initial), frozen, strict=True):
        assert before.keys() == model.state_dict().keys()
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert not anchor.training and not initial.training
    report = dict(
        passed=True, dt=env.dt, prefix_intervals=PREFIX_INTERVALS,
        native_worlds=16, native_world_intervals=PREFIX_INTERVALS * 16,
        captured_states=PREFIX_INTERVALS + 1, anchors=48, original_anchors=12,
        added_anchors=36, onsets=list(ONSETS), poses=list(POSES), ankles=env.ankles,
        original_indices=original.tolist(), exact_reference_fields=sorted(trace),
        archived_prefix_exact=True, original_features_exact=True,
        all_active=True, zero_forces=True, zero_correction=True,
        zero_action_memory=True, initial_targets_zero=True,
        frozen_source_unchanged=True, frozen_initial_unchanged=True,
        reserved_poses_in_bank=False, reserved_onsets_in_bank=False,
        optimizer_updates=0, critic_updates=0, new_checkpoints=0,
        phase='State index onset+1 after interval onset and before the continuation action.',
        feature_layout=dict(normalized_observation=[0, 374], balance_history=[374, 425],
                            previous_ankle_addition=[425, 426], previous_latent=[426, 427]),
        normalization='Frozen source normalization followed by [-20,20] feature clamp.',
        scope='One frozen quiet prefix; reserved worlds appear only in archived-parity evidence.',
    )
    return bank, prefix, report
