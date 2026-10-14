"""Frozen command measurements on the fixed 36 paired recovery observations."""
import numpy as np
import torch

HALF_SIGMA = .08717797887081347
RHO = .9
RADIUS = .1
SLEW = .02
SIGN_BAND = 1e-7
BRANCH_NAMES = ('quiet', 'positive_x', 'negative_x')


def validate_probes(probes):
    """Validate the fixed ordering and match each pushed input to its quiet pose/onset."""
    features = np.asarray(probes['probe_x'])
    assert features.shape == (36, 427) and features.dtype == np.float32
    assert np.isfinite(features).all() and not np.any(features[:, -2:])
    expected = np.asarray([(onset, branch, pose) for onset in (44, 74)
                           for branch in range(3) for pose in range(6)], dtype=np.int64)
    metadata = {}
    for name, column in (('onset', 0), ('branch', 1), ('pose', 2)):
        value = np.asarray(probes['probe_'+name])
        assert value.shape == (36,) and np.issubdtype(value.dtype, np.integer)
        assert np.array_equal(value, expected[:, column]), name
        metadata[name] = value.astype(np.int64, copy=True)
    quiet_index = np.asarray(probes['probe_quiet_index'])
    assert quiet_index.shape == (36,) and np.issubdtype(quiet_index.dtype, np.integer)
    expected_quiet = np.where(metadata['onset'] == 44, 0, 18)+metadata['pose']
    assert np.array_equal(quiet_index, expected_quiet)
    assert np.all(metadata['branch'][quiet_index] == 0)
    assert np.array_equal(metadata['pose'][quiet_index], metadata['pose'])
    assert np.array_equal(metadata['onset'][quiet_index], metadata['onset'])
    metadata['quiet_index'] = quiet_index.astype(np.int64, copy=True)
    reference = np.where(metadata['branch'] == 1, -1, np.where(metadata['branch'] == 2, 1, 0)).astype(np.int64)
    historical = np.asarray(probes['probe_preferred'])
    assert historical.shape == (36,) and np.issubdtype(historical.dtype, np.integer)
    push = metadata['branch'] != 0
    assert np.array_equal(historical[push], reference[push])
    # Quiet-row preferences came from older continuation returns and are not command labels.
    metadata['reference_sign'] = reference
    return features.copy(), metadata


def moments(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and len(values) and np.isfinite(values).all()
    return dict(count=len(values), mean=float(values.mean()), rms=float(np.sqrt(np.mean(values**2))),
                maximum_absolute=float(np.abs(values).max()))


def alignment(values, reference):
    directed = np.asarray(values, dtype=np.float64)*reference
    result = np.where(directed > SIGN_BAND, 1, np.where(directed < -SIGN_BAND, -1, 0)).astype(np.int64)
    result[reference == 0] = 0
    return result


def group_summary(evidence, mask):
    mask = np.asarray(mask, dtype=bool)
    assert mask.shape == (36,) and mask.any()
    push = mask & (evidence['branch'] != 0)
    def counts(key):
        values = evidence[key][push]
        return dict(cases=len(values), aligned=int((values == 1).sum()),
                    opposed=int((values == -1).sum()), within_band=int((values == 0).sum()))
    keys = ('mean', 'memory', 'innovation', 'paired_mean', 'paired_innovation',
            'mean_over_sigma', 'innovation_over_sigma', 'paired_mean_over_sigma',
            'desired_addition', 'requested_addition', 'paired_requested_addition')
    return dict(cases=int(mask.sum()), **{key: moments(evidence[key][mask]) for key in keys},
                raw_reference=counts('raw_alignment'), paired_reference=counts('paired_alignment'),
                slew_limited=int(evidence['slew_limited'][mask].sum()))


def compute(actor, probes):
    """Return arrays and descriptive summaries without changing the supplied frozen actor."""
    features, metadata = validate_probes(probes)
    assert not actor.training
    assert all(parameter.device.type == 'cpu' for parameter in actor.parameters())
    x = torch.from_numpy(features)
    with torch.no_grad():
        output = actor(x)
        assert output.shape == (36, 1) and output.dtype == torch.float32 and torch.isfinite(output).all()
        mean = output[:, 0]
        previous_latent = x[:, -1]
        memory = RHO*previous_latent
        innovation = mean-memory
        previous_addition = RADIUS*x[:, -2]
        desired = RADIUS*mean.tanh()
        requested = previous_addition+(desired-previous_addition).clamp(-SLEW, SLEW)
        limited = (desired-previous_addition).abs() > SLEW
    evidence = dict(**metadata, **{key: value.numpy().copy() for key, value in
        (('mean', mean), ('previous_latent', previous_latent), ('memory', memory),
         ('innovation', innovation), ('previous_addition', previous_addition),
         ('desired_addition', desired), ('requested_addition', requested), ('slew_limited', limited))})
    for key, paired in (('mean', 'paired_mean'), ('innovation', 'paired_innovation'),
                        ('requested_addition', 'paired_requested_addition')):
        values = evidence[key].astype(np.float64)
        evidence[paired] = values-values[evidence['quiet_index']]
    for key in ('mean', 'innovation', 'paired_mean'):
        evidence[key+'_over_sigma'] = evidence[key].astype(np.float64)/HALF_SIGMA
    evidence['raw_alignment'] = alignment(evidence['mean'], evidence['reference_sign'])
    evidence['paired_alignment'] = alignment(evidence['paired_mean'], evidence['reference_sign'])
    assert not evidence['previous_latent'].any() and not evidence['previous_addition'].any()
    assert np.array_equal(evidence['mean'], evidence['innovation'])
    assert np.max(np.abs(evidence['requested_addition'])) <= RADIUS+1e-7
    assert np.max(np.abs(evidence['requested_addition']-evidence['previous_addition'])) <= SLEW+1e-7
    masks = dict(all=np.ones(36, dtype=bool), quiet=evidence['branch'] == 0,
                 push=evidence['branch'] != 0, positive_x=evidence['branch'] == 1,
                 negative_x=evidence['branch'] == 2)
    labels = {-1: 'opposed', 0: 'within_band', 1: 'aligned'}
    per_probe = []
    for i in range(36):
        push = bool(evidence['branch'][i] != 0)
        row = dict(index=i, onset_frame=int(evidence['onset'][i]), pose=int(evidence['pose'][i]),
            branch=BRANCH_NAMES[evidence['branch'][i]], quiet_probe_index=int(evidence['quiet_index'][i]),
            reference_sign=int(evidence['reference_sign'][i]) if push else None,
            raw_reference=labels[int(evidence['raw_alignment'][i])] if push else None,
            paired_reference=labels[int(evidence['paired_alignment'][i])] if push else None,
            slew_limited=bool(evidence['slew_limited'][i]))
        row.update({key: float(evidence[key][i]) for key in
            ('mean', 'previous_latent', 'memory', 'innovation', 'paired_mean', 'paired_innovation',
             'mean_over_sigma', 'innovation_over_sigma', 'paired_mean_over_sigma',
             'previous_addition', 'desired_addition', 'requested_addition', 'paired_requested_addition')})
        per_probe.append(row)
    report = dict(probes=36, quiet_probes=12, pushed_probes=24, poses=list(range(6)), onsets=[44, 74],
        zero_action_memory_verified=True, pairing_verified=True, sigma=HALF_SIGMA, sign_band=SIGN_BAND,
        rho=RHO, addition_radius=RADIUS, addition_slew=SLEW,
        groups={name: group_summary(evidence, mask) for name, mask in masks.items()},
        by_onset={str(onset): group_summary(evidence, evidence['onset'] == onset) for onset in (44, 74)},
        per_probe=per_probe,
        scope='Frozen same-input predictions on seen poses/onsets only. Reference polarity comes from older '
              'weighted-policy continuations, not a current optimality claim. Requested additions precede '
              'source-target joint-limit clipping and the delayed control queue; no physics is integrated.')
    return evidence, report
