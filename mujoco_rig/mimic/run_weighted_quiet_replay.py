"""Frozen causal replay of the age-weighted actor's quiet ankle commands."""
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np
import torch

from . import run_direct_recovery as prior
from .rig import Rig
from .run_ankle_response_causal import QuietContinuation
from .run_learned_support import PARENT
from .run_support_skill_study import FOUNDATION, digest, save
from .stand_experiment_evidence import load_actor
from .train_coordinated_ankle import AnkleHead, AnkleRecovery

OUT = Path('logs/mimickit-training/recovery-weighted-quiet-replay-20261009-01')
CANDIDATE = Path('logs/mimickit-training/recovery-age-weighted-learning-20261009-01')
ZERO = Path('logs/mimickit-training/learned-ankle-20261009-01')
SEED = 931009


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'sources').mkdir()
    paths = sorted(Path(__file__).parent.glob('*.py'))
    for path in paths:
        shutil.copy2(path, OUT/'sources'/path.name)
    paths += [PARENT, PARENT.with_suffix('.amp.pt'), FOUNDATION/'dummy.xml',
        FOUNDATION/'protocol.json', FOUNDATION/'stationary_reference.npz',
        FOUNDATION/'stationary_reference.json',
        Path(__file__).with_name('RECOVERY_WEIGHTED_QUIET_REPLAY_PLAN.md')]
    for root in (CANDIDATE, ZERO):
        assert json.loads((root/'audit.json').read_text())['passed']
        paths += [root/'audit.json', root/'completed.json']
    paths += [CANDIDATE/'contract.json', CANDIDATE/f'fit-{SEED}'/'final.pt']
    references = [CANDIDATE/f'{SEED}-quiet']
    for timing in (0, 1):
        references += [CANDIDATE/f'{SEED}-t{timing}-g0',
            CANDIDATE/f'matched-{SEED}-t{timing}-g0', ZERO/f'baseline-t{timing}-g0']
    paths += [folder/name for folder in references for name in ('trace.npz', 'report.json')]
    inputs = {str(path): digest(path) for path in paths}
    for root in (CANDIDATE, ZERO):
        sealed = json.loads((root/'completed.json').read_text())['outputs']
        for path in paths:
            if path.is_relative_to(root) and str(path.relative_to(root)) in sealed:
                assert digest(path) == sealed[str(path.relative_to(root))], path
    save(OUT/'inputs.json', inputs)
    torch.set_num_threads(1)
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    anchor = load_actor(rig, FOUNDATION/'stationary_reference.npz',
        json.loads((FOUNDATION/'protocol.json').read_text()), PARENT)
    actor = AnkleHead().eval().requires_grad_(False)
    actor.load_state_dict(torch.load(CANDIDATE/f'fit-{SEED}'/'final.pt',
                                    weights_only=True, map_location='cpu'))
    assert not any(p.requires_grad for model in (actor, anchor) for p in model.parameters())
    frozen_actor = {key: value.clone() for key, value in actor.state_dict().items()}
    frozen_anchor = {key: value.clone() for key, value in anchor.state_dict().items()}
    prior_contract = json.loads((CANDIDATE/'contract.json').read_text())
    dt = prior_contract['dt']
    issues = [int(np.ceil(onset/dt))+1 for onset in (1., 1.7)]
    assert issues == [61, 103]
    save(OUT/'contract.json', dict(seed=SEED, candidate=str(CANDIDATE), zero=str(ZERO),
        dt=dt, ankles=prior_contract['ankles'], control=prior_contract['control'],
        issue_frames=issues, push_cases=32, quiet_control_cases=16, cases=48,
        training_transitions=0, source_feedback='live frozen source',
        replay='same-policy continuous quiet scalar latent after first force interval',
        diagnostic_only=True, new_policy_checkpoints=False))
    quiet_folder = CANDIDATE/f'{SEED}-quiet'
    quiet_trace = arrays(quiet_folder/'trace.npz')
    records, start = [], time.perf_counter()
    for timing, quiet_control in ((0, True), (0, False), (1, False)):
        issue = issues[timing]
        name = f'quiet-parity-t{timing}' if quiet_control else f'replay-t{timing}'
        live_folder = quiet_folder if quiet_control else CANDIDATE/f'{SEED}-t{timing}-g0'
        live_trace = quiet_trace if quiet_control else arrays(live_folder/'trace.npz')
        live_report = json.loads((live_folder/'report.json').read_text())
        env = AnkleRecovery(rig, anchor)

        def configure(current, group, quiet):
            assert group == 0 and quiet == quiet_control
            current.reset()
            current.force.zero_()
            if not quiet:
                current.force[:8, 0] = 150.
                current.force[8:, 0] = -150.
            current.onset[:] = issue-1

        intervention = QuietContinuation(actor, quiet_trace['latent'], issue)
        with patch.object(prior, 'configure', configure):
            report = prior.rollout(env, intervention, OUT/name, quiet=quiet_control)
        z = arrays(OUT/name/'trace.npz')
        frames = len(z['time'])
        assert intervention.frame == frames == int(np.ceil(6/dt))
        assert np.array_equal(z['latent'][issue:], quiet_trace['latent'][issue:frames])
        end = frames if quiet_control else issue
        assert all(np.array_equal(z[key][:end], live_trace[key][:end]) for key in z)
        assert all(np.array_equal(z[key][:issue+2], live_trace[key][:issue+2])
                   for key in ('qpos', 'qvel', 'active_target'))
        descriptor = dict(mode='quiet-parity' if quiet_control else 'replay', timing=timing,
            issue_frame=issue, calls=intervention.frame, reference_live=str(live_folder),
            reference_quiet=str(quiet_folder),
            reference_matched=str(CANDIDATE/f'matched-{SEED}-t{timing}-g0'),
            reference_zero=str(ZERO/f'baseline-t{timing}-g0'))
        save(OUT/name/'descriptor.json', descriptor)
        records.append(dict(name=name, timing=timing, quiet=quiet_control, issue_frame=issue,
            replay_settled=report['settled'], live_settled=live_report['settled'],
            replay_survived=report['survived'], live_survived=live_report['survived']))
    save(OUT/'summary.json', records)
    assert all(torch.equal(value, actor.state_dict()[key]) for key, value in frozen_actor.items())
    assert all(torch.equal(value, anchor.state_dict()[key]) for key, value in frozen_anchor.items())
    assert all(digest(Path(path)) == sha for path, sha in inputs.items())
    save(OUT/'completed.json', dict(wall_seconds=time.perf_counter()-start, traces=3,
        cases=48, push_cases=32, quiet_control_cases=16, training_transitions=0,
        frozen_source_unchanged=True, frozen_actor_unchanged=True, inputs_unchanged=True,
        outputs={str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob('*'))
                 if p.is_file() and 'sources' not in p.parts}))
    print('COMPLETE', json.dumps(records), flush=True)


if __name__ == '__main__':
    main()
