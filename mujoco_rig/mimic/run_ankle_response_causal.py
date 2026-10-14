"""Preserve learned prefixes and remove only the ankle head's impact response."""
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np
import torch

from . import run_direct_recovery as prior
from .rig import Rig
from .run_learned_support import PARENT
from .run_support_skill_study import FOUNDATION, digest, save
from .stand_experiment_evidence import load_actor
from .train_coordinated_ankle import AnkleHead, AnkleRecovery, OUT as PILOT

OUT = Path('logs/mimickit-training/ankle-response-causal-20261009-01')
SEEDS = (921009, 931009)


class QuietContinuation:
    """One-rollout intervention; the environment still evaluates its live anchor."""

    def __init__(self, actor, quiet_latents, switch):
        self.actor = actor
        self.quiet = torch.from_numpy(quiet_latents.copy())
        self.switch = switch
        self.frame = 0

    def __call__(self, features):
        frame = self.frame
        self.frame += 1
        return self.actor(features) if frame < self.switch else self.quiet[frame].clone()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'sources').mkdir()
    inputs = {}
    for path in sorted(Path(__file__).parent.glob('*.py')):
        inputs[str(path)] = digest(path)
        shutil.copy2(path, OUT/'sources'/path.name)
    paths = [PARENT, PARENT.with_suffix('.amp.pt'), FOUNDATION/'dummy.xml',
             FOUNDATION/'stationary_reference.npz', FOUNDATION/'stationary_reference.json',
             FOUNDATION/'protocol.json', PILOT/'contract.json', PILOT/'completed.json',
             PILOT/'audit.json', Path(__file__).with_name('ANKLE_RESPONSE_CAUSAL_PLAN.md')]
    for seed in SEEDS:
        paths += [PILOT/f'fit-{seed}'/'final.pt', PILOT/f'{seed}-quiet'/'trace.npz']
        for timing in (0, 1):
            paths += [PILOT/f'{seed}-t{timing}-g0'/name for name in ('trace.npz', 'report.json')]
    completed = json.loads((PILOT/'completed.json').read_text())
    for path in paths:
        inputs[str(path)] = digest(path)
        if path.is_relative_to(PILOT) and str(path.relative_to(PILOT)) in completed['outputs']:
            assert inputs[str(path)] == completed['outputs'][str(path.relative_to(PILOT))]
    assert json.loads((PILOT/'audit.json').read_text())['passed']
    save(OUT/'inputs.json', inputs)
    torch.set_num_threads(1)
    rig = Rig.load(FOUNDATION/'dummy.xml', policy_exclude=())
    anchor = load_actor(rig, FOUNDATION/'stationary_reference.npz',
        json.loads((FOUNDATION/'protocol.json').read_text()), PARENT)
    assert not any(p.requires_grad for p in anchor.parameters())
    contract = json.loads((PILOT/'contract.json').read_text())
    save(OUT/'contract.json', dict(pilot=str(PILOT), seeds=list(SEEDS), dt=contract['dt'],
        ankles=contract['ankles'], control=contract['control'], switch_offset_frames=1,
        source_feedback='live frozen source', replay='same-policy quiet scalar latent',
        new_cases=64, training_transitions=0))
    records = []
    start = time.perf_counter()
    for seed in SEEDS:
        actor = AnkleHead().eval().requires_grad_(False)
        actor.load_state_dict(torch.load(PILOT/f'fit-{seed}'/'final.pt', weights_only=True, map_location='cpu'))
        with np.load(PILOT/f'{seed}-quiet'/'trace.npz', allow_pickle=False) as archive:
            quiet = archive['latent']
        for timing, onset in enumerate((1., 1.7)):
            env = AnkleRecovery(rig, anchor)
            hit_frame = int(np.ceil(onset/env.dt))
            switch = hit_frame+1

            def configure(current, group, quiet):
                assert group == 0 and not quiet
                current.reset()
                current.force.zero_()
                current.force[:8, 0] = 150.
                current.force[8:, 0] = -150.
                current.onset[:] = hit_frame

            intervention = QuietContinuation(actor, quiet, switch)
            name = f'{seed}-t{timing}'
            with patch.object(prior, 'configure', configure):
                report = prior.rollout(env, intervention, OUT/name)
            normal = json.loads((PILOT/f'{seed}-t{timing}-g0'/'report.json').read_text())
            records.append(dict(name=name, seed=seed, timing=timing, hit_frame=hit_frame,
                switch_frame=switch, calls=intervention.frame, normal_settled=normal['settled'],
                replay_settled=report['settled'], normal_survived=normal['survived'],
                replay_survived=report['survived']))
    save(OUT/'summary.json', records)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    save(OUT/'completed.json', dict(wall_seconds=time.perf_counter()-start,
        inputs_unchanged=True, training_transitions=0, new_cases=64,
        outputs={str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob('*'))
                 if p.is_file() and 'sources' not in p.parts}))
    print('COMPLETE', json.dumps(records), flush=True)


if __name__ == '__main__':
    main()
