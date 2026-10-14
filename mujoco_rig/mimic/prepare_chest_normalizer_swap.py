"""Build explicit evaluation-only component interventions from sealed checkpoints."""
import argparse
from pathlib import Path
import shutil

import torch

from .audit_stand_failures import digest, read, save
from .replicate_stand_representations import tensor_hash
from .runtime import activate
from .stand_experiment_evidence import seal


def policy_state(state):
    prefixes = {'_obs_norm.': 'obs_norm.', '_a_norm.': 'action_norm.',
                '_model._actor_layers.': 'actor.', '_model._action_dist._mean_net.': 'mean.'}
    return {target + k[len(source):]: v for k, v in state.items()
            for source, target in prefixes.items() if k.startswith(source)}


def run(args):
    activate(args.mimickit)
    from .amp_recovery import actor_state_dict, validate_sidecar
    s = read(args.protocol)
    assert s['schema'] == 'chest_normalizer_swap_v1' and s['seeds'] == [301027, 301028]
    assert s['training_updates'] == s['evaluation_cases'] == 0 and s['evaluation_only']
    inputs = {str(args.protocol): digest(args.protocol), str(Path(__file__)): digest(Path(__file__))}

    def load(item):
        root, checkpoint = Path(item['root']), Path(item['checkpoint'])
        proof = read(root / 'final-verification.json')
        assert proof['verified'] and proof['inputs_and_sources_unchanged']
        hashes = {str(Path(p).resolve()): h for p, h in proof['output_hashes'].items()}
        inputs[str(root / 'final-verification.json')] = digest(root / 'final-verification.json')
        for p in (checkpoint, checkpoint.with_suffix('.amp.pt')):
            h = digest(p)
            assert hashes[str(p.resolve())] == h
            inputs[str(p)] = h
        full = torch.load(checkpoint.with_suffix('.amp.pt'), map_location='cpu', weights_only=True)
        validate_sidecar(torch.load(checkpoint, map_location='cpu', weights_only=True), full)
        assert all(torch.isfinite(v).all() for v in full.values())
        return full

    loaded = {seed: {kind: load(s[kind][str(seed)]) for kind in ('parent', 'continued')} for seed in s['seeds']}
    args.out.mkdir(exist_ok=False)
    shutil.copy2(args.protocol, args.out / 'protocol.json')
    shutil.copy2(Path(__file__), args.out / 'preparation-source.py')
    records = []
    for seed, pair in loaded.items():
        parent, continued = pair['parent'], pair['continued']
        assert parent.keys() == continued.keys()
        assert all(parent[k].shape == continued[k].shape and parent[k].dtype == continued[k].dtype for k in parent)
        assert all(torch.equal(parent[k], continued[k]) for k in parent if k.startswith('_a_norm.'))
        parent_hash, continued_hash = tensor_hash(policy_state(parent)), tensor_hash(policy_state(continued))
        assert parent_hash == s['parent_policy_hashes'][str(seed)]
        for label, prefixes in (('normalizer_only', ('_obs_norm.',)),
                                ('weights_only', ('_model._actor_layers.', '_model._action_dist._mean_net.'))):
            changed = [k for k in parent if k.startswith(prefixes)]
            assert changed
            mixed = {k: (continued[k] if k in changed else v).clone() for k, v in parent.items()}
            folder = args.out / str(seed) / label
            folder.mkdir(parents=True)
            torch.save(mixed, folder / 'model.amp.pt')
            torch.save(actor_state_dict(mixed), folder / 'model.pt')
            saved = torch.load(folder / 'model.amp.pt', map_location='cpu', weights_only=True)
            validate_sidecar(torch.load(folder / 'model.pt', map_location='cpu', weights_only=True), saved)
            assert all(torch.equal(v, saved[k]) for k, v in mixed.items())
            records.append(dict(seed=seed, label=label, parent_policy_hash=parent_hash,
                continued_policy_hash=continued_hash, hybrid_policy_hash=tensor_hash(policy_state(saved)),
                replaced_keys=changed, copied_keys=[k for k in parent if k not in changed],
                maximum_tensor_changes={k: float((continued[k]-parent[k]).abs().max()) for k in changed}))
        # Actor weights plus observation normalization reconstruct the continued deterministic policy.
        keys = ('_obs_norm.', '_model._actor_layers.', '_model._action_dist._mean_net.')
        both = {k: continued[k] if k.startswith(keys) else v for k, v in parent.items()}
        assert tensor_hash(policy_state(both)) == continued_hash
    save(args.out / 'report.json', dict(records=records, evaluation_only=True,
        training_updates=0, evaluation_cases=0, arena_changed=False,
        scope='Counterfactual deterministic policies; critic/discriminator remain from the parent. These are not coherent learner continuations or deployment candidates.'))
    seal(args.out, inputs, [], verified=True, evaluation_only=True, training_updates=0, evaluation_cases=0)
    print(dict(prepared=len(records), evaluation_only=True, training_updates=0), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mimickit', type=Path, default=Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    run(p.parse_args())
