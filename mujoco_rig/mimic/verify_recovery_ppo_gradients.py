"""Independent saved-data checks of labels, exact replay and analytic gradients."""
import json
from pathlib import Path

import numpy as np
import torch

from .run_support_skill_study import digest, save
from .train_coordinated_ankle import AnkleHead

BASE = Path('logs/mimickit-training/recovery-half-noise-learning-20261009-01')
OUT = Path('logs/mimickit-training/recovery-ppo-gradients-20261009-01')
WORLDS = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]


def read(path):
    return json.loads(path.read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main():
    torch.set_num_threads(1)
    inputs, done, results = (read(OUT/n) for n in ('inputs.json', 'completed.json', 'results.json'))
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    assert all(digest(OUT/p) == sha for p, sha in done['outputs'].items())
    errors = {}
    for seed in (921009, 931009):
        z = arrays(OUT/f'{seed}.npz')
        batch = arrays(BASE/f'fit-{seed}'/'collection-1.npz')
        resets = read(BASE/f'fit-{seed}'/'resets.json')
        history = read(BASE/f'fit-{seed}'/'history.json')
        consumed = 0
        for j, world in enumerate(WORLDS):
            per_world = [r for r in resets if r['world'] == world]
            episode, age = 0, 0
            for frame in range(64):
                row = per_world[episode]
                force = row['force_n'][0]
                assert row['start_frame'] == row['onset_frame']+1 and row['pose'] == world % 8 < 6
                assert z['branch'][frame, j] == (0 if force == 0 else 1 if force > 0 else 2)
                assert z['age'][frame, j] == age
                if age == 0:
                    assert not batch['x'][frame, j, -2:].any()
                else:
                    assert batch['x'][frame, j, -1] == batch['action'][frame-1, j, 0]
                age += 1
                if batch['terminal'][frame, j]:
                    episode += 1
                    age = 0
            consumed += episode+1
        assert consumed == results[str(seed)]['assignment']['records_consumed']
        assert np.array_equal(z['replay_mean'], batch['post_mean'])
        assert np.array_equal(z['replay_value'], batch['post_value'])
        assert z['kl'].tolist() == history[0]['accepted_kl'] and len(z['kl']) == 9
        torch.set_rng_state(torch.from_numpy(z['permutation_rng']))
        for order in z['permutations']:
            assert np.array_equal(np.sort(order), np.arange(768))
            assert np.array_equal(torch.randperm(768).numpy(), order)
        actor = AnkleHead().eval()
        actor.load_state_dict(torch.load(BASE/f'fit-{seed}'/'initial.pt', weights_only=True, map_location='cpu'))
        with torch.no_grad():
            x = torch.from_numpy(batch['x']).reshape(-1, 427)
            assert np.array_equal(actor.net[:-1](x).numpy(), z['hidden'])
            assert np.array_equal(actor(x).numpy().reshape(64, 12, 1), batch['mean'])
        assert not actor.net[-1].weight.any() and not actor.net[-1].bias.any()
        sigma = float(batch['sigma'])
        score = (batch['action'].flatten().astype(float)-batch['mean'].flatten())/sigma**2
        score *= batch['normalized_advantage']
        h = np.c_[z['hidden'].astype(float), np.ones(768)]
        groups = z['branch'].flatten()
        independent = np.stack([.1*(h[mask].T@score[mask])/768 for mask in
                                [np.ones(768, bool), *[groups == i for i in range(3)]]])
        assert np.allclose(independent, z['gradients'], atol=3e-6, rtol=3e-5)
        assert np.allclose(independent[1:].sum(0), independent[0], atol=1e-12, rtol=0)
        assert np.allclose(.1*z['gradients']@h.T, z['linear_response'], atol=1e-10, rtol=0)
        change = batch['post_mean'].flatten().astype(float)-batch['mean'].flatten()
        assert np.array_equal(change, z['change'])
        summary = results[str(seed)]['gradient']
        assert abs(summary['constant_change_energy_fraction']-change.mean()**2/np.mean(change**2)) < 1e-12
        assert abs(np.mean(change**2)-(change.mean()**2+change.var())) < 1e-15
        for i, name in enumerate(('quiet', '+X', '-X')):
            mask = groups == i
            row = summary['branches'][name]
            assert row['samples'] == int(mask.sum())
            assert abs(row['actual_mean_change']-change[mask].mean()) < 1e-12
            assert abs(row['output_bias_ascent']-independent[i+1, -1]) < 3e-6
        errors[str(seed)] = float(np.abs(independent-z['gradients']).max())
    assert done['new_training_transitions'] == done['physical_rollouts'] == 0
    assert done['saved_samples'] == 1536
    assert done['replay_actor_steps'] == done['replay_critic_steps'] == 18
    audit = dict(passed=True, saved_samples=1536, seeds=2, input_hashes=len(inputs),
        output_hashes=len(done['outputs']), exact_labels=True, exact_saved_optimizer_outputs=True,
        exact_minibatch_permutations=True, analytic_gradient_max_error=errors,
        scope='First collection only; existing data, no physics or new policy checkpoints.')
    save(OUT/'audit.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
