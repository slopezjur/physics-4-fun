"""Frozen-policy input sensitivity on saved real trajectories; no new rollout."""
from pathlib import Path
import numpy as np
import torch

from .audit_stand_failures import read, save
from .rig import Rig
from .runtime import activate
from .stand_experiment_evidence import protect, load_actor, seal
from .train_stand_imitation import tensor_hash

PRIOR = Path('logs/mimickit-training/stand-policy-feedback-20261005-02')
TRAINING = Path('logs/mimickit-training/stand-quiet-speed-20261003-01')
OUT = Path('logs/mimickit-training/stand-feedback-inputs-20261005-01')
GROUPS = dict(root_pose=(0, 9), root_velocity=(9, 15), joint_position=(15, 54),
    joint_velocity=(54, 93), foot_loads=(93, 95), pending_targets=(95, 229), reference=(229, 374))


def run():
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from util import mp_util
    mp_util.init(0, 1, 'cpu', None)
    torch.set_num_threads(1)
    inputs, sources = protect(PRIOR, OUT, [Path(__file__)])
    protocol, trained = read(PRIOR/'protocol.json'), read(TRAINING/'report.json')
    rig, reference = Rig.load(OUT/'dummy.xml', policy_exclude=()), OUT/'stationary_reference.npz'
    save(OUT/'protocol.json', dict(inputs=inputs, sources=sources, config=protocol['config'],
        demonstrations=protocol['demonstrations'], model_sha256=protocol['model_sha256'],
        reference_sha256=protocol['reference_sha256'], groups=GROUPS, states_per_actor=16,
        selection='Each native original start: initial and last frame before first failed gate',
        root_velocity_query_delta_m_s=.02, physics_steps=0, training_updates=0))
    results = {}
    for seed, pair in trained['arms'].items():
        for mode, arm in pair.items():
            key = f'{seed}-{mode}'
            actor = load_actor(rig, reference, protocol, Path(arm['final_checkpoint']))
            before = tensor_hash(actor.state_dict())
            with np.load(PRIOR/key/'live-native'/'trace.npz') as data:
                trace = {k: data[k] for k in data.files}
            points = [(0, case) for case in range(8)]
            for case in range(8):
                first = int(np.flatnonzero(trace['alive_before'][:, case] &
                    ~(trace['balance'][:, case] & trace['posture'][:, case]))[0])
                points.append((max(0, first-1), case))
            obs = torch.tensor(np.stack([trace['observation'][frame, case] for frame, case in points]), requires_grad=True)
            action = actor(obs)
            expected = np.stack([trace['action'][frame, case] for frame, case in points])
            assert np.allclose(action.detach().numpy(), expected, atol=1e-6, rtol=0)
            derivatives = []
            for motor in range(33):
                derivatives.append(torch.autograd.grad(action[:, motor].sum(), obs, retain_graph=motor < 32)[0])
            jacobian = torch.stack(derivatives, 1).detach()*actor.obs_norm.get_std()[None, None]
            live_obs = torch.from_numpy(trace['observation'][trace['alive_before']])
            normalized = (live_obs-actor.obs_norm.get_mean())/actor.obs_norm.get_std()
            statistics = {}
            energy = jacobian.square().sum((1, 2)).clamp_min(1e-20)
            for name, (lo, hi) in GROUPS.items():
                fraction = jacobian[:, :, lo:hi].square().sum((1, 2))/energy
                statistics[name] = dict(initial_jacobian_energy_fraction=float(fraction[:8].mean()),
                    predeparture_jacobian_energy_fraction=float(fraction[8:].mean()),
                    live_clip_fraction=float((normalized[:, lo:hi].abs() > 10).float().mean()))
            with torch.no_grad():
                responses = []
                for coordinate in (9, 10):
                    plus, minus = obs.detach().clone(), obs.detach().clone()
                    plus[:, coordinate] += .02
                    minus[:, coordinate] -= .02
                    responses.append((actor(plus)-actor(minus))/2)
            response = torch.stack(responses).numpy()
            directory = OUT/key
            directory.mkdir()
            np.savez_compressed(directory/'sensitivity.npz', observations=obs.detach().numpy(),
                actions=action.detach().numpy(), points=np.asarray(points), normalized_jacobian=jacobian.numpy(),
                horizontal_velocity_half_difference=response, obs_mean=actor.obs_norm.get_mean().numpy(),
                obs_std=actor.obs_norm.get_std().numpy())
            result = dict(actor_hash=before, groups=statistics,
                max_half_difference_target_radians=float(np.abs(response).max()*.25),
                predeparture_action_change_rms=float(np.sqrt(np.mean((expected[8:]-expected[:8])**2))),
                phase_always_zero=bool((trace['observation'][..., 373] == 0).all()),
                all_finite=bool(torch.isfinite(jacobian).all() and np.isfinite(response).all()),
                interpretation='Local network sensitivity on real saved states; not closed-loop stabilizing feedback proof')
            assert result['all_finite'] and tensor_hash(actor.state_dict()) == before
            save(directory/'report.json', result)
            results[key] = result
            print(dict(actor=key, max_target_response=result['max_half_difference_target_radians'],
                groups={k: round(v['predeparture_jacobian_energy_fraction'], 3) for k, v in statistics.items()}), flush=True)
    save(OUT/'report.json', dict(results=results, selected_states=64, new_physics_steps=0, training_updates=0,
        input_derivatives_only=True, optimizer_steps=0))
    seal(OUT, inputs, sources, selected_states=64, saved_action_reproduction_verified=True,
        actor_hashes_unchanged=True, new_physics_steps=0, training_updates=0)


if __name__ == '__main__':
    run()
