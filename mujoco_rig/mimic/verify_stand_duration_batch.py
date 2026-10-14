"""Close the second autonomous batch by checking sealed duration evidence."""
import argparse
from pathlib import Path
import shutil
import numpy as np

from .audit_stand_failures import digest, read, save
from .measure_stand_learning_curve import continuation


def evidence_protocol(root):
    """Recover diagnostic provenance from its sealed source snapshots and parent."""
    if (root/'protocol.json').exists():
        return read(root/'protocol.json')
    report, verification = read(root/'report.json'), read(root/'final-verification.json')
    parent = Path(report['results'][-1]['root'])
    prior, complete = read(parent/'protocol.json'), read(parent/'final-verification.json')
    inputs = {**prior['inputs'], **complete['output_hashes'],
        str(parent/'final-verification.json'): digest(parent/'final-verification.json')}
    paths = [Path(__file__).with_name('stand_experiment_evidence.py'),
        Path(__file__).with_name('audit_stand_duration_replication.py')]
    for source in prior['sources']:
        paths.append(Path(source['path']))
        for key in ('path', 'snapshot'):
            inputs[source[key]] = source['sha256']
    paths = list({str(path.resolve()): path for path in paths}.values())
    sources = []
    for index, path in enumerate(paths):
        snapshot = root/'sources'/f'{index:02d}-{path.name}'
        expected = verification['output_hashes'][str(snapshot)]
        assert str(path) not in inputs or inputs[str(path)] == expected
        inputs[str(path)] = expected
        sources.append(dict(path=str(path), snapshot=str(snapshot), sha256=expected))
    assert len(inputs) == verification['protected_inputs']
    assert len(sources) == verification['sources']
    return dict(inputs=inputs, sources=sources, reconstructed_diagnostic_manifest=True)


def run(args):
    hashes, metadata, protocols = {}, {}, {}
    previous = read(Path('logs/mimickit-training/stand-autonomous-batch-20261005-01/verification.json'))
    manifests = [{**previous['protected_hashes'], **previous['output_hashes']}]
    for root in args.roots:
        protocol, verification = evidence_protocol(root), read(root/'final-verification.json')
        protocols[str(root)] = protocol
        entries = {**protocol['inputs'], **verification['output_hashes'],
            str(root/'final-verification.json'): digest(root/'final-verification.json')}
        for source in protocol['sources']:
            entries.update({source[key]: source['sha256'] for key in ('path', 'snapshot')})
        manifests.append(entries)
        metadata[root.name] = {key: value for key, value in verification.items() if key != 'output_hashes'}
    for entries in manifests:
        for name, expected in entries.items():
            path = str(Path(name).resolve())
            assert path not in hashes or hashes[path] == expected, path
            hashes[path] = expected
    hashes[str(Path(__file__).resolve())] = digest(Path(__file__))
    for path, expected in hashes.items():
        assert digest(Path(path)) == expected, path
    runs, assessments, cycles, seconds = {}, 0, 0, 0.
    for root in args.roots:
        protocol, report = protocols[str(root)], read(root/'report.json')
        if 'assessments' not in report:
            continue
        assert protocol['initialize_from'] is None
        prior = read(Path(protocol['prior_root'])/'protocol.json')
        assert protocol['config'] == prior['config'] and protocol['demonstrations'] == prior['demonstrations']
        for name, key in (('dummy.xml', 'model_sha256'), ('stationary_reference.npz', 'reference_sha256')):
            assert digest(root/name) == protocol[key] == prior[key]
        contract = read(root/'contract.json')
        assert contract['control_timestep'] == .016668
        assert contract['num_obs'] == 374 and contract['representation'] == 'target_pd'
        assert contract['passive_joint_stiffness'] == 0
        stages = report['assessments']
        decisions = {f'{lo}_to_{hi}': continuation(stages[f'updates-{hi}'], stages[f'updates-{lo}'], contract['control_timestep'])
            for lo, hi in ((100, 300), (300, 600)) if f'updates-{hi}' in stages}
        full_interval_stand = {}
        final_stage = list(stages)[-1]
        for backend in ('native', 'gpu'):
            with np.load(root/final_stage/backend/'trace.npz', allow_pickle=False) as trace:
                full_interval_stand[backend] = bool(len(trace['alive_before']) == 720 and
                    trace['alive_before'].all() and all(trace[key].all() for key in
                    ('balance', 'posture', 'pre_balance', 'pre_posture')))
        runs[str(protocol['seed'])] = dict(root=str(root), decisions=decisions, updates=report['ppo_updates'],
            final_stand_passes={b: list(stages.values())[-1]['results'][b]['standing_passes'] for b in ('native', 'gpu')},
            continuously_balanced_natural_all_sampled_frames=full_interval_stand,
            stop_reason=report['stop_reason'])
        assessments += 16*len(stages)
        cycles += report['ppo_updates']
        seconds += report['training_seconds']
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.out/'verifier.py')
    report = dict(verified_steps=list(range(1, 20)), final_documentation_step=20,
        reconstructed_diagnostic_manifests=[path for path, protocol in protocols.items() if protocol.get('reconstructed_diagnostic_manifest')],
        roots=[str(p) for p in args.roots], protected_paths=len(hashes), phase_verification=metadata,
        runs=runs, new_assessment_cases=assessments, new_ppo_updates=cycles,
        new_transitions=cycles*4096, synchronized_training_seconds=seconds,
        sustained_stand_verified=False, perturbations_ready=False, arena_changed=False,
        verification_new_physics_steps=0, verification_new_forward_passes=0, verification_training_updates=0)
    save(args.out/'report.json', report)
    save(args.out/'verification.json', dict(protected_hashes=hashes, all_hashes_verified=True,
        output_hashes={str(p): digest(p) for p in args.out.iterdir() if p.is_file()}))
    print({key: report[key] for key in ('protected_paths', 'runs', 'new_assessment_cases', 'new_ppo_updates', 'new_transitions')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--roots', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
