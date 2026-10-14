"""Reconcile the bounded block, selection and on-disk Arena identity."""
from pathlib import Path

from .audit_stand_failures import digest, read, save
from .verify_perturb_response import verify_seal

BASE=Path('logs/mimickit-training')
OUT=BASE/'perturb-response-block-review-20261008-01'


def run():
    probe=BASE/'perturb-response-probe-20261008-01'
    final=BASE/'perturb-response-final-20261008-01'
    sources=[probe,final]
    comparisons={}
    updates=transitions=0
    training_seconds=0.
    physical=read(probe/'report.json')['physical_cases']+read(final/'report.json')['physical_cases']
    excluded=read(probe/'report.json')['excluded_cases']
    for index in (1,2):
        train=BASE/f'perturb-response-learning-20261008-{index:02d}'
        evaluate=BASE/f'perturb-response-evaluation-20261008-{index:02d}'
        sources.extend([train,evaluate])
        t,r=read(train/'report.json'),read(evaluate/'report.json')
        assert read(BASE/f'perturb-response-training-review-20261008-{index:02d}'/'report.json')['verified']
        assert read(BASE/f'perturb-response-evaluation-review-20261008-{index:02d}'/'report.json')['verified']
        updates+=t['training_updates']; transitions+=t['transitions']; physical+=r['physical_cases']
        training_seconds+=sum(v['training_seconds'] for v in t['results'].values())
        comparisons[str(index)]={}
        for arm,row in r['results'].items():
            quiet={b:v['standing_passes'] for b,v in row['quiet'].items()}
            impact={k:{b:v[b]['recovery_passes'] for b in ('native','gpu')} for k,v in row['impacts'].items()}
            admitted=all(v['parity']['passed'] and v['chest_first'] for v in row['impacts'].values())
            assert row['admitted']==admitted
            qualifies=admitted and all(v==8 for v in quiet.values()) and all(v==8 for x in impact.values() for v in x.values())
            assert row['qualifies']==qualifies
            excluded+=sum(16 for v in row['impacts'].values() if not(v['parity']['passed'] and v['chest_first']))
            comparisons[str(index)][arm]=dict(quiet=quiet,impacts=impact,admitted=admitted,qualifies=qualifies)
    for source in sources:
        verify_seal(source)
    completion=read(probe/'completion-protocol.json')
    for path,value in completion['inputs'].items():
        assert digest(Path(path))==value,path
    assert updates==400 and transitions==1638400 and physical<=640
    result=read(final/'report.json')
    assert read(BASE/'perturb-response-final-review-20261008-01/report.json')['verified']
    foot=read(BASE/'perturb-response-foot-audit-20261008-01/report.json')
    for path,value in foot['inputs'].items():
        assert digest(Path(path))==value,path
    viewer=read(Path('logs/mimickit-viewer/ball.json'))
    if result['selected'] is None:
        assert not result['candidates'] or not result['arena_change_eligible']
        assert viewer['checkpoint_sha256'].startswith('f832bab8')
    else:
        item=result['candidates'][result['selected']]
        assert viewer['checkpoint_sha256']==digest(Path(item['model']))
        assert all(r['standing_passes']==8 for r in result['retention'][result['selected']].values())
    bundle=Path(viewer['bundle'].removeprefix('res://'))
    assert digest(bundle/'stand.onnx')==viewer['actor_sha256']
    sizes={str(p):sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) for p in BASE.glob('perturb-response-*') if p.is_dir()}
    OUT.mkdir(exist_ok=False)
    save(OUT/'report.json',dict(verified=True,completed_substantive_steps=20,training_updates=updates,
        transitions=transitions,training_seconds=training_seconds,physical_cases=physical,
        excluded_cases=excluded,admitted_cases=physical-excluded,comparisons=comparisons,
        selected=result['selected'],recipe_improvement_replicated=result['recipe_improvement_replicated'],
        arena_checkpoint=viewer['checkpoint_sha256'],arena_onnx=viewer['actor_sha256'],
        step_signature_counts={k:v['recovered_step_cases'] for k,v in foot['results'].items()},
        folder_bytes=sizes,total_new_evidence_bytes=sum(sizes.values()),reviewer_sha256=digest(Path(__file__))))
    print(dict(verified=True,training_updates=updates,transitions=transitions,physical_cases=physical,
        excluded_cases=excluded,selected=result['selected'],new_bytes=sum(sizes.values())),flush=True)


if __name__=='__main__':
    run()
