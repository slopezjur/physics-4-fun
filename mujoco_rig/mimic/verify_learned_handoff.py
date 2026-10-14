"""Audit quiet identity, matched comparisons, and live-only handoff slew."""
import json
from pathlib import Path

import numpy as np
import torch

from .runtime import activate
from .run_support_skill_study import FOUNDATION, digest, save
from .run_learned_support import Learner, PARENT, evaluate
from .run_learned_handoff import GuardedLearner, PRIOR


def main():
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from .rig import Rig
    from .stand_experiment_evidence import load_actor
    from .run_single_support_feedback import make_environment
    torch.set_num_threads(2)
    runs=[Path('logs/mimickit-training')/f'learned-handoff{suffix}-20261008-01' for suffix in ('','-late')]
    out=Path('logs/mimickit-training/learned-handoff-review-20261008-01');out.mkdir(exist_ok=False)
    baseline=np.load('logs/mimickit-training/learned-support-review-20261008-01/zero-addition-quiet30/trace.npz')
    summary=[]
    for run in runs:
        report=json.loads((run/'report.json').read_text());protocol=json.loads((run/'protocol.json').read_text())
        assert all(digest(Path(p))==h for p,h in protocol['inputs'].items())
        assert all(digest(Path(p))==h for p,h in report['output_hashes'].items())
        for folder in run.glob('quiet-*'):
            z=np.load(folder/'trace.npz')
            assert np.array_equal(z['qpos'],baseline['qpos']) and np.array_equal(z['issued'],baseline['issued'])
        for folder in run.glob('entry-*'):
            z=np.load(folder/'trace.npz');r=json.loads((folder/'report.json').read_text())
            ends=[];first_invalid=[]
            for i,c in enumerate(r['cases']):
                live=z['alive'][:,i];last=np.flatnonzero(live)[-1]
                assert live[:last+1].all() and z['done'][last,i]==1 and not c['completed']
                ends.append(float(z['time'][last]))
                invalid=np.flatnonzero(live&~z['supported'][:,i])
                first_invalid.append(float(z['time'][invalid[0]]))
            summary.append(dict(run=run.name,condition=folder.name,completed=r['completed'],
                first_support_failure_range_s=[min(first_invalid),max(first_invalid)],physical_failure_range_s=[min(ends),max(ends)]))
    early=np.load(runs[0]/'entry-correction/trace.npz');late=np.load(runs[1]/'collection/trace.npz')
    assert np.array_equal(early['qpos'],late['qpos']) and np.array_equal(early['issued'],late['issued'])
    rig=Rig.load(FOUNDATION/'dummy.xml',policy_exclude=())
    anchor=load_actor(rig,FOUNDATION/'stationary_reference.npz',json.loads((FOUNDATION/'protocol.json').read_text()),PARENT)
    env=make_environment(rig,anchor,40.)
    poses=np.load(PRIOR/'goals.npz')['poses'];bank={i:dict(poses=poses[i]) for i in range(16)}
    audits=[]
    for run in runs:
        state=torch.load(run/'correction/final.pt',weights_only=True)
        net=Learner(state['mean'],state['scale'],state['output_scale']);net.load_state_dict(state);net.eval()
        guarded=GuardedLearner(net,anchor,env);additions=[]
        hook=guarded.register_forward_hook(lambda module,args,result:additions.append(module.previous.numpy().copy()))
        dest=out/run.name
        evaluate(env,anchor,bank,guarded,'goal',1.,False,dest)
        hook.remove()
        z=np.load(dest/'trace.npz');old=np.load(run/'entry-correction/trace.npz')
        assert np.array_equal(z['qpos'],old['qpos']) and np.array_equal(z['issued'],old['issued'])
        a=np.stack(additions);change=abs(np.diff(a,axis=0,prepend=np.zeros_like(a[:1])))
        maximum=float(change[z['alive']].max())
        assert maximum<=.020001
        audits.append(dict(run=run.name,maximum_live_addition_change_rad=maximum,
            frozen_rerun_exact=True,maximum_live_addition_rad=float(abs(a[z['alive']]).max())))
    results=dict(protected_inputs_and_outputs_unchanged=True,quiet_traces_exactly_match_anchor=True,
        private_queries_do_not_change_frozen_trajectory=True,entry_comparisons=summary,slew_audits=audits,
        telemetry_correction='Original all-world maximum-change counters include dead-world resets to zero. Live-only slew verified separately.',
        planned_physical_cases=144,extra_audit_cases=32,supervised_updates=6000,corrective_labels=277,
        arena_changed=False,parent_sha256=digest(PARENT),source_sha256=digest(Path(__file__)))
    save(out/'report.json',results);print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
