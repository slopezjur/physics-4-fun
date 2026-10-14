"""Reproduce the physical teacher under the original full command contract."""
import numpy as np

from .probe_arm_proposal_foundations import ROOT,TEACHER,PLAN
from .probe_arm_execution_distribution import setup,seal
from .train_arm_hold import episode
from .run_support_skill_study import digest,save


def main():
    out=ROOT/'step-05';paths=[TEACHER/f'teacher-{name}/trace.npz' for name in ('nominal','varied')]
    inputs=seal(out,[PLAN,*paths]);env,anchor,proposal,poses,_=setup();results=[]
    save(out/'protocol.json',dict(step=5,inputs=inputs,role='Physical teacher, not a learned actor',training_updates=0))
    for variant,label in ((False,'nominal'),(True,'varied')):
        r=episode(env,anchor,proposal,poses,out/label,teacher=True,variant=variant)
        old=np.load(TEACHER/f'teacher-{label}/trace.npz');new=np.load(out/label/'trace.npz')
        errors={k:float(abs(new[k]-old[k]).max()) for k in ('qpos','issued')}
        assert max(errors.values())<2e-6 and r['complete']==8
        results.append(dict(folder=label,replay_errors=errors,**r));save(out/'progress.json',results)
    assert all(digest(__import__('pathlib').Path(p))==h for p,h in inputs.items())
    save(out/'report.json',dict(step=5,results=results,inputs_unchanged=True,
        outputs={str(p):digest(p) for p in out.rglob('*') if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
