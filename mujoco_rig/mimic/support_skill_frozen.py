"""Frozen support-skill evaluation, including explicit repeated goal commands."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import torch

from .runtime import activate
from . import run_support_skill_study as study
from .support_skill_experiments import metric_adapter
from .support_skill_exact_goals import exact_goals


def load_policy(rig,anchor,folder):
    from .support_skill_core import SkillEnvironment,Goal
    from .support_skill_learning import build
    from .policy import StandPolicy
    folder=Path(folder);p=json.loads((folder/'protocol.json').read_text());arm=p['arm']
    env=SkillEnvironment(rig,study.FOUNDATION/'stationary_reference.npz',deepcopy(anchor),[Goal(**arm['goal'])]*8,'cpu',
        action_mode=arm.get('action_mode','basis'),radius=arm.get('radius',.5),seconds=arm.get('seconds',10.))
    agent=build(p['config'],env,folder/'final.pt')
    return StandPolicy(agent).eval().requires_grad_(False),arm


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();spec=json.loads(args.protocol.read_text())
    activate(Path('D:/Proyectos/Juegos/Tools/MimicKit'))
    from util import mp_util
    from . import support_skill_core as core
    mp_util.init(0,1,'cuda:0',None);torch.set_num_threads(1)
    rig,_,actors=study.setup(spec,args.out);results={}
    extras={str(Path(p['folder'])/name):study.digest(Path(p['folder'])/name)
            for p in spec['policies'] for name in ('final.pt','protocol.json')}
    study.save(args.out/'policy-inputs.json',extras)
    with exact_goals(),metric_adapter():
        base=core.SkillEnvironment
        for parent,anchor in actors.items():
            for item in spec['policies']:
                policy,trained=load_policy(rig,anchor,item['folder'])
                for arm in spec['arms']:
                    class RepeatEnvironment(base):
                        def __init__(self,*a,**kw):
                            super().__init__(*a,**kw)
                            self.first_onset=self.settings['onset'].clone()

                        def coefficients(self):
                            if arm.get('second_onset') is not None and hasattr(self,'first_onset'):
                                self.settings['onset']=torch.where(self.task.steps*self.dt>=arm['second_onset'],
                                    torch.full_like(self.first_onset,arm['second_onset']),self.first_onset)
                            return super().coefficients()

                        def contract(self):
                            return dict(super().contract(),second_command_onset=arm.get('second_onset'))

                    for backend in spec.get('backends',['native']):
                        label=f"{parent}-{item['name']}-{arm['name']}-{backend}"
                        with patch.object(core,'SkillEnvironment',RepeatEnvironment):
                            result=study.evaluate(rig,anchor,dict(arm,action_mode=trained.get('action_mode','basis'),
                                radius=trained.get('radius',.5)),backend,args.out/label,policy)
                        results[label]=result;study.save(args.out/'progress.json',results)
                        print(label,{k:v for k,v in result.items() if k!='cases'},flush=True)
    for path,expected in extras.items():assert study.digest(path)==expected
    p=json.loads((args.out/'protocol.json').read_text())
    for path,expected in {**p['sources'],**p['inputs']}.items():assert study.digest(path)==expected
    study.save(args.out/'report.json',dict(stage=spec['stage'],completed=True,results=results))
    study.save(args.out/'final-verification.json',dict(inputs_unchanged=True,outputs={str(p.relative_to(args.out)):study.digest(p)
        for p in sorted(args.out.rglob('*')) if p.is_file() and 'sources' not in p.parts}))


if __name__=='__main__':main()
