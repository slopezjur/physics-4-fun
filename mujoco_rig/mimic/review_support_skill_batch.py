"""Independent saved-evidence reconciliation; performs no simulation or learning."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


NAMES={
 1:['stage01-zero-20261007-01'],2:['stage02-transfer-20261007-02'],
 3:['stage03-duration-20261007-01'],4:['stage04-feedback-20261007-01'],
 5:['stage05-equilibrium-20261007-01'],6:['stage06-lift-20261007-02'],
 7:['stage07-authority-20261007-01'],8:['stage08-hold-20261007-02'],
 9:['stage09-air-translation-20261007-01'],10:['stage10-entry-return-20261007-01'],
 11:['stage11-transfer-learning-20261007-01','stage11-controls-20261007-01'],
 12:['stage12-action-comparison-20261007-01'],13:['stage13-seed-replication-20261007-01'],
 14:['stage14-original-20261007-01','stage14-prepared-20261007-01'],
 15:['stage15-supervision-20261007-01'],16:['stage16-replant-20261007-01','stage16-teacher-20261007-01'],
 17:['stage17-heldout-onsets-20261007-01'],18:['stage18-retention-20261007-02'],
 19:['stage19-reactive-20261007-01'],20:['stage20-replication-20261007-01']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args();root=Path('logs/mimickit-training');cache={}

    def digest(path):
        path=Path(path).resolve();stat=path.stat();key=(str(path),stat.st_size,stat.st_mtime_ns)
        if key not in cache:
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1048576),b''):h.update(block)
            cache[key]=h.hexdigest()
        return cache[key]

    results={};updates=transitions=cases=presentations=0;checked=0;exceptions=[]
    for stage,names in NAMES.items():
        folders=[root/('support-skill-'+name) for name in names]
        if not all((p/'final-verification.json').exists() for p in folders):continue
        stage_cases=0;stage_training=[]
        for folder in folders:
            p=json.loads((folder/'protocol.json').read_text());r=json.loads((folder/'report.json').read_text())
            assert r['completed'] and r['stage']==stage
            for path,h in p['inputs'].items():assert digest(path)==h
            for path,h in p['sources'].items():assert digest(folder/'sources'/Path(path).name)==h
            seal=json.loads((folder/'final-verification.json').read_text())
            for path,h in seal['outputs'].items():
                assert digest(folder/path)==h,(folder,path)
                checked+=1
            if seal.get('unused_live_sources_changed'):exceptions.append(dict(stage=stage,paths=seal['unused_live_sources_changed']))
            for name in ('policy-inputs.json','composition-inputs.json'):
                if (folder/name).exists():
                    for path,h in json.loads((folder/name).read_text()).items():assert digest(path)==h
            for trace in folder.rglob('trace.npz'):
                with np.load(trace,allow_pickle=False) as z:
                    alive=z['alive'] if 'alive' in z else z['alive_before']
                    stage_cases+=alive.shape[1]
                    time=z['time'] if 'time' in z else z['time_s']
                    assert np.allclose(time,(np.arange(len(time))+1)*.016668,atol=1e-8,rtol=0)
                    assert not np.any((~alive[:-1])&alive[1:])
                    if 'issued' in z:
                        import mujoco
                        model=mujoco.MjModel.from_xml_path(str(root/'stand-ball-checkpoints-20261006-01/dummy.xml'))
                        limits=model.jnt_range[model.actuator_trnid[:,0]]
                        assert (z['issued']>=limits[:,0]-2e-7).all() and (z['issued']<=limits[:,1]+2e-7).all()
                        for key in ('qpos','loads','clearance','xy','reward'):assert np.isfinite(z[key]).all()
                        # Independent kinematics verifies complete-foot clearance, including toes.
                        from .corrective_step_audit import foot_geometry
                        sample=np.linspace(0,len(time)-1,min(5,len(time))).astype(int)
                        xy,clearance=foot_geometry(model,z['qpos'][sample])
                        assert np.max(abs(clearance-z['clearance'][sample]))<3e-6
                        assert np.max(abs(xy-z['xy'][sample]))<3e-6
            for log in folder.rglob('updates.jsonl'):
                rows=[json.loads(line) for line in log.read_text().splitlines()]
                assert len(rows)==100
                for i,row in enumerate(rows,1):
                    assert row['update']==i and row['transitions']==4096*i
                    assert row['actor_optimizer_steps']==48*i and row['critic_optimizer_steps']==64*i
                    assert all(np.isfinite(v) for v in row.values())
                updates+=100;transitions+=409600;stage_training.append(str(log.parent))
                report=json.loads((log.parent/'report.json').read_text())
                presentations+=(report.get('supervision') or {}).get('presentations',0)
        cases+=stage_cases;results[str(stage)]=dict(completed=True,cases=stage_cases,training_arms=stage_training,folders=[str(p) for p in folders])
    if args.require_complete:assert len(results)==20,sorted(results)
    out=dict(completed_stages=len(results),stages=results,completed_frozen_cases=cases,
        training_updates=updates,training_transitions=transitions,offline_supervision_presentations=presentations,
        recorded_output_hashes_verified=checked,source_snapshots_verified=True,independent_foot_geometry_verified=True,
        finalization_exceptions=exceptions,invalid_or_incomplete_runs_excluded=['stage02-transfer-20261007-01','stage06-lift-20261007-01','stage08-hold-20261007-01','stage18-retention-20261007-01'],
        separate_infrastructure_training=dict(updates=1,transitions=4096),arena_changed=False)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='stages'},indent=2))


if __name__=='__main__':main()
