#!/usr/bin/env python3
"""Replay fixed-local-phase evidence; do not launch ASPECT or alter raw runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from assess_pf6_hanging import assess
from assess_pf6_local import LocalState, compare_exports
from assess_pf6_transfer import quadrature, norm2


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identical(first, second, label):
    files=sorted(p for p in (first/'run').glob(label+'-*') if p.is_file())
    assert files
    for directory in sorted((first/'run').glob(label+'-reference')):
        files.extend(p for p in sorted(directory.iterdir()) if p.is_file())
    result={str(p.relative_to(first/'run')):sha(p)==sha(second/'run'/p.relative_to(first/'run')) for p in files}
    assert all(result.values()),result
    return dict(exact_files=result,common_functional=compare_exports(first,second,label,label))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,default=Path('build-particle-free/runs'))
    p.add_argument('--protected',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();here=Path(__file__).resolve().parent
    run=lambda name:a.runs/(name+'-20261019')
    primary,parent,resumed=map(run,['hanging-final','hanging-parent','hanging-resumed-final'])
    cut,matched,half=map(run,['hanging-cutback','hanging-matched','hanging-half'])
    labels=['hanging-final','hanging-parent','hanging-resumed-final','hanging-cutback','hanging-matched','hanging-half']
    result=dict(runs=[assess(run(label)) for label in labels],
        strict_reference_match_passed=False,scope='Serial fixed-mesh master-space qualification; no spatial/branch/propagation qualification')
    baseline=json.loads((here/'uniform-baseline.json').read_text())
    uniform=run('hanging-uniform-final')/'run'
    result['uniform_identity']={name:sha(uniform/name)==value for name,value in baseline.items()}
    assert len(baseline)==162 and all(result['uniform_identity'].values())
    result['parent_prefix']=identical(primary,parent,'state-1')
    result['restart']=identical(primary,resumed,'state-2')
    result['cutback_same_accepted_sequence']={str(i):identical(cut,matched,f'state-{i}') for i in [1,2,3]}
    attempts=np.atleast_1d(np.genfromtxt(cut/'run/cutback.csv',delimiter=',',names=True))
    assert len(attempts)==1 and tuple(attempts[0])==(2,.25,.125,.1875,.0625,1)
    result['actual_cutback']={k:float(attempts[0][k]) for k in attempts.dtype.names}
    # Compare only fields at the same physical time. Old/older native levels
    # refer to different physical times when dt differs and are not errors.
    coarse=LocalState(primary/'run','state-2');fine=LocalState(half/'run','state-4')
    points,weights=quadrature([coarse,fine],24);c,f=coarse.evaluate(points),fine.evaluate(points)
    temporal={}
    for name in ['T','tau']:
        temporal[name+'_relative_L2']=float(np.sqrt(weights@norm2(c[name]-f[name])/(weights@norm2(f[name]))))
    temporal['physical_phase_Linf']=float(max(abs(c['phase'][:,0]-f['phase'][:,0])))
    temporal['physical_phase_L2']=float(np.sqrt(weights@((c['phase'][:,0]-f['phase'][:,0])**2)))
    temporal['phase_gradient_Linf']=float(np.max(abs(c['gradient']-f['gradient'])))
    result['bounded_dt_half_comparison']=temporal
    if a.protected:
        protected=json.loads(a.protected.read_text())['files']
        result['protected_BP3']={name:sha(Path(name))==item['sha256'] for name,item in protected.items()}
        assert len(protected)==34 and all(result['protected_BP3'].values())
    for r in result['runs']:
        # Normalize the initial raw diagnostic's mistaken dimensional label;
        # no raw bytes or values are changed. The final observer uses J/m.
        for row in r['trajectory']:
            if 'KKT_Pa_m' in row:row['KKT_J_per_m']=row.pop('KKT_Pa_m')
    result['passed']=True
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,uniform_files=162,restart_files=len(result['restart']['exact_files']),cutbacks=1,temporal=temporal)))

if __name__=='__main__':main()
