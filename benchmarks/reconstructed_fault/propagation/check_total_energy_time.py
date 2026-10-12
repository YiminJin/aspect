#!/usr/bin/env python3
"""Preservation and independent saved-state gates for prototype refinement."""
import hashlib
import json
import subprocess

from total_energy_step import ROOT, HERE
from assess_total_energy_time import checks


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_values(a,b,path='',diagnostic_changes=False):
    if isinstance(a,dict):
        for key,value in a.items():
            if diagnostic_changes and key in ['mechanical_kkt','free_bulk_residual','free_bulk_work']:
                assert abs(b[key]) < (1e-12 if key=='free_bulk_work' else 1e-9)
            else:original_values(value,b[key],path+'/'+key,diagnostic_changes)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for j,(x,y) in enumerate(zip(a,b)):original_values(x,y,path+'/'+str(j),diagnostic_changes)
    else:assert a==b,(path,a,b)


def main():
    out=HERE/'total_energy_time_evidence'
    manifest=json.loads((out/'manifest.json').read_text())
    baseline_dir=ROOT/'build-propagation-integration/total-energy-time-baseline'
    baseline=json.loads((baseline_dir/'baseline.json').read_text())
    allowed=set(manifest['authorized_existing_changes'])
    for name,value in baseline['hashes'].items():
        assert digest(baseline_dir/name)==value,('Preserved copy changed',name)
        if name not in allowed:assert digest(ROOT/name)==value,('Incoming artifact changed',name)
    for name,value in manifest['hashes'].items():
        assert digest(ROOT/name)==value,('Task artifact changed',name)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==baseline['head']
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()==baseline['branch']
    old=HERE/'total_energy_evidence/coupled_blocks'
    new=HERE/'total_energy_evidence/time_compatibility'
    original_values(json.loads((old/'summary.json').read_text())['results'],
                    json.loads((new/'summary.json').read_text())['results'])
    assert (old/'phase-fields.json').read_bytes()==(new/'phase-fields.json').read_bytes()
    final=HERE/'total_energy_evidence/time_compatibility_final'
    original_values(json.loads((old/'summary.json').read_text())['results'],
                    json.loads((final/'summary.json').read_text())['results'],diagnostic_changes=True)
    assert (old/'phase-fields.json').read_bytes()==(final/'phase-fields.json').read_bytes()
    assessment=json.loads((out/'assessment.json').read_text())
    assert assessment['prototype_only'] and not assessment['production_changed']
    for name,saved in assessment['checks'].items():
        original_values(saved,checks(name))
    print('PASS:',len(baseline['hashes']),'incoming artifacts preserved;',
          len(assessment['checks']),'common-path numerical gates; original 27-step controls reproduced.')
    print('Qualification remains isolated/discrete; production, native histories and P3 unchanged.')


if __name__=='__main__':main()
