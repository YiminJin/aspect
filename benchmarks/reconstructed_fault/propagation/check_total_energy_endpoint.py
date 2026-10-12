#!/usr/bin/env python3
"""Saved-field, common-fixture and preservation checks for endpoint comparison."""
import hashlib
import json
import subprocess
import numpy as np

from assess_total_energy_endpoint import (OUT, COUNTS, name, rates_comparison,
    field_locations, state_update_errors)
from assess_total_energy_time import load, checks, compare
from total_energy_step import ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    baseline_dir=ROOT/'build-propagation-integration/total-energy-endpoint-baseline'
    baseline=json.loads((baseline_dir/'baseline.json').read_text())
    manifest=json.loads((OUT/'manifest.json').read_text())
    allowed=set(manifest['authorized_existing_changes'])
    for path,value in baseline['hashes'].items():
        assert digest(baseline_dir/path)==value,('Preserved copy changed',path)
        if path not in allowed:assert digest(ROOT/path)==value,('Incoming artifact changed',path)
    for path,value in manifest['hashes'].items():
        assert digest(ROOT/path)==value,('New evidence changed',path)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==baseline['head']
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()==baseline['branch']
    assessment=json.loads((OUT/'assessment.json').read_text())
    assert assessment['prototype_only'] and not assessment['production_changed']
    audit=json.loads((OUT/'native-entry-rank-audit.json').read_text())
    assert not audit['native_model_implemented'] and not audit['native_eigenvalue_run']
    for path,value in audit['inputs'].items():assert digest(ROOT/path)==value,path
    state_audit=json.loads((OUT/'native-entry-state-audit.json').read_text())
    assert not state_audit['native_state_implemented']
    assert state_update_errors()==state_audit['accepted_R_update_error']
    for path,value in state_audit['inputs'].items():assert digest(ROOT/path)==value,path
    fixture=None
    for scheme in ['mean','endpoint']:
        for n in COUNTS+([512] if scheme=='endpoint' and name('endpoint',512) in assessment['checks'] else []):
            key=name(scheme,n)
            saved=assessment['checks'][key]
            result=checks(key)
            for field,value in result.items():assert saved[field]==value,(key,field)
            summary,records,fields=load(key)
            signature={field:summary[field] for field in ['amplitude','final_time','material',
                'aligned_bulk_quadrature','bulk_samples','input_sha256','initial_phi_max']}
            if fixture is None:fixture=signature
            assert signature==fixture,('Different physical fixtures',key)
            assert summary['steps']==n and summary['dt']==1/n and summary['phase_evolves']
            for j,r in enumerate(records):
                assert np.max(abs(fields['slip'][j+1]-fields['slip'][j]
                                  -r['dt']*np.array(r['slip_rate'])))<1e-13
    for scheme in ['mean','endpoint']:
        for a,b in zip(COUNTS[:-1],COUNTS[1:]):
            value=compare(name(scheme,a),name(scheme,b))
            value['rates']=rates_comparison(name(scheme,a),name(scheme,b))
            value['worst_locations']=field_locations(name(scheme,a),name(scheme,b))
            assert value==assessment['comparisons'][f'{scheme}{a}_vs_{b}']
    for n in COUNTS:
        value=compare(name('mean',n),name('endpoint',n))
        value['rates']=rates_comparison(name('mean',n),name('endpoint',n))
        value['worst_locations']=field_locations(name('mean',n),name('endpoint',n))
        assert value==assessment['comparisons'][f'mean_vs_endpoint{n}']
    if name('endpoint',512) in assessment['checks']:
        for scheme in ['mean','endpoint']:
            value=compare(name(scheme,256),name('endpoint',512))
            value['rates']=rates_comparison(name(scheme,256),name('endpoint',512))
            value['worst_locations']=field_locations(name(scheme,256),name('endpoint',512))
            assert value==assessment['comparisons'][f'{scheme}256_vs_endpoint512']
    print('PASS:',len(baseline['hashes']),'incoming artifacts preserved;',len(assessment['checks']),'saved-field '
        'bounds/KKT/Maxwell/work checks; identical fixtures and common-time/rate comparisons.')
    print('Numerical pass counts do not qualify temporal accuracy or native/P3 readiness.')


if __name__=='__main__':main()
