#!/usr/bin/env python3
"""Read-only evidence audit; writes only the current verification summary."""
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
E=ROOT/'benchmarks/reconstructed_fault/propagation/plane_strain_evidence'
B=ROOT/'build-propagation-integration/plane-strain-baseline'
RUNS=ROOT/'build-propagation-integration/runs'
EPS=np.finfo(float).eps

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(p.read_text())
def read(p):return np.genfromtxt(p,delimiter=',',names=True,ndmin=1)

def main():
    baseline=load(B/'baseline.json');changed=load(E/'task-changes.json')['modified_existing_files']
    for n,v in baseline['hashes'].items():
        assert sha(B/n)==v, 'snapshot '+n
        if n not in changed:assert sha(ROOT/n)==v, 'unchanged '+n
    for cmd,expected in [(['git','rev-parse','HEAD'],baseline['head']),(['git','branch','--show-current'],baseline['branch'])]:
        assert subprocess.check_output(cmd,cwd=ROOT,text=True).strip()==expected
    marker='diff --git a/doc/reconstructed_fault/pf_rsf.tex b/doc/reconstructed_fault/pf_rsf.tex\n'
    original=marker+(B/'incoming.diff').read_text().split(marker)[1].split('diff --git ',1)[0]
    assert subprocess.check_output(['git','diff','--binary','--','doc/reconstructed_fault/pf_rsf.tex'],cwd=ROOT,text=True)==original
    preserved=0
    for n,v in load(E/'incoming-localization-manifest.json')['hashes'].items():
        if n not in changed:
            assert sha(ROOT/n)==v, 'historical '+n
            preserved+=1
    for n,v in load(E/'native-build.json').items():assert sha(ROOT/n)==v, 'build '+n
    for n,v in load(E/'manifest.json')['hashes'].items():assert sha(ROOT/n)==v, 'current '+n
    # The consumer inventory covers every qualified helper call in source,
    # excluding their definitions and explicit template instantiations.
    with (E/'tensor-consumers.csv').open() as f:rows=list(csv.DictReader(f))
    recorded={(r['file'],int(r['line'])) for r in rows}
    actual=set()
    for p in (ROOT/'source').rglob('*.cc'):
        for i,line in enumerate(p.read_text().splitlines(),1):
            if 'Utilities::Tensors::consistent_deviator(' in line or 'Utilities::Tensors::consistent_second_invariant_of_deviatoric_tensor(' in line:
                actual.add((str(p.relative_to(ROOT)),i))
    assert recorded==actual, ('consumer inventory',recorded^actual)
    before=RUNS/'output-plane-strain-before';final=RUNS/'output-plane-strain-final-small'
    for run in [before,final]:
        q=read(run/'plane-strain-algebra.csv')
        assert max(q['helper_block_error'])==0 and max(q['invariant_error'])<=EPS
        assert q[-1]['repeat_deviator_error']>.1
    old=read(before/'plane-strain-operator.csv')[0];new=read(final/'plane-strain-operator.csv')[0]
    assert abs(old['native']-old['raw_sym'])<=256*EPS*old['raw_sym']
    assert abs(old['native_minus_plane']-old['missing_subtraction'])<=256*EPS*old['raw_sym']
    assert abs(new['native_minus_plane'])<=256*EPS*new['raw_sym']
    small_identical=0
    for j in range(5):
        for label in ['phase','surface']:
            n=f'fe-{label}-{j}.csv';assert sha(before/n)==sha(final/n);small_identical+=1
        x=read(before/f'fe-encoding-{j}.csv');y=read(final/f'fe-encoding-{j}.csv')
        assert max(abs(x['d_R']-y['d_R']))<5e-18
    local=read(final/'fe-local-0.csv')
    assert len(local)==6 and set(local['field_mode'])=={0.,1.}
    assert set(local['ratio'])=={.00125,1.,4.}
    assert all(local['stress_error_Pa']<=256*EPS*local['stress_scale_Pa'])
    assert max(local[local['field_mode']==1]['max_raw_trace'])>2e-5
    mature=RUNS/'output-plane-strain-mature-final';oldm=ROOT/'build-propagation-integration/output-native-fe-mature-final'
    for n in ['boundary_state.csv','ih_automatic_completion_rank0.csv']:assert sha(mature/n)==sha(oldm/n)
    algebra=load(E/'algebra.json')
    assert not algebra['pressure_redefined']
    assert all(r['energy_derivative_relative_error']<1e-9 and r['phase_derivative_relative_error']<1e-8 for r in algebra['cases'])
    assert all(r['FE_recurrence_relative_error']<2e-14 and r['finite_step_bulk_work_relative_error']<=256*EPS for r in algebra['general_Maxwell'])
    a=load(E/'assessment.json')
    for name in ['q2','q3']:
        d=a[name];run=RUNS/('output-plane-strain-compatible-'+name)
        log=(ROOT/'build-propagation-integration'/('plane-strain-compatible-'+name+'.log')).read_text()
        assert d['steps']==40 and d['time']==.625 and d['healing']==0 and d['phi_max']<1 and d['recurrence_error']==0
        assert max(d['mechanical_residuals'])<=1e-8 and d['source_error']<.005
        targets=[float(v) for v in re.findall(r'Phase box target:.*? target=([^\s]+)',log)]
        defects=[float(v) for v in re.findall(r'Phase box converged:.*? KKT=([^\s]+)',log)]
        assert len(targets)==len(defects)==41 and all(k<=t for k,t in zip(defects,targets))
        previous=read(run/'fe-phase-0.csv')['phi']
        for j in range(1,41):
            current=read(run/f'fe-phase-{j}.csv')['phi']
            assert min(current)>=0 and max(current)<1 and min(current-previous)>=-4.922e-12
            previous=current
        for c in d['curvature']:
            if c['free_above_healing_allowance']:
                assert c['min_eigenvalue']>0 and c['negative_count']==0
                assert min(c['energy_minus'],c['energy_plus'])>c['energy_arithmetic']
                assert max(abs(c['energy_minus']-c['quadratic_prediction']),abs(c['energy_plus']-c['quadratic_prediction']))<1e-12
        q=d['frozen_normal_quadrature']
        assert abs(q['12']['bulk']-q['24']['bulk'])<=256*EPS*max(1.,q['24']['bulk'])
    assert a['raw_vs_plane_q2_phase_difference']<4e-15
    assert a['q3']['frozen_normal_quadrature']['3']['stress_rms']<1e-7
    assert a['q3']['frozen_normal_quadrature']['24']['stress_rms']>400
    assert a['q3']['frozen_normal_quadrature']['3']['bulk']/a['q3']['frozen_normal_quadrature']['24']['bulk']<.91
    result=dict(snapshot_files=len(baseline['hashes']),preserved_historical_artifacts=preserved,consumer_sites=len(rows),native_helper_operator_general_Maxwell_pass=True,small_shear_identical_files=small_identical,mature_default_identical_files=2,loaded_controls_completed=2,unchanged_KKT_and_healing_allowances=True,FE_recurrence_exact=True,quadrature_aliasing_quantified=True,full_constrained_stability_verified=False,native_spatial_time_convergence_verified=False,P3_ready=False)
    (E/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
