#!/usr/bin/env python3
"""Verify current coupled-rule evidence and preservation; no P3 claim."""
import hashlib,json,re,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];E=ROOT/'benchmarks/reconstructed_fault/propagation/quadrature_evidence';B=ROOT/'build-propagation-integration/quadrature-baseline';RUNS=ROOT/'build-propagation-integration/runs'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(p.read_text())
def read(p):return np.genfromtxt(p,names=True,delimiter=',',ndmin=1)
def main():
    baseline=load(B/'baseline.json');changes=load(E/'task-changes.json');changed=changes['modified_existing_files']
    for n,v in baseline['hashes'].items():
        assert sha(B/n)==v,'snapshot '+n
        if n not in changed:assert sha(ROOT/n)==v,'unchanged incoming '+n
    for cmd,expected in [(['git','rev-parse','HEAD'],baseline['head']),(['git','branch','--show-current'],baseline['branch'])]:assert subprocess.check_output(cmd,cwd=ROOT,text=True).strip()==expected
    marker='diff --git a/doc/reconstructed_fault/pf_rsf.tex b/doc/reconstructed_fault/pf_rsf.tex\n';incoming=marker+(B/'incoming.diff').read_text().split(marker)[1].split('diff --git ',1)[0]
    assert subprocess.check_output(['git','diff','--binary','--','doc/reconstructed_fault/pf_rsf.tex'],cwd=ROOT,text=True)==incoming
    # The completed plane-strain task already updated this source after the
    # localization manifest. Verify its incoming diff exactly, rather than
    # treating that historical update as part of the quadrature task.
    historical_updates=changes['pre_existing_historical_source_updates']
    for n in historical_updates:
        marker=f'diff --git a/{n} b/{n}\n'
        incoming=marker+(B/'incoming.diff').read_text().split(marker)[1].split('diff --git ',1)[0]
        assert subprocess.check_output(['git','diff','--binary','--',n],cwd=ROOT,text=True)==incoming
    old_counts={}
    for label,path in [('plane',E/'incoming-plane-strain-manifest.json'),('localization',ROOT/'benchmarks/reconstructed_fault/propagation/localization_evidence/manifest.json')]:
        count=0
        for n,v in load(path)['hashes'].items():
            if n not in changed and not (label=='localization' and n in historical_updates):
                assert sha(ROOT/n)==v,'preserved '+n;count+=1
        old_counts[label]=count
    for label in ['native-build.json','manifest.json']:
        payload=load(E/label);hashes=payload['hashes'] if label=='manifest.json' else payload
        for n,v in hashes.items():assert sha(ROOT/n)==v,'current '+n
    selected=load(E/'selection.json');assert selected['selected']==6
    for q in [4,6,12]:
        for degree in [2,3]:
            out=E/f'one-d-n16-d{degree}-q{q}';s=load(out/'summary.json');a=s['audit_max'];f=np.load(out/'fields.npz');hist=load(out/'history.json')
            assert len(hist)==40 and hist[-1]['time']==.625 and all(row['KKT']<=1e-11 for row in hist)
            assert min(f['phase'].ravel())>=0 and max(f['phase'].ravel())<1
            assert min(np.diff(np.r_[f['initial'][None,:],f['phase']],axis=0).ravel())>=-4.922e-12
            assert a['higher_reference_energy_error']<1e-11 and a['higher_reference_gradient_error']<1e-11
            assert a['derivative_relative_error']<1e-7 and a['Hessian_derivative_relative_error']<1e-7
            if q==6:
                assert a['phase_gradient_error']<1e-5 and a['phase_Hessian_relative_error']<1e-4
                assert max(a['bulk_energy_error'],a['endpoint_bulk_energy_error'])<1e-5
                assert max(a['endpoint_bulk_force_error_Pa'],a['endpoint_surface_force_error_Pa'])<.01
                other=np.load(E/f'one-d-n16-d{degree}-q12'/'fields.npz');assert max(abs(f['phase']-other['phase']).ravel())<1e-5
            if q==4:assert a['phase_gradient_error']>1e-5 or a['endpoint_bulk_force_error_Pa']>.01
    assessment=load(E/'assessment.json')
    for label in ['q2','q3','q2-refined']:
        d=assessment[label];run=RUNS/('output-quadrature-'+label);log=(ROOT/'build-propagation-integration'/('quadrature-'+label+'.log')).read_text()
        assert d['cost']['returncode']==0 and d['cost']['threads']==1 and d['cost']['wall_seconds']>0 and d['cost']['peak_memory_kib']>0
        assert d['healing']==0 and d['recurrence_error']==0 and max(d['mechanical_residuals'])<1e-8
        assert d['matched_prototype_errors']['phi']<2e-8 and d['matched_prototype_errors']['q']<4e-5
        targets=[float(v) for v in re.findall(r'Phase box target:.*? target=([^\s]+)',log)];defects=[float(v) for v in re.findall(r'Phase box converged:.*? KKT=([^\s]+)',log)]
        assert len(targets)==len(defects)==41 and all(k<=t for k,t in zip(defects,targets))
        previous=read(run/'fe-phase-0.csv')['phi']
        for j in range(1,41):
            current=read(run/f'fe-phase-{j}.csv')['phi'];assert min(current)>=0 and max(current)<1 and min(current-previous)>=-4.922e-12;previous=current
        bulk=read(run/'fe-bulk-40.csv');assert min(bulk['w'])>0 and abs(sum(bulk['w'])-1)<1e-12
        assert all(v['source_error']<.005 and v['I_min']>0 for v in d['normalization'])
        assert max(d['independent_force'].values())<.01
        q=d['frozen_quadrature'];assert abs(q['12']['bulk']-q['24']['bulk'])<1e-11 and abs(q['6']['bulk']-q['24']['bulk'])<1e-5
        for a in d['incoming_anchor_audits']:
            assert a['incoming_step']==a['step']-1 and a['phase_gradient_error']<1e-5
            assert min(v['energy_derivative_relative_error'] for v in a['native_derivatives'])<1e-7
            assert min(v['Hessian_direction_relative_error'] for v in a['native_derivatives'])<1e-7
        for c in d['curvature']:
            if c['free_above_healing_allowance']:
                assert c['min_eigenvalue']>0 and c['negative_count']==0
                assert min(c['energy_minus'],c['energy_plus'])>c['energy_arithmetic']
                assert max(abs(c['energy_minus']-c['quadratic_prediction']),abs(c['energy_plus']-c['quadratic_prediction']))<1e-12
        w=d['prototype_work'];assert abs(w['absolute_boundary_error']-d['absolute_boundary_mechanical_error'])<1e-6
        assert abs(w['absolute_internal_error']-d['absolute_internal_mechanical_error'])<1e-6
        if label!='q2-refined':assert d['three_point_changes']['initial_phase_identical']
    for n in ['boundary_state.csv','ih_automatic_completion_rank0.csv']:
        assert sha(RUNS/'output-quadrature-mature'/n)==sha(ROOT/'build-propagation-integration/output-native-fe-mature-final'/n)
    assert assessment['recommendation']=='q2-refined' and assessment['q2-refined']['physical_target_pass']
    assert not assessment['q2']['physical_target_pass'] and not assessment['q3']['physical_target_pass']
    assert not assessment['P3_ready'] and not assessment['stress_accuracy_qualified'] and not assessment['time_accuracy_qualified']
    assert sha(ROOT/'source/utilities.cc')==baseline['hashes']['source/utilities.cc']
    result=dict(snapshot_files=len(baseline['hashes']),preserved_historical_artifacts=old_counts,new_manifest_files=len(load(E/'manifest.json')['hashes']),native_controls_completed=3,selected_rule=6,proper_incoming_anchor_checks=True,native_bounds_KKT_history_pass=True,physical_targets_met_by='16x32 Q2',mature_default_identical_files=2,stress_energy_spatial_qualification=False,full_stability_qualification=False,time_qualification=False,P3_ready=False)
    (E/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
