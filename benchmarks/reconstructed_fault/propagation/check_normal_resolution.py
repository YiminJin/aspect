#!/usr/bin/env python3
"""Read-only verification of the bounded Q2 resolution decision and preservation."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
from normal_resolution import E,OLD,FINE,load,directory
from assess_localization import read,grid
ROOT=Path(__file__).resolve().parents[3]
DOCS={'AGENTS.md','doc/reconstructed_fault/current_design.md','doc/reconstructed_fault/CURRENT_STATUS.md','doc/reconstructed_fault/propagation/STATUS.md','doc/reconstructed_fault/propagation/PLAN.md','benchmarks/reconstructed_fault/propagation/README.md'}
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    baseline=load(E/'baseline.json');preserved=0
    for n,v in load(E/'incoming-quadrature-manifest.json')['hashes'].items():
        if n in DOCS or n in {'source/simulator/phase_field.cc','build-propagation-integration/aspect-release','benchmarks/reconstructed_fault/propagation/native_fe_strain.cc','build-propagation-integration/plugin/libnative_fe_strain.release.so'}:continue
        assert sha(ROOT/n)==v,'preserved incoming '+n;preserved+=1
    assert sha(E/'phase_field.cc.before-iterator-fix')==load(E/'incoming-quadrature-manifest.json')['hashes']['source/simulator/phase_field.cc']
    assert sha(E/'native_fe_strain.cc.before-probe-fix')==load(E/'incoming-quadrature-manifest.json')['hashes']['benchmarks/reconstructed_fault/propagation/native_fe_strain.cc']
    for n,v in load(E/'native-build.json').items():assert sha(ROOT/n)==v,'unchanged production/build '+n
    for n,v in load(E/'manifest.json')['hashes'].items():assert sha(ROOT/n)==v,'current '+n
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==baseline['head']
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()==baseline['branch']
    selection=load(E/'selection.json');assert selection['dt']==1/64 and selection['final_time']==.625 and selection['normal_rule']==6
    ns=[16,32,64,128,256];assert selection['selected_cells']==256
    for n in ns:
        d=directory(n);f=np.load(d/'fields.npz');hist=load(d/'history.json');s=load(d/'summary.json')
        assert len(hist)==40 and hist[-1]['time']==.625 and s['normal_points']==6 and s['degree']==2
        assert max(r['KKT'] for r in hist)<=1e-11 and min(f['phase'].ravel())>=0 and max(f['phase'].ravel())<1
        assert min(np.diff(np.r_[f['initial'][None,:],f['phase']],axis=0).ravel())>=-4.922e-12
        summary=selection['evolving'][str(n)];err=summary['errors'];passed=all(err[k]<=v for k,v in baseline['targets'].items());assert passed==summary['targets_pass']==(n==256)
        assert summary['quadrature_energy_24_48']<1e-11
        rows=load(E/f'n{n}-stress-history.json');assert len(rows)==40
        for row in rows:
            assert abs(row['bulk_energy']-row['mean_stress_energy']-row['oscillation_energy'])<1e-11
            assert row['stress_min_Pa']<=row['mean_stress_Pa']<=row['stress_max_Pa'] and row['interface_jump_max_Pa']>=0
            assert abs(row['whole_error_RMS_Pa']**2-row['mean_error_Pa']**2-row['oscillation_RMS_Pa']**2)<1e-7*max(1.,row['whole_error_RMS_Pa']**2)
    pairs={(r['step'],r['cells']):r for r in selection['frozen'] if r['integration_points']==24}
    other={(r['step'],r['cells']):r for r in selection['frozen'] if r['integration_points']==48}
    for key,row in pairs.items():
        assert row['weak_force_Pa']<1e-7 and abs(row['source_integral']-1)<1e-12 and abs(row['mean_error_Pa'])<1e-7
        assert abs(row['bulk_energy']-other[key]['bulk_energy'])<1e-11 and abs(row['localized_error_RMS_Pa']-other[key]['localized_error_RMS_Pa'])<1e-7
    native=load(E/'native-assessment.json');assert native['cells']==256 and native['cost']['returncode']==0 and native['cost']['threads']==1
    assert native['matched_discrete_errors']['phi']<2e-8 and native['matched_discrete_errors']['q']<4e-5
    assert native['maximum_healing']==0 and native['recurrence_error']<1e-14 and native['normal_stress_max_Pa']<1e-7
    assert max(native['mechanical_residuals'])<1e-8
    for row in native['normalization']:assert row['min_I']>0 and row['maximum_source_error']<.005
    assert native['proper_incoming_steps']==[31,39]
    for row in native['incoming_anchor_audits']:
        assert row['incoming_step']==row['step']-1
        assert row['phase_gradient_error']<1e-5 and row['phase_Hessian_relative_error']<1e-4
    for row in native['phase_curvature']:
        if row['free_above_healing_allowance']:
            assert row['min_eigenvalue']>0 and row['negative_count']==0
            assert min(row['energy_minus'],row['energy_plus'])>row['energy_arithmetic']
    for d in native['phase_derivatives']:
        assert min(r['energy_derivative_relative_error'] for r in d['rows'])<1e-7
        assert min(r['Hessian_direction_relative_error'] for r in d['rows'])<1e-7
    for rows in native['independent_bulk_integration'].values():assert abs(rows['24']['bulk']-rows['48']['bulk'])<1e-11
    # Numerical publication/curvature passes are separate from spatial targets.
    history=load(E/'native-stress-history.json');assert len(history)==40
    assert max(r['whole_error_RMS_Pa'] for r in history)/selection['stress_scale_Pa']<.05
    assert max(r['localized_error_RMS_Pa'] for r in history)/selection['stress_scale_Pa']<.05
    assert max(abs(r['bulk_error']) for r in history)/selection['bulk_storage_scale_J_m']<.05
    assert native['fine_errors']['phi']<.01 and native['fine_errors']['q']<10 and native['fine_errors']['slip_relative']<.005
    result=dict(preserved_incoming_artifacts=preserved,production_change='Cached row-end iterators in phase constraint enforcement; rebuilt executable',physical_or_numerical_algorithm_change=False,selected_native_mesh='16x256 Q2/Q1',normal_quadrature=6,spatial_benchmark_targets_pass=True,proper_incoming_native_curvature_checked=True,time_qualification=False,full_energy_or_stability_qualification=False,P3_ready=False)
    (E/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
