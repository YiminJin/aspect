#!/usr/bin/env python3
"""Independent work/KKT/preservation checks; native adoption remains blocked."""
import hashlib
import json
import subprocess
import numpy as np
from numpy.polynomial.legendre import leggauss
from profile_temporal_diagnosis import E, OLD, directory, fields, Reference, G, ETA, h, load

ROOT=E.parents[3]
DOCS={'AGENTS.md','doc/reconstructed_fault/current_design.md',
      'doc/reconstructed_fault/CURRENT_STATUS.md',
      'doc/reconstructed_fault/propagation/STATUS.md',
      'doc/reconstructed_fault/propagation/PLAN.md',
      'benchmarks/reconstructed_fault/propagation/README.md'}


def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def reaction_trace(r,path,d):
    phi,R,Z=fields(path);hist=load(path/'history.json')
    diag=load(E/('diagnosis-'+str(d) if path==directory(d) else 'smooth-diagnosis-'+str(d))/'history.json')
    dt=1/d;beta=np.exp(-G*dt/ETA);kappa=-ETA*np.expm1(-G*dt/ETA)
    cb=ETA/G*(-np.expm1(-G*dt/ETA));ck=ETA*(dt-cb)
    trace_total=0.;reaction_total=0.;maxdifference=0.;maxquadrature=0.
    for j,row in enumerate(hist,1):
        I=r.direct(phi[j])[0];chi=h(r.B@phi[j])/I
        u=(R[j]-beta*R[j-1])*G/kappa
        tau_star=G*r.D@R[j-1]-Z[j-1]*chi
        integral=cb*tau_star+ck*(r.D@u-chi*row['V'])
        force=r.D[:,[0,-1]].T@(r.w*integral)
        W=u[[0,-1]]@force
        maximum=diag[j-1]['reaction_work']
        maxquadrature=max(maxquadrature,abs(W-maximum))
        # Direct off-assembly-point stress traces at the two normal boundaries.
        derivative=lambda v:np.array([-3*v[0]+4*v[1]-v[2],v[-3]-4*v[-2]+3*v[-1]])/r.dy
        chib=h(phi[j][[0,-1]])/I
        stress0=G*derivative(R[j-1])-Z[j-1]*chib
        stress_integral=cb*stress0+ck*(derivative(u)-chib*row['V'])
        trace=u[-1]*stress_integral[1]-u[0]*stress_integral[0]
        trace_total+=trace;reaction_total+=W
        maxdifference=max(maxdifference,abs(W-trace))
    return dict(boundary_reaction_work=reaction_total,boundary_stress_trace_work=trace_total,
                cumulative_trace_difference=trace_total-reaction_total,
                maximum_step_trace_difference=maxdifference,
                maximum_analytic_vs_time_quadrature_difference=maxquadrature)


def main():
    baseline=load(E/'baseline.json');preserved=0;unique=set()
    for manifest in [baseline['hashes'],load(E/'incoming-time-manifest.json')['hashes'],load(OLD/'incoming-normal-manifest.json')['hashes']]:
        for n,v in manifest.items():
            if n not in DOCS:
                assert sha(ROOT/n)==v,'incoming changed: '+n;preserved+=1;unique.add(n)
    for n,v in load(E/'manifest.json')['hashes'].items():assert sha(ROOT/n)==v,'current changed: '+n
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==baseline['head']
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()==baseline['branch']
    for n,before in load(E/'documentation-before.json').items():
        old=(ROOT/before).read_text();current=(ROOT/n).read_text()
        if n=='AGENTS.md':assert current.startswith(old)
        elif n=='doc/reconstructed_fault/current_design.md':
            addition=load(E/'documentation-additions.json')['design'];assert current.replace(addition,'',1)==old
        else:assert current.endswith(old)
    initial=np.load(E/'baseline-diagnostics/initial.npz');r=Reference(256,1.,6)
    assert np.array_equal(r.initial,initial['phi']) and np.count_nonzero(initial['r'])==0 and initial['Z']==0
    traces={};max_kkt=0.;max_identity=0.
    for d in [64,128,256,512,1024,2048,4096]:
        phi,R,Z=fields(directory(d));hist=load(directory(d)/'history.json')
        assert len(hist)==round(.625*d) and np.array_equal(phi[0],initial['phi'])
        assert min(phi.ravel())>=0 and max(phi.ravel())<1 and min(np.diff(phi,axis=0).ravel())>=-4.922e-12
        diagnosis=load(E/f'diagnosis-{d}/history.json')
        summary=load(E/f'diagnosis-{d}/summary.json')
        assert summary['maximum_incoming_phase_KKT']<=1e-11
        max_kkt=max(max_kkt,summary['maximum_incoming_phase_KKT'])
        assert summary['maximum_work_identity_error']<1e-10
        max_identity=max(max_identity,summary['maximum_work_identity_error'])
        for j,(row,probe) in enumerate(zip(hist,diagnosis),1):
            assert row['time']==probe['time']==j/d
            assert row['force']<1e-7 and abs(row['stick_slip'])<1e-7
            assert abs(probe['incoming_phase_KKT']-row['KKT'])<1e-13
        if d>1024:
            snapshot=np.load(directory(d)/'initial.npz')
            assert all(np.array_equal(snapshot[k],initial[k]) for k in initial.files)
            assert abs(load(directory(d)/'summary.json')['loading_displacement']-.05)<1e-14
        traces[str(d)]=reaction_trace(r,directory(d),d)
        assert traces[str(d)]['maximum_analytic_vs_time_quadrature_difference']<1e-8
    work=[];forces=[];rates=[]
    for d in [64,256,1024]:
        path=E/f'smooth-dt-1-{d}';phi,R,Z=fields(path)
        s=np.clip((np.arange(len(phi))/d-.5)/.125,0.,1.)
        profile=initial['phi']+.08*(10*s**3-15*s**4+6*s**5)[:,None]*np.exp(-((r.y-.5)/.15)**2)
        assert np.array_equal(phi,profile)
        assert min(np.diff(phi,axis=0).ravel())>=-1e-14 and max(phi.ravel())<1
        summary=load(E/f'smooth-diagnosis-{d}/summary.json')
        work.append(summary['totals']['actual_work_difference']['absolute'])
        forces.append(summary['maxima']['redistributed_interior_force_max'])
        rates.append(summary['maxima']['delta_chi_rate_max'])
        traces['smooth-'+str(d)]=reaction_trace(r,path,d)
    assert 3.5<work[0]/work[1]<4.5 and 3.5<work[1]/work[2]<4.5
    assert 3.5<forces[0]/forces[1]<4.5 and 3.5<forces[1]/forces[2]<4.5
    assert max(rates)/min(rates)<1.05
    derivative=load(E/'endpoint-replay-1024/derivative-verification.json')
    for name in ['sequential','coupled_trial']:
        rows=derivative[name]['derivative_checks']
        assert min(v['energy_derivative_error'] for v in rows)<1e-7
        assert min(v['Hessian_direction_error'] for v in rows)<1e-5
    candidate=load(E/'endpoint-replay-1024/summary.json')
    assert candidate['final_endpoint_KKT']<=1e-11 and candidate['mechanical']['force']<1e-7
    assert candidate['physical_history_published'] is False and candidate['profile_energy_change']>1
    assert candidate['phase_increment']>.01 and candidate['reduced_curvature_before']['minimum']<0
    f=np.load(E/'endpoint-replay-1024/trial.npz');phi,R,Z=fields(directory(1024))
    assert np.array_equal(f['lower'],phi[-2]) and np.array_equal(f['incoming_R'],R[-2]) and f['incoming_Z']==Z[-2]
    assert min(f['phase']-f['lower'])>=-4.922e-12
    comparison=load(E/'comparison.json')
    assert comparison['native_gate_pass'] is False and comparison['selected_native_timestep'] is None
    assert comparison['evolving']['4096']['maxima']['endpoint_phase_KKT']>1e-6
    (E/'reaction-trace-verification.json').write_text(json.dumps(traces,indent=2)+'\n')
    result=dict(incoming_hash_checks=preserved,unique_preserved_files=len(unique),old_trajectories_unchanged=True,
                production_and_governing_specifications_unchanged=True,
                maximum_solved_phase_KKT=max_kkt,maximum_work_identity_error=max_identity,
                endpoint_phase_diagnostic_is_not_a_solver_failure=True,
                smooth_profile_first_order_work_scaling=True,
                boundary_reaction_and_lifting_dependence_audited=True,
                consistent_endpoint_derivative_and_saved_entry_KKT_pass=True,
                saved_entry_energy_path_is_not_admissible=True,
                native_gate_pass=False,new_native_run=False,P3_ready=False)
    (E/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
