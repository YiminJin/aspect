#!/usr/bin/env python3
"""Independent saved-state checks for the restricted native FE representation."""
import csv, hashlib, json, math, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from total_energy_step import h, M, G, ETA, basis, N
from total_energy_time_comparison import make_energy
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
EVIDENCE=HERE/'native_fe_evidence'
OUT=ROOT/'build-propagation-integration/output-native-fe-endpoint64-pdas'
def read(path):
    return np.genfromtxt(path,delimiter=',',names=True,ndmin=1)
def phi(path):
    r=read(path); lookup={(float(x['x']),float(x['y'])):float(x['phi']) for x in r}
    return np.array([lookup[(i/N,j/N)] for i in range(N+1) for j in range(N+1)])
def direct(profile,x):
    a=np.minimum((N*x).astype(int),N-1);s=N*x-a
    f=(1-s[:,None])*profile[a]+s[:,None]*profile[a+1]
    u=1-f[:,:-1].astype(np.longdouble);v=1-f[:,1:].astype(np.longdouble)
    z=(v-u)/u
    ratio=np.ones_like(z);mask=z!=0;ratio[mask]=np.log1p(z[mask])/z[mask]
    return np.asarray(np.sum(M/N*(2/(u*v)-3*ratio/u+1),axis=1),float)
def main():
    rows=read(OUT/'fe-metrics.csv');count=int(rows[-1]['step'])
    log=(ROOT/'build-propagation-integration/native-fe-endpoint64-pdas.log').read_text()
    targets=[float(x) for x in re.findall(r'Phase box target:.*? target=([^\s]+)',log)]
    residuals=[float(x) for x in re.findall(r'Phase box converged:.*? KKT=([^\s]+)',log)]
    assert len(residuals)==count+1 # includes initial seeded-H relaxation
    assert all(r<=t for r,t in zip(residuals,targets))
    mechanics=re.findall(r'Relative nonlinear residuals \(bulk, fault\) after nonlinear iteration\s+2: ([^,]+), ([^\s]+)',log)
    assert len(mechanics)==count
    mechanical_max=[max(float(r[k]) for r in mechanics) for k in (0,1)]
    assert max(mechanical_max)<1e-10
    phases=np.array([phi(OUT/f'fe-phase-{j}.csv') for j in range(count+1)])
    surfaces=[read(OUT/f'fe-surface-{j}.csv') for j in range(count+1)]
    assert np.min(phases)>=0 and np.max(phases)<1
    assert np.min(np.diff(phases,axis=0))>=-4.922e-12
    assert max(rows['Maxwell_error_Pa'])<1e-7
    assert max(abs(rows['energy_reconstruction_error']))<1e-8
    assert max(rows['KKT'])<1e-11 # native solver has a stricter per-entry logged target
    assert min(s['I'].min() for s in surfaces)>0
    recurrence=0.;recurrence_relative=0.
    for j in range(1,count+1):
        old=read(OUT/f'fe-encoding-{j-1}.csv');new=read(OUT/f'fe-encoding-{j}.csv')
        assert np.array_equal(old['dof'],new['dof'])
        dt=rows[j]['dt'];beta=math.exp(-G*dt/ETA);kappa=-ETA*math.expm1(-G*dt/ETA)
        expected=beta*old['d_R']+kappa/G*new['u']
        err=np.max(abs(new['d_R']-expected));recurrence=max(recurrence,float(err))
        scale=np.max(abs(beta*old['d_R'])+abs(kappa/G*new['u']))
        recurrence_relative=max(recurrence_relative,float(err/max(scale,1e-300)))
        assert err<=64*np.finfo(float).eps*max(1.,scale)
        if j>1:
            assert np.max(abs(surfaces[j]['Z']-(beta*surfaces[j-1]['Z']+kappa*surfaces[j]['V_interval'])))<=1e-7
    x=np.unique(np.r_[np.linspace(0,1,4001),np.arange(65)/64]);norms=[]
    energy=make_energy(True)
    for j in sorted(set([0,min(12,count),count])):
        reference=direct(phases[j].reshape(N+1,N+1),x);projected=np.interp(x,surfaces[j]['x'],surfaces[j]['I'])
        err=reference/projected-1;worst=np.argmax(abs(err));positive=np.min(projected)>0
        # Separate quadrature-defined projection conservation from representation.
        expected=energy.profile(phases[j])['I']
        projection_error=float(np.max(abs(expected-surfaces[j]['I'])/expected))
        assert projection_error<1e-10
        direct_samples,_=energy.direct(phases[j]);projected_samples=energy.S@surfaces[j]['I']
        conservation=float(abs(energy.ws@(projected_samples-direct_samples))/abs(energy.ws@direct_samples))
        assert conservation<1e-12
        constant=np.linalg.solve(energy.mass,np.asarray(energy.S.T@(energy.ws*3.25)))
        constant_error=float(np.max(abs(constant-3.25)))
        assert constant_error<1e-12
        norms.append(dict(step=j,time=float(rows[j]['time']),source_error_max=float(max(abs(err))),worst_x=float(x[worst]),endpoint_error=float(max(abs(err[0]),abs(err[-1]))),positive=bool(positive),native_projection_error=projection_error,conservation=conservation,constant_error=constant_error))
        np.savetxt(EVIDENCE/f'normalization-{j}.csv',np.c_[x,reference,projected,err],delimiter=',',header='x,direct,consistent,source_error',comments='')
    local=[];derivatives=[]
    for p in sorted(OUT.glob('fe-local-*.csv')):
        r=read(p);assert max(r['relative_error'])<=256*np.finfo(float).eps
        local.append(dict(file=p.name,max_relative=float(max(r['relative_error'])),max_Pa=float(max(r['stress_error_Pa']))))
    for p in sorted(OUT.glob('fe-derivative-*.csv')):
        r=read(p);assert min(r['energy_relative'])<1e-7 and min(r['hessian_relative'])<1e-7
        derivatives.append(dict(file=p.name,best_gradient=float(min(r['energy_relative'])),best_hessian=float(min(r['hessian_relative']))))
    zero=ROOT/'build-propagation-integration/output-native-fe-zero-final';zr=read(zero/'fe-metrics.csv')
    assert np.all(zr['increment']==0) and np.all(zr['healing']==0) and np.all(zr['D_relax']==0)
    assert np.array_equal(phi(zero/'fe-phase-0.csv'),phi(zero/f'fe-phase-{int(zr[-1]["step"])}.csv'))
    reference=np.load(EVIDENCE/'prototype64/fields.npz');pr=json.loads((EVIDENCE/'prototype64/records.json').read_text());ps=json.loads((EVIDENCE/'prototype64/summary.json').read_text())
    assert np.array_equal(phases[0],reference['phi'][0])
    assert abs(energy.fracture(phases[0])-rows[0]['fracture'])<1e-10
    qt=np.array([s['q'][[0,32,-1]] for s in surfaces]);vt=np.array([s['V_interval'][[0,32,-1]] for s in surfaces]);st=np.array([s['slip'][[0,32,-1]] for s in surfaces])
    phi_diff=np.max(abs(phases-reference['phi'][:count+1]),axis=1)
    qdiff=np.max(abs(qt-reference['Z'][:count+1,[0,32,-1]]/reference['I'][:count+1,[0,32,-1]]),axis=1)
    sdiff=np.max(abs(st-reference['slip'][:count+1,[0,32,-1]]),axis=1)
    onset=next((j for j in range(1,count+1) if np.max(phases[j]-phases[0])>1e-6),None)
    bulk_comparisons=[]
    for j in [0,8,16,32,48,64]:
        path=OUT/f'fe-bulk-{j}.csv'
        if not path.exists():continue
        r=read(path);order=np.lexsort((energy.y,energy.x));native_order=np.lexsort((r['y'],r['x']))
        assert np.max(abs(r['x'][native_order]-energy.x[order]))<1e-12
        assert np.max(abs(r['y'][native_order]-energy.y[order]))<1e-12
        diff=r['tau_xy'][native_order]-reference['tau'][j,order]
        w=r['w'][native_order]
        bulk_comparisons.append(dict(step=j,time=float(rows[j]['time']),shear_max_difference_Pa=float(max(abs(diff))),shear_rms_difference_Pa=float(np.sqrt(sum(w*diff*diff))),normal_stress_rms_Pa=float(np.sqrt(sum(w*(r['tau_xx'][native_order]**2+r['tau_yy'][native_order]**2))))))
    sums={k:float(sum(rows[k][1:])) for k in ['D_relax','mechanical_work_error','bulk_work','boundary_work','bulk_viscous','cohesive_viscous','friction','bulk_profile_change','cohesive_profile_change']}
    sums['absolute_mechanical_work_error']=float(sum(abs(rows['mechanical_work_error'][1:])))
    sums['absolute_D_relax']=float(sum(abs(rows['D_relax'][1:])))
    summary=dict(qualification=dict(FE_representation_verified=True,frozen_normalization_target_met=bool(max(n['source_error_max'] for n in norms)<.005),coupled_completed=bool(rows[-1]['time']==1.),native_time_convergence_verified=False,P3_ready=False),mechanical_relative_residual_max=mechanical_max,accepted_phase_targets=targets[:count+1],failed_phase_target=targets[-1],completed_steps=count,final_time=float(rows[-1]['time']),initial_phi_max=float(max(phases[0])),final_phi_max=float(max(phases[-1])),largest_increment=float(max(rows['increment'])),healing=float(max(rows['healing'])),phase_KKT=float(max(rows['KKT'])),nodal_recurrence_error_m=recurrence,nodal_recurrence_relative=recurrence_relative,local_Maxwell=local,derivative=derivatives,normalization=norms,final_q_Pa=qt[-1].tolist(),final_slip_m=st[-1].tolist(),onset_time=None if onset is None else float(rows[onset]['time']),sums=sums,fracture_expenditure=float(rows[-1]['fracture']-rows[0]['fracture']),prototype_comparison=dict(initial_phase_identical=True,fracture_measure_identical=True,mechanical_spaces_and_lateral_boundary_values_differ=True,full_phi_common_time_max=float(max(phi_diff)),final_phi_max_difference=float(phi_diff[-1]),common_q_max_difference_Pa=float(max(qdiff)),final_q_difference_Pa=float(qdiff[-1]),common_slip_max_difference_m=float(max(sdiff)),final_slip_difference_m=float(sdiff[-1]),bulk=bulk_comparisons,reference=ps))
    selected=pr[:count]
    summary['prototype_same_accepted_interval']={k:float(sum(r[k] for r in selected)) for k in ['phase_release','mechanical_work_error','fracture_expenditure']}
    summary['prototype_same_accepted_interval']['absolute_mechanical_work_error']=float(sum(abs(r['mechanical_work_error']) for r in selected))
    delta=phases[-1]-phases[0];loc=int(np.argmax(delta));summary['largest_total_phase_gain']=float(delta[loc]);summary['phase_gain_location_m']=[(loc//17)/16,(loc%17)/16]
    summary['normal_boundary_phase_max']=float(np.max(phases[-1].reshape(17,17)[:,[0,16]]))
    (EVIDENCE/'assessment.json').write_text(json.dumps(summary,indent=2)+'\n')
    time=rows['time'];pt=reference['time']
    fig,axes=plt.subplots(2,3,figsize=(12,6.5));names=['left endpoint','center','right endpoint']
    for k,ix in enumerate([0,8,16]):
        axes[0,0].plot(time,phases.reshape(-1,17,17)[:,ix,8],label=names[k]);axes[0,0].plot(pt,reference['phi'].reshape(-1,17,17)[:,ix,8],'--',alpha=.5)
        axes[0,1].plot(time,qt[:,k]);axes[0,1].plot(pt,reference['Z'][:,[0,32,64][k]]/reference['I'][:,[0,32,64][k]],'--',alpha=.5)
        axes[0,2].plot(st[:,k],qt[:,k],label=names[k]);axes[0,2].plot(reference['slip'][:,[0,32,64][k]],reference['Z'][:,[0,32,64][k]]/reference['I'][:,[0,32,64][k]],'--',alpha=.5)
    axes[0,0].set(xlabel='Time (s)',ylabel='Core phase');axes[0,0].legend(fontsize=8)
    axes[0,1].set(xlabel='Time (s)',ylabel='Cohesive traction (Pa)');axes[0,2].set(xlabel='Cumulative local slip (m)',ylabel='Cohesive traction (Pa)')
    axes[1,0].plot(time,np.cumsum(rows['D_relax']),label='Phase relaxation release');axes[1,0].plot(time,np.cumsum(abs(rows['mechanical_work_error'])),label='Absolute mechanical discrepancy');axes[1,0].legend(fontsize=8);axes[1,0].set(xlabel='Time (s)',ylabel='Cumulative energy (J/m)')
    axes[1,1].plot(time,phi_diff);axes[1,1].set(xlabel='Time (s)',ylabel='Native/prototype full phase difference')
    im=axes[1,2].imshow((phases[-1]-phases[0]).reshape(17,17).T,origin='lower',extent=[0,1,0,1],vmin=0);fig.colorbar(im,ax=axes[1,2],label='Phase increase');axes[1,2].set(xlabel='x (m)',ylabel='y (m)')
    fig.suptitle('Native full Stokes (solid) versus one-mode endpoint prototype (dashed)');fig.tight_layout()
    fig.savefig(EVIDENCE/'native-comparison.png',dpi=160);fig.savefig(EVIDENCE/'native-comparison.pdf');plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['prototype_comparison','derivative','local_Maxwell']},indent=2))
if __name__=='__main__':main()
