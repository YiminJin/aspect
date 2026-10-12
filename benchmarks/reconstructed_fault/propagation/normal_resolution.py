#!/usr/bin/env python3
"""Q2 spatial errors and common-physical-profile controls; no history mutation."""
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve
from scipy.optimize import brentq
from localization_1d import Reference
from total_energy_step import G, h, dh
from assess_quadrature import prototype_work

HERE=Path(__file__).resolve().parent
E=HERE/'normal_resolution_evidence'
OLD=HERE/'quadrature_evidence'
FINE=HERE/'localization_evidence/one-d-h1-n256-dt0.015625-exact-bulk'
DT=1/64

def load(path):return json.loads(path.read_text())
def directory(n):return (OLD if n<=64 else E)/f'one-d-n{n}-d2-q6'

def derivative(n,y):
    """Evaluate the actual Q2 derivative away from assembly points."""
    cell=np.minimum((np.asarray(y)*n).astype(int),n-1)
    s=np.asarray(y)*n-cell
    d=np.c_[-3+4*s,4-8*s,-1+4*s]*n
    D=np.zeros((len(y),2*n+1))
    D[np.arange(len(y))[:,None],2*cell[:,None]+np.arange(3)]=d
    return D

def moments(tau,weights,chi,reference):
    length=sum(weights);mean=weights@tau/length
    osc2=weights@(tau-mean)**2/length
    C=weights@chi;localized_mean=(weights*chi)@tau/C
    localized_osc2=(weights*chi)@(tau-localized_mean)**2/C
    mean_energy=length*mean**2/(2*G);osc_energy=length*osc2/(2*G)
    energy=weights@(tau*tau)/(2*G)
    assert abs(energy-mean_energy-osc_energy)<1e-11*max(1.,energy)
    return dict(mean_stress_Pa=float(mean),mean_error_Pa=float(mean-reference),oscillation_RMS_Pa=float(np.sqrt(osc2)),whole_error_RMS_Pa=float(np.sqrt((mean-reference)**2+osc2)),localized_mean_Pa=float(localized_mean),localized_mean_error_Pa=float(localized_mean-reference),localized_oscillation_RMS_Pa=float(np.sqrt(localized_osc2)),localized_error_RMS_Pa=float(np.sqrt((localized_mean-reference)**2+localized_osc2)),bulk_energy=float(energy),mean_stress_energy=float(mean_energy),oscillation_energy=float(osc_energy),source_integral=float(C))

def extrema(n,phi,r,Z,I):
    """Cell one-sided limits and stationary points, not dense-sample extrema.

    tau''=-Z h''(phi) phi_y^2/I <=0 in each Q1-phase/Q2-strain cell.
    Thus both limits and at most one stationary point suffice (Z>=0).
    """
    assert Z>=0
    left=[];right=[];candidates=[]
    for cell in range(n):
        rr=r[2*cell:2*cell+3];p0,p1=phi[cell:cell+2];slope=p1-p0
        def value(s):return G*n*np.dot([-3+4*s,4-8*s,-1+4*s],rr)-Z*h(p0+s*slope)/I
        def first(s):return G*n*np.dot([4,-8,4],rr)-Z*dh(p0+s*slope)*slope/I
        l=value(0.);v=value(1.);left.append(l);right.append(v)
        candidates.extend([(l,cell/n,'right limit'),(v,(cell+1)/n,'left limit')])
        if first(0.)>0 and first(1.)<0:
            s=brentq(first,0.,1.);candidates.append((value(s),(cell+s)/n,'cell stationary point'))
    mn=min(candidates);mx=max(candidates);jump=np.asarray(left[1:])-np.asarray(right[:-1]);worst=int(np.argmax(abs(jump)))
    return dict(stress_min_Pa=float(mn[0]),stress_min_y=float(mn[1]),stress_max_Pa=float(mx[0]),stress_max_y=float(mx[1]),interface_jump_max_Pa=float(max(abs(jump))),interface_jump_worst_y=float((worst+1)/n),interface_jumps_Pa=jump.tolist())

def evolving(n,fine_fields,fine_history,stress_scale,energy_scale):
    path=directory(n);f=np.load(path/'fields.npz');hist=load(path/'history.json');r=Reference(n,1.,24);higher=Reference(n,1.,48)
    union=np.union1d(f['y'],fine_fields['y']);init=np.interp(union,f['y'],f['initial']);refinit=np.interp(union,fine_fields['y'],fine_fields['initial'])
    errors=dict(full_phase=0.,phase_change=0.,cohesive_q_Pa=0.,V_interval=0.,slip=0.,initial_phase=float(max(abs(init-refinit))))
    history=[];qerror=0.
    for j,row in enumerate(hist):
        phase=f['phase'][j];anchor=f['incoming_r'][j+1] if j<39 else f['final_r'];Z=f['incoming_Z'][j+1] if j<39 else f['final_Z']
        r.r=anchor;r.Z=Z;higher.r=anchor;higher.Z=Z
        terms=r.objective(phase)[2];hi=higher.objective(phase)[2];I=terms[6]
        m=moments(terms[-1],r.w,terms[-2],fine_history[j]['stress_min'])
        eh=fine_history[j]['bulk'];m.update(time=(j+1)*DT,bulk_reference=eh,bulk_error=m['bulk_energy']-eh,bulk_error_fixed_scale=abs(m['bulk_energy']-eh)/energy_scale,whole_error_fixed_scale=m['whole_error_RMS_Pa']/stress_scale,localized_error_fixed_scale=m['localized_error_RMS_Pa']/stress_scale)
        m.update(extrema(n,phase,anchor,Z,I));m.update(phi_max=float(max(phase)),q=row['q'],V_interval=row['V'],slip=row['slip'],I=I)
        qerror=max(qerror,abs(m['bulk_energy']-hi[0]))
        p=np.interp(union,f['y'],phase);pr=np.interp(union,fine_fields['y'],fine_fields['phase'][j]);errors['full_phase']=max(errors['full_phase'],float(max(abs(p-pr))))
        errors['phase_change']=max(errors['phase_change'],float(max(abs(p-init-pr+refinit))))
        for k,rk in [('cohesive_q_Pa','q'),('V_interval','V'),('slip','slip')]:errors[k]=max(errors[k],abs(row[rk]-fine_history[j][rk]))
        history.append(m)
    errors['slip_relative_to_reference_final']=errors['slip']/fine_history[39]['slip']
    for name,key in [('whole_stress_RMS_relative_to_reference_peak','whole_error_fixed_scale'),('localized_stress_RMS_relative_to_reference_peak','localized_error_fixed_scale'),('bulk_storage_error_relative_to_reference_peak','bulk_error_fixed_scale')]:
        worst=max(history,key=lambda r:r[key]);errors[name]=worst[key];errors[name+'_time']=worst['time']
    onset=next((j*DT for j in range(1,41) if max(f['phase'][j-1]-f['initial'])>1e-5),None)
    targets=load(E/'baseline.json')['targets'];passed=all(errors[key]<=value for key,value in targets.items())
    work=prototype_work(path)
    summary=dict(cells=n,errors=errors,targets_pass=passed,onset=onset,initial_max=float(max(f['initial'])),final=history[-1],quadrature_energy_24_48=qerror,phase_relaxation_release=load(path/'summary.json')['phase_release'],fracture_expenditure=hist[-1]['fracture']-r.fracture(f['initial']),work=work,prototype_runtime_seconds=load(path/'summary.json')['runtime_seconds'])
    (E/f'n{n}-stress-history.json').write_text(json.dumps(history,indent=2)+'\n')
    return summary

def frozen_comparison(fine_fields,fine_history,ns):
    """Same physical Q1 phi, Z and total elastic mean strain for every space.

    Solve only equilibrium in r at the prescribed endpoint difference. This
    is an isolated velocity/internal-strain-space comparison, not evolution
    with an interpolated or substituted irreversible obstacle.
    """
    result=[];fine_y=fine_fields['y'];nf=len(fine_y)-1
    for step in [32,35,40]:
        ph=fine_fields['phase'][step-1];ref=fine_history[step-1];Z=ref['q']*ref['I'];mean_R=(ref['stress_min']+Z)/G
        physical=Reference(nf,1.,24);I,_=physical.direct(ph)
        for points in [24,48]:
            z,w=leggauss(points);y=(fine_y[:-1,None]+np.diff(fine_y)[:,None]*(z+1)/2).ravel();weights=(np.diff(fine_y)[:,None]*w/2).ravel();pp=np.interp(y,fine_y,ph);chi=h(pp)/I
            assert abs(weights@chi-1)<1e-12
            for n in ns:
                D=derivative(n,y);fixed=np.array([-mean_R/2,mean_R/2]);lift=D[:,[0,-1]]@fixed;free=D[:,1:-1]
                K=G*free.T@(weights[:,None]*free);rhs=free.T@(weights*(Z*chi-G*lift));rr=np.r_[fixed[0],solve(K,rhs,assume_a='pos'),fixed[1]]
                tau=G*D@rr-Z*chi;m=moments(tau,weights,chi,ref['stress_min'])
                m.update(step=step,time=step*DT,cells=n,integration_points=points,physical_I=I,Z=Z,mean_R=mean_R,weak_force_Pa=float(max(abs(free.T@(weights*tau)))),fault_balance_defect_Pa=float(weights@(chi*tau)-Z/I-1000))
                # Cell-interface limits require the SAME finer physical phase.
                interfaces=np.arange(1,n)/n
                limits=[]
                for cell in range(n):
                    vals=rr[2*cell:2*cell+3];left=G*n*np.dot([-3,4,-1],vals)-Z*h(np.interp(cell/n,fine_y,ph))/I;right=G*n*np.dot([1,-4,3],vals)-Z*h(np.interp((cell+1)/n,fine_y,ph))/I;limits.append((left,right))
                m['interface_jump_max_Pa']=float(max(abs(np.array(limits)[1:,0]-np.array(limits)[:-1,1])))
                result.append(m)
    return result

def main():
    fine_fields=np.load(FINE/'fields.npz');fine_history=load(FINE/'history.json')[:40]
    stress_scale=max(abs(r['stress_min']) for r in fine_history);energy_scale=max(r['bulk'] for r in fine_history)
    ns=[16,32,64,128,256]
    result=dict(stress_scale_Pa=stress_scale,bulk_storage_scale_J_m=energy_scale,scales='fixed reference peak over 0--0.625 s; never startup denominators',dt=DT,final_time=.625,normal_rule=6,evolving={str(n):evolving(n,fine_fields,fine_history,stress_scale,energy_scale) for n in ns})
    result['frozen']=frozen_comparison(fine_fields,fine_history,ns)
    passing=[n for n in ns if result['evolving'][str(n)]['targets_pass']];result['selected_cells']=min(passing) if passing else None
    result['irreversible_obstacle_effect']='Subtracting initial offsets does not remove the effect of different accepted lower bounds on evolution. The separate frozen-profile equilibrium keeps physical phi and Z identical.'
    (E/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v['errors']|dict(pass_=v['targets_pass']) for k,v in result['evolving'].items()},indent=2));print('Selected',result['selected_cells'],'scales',stress_scale,energy_scale)

if __name__=='__main__':main()
