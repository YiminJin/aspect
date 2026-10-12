#!/usr/bin/env python3
"""Matched native accuracy, frozen-rule uncertainty, stability and measured cost."""
import json,re
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assess_plane_strain import stress,frozen
from assess_localization import read,grid,direct,width,transverse
from localization_1d import Reference
from total_energy_step import G,ETA,h

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
E=HERE/'quadrature_evidence';BUILD=ROOT/'build-propagation-integration';RUNS=BUILD/'runs'
DT=.015625

def coefficient(run,step):
    a=read(run/f'fe-encoding-{step}.csv');a=a[(a['component']==0)&(abs(a['x']-.5)<1e-13)]
    return a[np.argsort(a['y'])]['d_R']

def compare(run,reference):
    f=np.load(reference/'fields.npz');history=json.loads((reference/'history.json').read_text())
    errors=dict(phi=0.,phase_change=0.,q=0.,V_interval=0.,slip=0.,initial_phi=0.)
    xs,ys,p0=grid(run,0);y=np.union1d(ys,f['y']);initial=np.interp(y,ys,p0[len(xs)//2]);ri=np.interp(y,f['y'],f['initial'])
    errors['initial_phi']=float(max(abs(initial-ri)))
    for j in range(1,41):
        xs,ys,p=grid(run,j);s=read(run/f'fe-surface-{j}.csv');rh=history[j-1]
        ph=np.interp(y,ys,p[len(xs)//2]);ref=np.interp(y,f['y'],f['phase'][j-1])
        errors['phi']=max(errors['phi'],float(max(abs(ph-ref))))
        errors['phase_change']=max(errors['phase_change'],float(max(abs((ph-initial)-(ref-ri)))))
        for key,rkey in [('q','q'),('V_interval','V'),('slip','slip')]:errors[key]=max(errors[key],float(max(abs(s[key]-rh[rkey]))))
    errors['slip_relative']=errors['slip']/history[39]['slip']
    return errors

def timers(log):
    result={}
    for line in log.splitlines():
        if line.startswith('|'):
            items=[v.strip() for v in line.split('|')[1:-1]]
            if len(items)==4 and items[1].isdigit() and items[2].endswith('s'):
                result[items[0]]=dict(calls=int(items[1]),seconds=float(items[2][:-1]))
    return result

def prototype_work(reference):
    """Add the native boundary-lifting work measure without changing histories."""
    f=np.load(reference/'fields.npz');hist=json.loads((reference/'history.json').read_text());s=json.loads((reference/'summary.json').read_text())
    r=Reference(s['cells'],1.,s['normal_points'],degree=s['degree']);z,w=leggauss(12);beta=np.exp(-DT/100);kappa=-ETA*np.expm1(-DT/100)
    discrepancies=[];internal=[];works=[]
    for j,row in enumerate(hist):
        r.r=f['incoming_r'][j].copy();r.Z=f['incoming_Z'][j];ph=f['phase'][j];I,_=r.direct(ph);chi=h(r.B@ph)/I
        tau0=G*r.D@r.r-r.Z*chi
        next_r=f['incoming_r'][j+1] if j<39 else f['final_r']
        du=r.D@((next_r-beta*r.r)*G/kappa);forcing=du-chi*row['V'];integral=np.zeros(len(r.w))
        for t,weight in zip((z+1)*DT/2,w*DT/2):
            b=np.exp(-t/100);integral+=weight*(b*tau0-ETA*np.expm1(-t/100)*forcing)
        Wboundary=.256*(row['time']-DT/2)*np.dot(r.w,integral);Wbulk=np.dot(r.w*du,integral)
        jump=np.dot(r.w,tau0*tau0)/(2*G)+r.Z*r.Z/(2*G*I)
        delta=row['bulk']+row['cohesive']-jump
        discrepancies.append(Wboundary-delta-row['loss']);internal.append(Wbulk-delta-row['loss']);works.append(Wboundary)
    return dict(absolute_boundary_error=float(sum(abs(np.array(discrepancies)))),absolute_internal_error=float(sum(abs(np.array(internal)))),boundary_work=float(sum(works)),legacy_internal_error=float(s['absolute_mechanical_error']))

def main():
    result={};plots={};fine=HERE/'localization_evidence/one-d-h1-n256-dt0.015625-exact-bulk'
    fine_h=json.loads((fine/'history.json').read_text());target=json.loads((E/'selection.json').read_text())['physical_comparison_targets']
    for label,degree,n in [('q2',2,16),('q3',3,16),('q2-refined',2,32)]:
        run=RUNS/('output-quadrature-'+label);m=read(run/'fe-metrics.csv');assert len(m)==41 and m[-1]['time']==.625
        log=(BUILD/('quadrature-'+label+'.log')).read_text();assert 'Exception' not in log
        targets=[float(v) for v in re.findall(r'Phase box target:.*? target=([^\s]+)',log)];defects=[float(v) for v in re.findall(r'Phase box converged:.*? KKT=([^\s]+)',log)]
        assert len(targets)==len(defects)==41 and all(k<=t for k,t in zip(defects,targets))
        previous=grid(run,0)[2];recurrence=0.;beta=np.exp(-DT/100);kappa=-ETA*np.expm1(-DT/100)
        for j in range(1,41):
            current=grid(run,j)[2];assert min(current.ravel())>=0 and max(current.ravel())<1 and min((current-previous).ravel())>=-4.922e-12;previous=current
            a=read(run/f'fe-encoding-{j-1}.csv');b=read(run/f'fe-encoding-{j}.csv');recurrence=max(recurrence,float(max(abs(b['d_R']-(beta*a['d_R']+kappa/G*b['u'])))))
        assert recurrence==0
        mechanics=[]
        for chunk in log.split('*** Timestep')[1:]:
            found=re.findall(r'Relative nonlinear residuals \(bulk, fault\) after nonlinear iteration\s+\d+: ([^,]+), ([^\s]+)',chunk)
            if found:mechanics.append([float(v) for v in found[-1]])
        assert max(np.array(mechanics).ravel())<=1e-8
        normalization=[]
        for j in [0,32,40]:
            xs,ys,ph=grid(run,j);s=read(run/f'fe-surface-{j}.csv');x=np.linspace(0,1,4001);I=np.interp(x,s['x'],s['I']);err=float(max(abs(direct(xs,ys,ph,x)/I-1)))
            assert min(I)>0 and err<.005;normalization.append(dict(step=j,source_error=err,I_min=float(min(I))))
        proto=E/f'one-d-n{n}-d{degree}-q6';matched=compare(run,proto);errors=compare(run,fine)
        assert matched['phi']<2e-8 and matched['q']<4e-5
        xs,ys,ph=grid(run,40);s=read(run/'fe-surface-40.csv');y=np.linspace(0,1,20001);phase,chi,tau=stress(run,40,.5,y)
        np.savetxt(E/(label+'-transverse.csv'),np.c_[y,phase,chi,tau.T],delimiter=',',header='y,phi,chi,tau_xx,tau_yy,tau_xy',comments='')
        frozen_integrals={str(q):frozen(run,40,q) for q in [6,12,24]}
        # Same supplied current phase and proper PREVIOUS accepted R/Z anchors.
        audits=[]
        for j in [32,40]:
            xs,ys,p=grid(run,j);old=read(run/f'fe-surface-{j-1}.csv')
            low=Reference(n,1.,6,degree=degree);high=Reference(n,1.,24,degree=degree)
            for r in [low,high]:r.r=coefficient(run,j-1);r.Z=float(np.interp(.5,old['x'],old['Z']))
            v,g,t=low.objective(p[len(xs)//2]);vv,gg,tt=high.objective(p[len(xs)//2]);H=low.phase_hessian(p[len(xs)//2]);HH=high.phase_hessian(p[len(xs)//2])
            fd=read(run/f'phase-derivative-{j}.csv');assert min(fd['energy_derivative_relative_error'])<1e-7 and min(fd['Hessian_direction_relative_error'])<1e-7
            audits.append(dict(step=j,incoming_step=j-1,bulk_energy_error=float(abs(t[0]-tt[0])),phase_gradient_error=float(max(abs(g-gg))),phase_Hessian_relative_error=float(np.max(abs(H-HH))/max(1.,np.max(abs(HH)))),native_derivatives=[{key:float(row[key]) for key in fd.dtype.names} for row in fd]))
        high=Reference(n,1.,24,degree=degree);high.r=coefficient(run,40);high.Z=float(np.interp(.5,s['x'],s['Z']));I,_=high.direct(ph[len(xs)//2]);hh=high.objective(ph[len(xs)//2])[2];sigma=hh[-1]
        forces=dict(bulk_max_Pa=float(max(abs(high.D[:,1:-1].T@(high.w*sigma)))),fault_Pa=float(abs(np.dot(high.w*h(high.B@ph[len(xs)//2])/I,sigma)-high.Z/I-1000)))
        curvature=[{key:float(row[key]) for key in row.dtype.names} for j in [32,40] for row in read(run/f'phase-curvature-{j}.csv')]
        cost=json.loads((E/(label+'-run.json')).read_text());cost['timers']=timers(log)
        cost['phase_iterations']=sum(int(v) for v in re.findall(r'Phase box converged: iterations=(\d+)',log))
        cost['mechanical_linear_iterations']=sum(int(v) for v in re.findall(r'Fault linear solve: iterations=(\d+)',log))
        cost['mechanical_nonlinear_evaluations']=len(re.findall(r'Relative nonlinear residuals \(bulk, fault\)',log))
        cost['degree']=degree;cost['bulk_dofs']=len(read(run/'fe-encoding-0.csv'))
        initial_phase=grid(run,0)[2]
        onset=next((j*DT for j in range(1,41) if np.max(grid(run,j)[2]-initial_phase)>1e-5),None)
        core_onset=next((float(row['time']) for row in m if row['phi_max']-m[0]['phi_max']>1e-5),None)
        result[label]=dict(initial_max=float(m[0]['phi_max']),initial_H_max=float(m[0]['Hmax_Pa']),phi_max=float(m[-1]['phi_max']),maximum_increment=float(max(m['increment'])),healing=float(max(m['healing'])),phase_spread_x=float(np.max(np.ptp(ph,axis=0))),onset=onset,q=float(np.interp(.5,s['x'],s['q'])),V_interval=float(np.interp(.5,s['x'],s['V_interval'])),slip=float(np.interp(.5,s['x'],s['slip'])),widths=width(y,phase,chi),dense_stress_range=[float(min(tau[2])),float(max(tau[2]))],independent_stress_RMS=float(frozen_integrals['24']['stress_rms']),uniform_reference_stress=float(fine_h[39]['stress_min']),fine_errors=errors,matched_prototype_errors=matched,normalization=normalization,frozen_quadrature=frozen_integrals,incoming_anchor_audits=audits,independent_force=forces,curvature=curvature,cost=cost,phase_release=float(sum(m['D_relax'][1:])),absolute_boundary_mechanical_error=float(sum(abs(m['mechanical_work_error'][1:]))),fracture_expenditure=float(m[-1]['fracture']-m[0]['fracture']),bulk_profile_change=float(sum(m['bulk_profile_change'][1:])),cohesive_profile_change=float(sum(m['cohesive_profile_change'][1:])),bulk_viscous=float(sum(m['bulk_viscous'][1:])),cohesive_viscous=float(sum(m['cohesive_viscous'][1:])),friction=float(sum(m['friction'][1:])),bulk_work=float(sum(m['bulk_work'][1:])),boundary_work=float(sum(m['boundary_work'][1:])),absolute_internal_mechanical_error=float(sum(abs(m['mechanical_work_error'][1:]+m['bulk_work'][1:]-m['boundary_work'][1:]))),prototype_work=prototype_work(proto),recurrence_error=recurrence,KKT_target_range=[min(targets),max(targets)],mechanical_residuals=np.max(mechanics,axis=0).tolist())
        result[label]['physical_target_pass']=errors['phi']<=target['full_phi'] and errors['q']<=target['cohesive_q_Pa'] and errors['slip_relative']<=target['slip_relative_to_reference_final']
        result[label]['maximum_growth_onset']=core_onset
        if label!='q2-refined':
            baseline=RUNS/('output-plane-strain-compatible-'+label);base=read(baseline/'fe-metrics.csv')
            result[label]['three_point_changes']=dict(phi_final=float(m[-1]['phi_max']-base[-1]['phi_max']),q_final=float(result[label]['q']-np.interp(.5,read(baseline/'fe-surface-40.csv')['x'],read(baseline/'fe-surface-40.csv')['q'])),initial_phase_identical=(run/'fe-phase-0.csv').read_bytes()==(baseline/'fe-phase-0.csv').read_bytes())
        plots[label]=(y,phase,chi,tau[2],m)
    passing=[k for k,v in result.items() if v['physical_target_pass']]
    result['recommendation']=min(passing,key=lambda k:result[k]['cost']['wall_seconds']) if passing else None
    result['targets']=target;result['stress_accuracy_qualified']=False;result['time_accuracy_qualified']=False;result['P3_ready']=False
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for label,(y,p,c,t,m) in plots.items():
        for k,a in enumerate([p,c,t]):axes[0,k].plot(y,a,label=label)
        axes[1,0].plot(m['time'],m['phi_max'],label=label)
        q=[];ss=[];V=[]
        for j in range(41):
            s=read(RUNS/('output-quadrature-'+label)/f'fe-surface-{j}.csv');q.append(np.interp(.5,s['x'],s['q']));ss.append(np.interp(.5,s['x'],s['slip']));V.append(np.interp(.5,s['x'],s['V_interval']))
        axes[1,1].plot(ss,q,label=label);axes[1,2].plot(m['time'],V,label=label)
    rf=np.load(fine/'fields.npz');pr=rf['phase'][39];I=fine_h[39]['I']
    axes[0,0].plot(rf['y'],pr,'k:',label='rich1D');axes[0,1].plot(rf['y'],h(pr)/I,'k:',label='rich1D');axes[0,2].axhline(fine_h[39]['stress_min'],color='k',ls=':',label='rich1D')
    for k,name in enumerate(['phi','chi [1/m]','tau_xy [Pa]']):axes[0,k].set(xlabel='y [m]',ylabel=name);axes[0,k].legend()
    for k,(x,name) in enumerate([('t [s]','phi maximum'),('slip [m]','q [Pa]'),('t [s]','interval V [m/s]')]):axes[1,k].set(xlabel=x,ylabel=name);axes[1,k].legend()
    fig.savefig(E/'accuracy-cost.png',dpi=180);fig.savefig(E/'accuracy-cost.pdf')
    (E/'assessment.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:{key:v[key] for key in ['phi_max','fine_errors','independent_stress_RMS','phase_release','absolute_boundary_mechanical_error','physical_target_pass']} for k,v in result.items() if isinstance(v,dict) and 'phi_max' in v},indent=2))

if __name__=='__main__':main()
