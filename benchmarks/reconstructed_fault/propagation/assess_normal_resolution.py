#!/usr/bin/env python3
"""Native confirmation, common-time stress/storage errors, and compact plots."""
import json,re
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from normal_resolution import E,OLD,FINE,DT,G,h,Reference,load,moments,extrema
from assess_localization import read,grid,direct,width
from assess_plane_strain import stress,frozen
from assess_quadrature import compare,timers,coefficient
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3];BUILD=ROOT/'build-propagation-integration'

def native(selection):
    n=selection['selected_cells'];run=BUILD/'runs'/f'output-normal-resolution-q2-n{n}';cost=load(E/'native-run.json');assert cost['returncode']==0
    metrics=read(run/'fe-metrics.csv');assert len(metrics)==41 and metrics[-1]['time']==.625
    log=(BUILD/'normal-resolution-native.log').read_text();assert 'Exception' not in log
    targets=[float(v) for v in re.findall(r'Phase box target:.*? target=([^\s]+)',log)];defects=[float(v) for v in re.findall(r'Phase box converged:.*? KKT=([^\s]+)',log)]
    assert len(targets)==len(defects)==41 and all(k<=t for k,t in zip(defects,targets))
    mechanical=[]
    for part in log.split('*** Timestep')[1:]:
        found=re.findall(r'Relative nonlinear residuals \(bulk, fault\) after nonlinear iteration\s+\d+: ([^,]+), ([^\s]+)',part)
        if found:mechanical.append([float(v) for v in found[-1]])
    # Include the unloaded initialization solve as well as the 40 physical steps.
    assert len(mechanical)==41 and max(np.array(mechanical).ravel())<1e-8
    previous=grid(run,0)[2];recurrence=0.;normal=0.;spread=0.;hist=[]
    fine_history=load(FINE/'history.json')[:40];r=Reference(n,1.,24)
    beta=np.exp(-DT/100);kappa=-1e8*np.expm1(-DT/100)
    normalization=[]
    oldencoding=read(run/'fe-encoding-0.csv')
    for j in range(1,41):
        xs,ys,p=grid(run,j);spread=max(spread,float(np.max(np.ptp(p,axis=0))))
        assert min(p.ravel())>=0 and max(p.ravel())<1 and min((p-previous).ravel())>=-4.922e-12;previous=p
        enc=read(run/f'fe-encoding-{j}.csv');recurrence=max(recurrence,float(max(abs(enc['d_R']-beta*oldencoding['d_R']-kappa/G*enc['u']))));oldencoding=enc
        selected=enc[(abs(enc['x']-.5)<1e-13)&(enc['component']==0)];rr=selected[np.argsort(selected['y'])]['d_R']
        surf=read(run/f'fe-surface-{j}.csv');Z=float(np.interp(.5,surf['x'],surf['Z']));I=float(np.interp(.5,surf['x'],surf['I']))
        ph,chi,tau=stress(run,j,.5,r.yq);normal=max(normal,float(max(abs(tau[:2]).ravel())))
        m=moments(tau[2],r.w,chi,fine_history[j-1]['stress_min']);m.update(time=j*DT,bulk_reference=fine_history[j-1]['bulk'],bulk_error=m['bulk_energy']-fine_history[j-1]['bulk'])
        m.update(extrema(n,p[len(xs)//2],rr,Z,I));hist.append(m)
        if j in [32,40]:
            x=np.array([0.,.5,1.]);di=direct(xs,ys,p,x);ii=np.interp(x,surf['x'],surf['I']);normalization.append(dict(step=j,min_I=float(min(ii)),maximum_source_error=float(max(abs(di/ii-1)))))
            for xx in [0.,.5,1.]:
                y=np.linspace(0,1,32769);phi,cc,t=stress(run,j,xx,y)
                np.savetxt(E/f'native-step{j}-x{xx:g}-profile.csv',np.c_[y,phi,cc,t.T],header='y,phi,chi,tau_xx,tau_yy,tau_xy',delimiter=',',comments='')
    assert recurrence<1e-14 and spread<1e-8 and normal<1e-7
    xs,ys,p=grid(run,0);surf=read(run/'fe-surface-0.csv');x=np.array([0.,.5,1.]);ii=np.interp(x,surf['x'],surf['I']);normalization.insert(0,dict(step=0,min_I=float(min(ii)),maximum_source_error=float(max(abs(direct(xs,ys,p,x)/ii-1)))))
    assert all(v['min_I']>0 and v['maximum_source_error']<.005 for v in normalization)
    matched=compare(run,E/f'one-d-n{n}-d2-q6');fine_errors=compare(run,FINE)
    assert matched['phi']<2e-8 and matched['q']<4e-5
    curvature=[{name:float(row[name]) for name in row.dtype.names} for j in [32,40] for row in read(run/f'phase-curvature-{j}.csv')]
    derivatives=[dict(step=j,rows=[{name:float(row[name]) for name in row.dtype.names} for row in read(run/f'phase-derivative-{j}.csv')]) for j in [32,40]]
    integrations={str(j):{str(q):frozen(run,j,q) for q in [6,24,48]} for j in [32,40]}
    audits=[]
    for j in [32,40]:
        xs,ys,p=grid(run,j);old=read(run/f'fe-surface-{j-1}.csv')
        low=Reference(n,1.,6);high=Reference(n,1.,24)
        for a in [low,high]:a.r=coefficient(run,j-1);a.Z=float(np.interp(.5,old['x'],old['Z']))
        v,g,t=low.objective(p[len(xs)//2]);vv,gg,tt=high.objective(p[len(xs)//2]);H=low.phase_hessian(p[len(xs)//2]);HH=high.phase_hessian(p[len(xs)//2])
        audits.append(dict(step=j,incoming_step=j-1,bulk_energy_error=float(abs(t[0]-tt[0])),phase_gradient_error=float(max(abs(g-gg))),phase_Hessian_relative_error=float(np.max(abs(H-HH))/max(1.,np.max(abs(HH))))))
    (E/'native-stress-history.json').write_text(json.dumps(hist,indent=2)+'\n')
    cost['timers']=timers(log);cost['phase_iterations']=sum(int(v) for v in re.findall(r'Phase box converged: iterations=(\d+)',log));cost['mechanical_linear_iterations']=sum(int(v) for v in re.findall(r'Fault linear solve: iterations=(\d+)',log));cost['mechanical_nonlinear_evaluations']=len(re.findall(r'Relative nonlinear residuals \(bulk, fault\)',log));cost['velocity_DoFs']=len(read(run/'fe-encoding-0.csv'))
    return dict(cells=n,cost=cost,matched_discrete_errors=matched,fine_errors=fine_errors,maximum_healing=float(max(metrics['healing'])),recurrence_error=recurrence,normal_stress_max_Pa=normal,tangential_phase_spread=spread,normalization=normalization,phase_curvature=curvature,phase_derivatives=derivatives,proper_incoming_steps=[31,39],incoming_anchor_audits=audits,independent_bulk_integration=integrations,mechanical_residuals=np.max(mechanical,axis=0).tolist(),KKT_target_range=[min(targets),max(targets)],phase_relaxation_release=float(sum(metrics['D_relax'][1:])),absolute_boundary_work_discrepancy=float(sum(abs(metrics['mechanical_work_error'][1:]))),absolute_internal_work_discrepancy=float(sum(abs(metrics['mechanical_work_error'][1:]+metrics['bulk_work'][1:]-metrics['boundary_work'][1:]))),fracture_expenditure=float(metrics[-1]['fracture']-metrics[0]['fracture']),bulk_profile_change=float(sum(metrics['bulk_profile_change'][1:])),cohesive_profile_change=float(sum(metrics['cohesive_profile_change'][1:])),maximum_phase_increment=float(max(metrics['increment'])),final=hist[-1],widths=width(y,phi,cc))

def plots(selection):
    ns=[int(n) for n in selection['evolving']];fine_fields=np.load(FINE/'fields.npz');fine_hist=load(FINE/'history.json')[:40]
    fig,ax=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for n in ns:
        row=selection['evolving'][str(n)];hist=load(E/f'n{n}-stress-history.json');out=(OLD if n<=64 else E)/f'one-d-n{n}-d2-q6';f=np.load(out/'fields.npz');r=Reference(n,1.,24);r.r=f['final_r'];r.Z=f['final_Z'];phi=f['phase'][-1];I,_=r.direct(phi)
        yy=np.linspace(0,1,16385);from normal_resolution import derivative
        sigma=G*derivative(n,yy)@r.r-r.Z*h(np.interp(yy,f['y'],phi))/I
        ax[0,0].plot(yy,sigma,label=f'{n} cells');ax[0,1].plot(f['y'],phi,label=f'{n} cells');ax[0,2].plot(f['y'],h(phi)/I,label=f'{n} cells')
        ax[1,0].plot([v['time'] for v in hist],[100*v['localized_error_fixed_scale'] for v in hist],label=f'{n} cells');ax[1,1].plot([v['time'] for v in hist],[v['oscillation_energy'] for v in hist],label=f'{n} cells');ax[1,2].plot([v['time'] for v in hist],[v['mean_error_Pa'] for v in hist],label=f'{n} cells')
    ax[0,0].axhline(fine_hist[-1]['stress_min'],color='k',ls=':',label='reference');ax[0,1].plot(fine_fields['y'],fine_fields['phase'][39],'k:',label='reference');ax[0,2].plot(fine_fields['y'],h(fine_fields['phase'][39])/fine_hist[-1]['I'],'k:',label='reference')
    for j,label in enumerate(['shear stress [Pa]','phase','chi [1/m]']):ax[0,j].set(xlabel='y [m]',ylabel=label);ax[0,j].legend(fontsize=8)
    for j,label in enumerate(['localized stress error [% peak reference]','oscillation storage [J/m]','mean stress error [Pa]']):ax[1,j].set(xlabel='time [s]',ylabel=label);ax[1,j].legend(fontsize=8)
    ax[1,0].axhline(5,color='k',ls='--');ax[1,1].set_yscale('symlog',linthresh=1e-6)
    fig.savefig(E/'normal-convergence.png',dpi=180);fig.savefig(E/'normal-convergence.pdf');plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    frozen=[v for v in selection['frozen'] if v['step']==40 and v['integration_points']==48]
    ax[0].loglog(ns,[v['localized_error_RMS_Pa'] for v in frozen],'o-',label='same physical phase/Z');ax[0].loglog(ns,[selection['evolving'][str(n)]['final']['localized_error_RMS_Pa'] for n in ns],'s-',label='evolving with own obstacle');ax[0].set(xlabel='normal cells',ylabel='localized RMS error [Pa]');ax[0].set_xticks(ns,[str(n) for n in ns]);ax[0].minorticks_off();ax[0].legend()
    for n in ns:
        out=(OLD if n<=64 else E)/f'one-d-n{n}-d2-q6';hist=load(out/'history.json');ax[1].plot([v['slip'] for v in hist],[v['q'] for v in hist],label=f'{n} cells');ax[2].plot([v['time'] for v in hist],[v['release'] for v in hist],label=f'{n} cells')
    ax[1].set(xlabel='cumulative slip [m]',ylabel='cohesive traction [Pa]');ax[1].legend(fontsize=8);ax[2].set(xlabel='time [s]',ylabel='cumulative phase-relaxation release [J/m]');ax[2].legend(fontsize=8)
    fig.savefig(E/'frozen-feedback.png',dpi=180);fig.savefig(E/'frozen-feedback.pdf');plt.close(fig)
    n=selection['selected_cells'];run=BUILD/'runs'/f'output-normal-resolution-q2-n{n}'
    p=np.loadtxt(E/'native-step40-x0.5-profile.csv',delimiter=',',skiprows=1)
    m=read(run/'fe-metrics.csv');prototype=load(E/f'one-d-n{n}-d2-q6/history.json')
    fig,ax=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for j,k in enumerate([5,1,2]):ax[0,j].plot(p[:,0],p[:,k],label='native Q2, 256 cells')
    ax[0,0].axhline(fine_hist[-1]['stress_min'],color='k',ls=':',label='exact-bulk reference')
    ax[0,1].plot(fine_fields['y'],fine_fields['phase'][39],'k:',label='reference')
    ax[0,2].plot(fine_fields['y'],h(fine_fields['phase'][39])/fine_hist[-1]['I'],'k:',label='reference')
    xs,ys,initial=grid(run,0);ax[0,1].plot(ys,initial[len(xs)//2],'--',label='native initial seed')
    for j,label in enumerate(['shear stress [Pa]','phase','chi [1/m]']):ax[0,j].set(xlabel='y [m]',ylabel=label);ax[0,j].legend(fontsize=8)
    ax[1,0].plot(m['time'],m['phi_max'],label='native');ax[1,0].plot([v['time'] for v in prototype],[v['phi_max'] for v in prototype],'--',label='matching discrete 1D')
    surface=[read(run/f'fe-surface-{j}.csv') for j in range(41)]
    q=[np.interp(.5,s['x'],s['q']) for s in surface];slip=[np.interp(.5,s['x'],s['slip']) for s in surface]
    ax[1,1].plot(slip,q,label='native');ax[1,1].plot([v['slip'] for v in prototype],[v['q'] for v in prototype],'--',label='matching discrete 1D')
    history=load(E/'native-stress-history.json');ax[1,2].plot([v['time'] for v in history],[v['mean_stress_energy'] for v in history],label='mean-stress storage');ax[1,2].plot([v['time'] for v in history],[v['oscillation_energy'] for v in history],label='oscillation storage');ax[1,2].set_yscale('symlog',linthresh=1e-5)
    for j,(x,y) in enumerate([('time [s]','maximum phase'),('cumulative slip [m]','cohesive traction [Pa]'),('time [s]','bulk storage [J/m]')]):ax[1,j].set(xlabel=x,ylabel=y);ax[1,j].legend(fontsize=8)
    fig.savefig(E/'selected-native.png',dpi=180);fig.savefig(E/'selected-native.pdf');plt.close(fig)

def main():
    selection=load(E/'selection.json');result=native(selection);plots(selection)
    (E/'native-assessment.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
