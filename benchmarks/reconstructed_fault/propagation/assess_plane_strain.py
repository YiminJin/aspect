#!/usr/bin/env python3
"""Native plane-strain controls, velocity enrichment and frozen integration audit."""
import json,re
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assess_localization import read,grid,transverse,direct,width
from total_energy_step import h,G

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OUT=HERE/'plane_strain_evidence';RUNS=ROOT/'build-propagation-integration/runs'

def basis(nodes,z,length):
    coefficients=np.linalg.solve(np.vander(nodes,len(nodes),increasing=True),np.eye(len(nodes)))
    b=np.polynomial.polynomial.polyval(z,coefficients)
    d=np.polynomial.polynomial.polyval(z,np.arange(1,len(nodes))[:,None]*coefficients[1:])/length
    return b,d

def stress(run,step,x,y):
    # Use actual FE support coordinates: Q3 has Gauss-Lobatto nodes, not thirds.
    xs,ys,phi=grid(run,step);dx=xs[1]-xs[0];dy=ys[1]-ys[0]
    encoding=read(run/f'fe-encoding-{step}.csv')
    lookup={(round(v['x'],13),round(v['y'],13),int(v['component'])):v['d_R'] for v in encoding}
    ex=np.unique(encoding['x']);ey=np.unique(encoding['y'])
    ix=min(int((x-xs[0])/dx),len(xs)-2);selected_x=ex[(ex>=xs[ix]-1e-13)&(ex<=xs[ix+1]+1e-13)]
    bx,dx_basis=basis((selected_x-xs[ix])/dx,(x-xs[ix])/dx,dx)
    a=np.zeros((3,len(y)))
    cells=np.minimum(((y-ys[0])/dy).astype(int),len(ys)-2)
    for iy in np.unique(cells):
        pick=cells==iy;selected_y=ey[(ey>=ys[iy]-1e-13)&(ey<=ys[iy+1]+1e-13)]
        by,dy_basis=basis((selected_y-ys[iy])/dy,(y[pick]-ys[iy])/dy,dy)
        for c in [0,1]:
            values=np.array([[lookup[round(xx,13),round(yy,13),c] for yy in selected_y] for xx in selected_x])
            if c==0:
                a[0,pick]=dx_basis@values@by
                a[2,pick]+=bx@values@dy_basis
            else:
                a[1,pick]=bx@values@dy_basis
                a[2,pick]+=dx_basis@values@by
    theta=a[0]+a[1];surf=read(run/f'fe-surface-{step}.csv')
    I=np.interp(x,surf['x'],surf['I']);Z=np.interp(x,surf['x'],surf['Z'])
    ph=np.interp(y,ys,transverse(xs,ys,phi,np.array([x]))[0]);chi=h(ph)/I
    tau=np.array([2*G*(a[0]-theta/3),2*G*(a[1]-theta/3),G*a[2]-Z*chi])
    return ph,chi,tau

def frozen(run,step,count):
    xs,ys,p=grid(run,step);z,w=leggauss(count);s=(z+1)/2;dy=ys[1]-ys[0]
    y=(ys[:-1,None]+dy*s).ravel();w=np.tile(w*dy/2,len(ys)-1)
    ph,chi,t=stress(run,step,.5,y)
    norm=t[0]**2+t[1]**2+2*t[2]**2+(t[0]+t[1])**2
    return dict(bulk=float(w@norm/(4*G)),shear_mean=float(w@t[2]),stress_rms=float(np.sqrt(w@(t[2]-w@t[2])**2)),source_integral=float(w@chi))

def main():
    result={};profiles={};histories={}
    for name in ['q2','q3']:
        label='compatible-'+name;run=RUNS/('output-plane-strain-'+label)
        metrics=read(run/'fe-metrics.csv');step=int(metrics[-1]['step']);log=(ROOT/'build-propagation-integration'/('plane-strain-'+label+'.log')).read_text()
        assert step==40 and metrics[-1]['time']==.625 and 'Exception' not in log
        target=[float(v) for v in re.findall(r'Phase box target:.*? target=([^\s]+)',log)]
        defect=[float(v) for v in re.findall(r'Phase box converged:.*? KKT=([^\s]+)',log)]
        assert len(target)==len(defect)==41 and all(a<=b for a,b in zip(defect,target))
        previous=grid(run,0)[2];recurrence=0.
        for j in range(1,step+1):
            current=grid(run,j)[2];assert min(current.ravel())>=0 and max(current.ravel())<1
            assert min((current-previous).ravel())>=-4.922e-12;previous=current
            a=read(run/f'fe-encoding-{j-1}.csv');b=read(run/f'fe-encoding-{j}.csv')
            beta=np.exp(-.015625/100);kappa=-1e8*np.expm1(-.015625/100)
            recurrence=max(recurrence,float(max(abs(b['d_R']-(beta*a['d_R']+kappa/G*b['u'])))))
        assert recurrence <= 256*np.finfo(float).eps*max(max(abs(b['d_R'])),1e-12)
        mechanics=[]
        for chunk in log.split('*** Timestep')[1:]:
            found=re.findall(r'Relative nonlinear residuals \(bulk, fault\) after nonlinear iteration\s+\d+: ([^,]+), ([^\s]+)',chunk)
            if found:mechanics.append([float(v) for v in found[-1]])
        assert all(max(v)<=1e-8 for v in mechanics)
        xs,ys,ph=grid(run,step);xx=np.linspace(0,1,4001);surf=read(run/f'fe-surface-{step}.csv')
        I=np.interp(xx,surf['x'],surf['I']);source_error=float(max(abs(direct(xs,ys,ph,xx)/I-1)))
        assert min(I)>0 and source_error<.005
        y=np.linspace(0,1,20001);phase,chi,tau=stress(run,step,.5,y);profiles[name]=(y,phase,chi,tau[2]);histories[name]=metrics
        np.savetxt(OUT/(name+'-transverse.csv'),np.c_[y,phase,chi,tau.T],delimiter=',',header='y,phi,chi,tau_xx,tau_yy,tau_xy',comments='')
        # Q24 resolves the specific Q6/Q12 remainder, without changing the solve.
        integration={str(q):frozen(run,step,q) for q in [3,6,12,24]}
        # Physical readings at center; compatible fields are uniform along x.
        onset=next((float(row['time']) for row in metrics if row['increment']>1e-5),None)
        local=[]
        for x in [0.,.5,1.]:
            pp=np.interp(y,ys,transverse(xs,ys,ph,np.array([x]))[0]);ii=np.interp(x,surf['x'],surf['I']);ch=h(pp)/ii
            local.append(dict(x=x,phi_max=float(max(pp)),q=float(np.interp(x,surf['x'],surf['q'])),V_interval=float(np.interp(x,surf['x'],surf['V_interval'])),slip=float(np.interp(x,surf['x'],surf['slip'])),**width(y,pp,ch)))
        curvature=[{n:float(v[n]) for n in v.dtype.names} for j in [32,40] for v in read(run/f'phase-curvature-{j}.csv')]
        result[name]=dict(steps=step,time=float(metrics[-1]['time']),onset=onset,phi_max=float(max(ph.ravel())),maximum_increment=float(max(metrics['increment'])),healing=float(max(metrics['healing'])),phase_spread_x=float(np.max(np.ptp(ph,axis=0))),local=local,recurrence_error=recurrence,source_error=source_error,phase_release=float(sum(metrics['D_relax'][1:])),absolute_mechanical_error=float(sum(abs(metrics['mechanical_work_error'][1:]))),fracture_expenditure=float(metrics[-1]['fracture']-metrics[0]['fracture']),native_bulk=float(metrics[-1]['bulk_energy']),frozen_normal_quadrature=integration,curvature=curvature,mechanical_residuals=np.max(mechanics,axis=0).tolist(),dense_stress_range=[float(min(tau[2])),float(max(tau[2]))],chi_max=float(max(chi)))
    # Same physical times; fine continuum reference has its own relaxed fine seed.
    fine=HERE/'localization_evidence/one-d-h1-n256-dt0.015625-exact-bulk';fields=np.load(fine/'fields.npz');reference=json.loads((fine/'history.json').read_text())
    yy=np.linspace(0,1,4001)
    for name in ['q2','q3']:
        run=RUNS/('output-plane-strain-compatible-'+name);ep=eq=ev=es=0.
        for step in range(1,41):
            xs,ys,p=grid(run,step);s=read(run/f'fe-surface-{step}.csv')
            a=np.interp(yy,ys,p[len(xs)//2]);b=np.interp(yy,fields['y'],fields['phase'][step-1])
            ep=max(ep,float(max(abs(a-b))));eq=max(eq,float(max(abs(s['q']-reference[step-1]['q']))));ev=max(ev,float(max(abs(s['V_interval']-reference[step-1]['V']))));es=max(es,float(max(abs(s['slip']-reference[step-1]['slip']))))
        result[name]['fine_reference_common_time_errors']=dict(phi=ep,q=eq,V_interval=ev,slip=es)
    coarse=HERE/'localization_evidence/one-d-h1-n16-dt0.015625-q3-exact-bulk'
    cf=np.load(coarse/'fields.npz');ch=json.loads((coarse/'history.json').read_text())
    for name in ['q2','q3']:
        run=RUNS/('output-plane-strain-compatible-'+name);ep=eq=0.
        for step in range(1,41):
            xs,ys,p=grid(run,step);s=read(run/f'fe-surface-{step}.csv')
            ep=max(ep,float(max(abs(p[len(xs)//2]-cf['phase'][step-1]))))
            eq=max(eq,float(max(abs(s['q']-ch[step-1]['q']))))
        result[name]['matching_phase_grid_exact_bulk_reference']=dict(phi=ep,q=eq)
    old=RUNS/'output-localization-compatible-operator-check';new=RUNS/'output-plane-strain-compatible-q2';difference=0.;surface=np.zeros(5)
    for step in range(41):
        difference=max(difference,float(max(abs(grid(old,step)[2]-grid(new,step)[2]).ravel())))
        left=read(old/f'fe-surface-{step}.csv');right=read(new/f'fe-surface-{step}.csv')
        surface=np.maximum(surface,[max(abs(left[z]-right[z])) for z in ['q','V_interval','slip','I','Z']])
    result['raw_vs_plane_q2_phase_difference']=difference
    result['raw_vs_plane_q2_surface_difference']=dict(zip(['q','V_interval','slip','I','Z'],surface.tolist()))
    # Compact profiles and common-time physical histories.
    fig,ax=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for name in ['q2','q3']:
        y,p,c,t=profiles[name];m=histories[name]
        for k,v in enumerate([p,c,t]):ax[0,k].plot(y,v,label=name.upper())
        ax[1,0].plot(m['time'],m['phi_max'],label=name.upper())
        rates=[];qs=[]
        for step in range(41):
            s=read(RUNS/('output-plane-strain-compatible-'+name)/f'fe-surface-{step}.csv');qs.append(np.interp(.5,s['x'],s['q']));rates.append(np.interp(.5,s['x'],s['V_interval']))
        ax[1,1].plot(m['time'],qs,label=name.upper());ax[1,2].plot(m['time'],rates,label=name.upper())
    row=reference[39];yr=fields['y'];pr=fields['phase'][39];ir=row['I'];cr=h(pr)/ir
    ax[0,0].plot(yr,pr,'k:',label='rich 1D');ax[0,1].plot(yr,cr,'k:',label='rich 1D');ax[0,2].axhline(row['stress_min'],color='k',ls=':',label='rich 1D')
    times=[r['time'] for r in reference[:40]]
    for k,key in enumerate(['phi_max','q','V']):ax[1,k].plot(times,[r[key] for r in reference[:40]],'k:',label='rich 1D')
    for k,label in enumerate(['phase','chi [1/m]','tau_xy [Pa]']):ax[0,k].set(xlabel='y [m]',ylabel=label);ax[0,k].legend()
    for k,label in enumerate(['phase maximum','q [Pa]','interval V [m/s]']):ax[1,k].set(xlabel='time [s]',ylabel=label);ax[1,k].legend()
    fig.savefig(OUT/'spatial-comparison.png',dpi=180);fig.savefig(OUT/'spatial-comparison.pdf')
    (OUT/'assessment.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
