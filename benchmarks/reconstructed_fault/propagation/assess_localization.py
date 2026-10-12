#!/usr/bin/env python3
"""Frozen accepted fields: profiles, widths, exact normalization and common times."""
import json,re,hashlib
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid
from total_energy_step import h,G,M
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/'localization_evidence'

def read(file):return np.genfromtxt(file,delimiter=',',names=True,ndmin=1)
def grid(run,step):
    a=read(run/f'fe-phase-{step}.csv');xs=np.unique(a['x']);ys=np.unique(a['y']);lookup={(r['x'],r['y']):r['phi'] for r in a}
    return xs,ys,np.array([[lookup[(x,y)] for y in ys] for x in xs])
def transverse(xs,ys,phi,x):
    i=np.minimum(np.searchsorted(xs,x,side='right')-1,len(xs)-2);s=(x-xs[i])/(xs[i+1]-xs[i]);return (1-s[:,None])*phi[i]+s[:,None]*phi[i+1]
def direct(xs,ys,phi,x):
    ph=transverse(xs,ys,phi,np.asarray(x));u=1-ph[:,:-1].astype(np.longdouble);v=1-ph[:,1:].astype(np.longdouble);z=(v-u)/u
    ratio=np.ones_like(z);nz=z!=0;ratio[nz]=np.log1p(z[nz])/z[nz]
    return np.asarray(np.sum(np.longdouble(M)*np.diff(ys)*(2/(u*v)-3*ratio/u+1),axis=1),float)
def width(y,phi,chi):
    mask=phi>=max(phi)/2;fwhm=float(np.trapezoid(mask.astype(float),y))
    cdf=cumulative_trapezoid(chi,y,initial=0.);cdf/=cdf[-1]
    cwhm=float(np.trapezoid((chi>=max(chi)/2).astype(float),y))
    center=float(np.trapezoid(y*chi,y)/np.trapezoid(chi,y));sigma=float(np.sqrt(np.trapezoid((y-center)**2*chi,y)/np.trapezoid(chi,y)))
    return dict(phi_FWHM=fwhm,chi_FWHM=cwhm,chi_95_width=float(np.interp(.975,cdf,y)-np.interp(.025,cdf,y)),chi_centroid=center,chi_two_sigma=2*sigma)
def stress(run,step,x,y):
    xs,ys,p=grid(run,step);a=read(run/f'fe-encoding-{step}.csv');nx=len(xs)-1;ny=len(ys)-1;dx=xs[1]-xs[0];dy=ys[1]-ys[0]
    r=np.zeros((2*nx+1,2*ny+1,2))
    for v in a:r[round((v['x']-xs[0])/dx*2),round((v['y']-ys[0])/dy*2),int(v['component'])]=v['d_R']
    i=min(int((x-xs[0])/dx),nx-1);j=np.minimum(((y-ys[0])/dy).astype(int),ny-1);s=(x-xs[i])/dx;t=(y-ys[j])/dy
    def b(z):return np.array([(1-z)*(1-2*z),4*z*(1-z),z*(2*z-1)])
    def d(z,ds):return np.array([-3+4*z,4-8*z,-1+4*z])/ds
    bs,bt,ds,dt=b(s),b(t),d(s,dx),d(t,dy);shear=np.zeros_like(y)
    for l in range(3):
        for k in range(3):shear+=bs[l]*dt[k]*r[2*i+l,2*j+k,0]+ds[l]*bt[k]*r[2*i+l,2*j+k,1]
    surf=read(run/f'fe-surface-{step}.csv');I=np.interp(x,surf['x'],surf['I']);Z=np.interp(x,surf['x'],surf['Z']);ph=np.interp(y,ys,transverse(xs,ys,p,np.array([x]))[0]);return G*shear-Z*h(ph)/I

def assess(name,run,log):
    m=read(run/'fe-metrics.csv');end=int(m[-1]['step']);xs,ys,ph=grid(run,end)
    phases=[grid(run,j)[2] for j in range(end+1)];assert min(p.min() for p in phases)>=0 and max(p.max() for p in phases)<1
    assert min(np.diff(phases,axis=0).ravel())>=-4.922e-12
    text=log.read_text();targets=[float(x) for x in re.findall(r'Phase box target:.*? target=([^\s]+)',text)];res=[float(x) for x in re.findall(r'Phase box converged:.*? KKT=([^\s]+)',text)]
    assert len(res)>=end+1 and all(r<=t for r,t in zip(res,targets))
    errors=[]
    for j in [0,end]:
        xx,yy,p=grid(run,j);s=read(run/f'fe-surface-{j}.csv');x=np.unique(np.r_[np.linspace(0,1,4001),s['x']]);reference=direct(xx,yy,p,x);I=np.interp(x,s['x'],s['I']);err=reference/I-1;worst=np.argmax(abs(err))
        assert min(I)>0
        errors.append(dict(step=j,max_source_error=float(max(abs(err))),worst_x=float(x[worst]),endpoint_source_error=float(max(abs(err[[0,-1]])))))
        np.savetxt(OUT/f'{name}-normalization-{j}.csv',np.c_[x,reference,I,err],delimiter=',',header='x,direct,consistent,source_error',comments='')
    local=[];y=np.linspace(ys[0],ys[-1],20001);s=read(run/f'fe-surface-{end}.csv')
    for x in [0.,.5,1.]:
        phi=np.interp(y,ys,transverse(xs,ys,ph,np.array([x]))[0]);I=np.interp(x,s['x'],s['I']);chi=h(phi)/I
        local.append(dict(x=x,q=float(np.interp(x,s['x'],s['q'])),V=float(np.interp(x,s['x'],s['V_interval'])),slip=float(np.interp(x,s['x'],s['slip'])),phase_max=float(max(phi)),normal_boundary_phase=phi[[0,-1]].tolist(),**width(y,phi,chi)))
        np.savetxt(OUT/f'{name}-profile-x{x:g}.csv',np.c_[y,phi,chi,stress(run,end,x,y)],delimiter=',',header='y,phi,chi,tau_xy',comments='')
    rec=0.
    for j in range(1,end+1):
        old=read(run/f'fe-encoding-{j-1}.csv');new=read(run/f'fe-encoding-{j}.csv');dt=m[j]['dt'];beta=np.exp(-G*dt/1e8);kappa=-1e8*np.expm1(-G*dt/1e8)
        e=max(abs(new['d_R']-(beta*old['d_R']+kappa/G*new['u'])));rec=max(rec,e)
    assert rec<1e-14 and max(m['Maxwell_error_Pa'])<1e-7 and max(abs(m['energy_reconstruction_error']))<1e-8
    mechanics=[]
    for chunk in text.split('*** Timestep')[1:]:
        found=re.findall(r'Relative nonlinear residuals \(bulk, fault\) after nonlinear iteration\s+\d+: ([^,]+), ([^\s]+)',chunk)
        if found:mechanics.append(found[-1])
    assert all(max(float(v) for v in pair)<=1e-8 for pair in mechanics)
    onset=next((float(m[j]['time']) for j,p in enumerate(phases) if max((p-phases[0]).ravel())>1e-5),None)
    return dict(steps=end,time=float(m[-1]['time']),completed=bool(end==64 and 'Exception' not in text),height=float(ys[-1]-ys[0]),mesh_dx=float(xs[1]-xs[0]),mesh_dy=float(ys[1]-ys[0]),phi_max=float(ph.max()),phi_spread_x=float(np.max(np.ptp(ph,axis=0))),maximum_increment=float(max(m['increment'])),healing=float(max(m['healing'])),onset=onset,normalization=errors,local=local,recurrence_error=rec,Maxwell_error=float(max(m['Maxwell_error_Pa'])),phase_release=float(sum(m['D_relax'][1:])),absolute_mechanical_error=float(sum(abs(m['mechanical_work_error'][1:]))),fracture_expenditure=float(m[-1]['fracture']-m[0]['fracture']),mechanics_max=[max(float(v[k]) for v in mechanics) for k in [0,1]] if mechanics else None)

def main():
    names={'affine':'affine-final-executable','compatible':'compatible-operator-check','wide':'wide-operator-check'}
    result={};runs={}
    for name,label in names.items():
        run=ROOT/'build-propagation-integration/runs'/('output-localization-'+label);runs[name]=run;result[name]=assess(name,run,ROOT/'build-propagation-integration'/('localization-'+label+'.log'))
    common=min(r['time'] for r in result.values());j=round(common/.015625);fields={k:grid(v,j) for k,v in runs.items()}
    result['common_time']=common;result['boundary_difference']=float(max(abs(fields['affine'][2]-fields['compatible'][2]).ravel()))
    result['normal_size_difference_center']=float(max(abs(fields['compatible'][2][8]-fields['wide'][2][8,8:25])))
    # Complete native-vs-1D coarse Q3 and fine histories at common times.
    comparisons={}
    for cells,q in [(16,3),(128,6),(256,6)]:
        path=OUT/f'one-d-h1-n{cells}-dt0.015625'
        if q!=6:path=path.with_name(path.name+f'-q{q}')
        data=np.load(path/'fields.npz');hist=json.loads((path/'history.json').read_text());maxphi=maxq=maxV=maxslip=0.
        for step in range(1,min(64,len(hist),result['compatible']['steps'])+1):
            xs,ys,p=grid(runs['compatible'],step);s=read(runs['compatible']/f'fe-surface-{step}.csv');reference=np.interp(ys,data['y'],data['phase'][step-1]);maxphi=max(maxphi,float(np.max(abs(p-reference[None,:]))));maxq=max(maxq,float(max(abs(s['q']-hist[step-1]['q']))));maxV=max(maxV,float(max(abs(s['V_interval']-hist[step-1]['V']))));maxslip=max(maxslip,float(max(abs(s['slip']-hist[step-1]['slip']))))
        comparisons[str(cells)]=dict(full_phi=maxphi,q=maxq,V=maxV,slip=maxslip)
    result['native_reference']=comparisons
    rich={}
    yy=np.linspace(0,1,4001)
    for cells,q in [(16,3),(128,6),(256,6)]:
        path=OUT/f'one-d-h1-n{cells}-dt0.015625'
        if q!=6:path=path.with_name(path.name+f'-q{q}')
        path=path.with_name(path.name+'-exact-bulk');data=np.load(path/'fields.npz');hist=json.loads((path/'history.json').read_text())
        errphi=errq=errV=errslip=0.
        for j in range(1,65):
            xx,y,p=grid(runs['compatible'],j);native=np.array([np.interp(yy,y,row) for row in p]);reference=np.interp(yy,data['y'],data['phase'][j-1]);s=read(runs['compatible']/f'fe-surface-{j}.csv')
            errphi=max(errphi,float(max(abs(native-reference[None,:]).ravel())));errq=max(errq,float(max(abs(s['q']-hist[j-1]['q']))));errV=max(errV,float(max(abs(s['V_interval']-hist[j-1]['V']))));errslip=max(errslip,float(max(abs(s['slip']-hist[j-1]['slip']))))
        y=data['y'];p=data['phase'][-1];I=float(data['Z'])/hist[-1]['q']
        fine=np.linspace(0,1,20001);phi=np.interp(fine,y,p);chi=h(phi)/I
        rich[str(cells)]=dict(full_phi=errphi,q=errq,V=errV,slip=errslip,local_widths=width(fine,phi,chi),last=hist[-1])
    result['native_rich_reference']=rich
    (OUT/'assessment.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(2,3,figsize=(14,8))
    for name,run in runs.items():
        rows=read(run/'fe-metrics.csv');end=int(rows[-1]['step']);xx,yy,p=grid(run,end);s=read(run/f'fe-surface-{end}.csv');y=np.linspace(yy[0],yy[-1],4001)
        for x,style in [(0.,'--'),(.5,'-')]:
            phi=np.interp(y,yy,transverse(xx,yy,p,np.array([x]))[0]);chi=h(phi)/np.interp(x,s['x'],s['I']);label=name+(' endpoint' if x==0 else ' center')
            ax[0,0].plot(y,phi,style,label=label);ax[0,1].plot(y,chi,style,label=label);ax[0,2].plot(y,stress(run,end,x,y),style,label=label)
        surfaces=[read(run/f'fe-surface-{i}.csv') for i in range(end+1)]
        for x,style in [(0.,'--'),(.5,'-')]:
            ax[1,0].plot(rows['time'],[np.interp(x,v['x'],v['q']) for v in surfaces],style,label=name+f' x={x}')
            ax[1,1].plot(rows['time'],[np.interp(x,v['x'],v['V_interval']) for v in surfaces],style,label=name+f' x={x}')
        ax[1,2].plot(rows['time'],rows['phi_max'],label=name)
    ref=OUT/'one-d-h1-n256-dt0.015625-exact-bulk';d=np.load(ref/'fields.npz');hist=json.loads((ref/'history.json').read_text());time=[v['time'] for v in hist]
    ax[0,2].axhline(hist[-1]['stress_max'],color='k',linestyle=':',label='1D256 uniform stress')
    ax[0,0].plot(d['y'],d['phase'][-1],'k:',label='1D256');ax[0,1].plot(d['y'],d['chi'],'k:',label='1D256')
    ax[1,0].plot(time,[v['q'] for v in hist],'k:',label='1D256');ax[1,1].plot(time,[v['V'] for v in hist],'k:',label='1D256');ax[1,2].plot(time,[v['phi_max'] for v in hist],'k:',label='1D256')
    for a,label in zip(ax.ravel(),['phase','chi [1/m]','tau_xy [Pa]','cohesive q [Pa]','interval V [m/s]','maximum phase']):a.set_ylabel(label);a.legend(fontsize=6)
    for a in ax[0]:a.set_xlabel('y [m]')
    for a in ax[1]:a.set_xlabel('time [s]')
    fig.tight_layout();fig.savefig(OUT/'native-boundary-comparison.png',dpi=160);fig.savefig(OUT/'native-boundary-comparison.pdf');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
