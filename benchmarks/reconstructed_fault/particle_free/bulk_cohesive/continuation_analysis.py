#!/usr/bin/env python3
"""Accepted native P4 response in physical coordinates; no history remapping."""
from pathlib import Path
import os
import argparse,json
import numpy as np
from fracture_analysis import read as slowread,save_csv,grid_value,crossing,fracture,initial_error
ROOT=Path(os.environ.get('ASPECT_P4_ROOT',Path(__file__).resolve().parents[4])).resolve();BASE=ROOT/'build-particle-free/bulk-continuation';OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/bulk_continuation';OLD=ROOT/'build-particle-free/bulk-fracture/runs/candidate'
LEVELS=[.001,.01,.05,.1,.3,.5]
def read(path):
    with path.open() as f:names=f.readline().strip().split(',')
    if names[0]=='gate':return slowread(path)
    cols=list(range(len(names)))
    if names[0]=='cell':cols=cols[1:];names=names[1:]
    data=np.loadtxt(path,delimiter=',',skiprows=1,usecols=cols,ndmin=2)
    return np.rec.fromarrays(data.T,names=names)
def displacement(t):return .018*np.clip(np.asarray(t)-.05,0,.125)
def initial_forces(phase,dx,dy,Kf=8.,ell=.046875):
    mx=dx*np.array([[2.,1.],[1.,2.]])/6;my=dy*np.array([[2.,1.],[1.,2.]])/6
    kx=np.array([[1.,-1.],[-1.,1.]])/dx;ky=np.array([[1.,-1.],[-1.,1.]])/dy
    stiffness=np.kron(ky,mx)+np.kron(my,kx)
    cell=np.stack([phase[:-1,:-1],phase[:-1,1:],phase[1:,:-1],phase[1:,1:]],axis=-1)
    g=Kf*(dx*dy/4+2*ell*ell*np.einsum('ij,abj->abi',stiffness,cell))
    gradient=np.zeros_like(phase);gradient[:-1,:-1]+=g[...,0];gradient[:-1,1:]+=g[...,1];gradient[1:,:-1]+=g[...,2];gradient[1:,1:]+=g[...,3]
    mass=np.full_like(phase,dx*dy);mass[[0,-1],:]*=.5;mass[:,[0,-1]]*=.5
    return np.stack([mass,gradient,np.zeros_like(phase),gradient],axis=-1)

def load(run):
    res=read(run/'resources.csv');steps=res['step'].astype(int);times=res['time'].astype(float)
    first=read(run/f'phase-{steps[0]:04d}.csv');first.sort(order='dof');x,y=np.unique(first['x']),np.unique(first['y']);order=np.lexsort((first['x'],first['y']));shape=(len(y),len(x))
    phase=[];stress=[];tauxx=[];tauyy=[];Y=[];rate=[];maxrate=[];reaction=[];increments=[];forces=[]
    for step in steps:
        a=read(run/f'phase-{step:04d}.csv');a.sort(order='dof');phase.append(a['phi'][order].reshape(shape))
        a=read(run/f'means-{step:04d}.csv');a.sort(order=['y','x']);stress.append(a['tauxy'].reshape(len(y)-1,len(x)-1));tauxx.append(a['tauxx'].reshape(len(y)-1,len(x)-1));tauyy.append(a['tauyy'].reshape(len(y)-1,len(x)-1));Y.append(a['Y'].reshape(len(y)-1,len(x)-1));rate.append(a['abs_rate'].reshape(len(y)-1,len(x)-1));maxrate.append(max(a['max_rate']))
        a=read(run/f'boundary-reactions-{step:04d}.csv');mask=(a['prescribed']==1)&(a['component']==0);b=(a['y'][mask]-.5)/(y[-1]-y[0]);reaction.append(float(np.sum(a['force'][mask]*b)));increments.append(float(np.sum(a['actual_increment'][mask]*b)/np.sum(b*b)))
        forcefile=run/f'phase-forces-{step:04d}.csv'
        if step==0 and not forcefile.exists():
            forces.append(initial_forces(phase[-1],x[1]-x[0],y[1]-y[0]))
        else:
            a=read(forcefile);a.sort(order='dof');forces.append(np.column_stack([a[k][order] for k in ('mass','fracture_gradient','mechanical_gradient','residual')]).reshape(*shape,4))
    ledger=read(run/'bulk-accepted.csv') if (run/'bulk-accepted.csv').exists() else np.empty(0)
    if len(ledger):
        assert np.all(ledger['publications']==ledger['step'])
        assert max(ledger['mechanical_residual'])<=1e-9 and max(ledger['phase_KKT'])<=1e-7 and max(ledger['projection_error'])<=1e-12
    assert np.all(read(run/'gates.csv')['passed']==1)
    return dict(x=x,y=y,phase=np.array(phase),stress=np.array(stress),tauxx=np.array(tauxx),tauyy=np.array(tauyy),Y=np.array(Y),rate=np.array(rate),maxrate=np.array(maxrate),forces=np.array(forces),steps=steps,times=times,reaction=np.array(reaction),increment=np.array(increments),ledger=ledger)
def initial_run(run):
    seen=set()
    while not (run/'phase-0000.csv').exists():
        assert str(run) not in seen;seen.add(str(run))
        provenance=BASE/(run.name+'-input.json')
        if not provenance.exists():
            assert run.name=='half';return OLD
        run=(ROOT/json.load(open(provenance))['source']).parents[1]
    return run

def prepare(name):
    run=OLD if name=='original' else BASE/'runs'/name;d=load(run);x,y=d['x'],d['y'];dx=x[1]-x[0];dy=y[1]-y[0]
    if d['steps'][0]>0:
        # Resumes have a physically matching parent from the pinned original
        # prefix or the same-mesh run named in the checkpoint manifest.
        manifest=BASE/(name+'-input.json')
        info=json.load(open(manifest)) if manifest.exists() else dict(source='build-particle-free/bulk-fracture/runs/candidate/restart/02',accepted_step=127,accepted_time=.15875)
        source=ROOT/info['source'];parentrun=source.parents[1];s=info['accepted_step']
        a=read(parentrun/f'phase-{s:04d}.csv');a.sort(order='dof');parent=a['phi'][np.lexsort((a['x'],a['y']))].reshape(len(y),len(x))
        delta=np.diff(np.concatenate([parent[None],d['phase']]),axis=0)
        a=read(parentrun/f'boundary-reactions-{s:04d}.csv');mask=(a['prescribed']==1)&(a['component']==0);qprev=float(np.sum(a['force'][mask]*(a['y'][mask]-.5))/(y[-1]-y[0]))
        tprev=info['accepted_time']
        init=read(initial_run(parentrun)/'phase-0000.csv');init.sort(order='dof');p0=init['phi'][np.lexsort((init['x'],init['y']))].reshape(len(y),len(x))
    else:parent=d['phase'][0];p0=parent;delta=np.diff(d['phase'],axis=0,prepend=parent[None]);qprev=d['reaction'][0];tprev=d['times'][0]
    assert np.min(delta)>=0
    growth=d['phase']-p0;mass=np.ones_like(p0)*dx*dy;mass[[0,-1],:]*=.5;mass[:,[0,-1]]*=.5
    center=np.argmin(abs(y-.5));contours=[];history=[];worktr=0.
    for k,(step,t) in enumerate(zip(d['steps'],d['times'])):
        for angle in (0.,-45.,45.):
            rr=np.linspace(0,min(x[-1]-1.,.5)*np.sqrt(2) if angle else x[-1]-1.,5001);theta=np.deg2rad(angle);xx=1+rr*np.cos(theta);yy=.5+rr*np.sin(theta);valid=(xx<=x[-1])&(yy>=y[0])&(yy<=y[-1]);rr=rr[valid];vals=grid_value(x,y,d['phase'][k],xx[valid],yy[valid])
            for level in LEVELS:
                r=crossing(rr,vals,level);contours.append([t,displacement(t),angle,level,r,1+r*np.cos(theta),.5+r*np.sin(theta)])
        row=d['ledger'][d['ledger']['step']==step] if len(d['ledger']) else []
        if len(row):
            a=row[0];dt_value=float(a['dt']);wtr=.5*(qprev+d['reaction'][k])*d['increment'][k];worktr+=wtr
            remainder=wtr-a['energy_change']-a['Db']-a['Dc']-a['Df'];potential=a['bulk']+a['cohesive']+a['fracture']+a['Df']/(1+a['dt']*.01)
            energies=[a[c] for c in ('bulk','cohesive','fracture','Wext','Db','Dc','Df','energy_change','ledger_error','stress_loss','traction_loss','phase_remainder','fracture_loss','phase_work','mechanical_work')]
        else:
            wtr=remainder=dt_value=0.;energies=[0.]*15
            energies[2]=fracture(p0,dx,dy,8.,.046875);potential=energies[2]
        history.append([t,displacement(t),step,np.max(delta[k]),np.max(growth[k]),np.sum(mass*growth[k]),np.sum(d['rate'][k])*dx*dy,d['maxrate'][k],d['reaction'][k],wtr,remainder,potential]+energies+[dt_value,d['increment'][k]])
        qprev=d['reaction'][k];tprev=t
    save_csv(OUT/(name+'-history.csv'),['time_s','imposed_relative_displacement_m','step','max_step_phase_increment','max_total_phase_increment','integrated_phase_growth_m2','integrated_abs_w_m2_s','max_abs_w_s_inv','shear_reaction_N_m','trapezoidal_work_J_m','trapezoidal_remainder_J_m','incremental_potential_J_m','bulk_J_m','cohesive_J_m','fracture_J_m','endpoint_work_J_m','Db_J_m','Dc_J_m','friction_J_m','energy_change_J_m','ledger_error_J_m','stress_loss_J_m','traction_loss_J_m','phase_remainder_J_m','fracture_loss_J_m','phase_work_J_m','mechanical_work_J_m','dt_s','native_prescribed_increment_m'],history)
    save_csv(OUT/(name+'-contours.csv'),['time_s','imposed_relative_displacement_m','ray_angle_deg','phi_contour','distance_from_original_endpoint_m','x_m','y_m'],contours)
    np.savez_compressed(BASE/(name+'-plot-data.npz'),**{k:v for k,v in d.items() if k!='ledger'},initial=p0,parent=parent,growth=growth,delta=delta)
    summary=dict(name=name,first_time=float(d['times'][0]),last_time=float(d['times'][-1]),dx=dx,dy=dy,cells=(len(x)-1)*(len(y)-1),max_step_phase_increment=float(delta.max()),max_slip_activity=float(d['maxrate'].max()),trapezoidal_work=float(worktr),initial_representation=initial_error(x,y,p0,.046875,1.))
    unloaded=d['times']<=.05+1e-12
    summary['unloaded_phase_drift']=float(np.max(abs(d['phase'][unloaded]-p0))) if np.any(unloaded) else None
    summary['initial_unloaded_KKT_gap_density']=float(np.min(d['forces'][0,...,3]/(8*d['forces'][0,...,0]))) if d['steps'][0]==0 else None
    (OUT/(name+'-summary.json')).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
    return d

def compare(a,b,label):
    with np.load(BASE/(a+'-plot-data.npz')) as z:a={k:z[k] for k in ['x','y','phase','times','initial','parent','stress','tauxx','tauyy']}
    with np.load(BASE/(b+'-plot-data.npz')) as z:b={k:z[k] for k in ['x','y','phase','times','initial','parent','stress','tauxx','tauyy']}
    rows=[]
    # Common physical Gauss3 points in the original 1.5x1 specimen. Fields are
    # interpolated as physical Q1 fields; no constitutive histories are remapped.
    bx=np.unique(np.r_[a['x'][a['x']<=1.5],b['x'][b['x']<=1.5]]);by=np.unique(np.r_[a['y'],b['y']]);z,w=np.polynomial.legendre.leggauss(3)
    xx=.5*(bx[:-1,None]+bx[1:,None])+np.diff(bx)[:,None]*z/2;yy=.5*(by[:-1,None]+by[1:,None])+np.diff(by)[:,None]*z/2;X,Y=np.meshgrid(xx.ravel(),yy.ravel());W=np.outer((np.diff(by)[:,None]*w/2).ravel(),(np.diff(bx)[:,None]*w/2).ravel())
    a0=grid_value(a['x'],a['y'],a['initial'],X,Y);b0=grid_value(b['x'],b['y'],b['initial'],X,Y)
    # A Q1 field difference is Q1 on the common Cartesian subdivision. Its
    # absolute maximum is attained at a subdivision vertex, not a Gauss point.
    NX,NY=np.meshgrid(bx,by)
    na0=grid_value(a['x'],a['y'],a['initial'],NX,NY);nb0=grid_value(b['x'],b['y'],b['initial'],NX,NY)
    for i,t in enumerate(a['times']):
        j=np.argmin(abs(b['times']-t))
        if abs(b['times'][j]-t)>1e-10:continue
        ap=grid_value(a['x'],a['y'],a['phase'][i],X,Y);bp=grid_value(b['x'],b['y'],b['phase'][j],X,Y)
        ga=ap-a0;gb=bp-b0
        nga=grid_value(a['x'],a['y'],a['phase'][i],NX,NY)-na0;ngb=grid_value(b['x'],b['y'],b['phase'][j],NX,NY)-nb0
        # Temporal branches share the accepted .15875 state: separately report
        # interval-growth normalization, not dilution by earlier cap growth.
        pa=grid_value(a['x'],a['y'],a['phase'][np.argmin(abs(a['times']-.15875))] if np.min(abs(a['times']-.15875))<1e-10 else a['parent'],X,Y);pb=grid_value(b['x'],b['y'],b['phase'][np.argmin(abs(b['times']-.15875))] if np.min(abs(b['times']-.15875))<1e-10 else b['parent'],X,Y)
        ia=ap-pa;ib=bp-pb
        interval_norm=float(np.sum(W*ib**2))
        # At the common parent there is no interval growth to normalize by.
        # Record an explicit validity flag; avoid subtracting/re-adding the
        # seed, which creates roundoff-sized false differences at this time.
        interval_defined=interval_norm>0
        interval_error=np.sqrt(np.sum(W*(ia-ib)**2)/interval_norm) if interval_defined else 0.
        def means_at(d,k):
            ix=np.clip(np.searchsorted(d['x'],X,side='right')-1,0,len(d['x'])-2);iy=np.clip(np.searchsorted(d['y'],Y,side='right')-1,0,len(d['y'])-2)
            return [d[key][k,iy,ix] for key in ['tauxx','tauyy','stress']]
        aa=means_at(a,i);bb=means_at(b,j);delta=[u-v for u,v in zip(aa,bb)]
        tensor=lambda v:v[0]**2+v[1]**2+(v[0]+v[1])**2+2*v[2]**2
        stress_difference=np.sqrt(np.sum(W*tensor(delta))/max(1e-100,np.sum(W*tensor(bb))))
        rows.append([t,displacement(t),np.sqrt(np.sum(W*(ga-gb)**2)/max(1e-100,np.sum(W*gb**2))),np.max(abs(nga-ngb))/max(1e-100,np.max(abs(ngb))),interval_error,np.sum(W*ga),np.sum(W*gb),np.sqrt(np.sum(W*(a0-b0)**2)/max(1e-100,np.sum(W*b0*b0))),stress_difference,int(interval_defined)])
    save_csv(OUT/(label+'.csv'),['time_s','imposed_relative_displacement_m','growth_relative_L2','growth_relative_peak_difference','interval_growth_relative_L2','growth_integral_a_m2','growth_integral_b_m2','initial_field_relative_L2_difference','cell_mean_full_3D_tensor_relative_L2_difference','interval_growth_defined'],rows)
    print(label,rows[-1])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('names',nargs='+');p.add_argument('--compare',nargs=3);args=p.parse_args()
    for name in args.names:prepare(name)
    if args.compare:compare(*args.compare)
