#!/usr/bin/env python3
"""Streaming reductions of rectangular layered coupled histories and common-label comparisons."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-layered-mpl')
import argparse,json,re
from pathlib import Path
import numpy as np
from layered_fields import State,read,dot3,powers,surface,work_traction,surface_rule,localization
from compare_bounded_profile import ROOT,RUNS
from assess_sheared_moving_fault import metric,nearest_labels,masks
OUT=Path(os.environ.get('LAYERED_EVIDENCE',str(ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/layered_campaign')))
OUT.mkdir(parents=True,exist_ok=True)
LABELS=(np.arange(400)+.5)/400

def load_case(name):
    path=RUNS/name;cmd=json.loads((path/'command.json').read_text());assert cmd['exit_code']==0,name
    rows=read(path/'output/material-summary.csv')
    if rows and 'step' not in rows[0]:
        import csv
        with (path/'output/material-summary.csv').open() as f:rows=list(csv.DictReader(f,fieldnames=['step','time','dt','history_publications','geometry_publications','retries']))
    return path,cmd,rows

def state_at(path,cmd,step):return State(path/'output',step,4/(2**cmd['arguments']['level']*cmd['arguments'].get('depth_repetitions',1)))

def exterior_flux(path,step,s,order=8):
    bg=read(path/'output'/f'background-{step}.csv')
    g,_=np.polynomial.legendre.leggauss(8)
    inv=np.linalg.inv(np.polynomial.legendre.legvander(g,7))
    a=np.array([float(r['xx']) for r in bg]).reshape(64,8)@inv.T
    xy=np.array([float(r['xy']) for r in bg]).reshape(64,8)@inv.T
    supplied=penalty=0.
    for axis,side,x,w in s.faces(order):
        u=(x[:,1]+2)/.0625;j=np.minimum(63,np.maximum(0,np.floor(u).astype(int)))
        b=np.polynomial.legendre.legvander(2*(u-j)-1,7)
        aa=np.sum(b*a[j],axis=1);ss=np.sum(b*xy[j],axis=1);ext=np.stack([aa,-aa,ss],axis=1)
        v=s.at(x);tau=v[:,3:6];vn=side*v[:,axis+1];up=np.where((vn<0)[:,None],ext,tau)
        supplied+=w@(vn*dot3(up,up)/40);jump=tau-ext
        penalty+=w@(-np.minimum(vn,0)*dot3(jump,jump)/40)
    return dict(endpoint_supplied_flux=float(supplied),incoming_penalty=float(penalty))

def reduce_case(name):
    path,cmd,rows=load_case(name);initial=state_at(path,cmd,int(rows[0]['step']));old=initial
    cum=np.zeros(len(initial.surface));spin=cum.copy();fault_spin=cum.copy();envelope=0.;records=[];prior=None;signed=absolute=active=trap_signed=trap_abs=0.
    geometry_error=age_error=0.;U=cmd['arguments']['translation']
    for row in rows:
        step=int(row['step']);dt=float(row['dt']);time=float(row['time']);s=state_at(path,cmd,step)
        assert int(row['history_publications'])==int(row['geometry_publications'])==step
        if prior is not None:
            X=old.surface[:,1:3];v=old.at(X);geometry_error=max(geometry_error,float(np.max(np.linalg.norm(s.surface[:,1:3]-X-dt*v[:,1:3],axis=1))))
            spin+=.5*dt*(v[:,14]-v[:,13]);cum+=dt*s.surface[:,6]
            fault_spin+=dt*np.asarray(prior['spin_diagnostics']['label_omega_f'])
            envelope+=dt*prior['energy']['sampled_max_abs_omega_f']
            V=s.surface[:,6];factor=V*dt/.008
            predicted=old.surface[:,3]*np.exp(-factor)+dt*(-np.expm1(-factor))/factor
            age_error=max(age_error,float(np.max(abs(predicted-s.surface[:,3]))))
        e=powers(s,old,initial);e.update(step=step,time=time,dt=dt,**exterior_flux(path,step,s))
        rhs=e['mechanical_power']+e['background_power']-e['friction_power']-e['radiation_power']-e['maxwell_loss']+e['dilatation_power']-e['flux']
        e['rhs']=rhs
        if prior is not None:
            defect=e['storage']-prior['energy']['storage']-dt*rhs
            trapezoid=e['storage']-prior['energy']['storage']-.5*dt*(rhs+prior['energy']['rhs'])
            signed+=defect;absolute+=abs(defect);trap_signed+=trapezoid;trap_abs+=abs(trapezoid)
            active+=.5*dt*sum(abs(z[k]) for z in (e,prior['energy']) for k in ['mechanical_power','slip_power','maxwell_loss','flux','dilatation_power'])
            e.update(endpoint_remainder=defect,trapezoid_remainder=trapezoid,
                     endpoint_supplied_flux_remainder=defect+dt*(e['endpoint_supplied_flux']-e['flux']))
            e['phase_transport_remainder']=e['profile_mass']-prior['energy']['profile_mass']-.5*dt*sum(z['phase_dilatation']-z['phase_flux'] for z in (e,prior['energy']))
        work,measure=work_traction(s);vals=surface(s,LABELS,cum,work)
        axes=[np.linspace(s.lo[j],s.hi[j],s.shape[j]+1) for j in (0,1)]
        nodes=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,2)
        nodal=s.at(nodes)[:,0];boundary=np.any((nodes==s.lo)|(nodes==s.hi),axis=1)
        e.update(nodal_phase_min=float(nodal.min()),nodal_phase_max=float(nodal.max()),boundary_phase_max=float(nodal[boundary].max()))
        geometry=s.surface[:,1:3];clearance=np.min(np.minimum(geometry-s.lo,s.hi-geometry))
        local_v=s.at(geometry);phi=local_v[:,0]
        chi=s.m*phi*(1+phi)/(1-phi)**2/s.surface[:,5]
        sense=float(read(path/'output/material-model.csv')[0]['sense'])
        omega=.5*(local_v[:,14]-local_v[:,13]);omega_f=-.5*sense*chi*s.surface[:,6]
        tau=local_v[:,3:6];anis=np.hypot(tau[:,0]-tau[:,1],2*tau[:,2])
        tendency=np.sqrt(2.)*abs(omega_f)*anis
        sd=dict(label_chi=chi.tolist(),label_omega=omega.tolist(),label_omega_f=omega_f.tolist(),
                label_omega_b=(omega-omega_f).tolist(),fault_spin_exposure=fault_spin.tolist(),
                bulk_spin_exposure=(spin-fault_spin).tolist(),sampled_max_envelope=envelope,
                label_stress=tau.tolist(),principal_angle=(.5*np.arctan2(2*tau[:,2],tau[:,0]-tau[:,1])).tolist(),
                principal_gap=anis.tolist(),stress_weighted_tendency=tendency.tolist())
        record=dict(time=time,step=step,dt=dt,energy=e,spin_diagnostics=sd,profile={k:v.tolist() for k,v in vals.items()},
          curve=geometry.tolist(),slip_vertices=s.surface[:,6].tolist(),cumulative_vertices=cum.tolist(),
          age_vertices=s.surface[:,3].tolist(),spin=spin.tolist(),clearance=float(clearance),retries=int(row['retries']))
        records.append(record);prior=record;old=s
    last=records[-1];q12=powers(old, state_at(path,cmd,int(rows[-2]['step'])),initial,12)
    qdiff={k:float(q12[k]-last['energy'][k]) for k in q12 if isinstance(q12[k],(float,int))}
    log=(path/'run.log').read_text();iters=[int(v) for v in re.findall(r'Fault linear solve: iterations=(\d+)',log)]
    stages={'transport':[],'reaction':[]}
    for line in log.splitlines():
        if 'Direct transport energy:' in line or 'Direct source energy:' in line:
            d={k:float(v) for k,v in re.findall(r'(\w+)=([^\s]+)',line)}
            if 'Direct transport energy:' in line:stages['transport'].append(d)
            elif d.get('publication')==1:stages['reaction'].append(d)
    result=dict(case=name,arguments=cmd['arguments'],wall_s=cmd['wall_s'],labels=LABELS.tolist(),records=records,
      energy=dict(signed=signed,absolute=absolute,active=active,relative=absolute/active,trapezoid_signed=trap_signed,trapezoid_absolute=trap_abs,final_q12_minus_q8=qdiff),
      checks=dict(geometry_replay_Linf=geometry_error,age_replay_Linf=age_error,minimum_curve_clearance=min(r['clearance'] for r in records)),
      cost=dict(stokes_solves=len(iters),stokes_iterations=sum(iters),advection_substeps=[int(v) for v in re.findall(r'advection substeps=(\d+)',log)]),stage_energy=stages)
    (OUT/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(name,'energy',result['energy']['relative'],'checks',result['checks'],flush=True)
    return result

def compare_records(a,b,translate=False):
    by={round(r['time'],10):r for r in b['records']};result=[]
    for ra in a['records']:
        key=round(ra['time'],10)
        if key not in by:continue
        rb=by[key];A={k:np.array(v) for k,v in ra['profile'].items()};B={k:np.array(v) for k,v in rb['profile'].items()}
        if translate:
            A['geometry'][:,0]-=a['arguments']['translation']*ra['time'];B['geometry'][:,0]-=b['arguments']['translation']*rb['time']
        regions={}
        for region,mask in masks(np.array(a['labels'])).items():
            w=B['jacobian'][mask]/len(mask);values={}
            for k in A:
                if k=='jacobian':continue
                err=A[k][mask]-B[k][mask];base=B[k][mask]
                if k=='stress':num,den=dot3(err,err),dot3(base,base)
                elif err.ndim==2:num,den=(err*err).sum(axis=1),(base*base).sum(axis=1)
                else:num,den=err*err,base*base
                values[k]=metric(num,den,w)
            regions[region]=values
        result.append(dict(time=ra['time'],regions=regions))
    return result

def common_bulk_quadrature(sa,sb,shift,order=8):
    """Integrate in b coordinates on the union of both Cartesian cell partitions.

    Shifted and non-nested DG meshes have internal jumps on different lines.
    A rule on b cells alone can straddle jumps in a; splitting at both sets
    of lines makes whole-domain Q2 stress differences polynomial per panel.
    """
    g,w=np.polynomial.legendre.leggauss(order)
    axes=[];weights=[]
    for j in (0,1):
        offset=shift if j==0 else 0.
        lower=max(sa.lo[j]-offset,sb.lo[j]);upper=min(sa.hi[j]-offset,sb.hi[j])
        assert upper>lower
        ea=sa.lo[j]+sa.h*np.arange(sa.shape[j]+1)-offset
        eb=sb.lo[j]+sb.h*np.arange(sb.shape[j]+1)
        edges=np.sort(np.r_[lower,ea[(ea>lower)&(ea<upper)],eb[(eb>lower)&(eb<upper)],upper])
        edges=edges[np.r_[True,np.diff(edges)>1e-11*max(1.,upper-lower)]]
        widths=np.diff(edges)
        axes.append((edges[:-1,None]+.5*widths[:,None]*(g+1)).ravel())
        weights.append((.5*widths[:,None]*w).ravel())
    return np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,2),np.outer(*weights).ravel()


def compare_final(a,b,translate=False,at_time=None):
    pa,ca,ra=load_case(a['case']);pb,cb,rb=load_case(b['case'])
    ar=a['records'][-1] if at_time is None else next(r for r in a['records'] if abs(r['time']-at_time)<1e-10)
    br=b['records'][-1] if at_time is None else next(r for r in b['records'] if abs(r['time']-at_time)<1e-10)
    assert abs(ar['time']-br['time'])<1e-10, 'Compare matching accepted physical times'
    sa=state_at(pa,ca,int(ar['step']));sb=state_at(pb,cb,int(br['step']))
    time=br['time']
    shift=(a['arguments']['translation']-b['arguments']['translation'])*time if translate else 0.
    x,w=common_bulk_quadrature(sa,sb,shift);xa=x.copy();xa[:,0]+=shift
    ta,tb=sa.stress(xa),sb.stress(x);num,den=dot3(ta-tb,ta-tb),dot3(tb,tb);labels,distance=nearest_labels(sb,x)
    result={'common_domain':metric(num,den,w)}
    for region,mask in masks(labels).items():
        mask&=distance<=.2;result[region]=metric(num[mask],den[mask],w[mask])
    va,vb=sa.at(xa),sb.at(x)
    phi_a,phi_b=va[:,0],vb[:,0];result['phase']=metric((phi_a-phi_b)**2,phi_b**2,w)
    ua,ub=va[:,1:3].copy(),vb[:,1:3].copy()
    ua[:,0]-=a['arguments']['translation'];ub[:,0]-=b['arguments']['translation']
    result['velocity_without_uniform_translation']=metric(np.sum((ua-ub)**2,axis=1),np.sum(ub**2,axis=1),w)
    result['pressure']=metric((va[:,17]-vb[:,17])**2,vb[:,17]**2,w)
    labels,weights=surface_rule([sa,sb]);A=surface(sa,labels,ar['cumulative_vertices']);B=surface(sb,labels,br['cumulative_vertices'])
    if translate:
        A['geometry'][:,0]-=a['arguments']['translation']*time;B['geometry'][:,0]-=b['arguments']['translation']*time
    result['surface_common_quadrature']={}
    for region,mask in masks(labels).items():
        vals={};ww=weights[mask]*B['jacobian'][mask]
        for k in A:
            if k=='jacobian':continue
            e=A[k][mask]-B[k][mask];base=B[k][mask]
            if k=='stress':num,den=dot3(e,e),dot3(base,base)
            elif e.ndim==2:num,den=np.sum(e*e,axis=1),np.sum(base*base,axis=1)
            else:num,den=e*e,base*base
            vals[k]=metric(num,den,ww)
        result['surface_common_quadrature'][region]=vals
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('cases',nargs='+');p.add_argument('--compare',action='store_true');a=p.parse_args()
    if not a.compare:
        for name in a.cases:reduce_case(name)
    else:
        cases=[json.loads((OUT/(n+'.json')).read_text()) for n in a.cases];pairs=[]
        for a,b in zip(cases[:-1],cases[1:]):
            shifted=a['arguments']['translation']!=b['arguments']['translation']
            pairs.append(dict(a=a['case'],b=b['case'],shifted=shifted,history=compare_records(a,b,shifted),bulk_final=compare_final(a,b,shifted)))
        (OUT/('comparison-'+cases[0]['case']+'-'+cases[-1]['case']+'.json')).write_text(json.dumps(pairs,indent=2)+'\n')
if __name__=='__main__':main()
