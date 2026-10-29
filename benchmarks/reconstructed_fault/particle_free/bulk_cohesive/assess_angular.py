#!/usr/bin/env python3
"""Read-only angular diagnosis. All constitutive samples stay at saved QPs.

Arc quantities qbar/Ybar/F/R are the same positive, unit-mass weak averaging
functional. Y is averaged AFTER evaluation at each saved state, never rebuilt
from interpolated q or phase. The actual phase geometry remains the Q1 field.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from assess_native import csv,gauss,KF,ELL,G,M,SEED_R,SEED_P
from assess_sensitivity import fields
from compare_native import grid,interpolate

RADII=[.15,.225,float(np.hypot(.1875,.21875)),float(np.hypot(.21875,.203125)),.325]
THETA=np.linspace(-90.,90.,721)
PATHS=dict(wide=Path('build-particle-free/bulk-sensitivity/runs/right-distance'),
           fine=Path('build-particle-free/bulk-sensitivity/runs/fine-gauss8'),
           square=Path('build-particle-free/bulk-sensitivity/runs/square'))


def smooth_table(n=262145):
    theta=np.linspace(0,np.pi/2,n);p=.6*np.cos(theta)**2
    v=2*ELL*(1-p)/np.sqrt(1.5-p);h=theta[1]-theta[0]
    r=np.zeros(n);r[2::2]=np.cumsum(h*(v[:-2:2]+4*v[1:-1:2]+v[2::2])/3)
    r[1::2]=r[:-2:2]+h*(5*v[:-2:2]+8*v[1:-1:2]-v[2::2])/12
    return r,p


SMOOTH_R,SMOOTH_P=smooth_table()


def smooth_profile(radius):return np.interp(radius,SMOOTH_R,SMOOTH_P,right=0.)


def smooth_force(radius):
    radius=np.asarray(radius);p=smooth_profile(radius)
    F=p*(.6-p)*(1.5-p)/(1-p)**2
    hp=128*(1+3*p)/(1-p)**3
    band=KF*hp/1280
    curvature=np.divide(2*KF*ELL*np.sqrt(np.maximum(F,0)),radius,out=np.full_like(p,216.),where=radius>0)
    cap=band+curvature
    return np.where(radius<SMOOTH_R[-1],band,KF),np.where(radius<SMOOTH_R[-1],cap,KF)


def qp_state(path,step):
    # Fixed mesh/QP geometry comes from a complete existing export. CellId's
    # comparator uses numeric coarse id, not string order or active iteration.
    geo=csv(path/'bulk-0052.csv')
    assert all(c.endswith('_0:') for c in np.unique(geo['cell']))
    coarse=np.array([int(c.split('_')[0]) for c in geo['cell']]);geo=geo[np.argsort(coarse,kind='stable')]
    metrics=csv(path/'fixture.csv');meta=metrics[0];ndg=int(meta['DG_DoFs']);nq=len(geo)
    raw=np.fromfile(path/f'state-{step:04d}.bin',dtype=np.float64)
    local=raw[ndg:ndg+5*nq].reshape(nq,5)
    phase=csv(path/f'phase-{step:04d}.csv');phase.sort(order='dof')
    assert np.array_equal(raw[ndg+5*nq:-3],phase['phi'])
    assert raw[-3]==step and raw[-1]==step
    assert raw[-2]==metrics['time'][metrics['step']==step][0]
    x,y,p=grid(path,step);phi=interpolate((x,y,p),geo['x'],geo['y'])
    q=local[:,3];Y=M*(1+3*phi)/(1-phi)**3*q*q/(2*G)
    return dict(x=geo['x'],y=geo['y'],weight=geo['JxW'],phi=phi,q=q,Y=Y),geo,local


def average_nodes(samples,values,x,y):
    nx,ny=len(x)-1,len(y)-1;dx=x[1]-x[0];dy=y[1]-y[0]
    i=np.clip(np.searchsorted(x,samples['x'],side='right')-1,0,nx-1)
    j=np.clip(np.searchsorted(y,samples['y'],side='right')-1,0,ny-1)
    u=(samples['x']-x[i])/dx;v=(samples['y']-y[j])/dy
    out=np.zeros((len(y),len(x)));mass=np.zeros_like(out)
    for a,b,N in [(0,0,(1-u)*(1-v)),(1,0,u*(1-v)),(0,1,(1-u)*v),(1,1,u*v)]:
        np.add.at(out,(j+b,i+a),samples['weight']*N*values)
        np.add.at(mass,(j+b,i+a),samples['weight']*N)
    return out/mass,mass


def smooth_weak(x,y,order=24,tabulated=False):
    """Exact weak form via edge integrals: Delta(Q1 basis)=0 per affine cell.

    Split every edge at compact support, band/cap join, and band center.
    No pointwise second derivative is assigned to a tabulated/Q1 function.
    """
    radius=SEED_R[-1] if tabulated else SMOOTH_R[-1]
    tab_r,tab_p=(SEED_R,SEED_P) if tabulated else (SMOOTH_R,SMOOTH_P)
    z,w=gauss(order);dx=x[1]-x[0];dy=y[1]-y[0];ny,nx=len(y)-1,len(x)-1
    vertical=np.zeros((len(x),ny,2));horizontal=np.zeros((len(y),nx,2))
    for direction,constants,intervals,target in [('v',x,y,vertical),('h',y,x,horizontal)]:
        for a,c in enumerate(constants):
            cuts=[.5]
            if direction=='v':
                d=max(c-.5,0.)
                if d<radius:
                    t=np.sqrt(radius*radius-d*d);cuts += [.5-t,.5+t]
            else:
                d=abs(c-.5)
                if d<radius:cuts += [.5+np.sqrt(radius*radius-d*d)]
            for b,(lo,hi) in enumerate(zip(intervals[:-1],intervals[1:])):
                split=sorted(set([lo,hi]+[v for v in cuts if lo<v<hi]))
                for left,right in zip(split[:-1],split[1:]):
                    s=left+(right-left)*z;t=(s-lo)/(hi-lo)
                    X,Y=(np.full_like(s,c),s) if direction=='v' else (s,np.full_like(s,c))
                    p=np.interp(np.hypot(np.maximum(X-.5,0),Y-.5),tab_r,tab_p,right=0.)
                    target[a,b,0]+=(right-left)*np.sum(w*p*(1-t))
                    target[a,b,1]+=(right-left)*np.sum(w*p*t)
    grad=np.zeros((len(y),len(x)));mass=np.zeros_like(grad)
    for i in range(nx):
        for j in range(ny):
            for a in range(2):
                for b in range(2):
                    grad[j+b,i+a]+=(2*a-1)/dx*(vertical[i+1,j,b]-vertical[i,j,b])+(2*b-1)/dy*(horizontal[j+1,i,a]-horizontal[j,i,a])
                    mass[j+b,i+a]+=dx*dy/4
    return KF*(1+2*ELL**2*grad/mass)


def locate_first(path):
    x,y,p0=grid(path,0);X,Y=np.meshgrid(x,y);r=np.hypot(X-.5,Y-.5)
    region=(X>.5)&(r<.35);previous=p0
    for step in range(1,65):
        ph=grid(path,step)[2];new=region&((ph-previous)>1e-10)
        if np.any(new):
            j,i=np.where(new)
            return step,[dict(i=int(ii),j=int(jj),x=float(x[ii]),y=float(y[jj]),radius=float(r[jj,ii]),
                              angle=float(np.rad2deg(np.arctan2(y[jj]-.5,x[ii]-.5))),increment=float(ph[jj,ii]-previous[jj,ii])) for jj,ii in zip(j,i)]
        previous=ph
    raise AssertionError('No cap activation in saved interval')


def analyze(path,include_reference=False):
    first,nodes=locate_first(path);pre=first-1;d=fields(path,pre);init=fields(path,1)
    assert np.array_equal(init['phi'],init['initial'])
    samples,geo,local=qp_state(path,52)
    decode={key:float(np.max(abs(local[:,index]-geo[key]))) for key,index in [('a',0),('k',1),('L',2),('q',3)]}
    assert max(decode.values())==0.
    data={};checks={};records=[]
    if include_reference:
        ref=smooth_weak(d['x'],d['y']);check=smooth_weak(d['x'],d['y'],order=32)
        tab=smooth_weak(d['x'],d['y'],tabulated=True)
        checks['smooth_edge_G24_G32_Pa']=float(np.max(abs(check-ref)))
        checks['tabulated_vs_smooth_weak_Pa']=float(np.max(abs(tab-ref)))
    for step in sorted(set([0,54,pre,first])):
        s,_,_=qp_state(path,step)
        qbar,mass=average_nodes(s,s['q'],d['x'],d['y']);Ybar,_=average_nodes(s,s['Y'],d['x'],d['y'])
        force=init if step==0 else fields(path,step)
        checks[str(step)]=dict(mass=float(np.max(abs(mass-force['mass']))),
                               mechanical=float(np.max(abs(Ybar-force['Yweak'])))/KF)
        if step==0:assert np.all(s['q']==0.)
        else:assert checks[str(step)]['mechanical']<1e-10
        data[str(step)]={**force,'qbar':qbar,'Ybar':Ybar,'samples':s}
        for radius in RADII:
            # Include actual grid crossings: arc densities can have sharp
            # derivative changes at native nodes. A uniform angle sample can
            # otherwise miss the first-growing node's exact residual minimum.
            cx=(force['x']-.5)/radius;cx=cx[(cx>=0)&(cx<=1)]
            cy=(force['y']-.5)/radius;cy=cy[abs(cy)<=1]
            angles=np.unique(np.round(np.r_[THETA,np.rad2deg(np.arccos(cx)),-np.rad2deg(np.arccos(cx)),np.rad2deg(np.arcsin(cy))],12))
            t=np.deg2rad(angles);X=.5+radius*np.cos(t);Y=.5+radius*np.sin(t)
            record=dict(step=np.full(len(t),step),radius=np.full(len(t),radius),angle=angles,
                        phi_smooth=np.full(len(t),smooth_profile(radius)),F_point_cap=np.full(len(t),smooth_force(radius)[1]))
            for name,key in [('phi','phi'),('qbar','qbar'),('Ybar','Ybar'),('F','Fweak'),('R','Rweak')]:
                record[name]=interpolate((force['x'],force['y'],data[str(step)][key]),X,Y)
            if include_reference:
                record['F_smooth_weak']=interpolate((force['x'],force['y'],ref),X,Y)
                record['F_tabulated_weak']=interpolate((force['x'],force['y'],tab),X,Y)
            records.append(record)
    for n in nodes:
        i,j=n['i'],n['j'];n.update(initial_phi=float(init['phi'][j,i]),F0=float(init['Fweak'][j,i]),
                                Fpre=float(d['Fweak'][j,i]),Ypre=float(d['Yweak'][j,i]),Rpre=float(d['Rweak'][j,i]))
    return dict(first_step=first,pre_step=pre,nodes=nodes,decode_errors=decode,checks=checks),records,data


def main(out):
    out.mkdir(exist_ok=True);summary={}
    for name,path in PATHS.items():
        result,records,data=analyze(path,name=='wide');summary[name]=result
        keys=list(records[0]);matrix=np.concatenate([np.column_stack([r[k] for k in keys]) for r in records])
        np.savetxt(out/(name+'-arcs.csv'),matrix,delimiter=',',header=','.join(keys),comments='',fmt='%.17g')
        if name=='wide':
            arcs=[r for r in records if r['step'][0]==54];extrema=[];nodes=[dict(n) for n in result['nodes']]
            for r in arcs:
                positive=r['angle']>=0;theta=r['angle'][positive]
                entry=dict(radius=float(r['radius'][0]),point_cap_force=float(r['F_point_cap'][0]))
                for key in ['phi','qbar','Ybar','F','F_smooth_weak','R']:
                    v=r[key][positive]
                    entry[key]=dict(minimum=float(min(v)),minimum_angle=float(theta[np.argmin(v)]),maximum=float(max(v)),maximum_angle=float(theta[np.argmax(v)]))
                extrema.append(entry)
            for n in nodes:
                r=next(r for r in arcs if abs(r['radius'][0]-n['radius'])<1e-12)
                zero=np.argmin(abs(r['angle']));at=np.argmin(abs(r['angle']-n['angle']))
                n.update(F_center=float(r['F'][zero]),Y_center=float(r['Ybar'][zero]),R_center=float(r['R'][zero]),F_smooth_weak=float(r['F_smooth_weak'][at]),
                         fracture_help=float(r['F'][zero]-n['Fpre']),mechanical_help=float(n['Ypre']-r['Ybar'][zero]))
            (out/'mechanism.json').write_text(json.dumps(dict(arc_extrema=extrema,nodes=nodes),indent=2)+'\n')
        np.savez_compressed(out/(name+'-nodal.npz'),**{f'{s}_{k}':v for s,d in data.items() for k,v in d.items() if k!='samples'})
        if name=='wide':
            for step,d in data.items():np.savez_compressed(out/f'wide-qps-{step}.npz',**d['samples'])
    summary['seed']=dict(ell=ELL,peak=.6,p=1,m=128,native_table_size=5000,native_support_radius=float(SEED_R[-1]),
                         smooth_support_radius=float(SMOOTH_R[-1]),
                         tabulated_phase_max_error=float(np.max(abs(np.interp(SMOOTH_R,SEED_R,SEED_P,right=0)-SMOOTH_P))))
    summary['strict_reference_match_passed']=False
    (out/'assessment.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();main(a.output)
