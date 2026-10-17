#!/usr/bin/env python3
"""Assess actual common-entry coupled advancement, not a diagnostic time driver."""
import argparse
import json
from functools import cache
from pathlib import Path
import numpy as np
from assess_pf6a import csv, norm2, grids
from assess_phase_energy import lagrange2
from assess_pf6a_spatial import fields, gradient, compare, Frozen
from total_energy_step import G, ETA, M, ELL, FRACTURE_SCALE as A


@cache
def independent(path):
    f=fields(path); w=f['weight']; t=np.stack([f[k] for k in ['xx','yy','xy']],axis=-1)
    bulk=float(w@norm2(t)/(4*G))
    # Q1 phase gradients, independently differentiated at the common points.
    nodes=csv(path/'gradient.csv'); nx=2; ny=round(1/f['dy'][0]); p=np.full((ny+1,nx+1),np.nan)
    for a in nodes:p[round(a['y']*ny),round(a['x']*nx)]=a['phi']
    i=np.minimum((nx*f['x']).astype(int),nx-1);j=np.minimum((ny*f['y']).astype(int),ny-1)
    r=nx*f['x']-i;s=ny*f['y']-j
    gx=nx*((1-s)*(p[j,i+1]-p[j,i])+s*(p[j+1,i+1]-p[j+1,i]))
    gy=ny*((1-r)*(p[j+1,i]-p[j,i])+r*(p[j+1,i+1]-p[j,i+1]))
    frac=float(A*w@(f['phi']+ELL**2*(gx*gx+gy*gy)))
    sf=csv(path/'surface.csv');x,qw=np.polynomial.legendre.leggauss(8);x=(x+1)/2;qw=qw/2
    coh=0.
    for a,b in zip(sf[:-1],sf[1:]):
        z=(1-x)*a['Z']+x*b['Z'];I=(1-x)*a['I']+x*b['I']
        coh+=float((b['x']-a['x'])*(qw@(z*z/I))/(2*G))
    energy=csv(path/'energy.csv');err=max(abs(v-float(energy[k])) for k,v in [('bulk',bulk),('cohesive',coh),('FE_fracture',frac)])
    # Same functional integrated on common high-order points; nine surface Q1 loads.
    ns=len(sf);seg=np.minimum((f['x']*(ns-1)).astype(int),ns-2);xi=f['x']*(ns-1)-seg
    loads=np.zeros(ns)
    for side,shape in [(0,1-xi),(1,xi)]:np.add.at(loads,seg+side,w*shape*f['chi']*f['xy'])
    # Reuse the inexpensive 1D functional with the measured shear derivative.
    # Quantify the discarded transverse/non-shear content, rather than imply a
    # general 2D reference. The native fixture is homogeneous along the fault.
    raw=csv(path/'tensor-nodes.csv');nc=len(np.unique(raw['cell']));tg=grids(raw,['xx','yy','xy'],nc)
    geometry=raw[np.unique(raw['cell'],return_index=True)[1]]
    incoming=np.zeros(2*ny+1);p2=0.;transverse=0.
    for j in range(ny):
        ids=np.flatnonzero(np.abs(geometry['y0']-j/ny)<1e-14)
        tv=tg[ids,:,:,2].mean(axis=(0,1));p2=max(p2,abs(tv[1]-.5*(tv[0]+tv[2])))
        transverse=max(transverse,float(np.max(abs(tg[ids,:,:,2]-tv[None,None,:]))))
        slope=tv[2]-tv[0]
        incoming[2*j+1]=incoming[2*j]+(.5*tv[0]+.125*slope)/(G*ny)
        incoming[2*j+2]=incoming[2*j]+(tv[0]+.5*slope)/(G*ny)
    refs=[]
    for order in [24,48]:
        ref=Frozen(ny,order,p.mean(axis=1),incoming,float(sf['Z'].mean()))
        value,g,terms=ref.objective(ref.phi);refs.append((A*value,A*g))
    gg=nodes['elastic']+nodes['FE_fracture'];native=np.zeros(ny+1)
    np.add.at(native,np.rint(nodes['y']*ny).astype(int),gg)
    gerror=float(np.max(abs(native-refs[1][1])))
    assert gerror<1e-6 and p2<1e-6 and transverse<1e-5,(path,gerror,p2,transverse)
    return dict(bulk=bulk,cohesive=coh,fracture=frac,energy=bulk+coh+frac,
                oneD_gradient_error_Linf=gerror,
                oneD_energy_difference=abs(refs[1][0]-(bulk+coh+frac)),
                reference_quadrature_24_48_gradient=float(np.max(abs(refs[0][1]-refs[1][1]))),
                reference_quadrature_24_48_energy=abs(refs[0][0]-refs[1][0]),
                transverse_tensor_variation_Pa=transverse,normal_P2_shear_defect_Pa=p2,
                transverse_phase_variation=float(np.max(np.ptp(p,axis=1))),
                common_rule_energy_difference_J_per_m=float(err),shear_weak_loads=loads.tolist())


def recurrence(old,new,dt):
    a,b=csv(old/'tensor-nodes.csv'),csv(new/'tensor-nodes.csv');nc=len(np.unique(b['cell']))
    Tn=grids(a,['xx','yy','xy'],nc);Te=grids(b,['xx','yy','xy'],nc);u=grids(b,['ux','uy'],nc)
    r=np.repeat([0.,.5,1.],3);s=np.tile([0.,.5,1.],3)
    dr=np.array([4*r-3,4-8*r,4*r-1]).T;ds=np.array([4*s-3,4-8*s,4*s-1]).T
    cells=b[np.unique(b['cell'],return_index=True)[1]]
    ux=np.einsum('qi,qj,cijk->cqk',dr,lagrange2(s),u)/cells['dx'][:,None,None]
    uy=np.einsum('qi,qj,cijk->cqk',lagrange2(r),ds,u)/cells['dy'][:,None,None]
    div=ux[...,0]+uy[...,1]
    E=np.stack([ux[...,0]-div/3,uy[...,1]-div/3,.5*(ux[...,1]+uy[...,0])],axis=-1)
    expected=np.exp(-dt/100)*Tn.reshape(nc,9,3)+2*(-ETA*np.expm1(-dt/100))*E
    error=Te.reshape(nc,9,3)-expected
    return dict(max_Pa=float(np.max(abs(error))),scaled=float(np.max(abs(error))/max(1.,np.max(abs(expected)))))


def difference(a,b):
    fa,fb=fields(a),fields(b);w=fa['weight'];chi=fb['chi']
    delta=np.stack([fa[k]-fb[k] for k in ['xx','yy','xy']],axis=-1)
    stress=np.stack([fb[k] for k in ['xx','yy','xy']],axis=-1)
    result=compare(a,b)
    result.update(whole_full_tensor_L2_Pa=float(np.sqrt(w@norm2(delta))),
                  relative_full_tensor_L2=float(np.sqrt(w@norm2(delta)/(w@norm2(stress)))),
                  localized_full_tensor_RMS_Pa=float(np.sqrt((w*chi)@norm2(delta)/(w@chi))),
                  phase_L2=float(np.sqrt(w@(fa['phi']-fb['phi'])**2)),
                  phase_Linf=float(max(abs(fa['phi']-fb['phi']))),
                  total_energy_difference_J_per_m=abs(independent(a)['energy']-independent(b)['energy']))
    sa,sb=csv(a/'surface.csv'),csv(b/'surface.csv')
    for k in ['I','Z']:result[k+'_Linf']=float(max(abs(sa[k]-sb[k])))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=dict(runs=[],pairs={});lookup={}
    for path in a.runs:
        record=json.loads((path/'record.json').read_text());assert record['gate_passed']
        case=record['spatial_case'];assert case['event']
        summary=csv(path/'run/event-summary.csv');steps=len(summary)-1;assert steps==case['divisor']//32
        assert np.array_equal(summary['step'],summary['publications']) and max(summary['healing'])==0
        entries=[]
        for k,row in enumerate(summary):
            state=path/'run'/f'event-{k}';r=dict(step=k,time=float(row['time']),KKT=float(row['KKT']),independent=independent(state))
            assert r['independent']['common_rule_energy_difference_J_per_m']<1e-8,r
            if k:
                r['tensor_recurrence']=recurrence(path/'run'/f'event-{k-1}',state,float(row['dt']));assert r['tensor_recurrence']['scaled']<1e-11,r
                assert row['KKT']<1e-8 and row['Zerror']<1e-7,row
            entries.append(r)
        out['runs'].append(dict(label=path.name,case=case,seconds=record['seconds'],states=entries,
                               phase_peak=summary['phase_max'].tolist(),publications=int(summary[-1]['publications'])))
        lookup[(case['normal_cells'],case['divisor'])]=(path,steps)
    baseline=next(iter(lookup.values()))[0]
    for key,(path,steps) in lookup.items():
        d=difference(baseline/'run/event-0',path/'run/event-0');out['pairs']['entry_'+path.name]=d
        assert max(d[k] for k in ['physical_stress_scaled','gradient_scaled','energy'])<1e-9
        if key[0]!=128 and key[1]==128 and (128,128) in lookup:
            ref,nref=lookup[(128,128)];out['pairs']['spatial_'+str(key[0])]=difference(path/'run'/f'event-{steps}',ref/'run'/f'event-{nref}')
        if key==(128,256):
            ref,nref=lookup[(128,128)];out['pairs']['temporal_128']=difference(ref/'run'/f'event-{nref}',path/'run'/f'event-{steps}')
    out['passed']=True;a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
