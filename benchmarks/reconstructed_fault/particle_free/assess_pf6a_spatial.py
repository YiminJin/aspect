#!/usr/bin/env python3
"""Independent fixed-field, 1D mechanical/projection, and common-functional checks."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from assess_pf6a import csv, norm2

PROP=Path(__file__).resolve().parents[1]/'propagation'
sys.path.insert(0,str(PROP))
from localization_1d import Reference
from normal_resolution import derivative, moments
from total_energy_step import G, ETA, FRACTURE_SCALE as A, h


def q2sample(values,y):
    n=(len(values)-1)//2;c=np.minimum((np.asarray(y)*n).astype(int),n-1);s=np.asarray(y)*n-c
    b=np.array([2*(s-.5)*(s-1),4*s*(1-s),2*s*(s-.5)]).T
    return np.sum(b*values[2*c[:,None]+np.arange(3)],axis=1)


class Frozen(Reference):
    def initialize(self):return np.interp(self.y,self.input_y,self.input_phase)
    def __init__(self,n,order,phase,incoming,Z,exact=False):
        self.input_y=np.linspace(0,1,len(phase));self.input_phase=phase
        super().__init__(n,1.,order,exact_bulk=exact)
        self.r=q2sample(incoming,np.linspace(0,1,2*n+1));self.Z=Z
        if exact:self.rq=derivative(len(phase)-1,self.yq)@incoming


def reference(n,order,common,exact=False):
    phase=common['phase'][0].mean(axis=1);incoming=common['R'][1,...,0].mean(axis=1);Z=float(common['Z'][1].mean())
    r=Frozen(n,order,phase,incoming,Z,exact)
    info=r.mechanical(.5625,1/128)
    value,gradient,terms=r.objective(r.phi)
    y=r.yq;w=r.w;I=terms[6];chi=terms[7];tau=terms[8]
    # Independent cellwise P1 L2 projection, not the mechanical linear solve.
    s=y*n-np.minimum((y*n).astype(int),n-1);xi=2*s-1
    cc=chi.reshape(n,-1);ww=w.reshape(n,-1);xx=xi.reshape(n,-1)
    mean=np.sum(ww*cc,axis=1)*n;slope=3*np.sum(ww*cc*xx,axis=1)*n
    projected=np.repeat(mean,len(y)//n)+np.repeat(slope,len(y)//n)*xi
    a=np.exp(-1/128/100)*G*(derivative(len(phase)-1,y)@incoming)
    aa=a.reshape(n,-1);ma=np.sum(ww*aa,axis=1)*n;sa=3*np.sum(ww*aa*xx,axis=1)*n
    pa=np.repeat(ma,len(y)//n)+np.repeat(sa,len(y)//n)*xi
    mean_tau=float(w@tau);predicted=mean_tau+(a-pa)-r.Z*(chi-projected)
    if not exact:assert max(abs(predicted-tau))<1e-3,('projection identity',n,max(abs(predicted-tau)))
    # Pull physical Q1 gradient back to the identical common phase test space.
    back=pullback_1d(r.y,len(phase)-1)
    return r,dict(cells=n,order=order,exact_bulk=exact,I=float(I),Z=float(r.Z),V=float(info['V']),
                  mean_tau=mean_tau,bulk=float(terms[0]),cohesive=float(terms[1]),fracture=float(terms[2]),
                  projection_identity_max_Pa=float(max(abs(predicted-tau))) if not exact else None,
                  incoming_projection_remainder_max_Pa=float(max(abs(a-pa))),
                  chi_projection_remainder_L2=float(np.sqrt(w@((chi-projected)**2))),
                  common_gradient=(back.T@(A*gradient)).tolist(),force=float(info['force']),contact=float(info['stick_slip']))


def pullback_1d(y,n):
    k=np.minimum((y*n).astype(int),n-1);s=y*n-k
    P=np.zeros((len(y),n+1));P[np.arange(len(y)),k]=1-s;P[np.arange(len(y)),k+1]=s
    return P


def gradient(path):
    g=csv(path/'gradient.csv');n=32;P=pullback_1d(g['y'],n)
    # X mesh is identical on every level; retain all three x test functions.
    out={}
    for name in ['elastic','FE_fracture','CPDI_fracture']:
        value=np.zeros((n+1,3))
        for j in range(3):value[:,j]=P.T@(g[name]*(np.rint(2*g['x']).astype(int)==j))
        out[name]=value
    out['total']=out['elastic']+out['FE_fracture']
    return out


def fields(path):
    f=csv(path/'field.csv');f=np.sort(f,order='id')
    assert len(f)==18432 and np.array_equal(f['id'],np.arange(18432))
    assert max(abs(f['Txy']-f['Zchi']-f['xy']))<1e-6
    return f


def native_metrics(path,common,continuum,write_moments=True):
    f=fields(path);g=gradient(path);e=csv(path/'energy.csv');sf=csv(path/'surface.csv')
    n=round(1/f['dy'][0]);r,ref=reference(n,48,common)
    I=float(np.mean(sf['I']));Z=float(np.mean(sf['Z']));V=float(np.mean(sf['V']))
    txy=G*derivative(n,f['y'])@r.r-Z*h(np.interp(f['y'],r.y,r.phi))/ref['I']
    d=np.stack([f['xx'],f['yy'],f['xy']-continuum['mean_tau']],axis=-1)
    w=f['weight'];chi=f['chi'];C=w@chi
    total=dict(cells=n,I=I,Z=Z,V=V,whole_full_tensor_L2_Pa=float(np.sqrt(w@norm2(d))),
               localized_full_tensor_RMS_Pa=float(np.sqrt((w*chi)@norm2(d)/C)),
               whole_shear_RMS_Pa=float(np.sqrt(w@(f['xy']-continuum['mean_tau'])**2)),
               localized_shear_RMS_Pa=float(np.sqrt((w*chi)@(f['xy']-continuum['mean_tau'])**2/C)),
               stress_mean_Pa=float(w@f['xy']),stress_min_Pa=float(min(f['xy'])),stress_max_Pa=float(max(f['xy'])),
               tensor_L2_Pa=float(np.sqrt(w@norm2(np.stack([f['Txx'],f['Tyy'],f['Txy']],axis=-1)))),
               Zchi_L2_Pa=float(np.sqrt(w@(f['Zchi']**2))),native_vs_1D_stress_max_Pa=float(max(abs(f['xy']-txy))),
               native_vs_1D_Z_Pa_m=abs(Z-ref['Z']),native_vs_1D_V=abs(V-ref['V']),
               energy={k:float(e[k]) for k in e.dtype.names},reference=ref,
               common_gradient_error_Linf=float(np.max(abs(g['total'].sum(axis=1)-np.array(continuum['common_gradient'])))),
               native_vs_1D_gradient_Linf=float(np.max(abs(g['total'].sum(axis=1)-np.array(ref['common_gradient'])))),
               native_vs_1D_bulk_energy=abs(float(e['bulk'])-ref['bulk']),
               fracture_difference=float(e['CPDI_fracture']-e['FE_fracture']) if e['CPDI_available'] else None,
               fracture_gradient_difference_Linf=float(np.max(abs(g['CPDI_fracture']-g['FE_fracture']))) if e['CPDI_available'] else None)
    total['legacy_moments']=moments(f['xy'],w,chi,continuum['mean_tau'])
    assert abs(total['legacy_moments']['whole_error_RMS_Pa']-total['whole_shear_RMS_Pa'])<1e-8
    assert total['native_vs_1D_stress_max_Pa']/1000<1e-6,total
    assert total['native_vs_1D_gradient_Linf']/max(1.,np.max(np.abs(ref['common_gradient'])))<1e-6,total
    traces=csv(path/'traces.csv');jumps=[]
    for x in np.unique(traces['x']):
        t=traces[traces['x']==x]
        for y in np.unique(t['y']):
            if y==0 or y==1:continue
            pair=t[t['y']==y];assert len(pair)==2
            left=pair[pair['side']==1][0];right=pair[pair['side']==0][0]
            jumps.append((x,y,right['Txy']-left['Txy'],right['Zchi']-left['Zchi'],right['tau_xy']-left['tau_xy']))
    total['interface_jump_max_Pa']=max(abs(a[-1]) for a in jumps)
    total['Zchi_jump_max_Pa']=max(abs(a[-2]) for a in jumps)
    if write_moments:
        with (path/'cell-moments.csv').open('w') as out:
            out.write('cell,a,b,Txy,Zchi,tau_xy\n')
            for cell in np.unique(f['cell']):
                a=f[f['cell']==cell];xr=2*(a['x']-a['x0'])/a['dx']-1;yr=2*(a['y']-a['y0'])/a['dy']-1
                for i in range(3):
                    for j in range(3):
                        p=np.polynomial.legendre.legval(xr,np.eye(3)[i])*np.polynomial.legendre.legval(yr,np.eye(3)[j]);ww=a['weight']*p/(a['dx'][0]*a['dy'][0])
                        values=[float(ww@a[k]) for k in ['Txy','Zchi','xy']]
                        out.write(','.join(map(str,[int(cell),i,j]+values))+'\n')
        np.savetxt(path/'interface-jumps.csv',jumps,delimiter=',',header='x,y,jump_Txy,jump_Zchi,jump_tau_xy',comments='')
    if write_moments:
        m=csv(path/'cell-moments.csv');means=m[(m['a']==0)&(m['b']==0)]
        second=m[(m['a']==0)&(m['b']==2)]
        total['physical_cell_mean_range_Pa']=float(np.ptp(means['tau_xy']))
        total['cancellation_moment_identity_max_Pa']=float(max(abs(m['Txy']-m['Zchi']-m['tau_xy'])))
        total['normal_P2_Txy_moment_max_Pa']=float(max(abs(second['Txy'])))
        total['normal_P2_tau_xy_moment_max_Pa']=float(max(abs(second['tau_xy'])))
    defect=csv(path/'residual.csv')
    total['mechanical_residual']={k:float(defect[k]) for k in defect.dtype.names}
    return total


def compare(a,b,scale=1000.):
    fa,fb=fields(a),fields(b);assert max(abs(fa['x']-fb['x']))<1e-14 and max(abs(fa['y']-fb['y']))<1e-14
    result={k:float(max(abs(fa[k]-fb[k]))) for k in ['phi','chi','Txx','Tyy','Txy','Zchi','xx','yy','xy']}
    ga,gb=gradient(a),gradient(b)
    result['gradient_Linf']=float(np.max(abs(ga['total']-gb['total'])))
    result['gradient_scaled']=result['gradient_Linf']/max(1.,np.max(abs(gb['total'])))
    ea,eb=csv(a/'energy.csv'),csv(b/'energy.csv')
    result['energy']=max(abs(float(ea[k]-eb[k]))/max(1.,abs(float(eb[k]))) for k in ['bulk','cohesive','FE_fracture'])
    result['physical_stress_scaled']=max(result[k] for k in ['xx','yy','xy'])/scale
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',nargs='+',type=Path);p.add_argument('--common',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    common=np.load(a.common);exact,continuum=reference(32,48,common,True)
    exact_hi,hi=reference(32,96,common,True)
    assert abs(hi['bulk']-continuum['bulk'])<1e-9
    out=dict(continuum=continuum,reference_quadrature={k:abs(hi[k]-continuum[k]) for k in ['I','Z','mean_tau','bulk','cohesive','fracture']},runs=[],pairs={})
    lookup={}
    for path in a.runs:
        record=json.loads((path/'record.json').read_text());assert record['gate_passed']
        case=record['spatial_case'];gates=(path/'run/gates.csv').read_text().splitlines();assert all(x.endswith(',1') for x in gates)
        entry=dict(label=path.name,seconds=record['seconds'],case=case,resolved=native_metrics(path/'run/resolved',common,continuum))
        f=fields(path/'run/identical');entry['identical_cancellation_condition']=(np.sqrt(f['weight']@f['Txy']**2)+np.sqrt(f['weight']@f['Zchi']**2))/np.sqrt(f['weight']@f['xy']**2)
        out['runs'].append(entry);lookup[(case['normal_cells'],case['backend'],case['quadrature'])]=path
    # A second, independent per-cell quadrature comparison on the frozen solve.
    out['mechanical_reference_quadrature_24_48']={}
    for n in sorted({key[0] for key in lookup}):
        lo,lr=reference(n,24,common);hi,hr=reference(n,48,common)
        delta={k:abs(lr[k]-hr[k]) for k in ['I','Z','mean_tau','bulk','cohesive','fracture']}
        delta['common_gradient_Linf']=float(np.max(abs(np.array(lr['common_gradient'])-np.array(hr['common_gradient']))))
        assert delta['bulk']<1e-8 and delta['common_gradient_Linf']<1e-7,delta
        out['mechanical_reference_quadrature_24_48'][str(n)]=delta
    baseline=next(p for key,p in lookup.items() if key==(32,'DG',12))
    for key,path in lookup.items():
        out['pairs']['identical_'+path.name]=compare(baseline/'run/identical',path/'run/identical')
        assert max(out['pairs']['identical_'+path.name][k] for k in ['physical_stress_scaled','gradient_scaled','energy'])<1e-9
        if (path/'run/entry_identical').exists():
            out['pairs']['entry_identical_'+path.name]=compare(baseline/'run/entry_identical',path/'run/entry_identical')
            assert max(out['pairs']['entry_identical_'+path.name][k] for k in ['physical_stress_scaled','gradient_scaled','energy'])<1e-9
        if key[2]==24:
            q12=lookup[(key[0],key[1],12)]
            out['pairs']['quadrature_'+path.name]=compare(q12/'run/resolved',path/'run/resolved')
        if key[1]=='FE':
            dg=lookup[(key[0],'DG',key[2])]
            out['pairs']['backend_'+str(key)]=compare(dg/'run/resolved',path/'run/resolved')
            assert max(out['pairs']['backend_'+str(key)][k] for k in ['physical_stress_scaled','gradient_scaled','energy'])<1e-9
    out['passed']=True;a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
