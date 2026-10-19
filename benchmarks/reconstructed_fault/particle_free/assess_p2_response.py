#!/usr/bin/env python3
"""Measure the bounded P2 native response on common physical fields/functionals.

Reuses PF6a reference geometry, tensor interpolation, native-Q2 differentiation
and work ledger. No branch alignment and no native history modification.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from assess_pf6a import Ref, grids, evaluate_grid, strains, inner, norm2, ledger, state
from assess_phase_energy import csv, gauss, q1, lagrange2
from pf6a_spatial_reference import Endpoint, G, A, ELL, h, PROP, SEED, evaluate_reference
from assess_pf6a_reference import pullback, seed_errors
from assess_pf6a_reference import load as load_reference


def write(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def fracture_operator(ref):
    """Raw Q1 cell derivative, independent of native phase matrices."""
    z,w=gauss(2);r,s=np.repeat(z,2),np.tile(z,2)
    weights=np.repeat(w,2)*np.tile(w,2)
    dx=np.array([-(1-s),1-s,-s,s]).T
    dy=np.array([-(1-r),-r,1-r,r]).T
    rows,cols,values=[],[],[]
    linear=np.zeros(ref.n)
    for ids,cell in zip(ref.indices,ref.cells):
        area=cell['dx']*cell['dy']
        local=2*ref.A*ref.ell**2*area*(dx.T@(weights[:,None]*dx)/cell['dx']**2+dy.T@(weights[:,None]*dy)/cell['dy']**2)
        np.add.at(linear,ids,ref.A*area/4)
        rows.extend(np.repeat(ids,4));cols.extend(np.tile(ids,4));values.extend(local.ravel())
    return linear,sparse.csr_matrix((values,(rows,cols)),shape=(ref.n,ref.n))


def static_work(ref,old,velocity,pressure,rates,dt,order=24):
    """Exact fixed-phase Maxwell path using only scalar time moments.

    Same full native bulk/surface quadrature, pressure and boundary lift as the
    existing PF6a ledger. This avoids recomputing unchanged I/DI at every time
    sample. Independently checked against that ledger before use.
    """
    rule=csv(ref.path/'rule.csv');r,s=rule['r'],rule['s']
    E,div=strains(ref,velocity,r,s)
    boundary=velocity.copy()
    for c,cell in enumerate(ref.cells):
        for i in range(3):
            for j in range(3):
                x=cell['x0']+.5*i*cell['dx'];y=cell['y0']+.5*j*cell['dy']
                if y not in (0.,1.):
                    boundary[c,i,j,0]=0.
                    if x not in (0.,1.):boundary[c,i,j,1]=0.
    ED,divD=strains(ref,boundary,r,s)
    E,ED=E.reshape(-1,3),ED.reshape(-1,3)
    div,divD=div.ravel(),divD.ravel()
    pressure=evaluate_grid(pressure,r,s).ravel()
    ix,N,_,w,sf,T=ref.bulk_rule()
    assert np.array_equal(ref.nodes['phase'],old.nodes['phase'])
    p=ref.nodes['phase'];phi=np.sum(p[ix]*N,axis=1)
    I=ref.surface['I_accepted'];chi=ref.m*phi*(1+phi)/(1-phi)**2/(sf@I)
    Z0=old.surface['Z'];Z1=ref.surface['Z']
    tau0=old.bulk_rule()[-1].copy();tau0[:,2]-=(sf@Z0)*chi
    tau1=T.copy();tau1[:,2]-=(sf@Z1)*chi
    Q=2*E;Q[:,2]-=(sf@rates)*chi
    z,wt=gauss(order);wt*=dt
    beta=np.exp(-dt*z/100.);kappa=-1e8*np.expm1(-dt*z/100.)
    b,k=wt@beta,wt@kappa
    bb,bk,kk=wt@(beta*beta),wt@(beta*kappa),wt@(kappa*kappa)
    integral=b*tau0+k*Q
    ws=ref.profiles['weight'];fs=ref.sf@I;zs0=ref.sf@Z0;zs1=ref.sf@Z1;vs=ref.sf@rates
    storage=(w@(norm2(tau1)-norm2(tau0)))/(4*G)+ws@((zs1*zs1-zs0*zs0)/fs)/(2*G)
    terms=dict(boundary_work=float(w@(inner(integral,ED)-dt*pressure*divD)),
        free_mechanical_work=float(w@(inner(integral,E-ED)-dt*pressure*(div-divD))),
        pressure_work=float(dt*(w@(pressure*div))),
        bulk_viscous=float(w@(bb*norm2(tau0)+2*bk*inner(tau0,Q)+kk*norm2(Q))/(2e8)),
        cohesive_viscous=float(ws@((bb*zs0**2+2*bk*zs0*vs+kk*vs**2)/fs)/1e8),
        friction=float(dt*(ws@(1000*vs))),
        contact_residual_work=float(ws@((b*zs0+k*vs)*vs/fs)+dt*(ws@(1000*vs))-w@(chi*(sf@rates)*integral[:,2])),
        phase_work=0.)
    remainder=storage-terms['boundary_work']+terms['bulk_viscous']+terms['cohesive_viscous']+terms['friction']
    identity=remainder-sum(terms[k] for k in ['free_mechanical_work','pressure_work','contact_residual_work','phase_work'])
    return dict(storage_change=float(storage),external_remainder=float(remainder),identity_remainder=float(identity),**terms)


def extract(run,output,full_work_checks=False):
    output.mkdir(parents=True,exist_ok=False)
    record=json.loads((run/'record.json').read_text());assert record['gate_passed']
    folder=run/'run';metrics=np.atleast_1d(csv(folder/'metrics.csv'))
    base=Ref(folder/'state-0');geometry=base.bulk_rule()
    rule=csv(base.path/'rule.csv');r,s=rule['r'],rule['s']
    linear,F=fracture_operator(base)
    nx=round(1/base.cells['dx'][0]);ny=round(1/base.cells['dy'][0])
    node_grid=np.empty((ny+1,nx+1),dtype=int)
    node_grid[np.rint(base.nodes['y']*ny).astype(int),np.rint(base.nodes['x']*nx).astype(int)]=np.arange(base.n)
    cell_grid=np.empty((ny,nx),dtype=int)
    cell_grid[np.rint(base.cells['y0']*ny).astype(int),np.rint(base.cells['x0']*nx).astype(int)]=np.arange(len(base.cells))
    P=pullback(base.nodes['y'])
    result=dict(label=run.name,metrics=[],work=[],cross_checks=[],normalization_checks=[])
    fields={k:[] for k in ['phase','tensor','Z','I','V','slip','driving']}
    old=None
    for m in metrics:
        step=int(m['step']);path=folder/f'state-{step}'
        ref=copy.copy(base);ref.path=path
        ref.nodes=csv(path/'nodes.csv');ref.tensor=csv(path/'tensor.csv');ref.surface=csv(path/'surface.csv')
        tensor=grids(ref.tensor,['xx','yy','xy'],len(ref.cells))
        T=evaluate_grid(tensor,r,s).reshape(-1,3)
        ref.cache={None:(*geometry[:-1],T)}
        phase=ref.nodes['phase'];surf=csv(path/'accepted-surface.csv')
        gradient=csv(path/'gradient.csv')['gradient']
        gf=linear+F@phase
        driving=np.stack([(gradient-gf)@P,gf@P])
        for k,v in dict(phase=phase,tensor=tensor,Z=surf['Z'],I=surf['I'],V=surf['V'],slip=surf['slip'],driving=driving).items():fields[k].append(v.copy())
        row={k:float(m[k]) for k in metrics.dtype.names}
        row['phase_x_spread']=float(np.max(np.ptp(phase[node_grid],axis=1)))
        row['phase_centroid']=float(np.trapezoid(np.linspace(0,1,ny+1)*phase[node_grid].mean(axis=1),dx=1/ny)/np.trapezoid(phase[node_grid].mean(axis=1),dx=1/ny))
        row['cumulative_growth']=float(max(phase-fields['phase'][0]))
        row['q_mean']=float(np.mean(surf['Z']/surf['I']))
        row['slip_mean']=float(np.mean(surf['slip']))
        row['V_mean']=float(np.mean(surf['V']))
        row['I_mean']=float(np.mean(surf['I']))
        row['total_energy']=row['bulk']+row['cohesive']+row['fracture']
        result['metrics'].append(row)
        if old is not None:
            velocity=grids(csv(path/'velocity.csv'),['ux','uy'],len(ref.cells))
            pressure=grids(csv(path/'velocity.csv'),['p'],len(ref.cells))
            dt=float(m['dt'])
            unchanged=np.array_equal(phase,old.nodes['phase'])
            if unchanged:
                work=static_work(ref,old,velocity,pressure,surf['V'],dt)
                # One nonzero-history sticking/sliding interval checks the
                # scalar moment formula independently against the existing path.
                if full_work_checks and step==64:
                    check=ledger(ref,old,velocity,pressure,surf['V'],dt,24)
                    error=max(abs(work[k]-check[k]) for k in work)
                    result['cross_checks'].append(dict(step=step,static_vs_full_ledger=error))
                    assert error<1e-7
            else:
                work=ledger(ref,old,velocity,pressure,surf['V'],dt,12)
                # Compare one strongly growing interval's time quadrature.
                if full_work_checks and abs(m['time']-.5625)<1e-14:
                    high=ledger(ref,old,velocity,pressure,surf['V'],dt,24)
                    result['cross_checks'].append(dict(step=step,time_quadrature_uncertainty=max(abs(work[k]-high[k]) for k in work)))
            assert abs(work['identity_remainder'])<1e-7,(step,work)
            work.update(step=step,time=float(m['time']),unchanged_phase=unchanged)
            result['work'].append(work)
        # Independent original high-order normal integration and bulk-rule
        # refinement only on startup, pre-onset, growth-end and hold-end.
        if full_work_checks and (step==0 or m['time'] in [.5,.5625,.59375]):
            e=ref.evaluate(phase);high=ref.evaluate(phase,24);higher=ref.evaluate(phase,32)
            result['normalization_checks'].append(dict(step=step,
                normalization=float(max(abs(e['I']-surf['I']))/max(1.,max(abs(surf['I'])))),
                energy=float(abs(e['energy']-row['total_energy'])/max(1.,abs(e['energy']))),
                quadrature_energy=float(abs(e['energy']-higher['energy'])/max(1.,abs(higher['energy']))),
                quadrature_load=float(max(abs(e['loads']-higher['loads']))/max(1.,max(abs(higher['loads'])))),
                quadrature_uncertainty=float(abs(high['energy']-higher['energy'])),
                gradient=float(max(abs(e['gradient']-gradient))/max(1.,max(abs(gradient))))))
            assert max(result['normalization_checks'][-1][k] for k in ['normalization','energy','quadrature_energy','quadrature_load','gradient'])<1e-9
        old=ref
        if step%16==0:print(run.name,step,flush=True)
    for k in fields:fields[k]=np.array(fields[k])
    np.savez_compressed(output/'fields.npz',**fields,cell_grid=cell_grid,node_grid=node_grid,indices=base.indices,
        cells=base.cells,nodes=base.nodes,time=metrics['time'],dt=metrics['dt'])
    result['onset']=next((row['time'] for row in result['metrics'] if row['cumulative_growth']>1e-5),None)
    result['phase_growth']=max(row['cumulative_growth'] for row in result['metrics'])
    result['max_x_spread']=max(row['phase_x_spread'] for row in result['metrics'])
    result['native_gates']=len(np.atleast_1d(csv(folder/'gates.csv')))
    assert all(csv(folder/'gates.csv')['passed'])
    write(output/'assessment.json',result)
    return result


class PhysicalCase:
    """Common physical sampling; native tensors retain all exported DG modes."""
    def __init__(self,path,native):
        self.native=native;self.path=path
        if native:
            self.f=dict(np.load(path/'fields.npz'))
            self.report=json.loads((path/'assessment.json').read_text())
            self.rows=self.report['metrics']
            self.q=self.f['Z']/self.f['I'];self.V=self.f['V'];self.slip=self.f['slip']
            self.energy=np.array([r['total_energy'] for r in self.rows])
            self.driving=self.f['driving']
            self.onset=self.report['onset']
        else:
            self.ref=load_reference(path);self.f=self.ref['fields'];self.rows=self.ref['history']
            initial=Endpoint(len(self.f['y'])-1);I0=initial.direct(self.f['phase'][0])[0]
            self.I=np.r_[I0,[r['I'] for r in self.rows]]
            self.q=np.r_[0.,[r['q'] for r in self.rows]][:,None]
            self.V=np.r_[0.,[r['V'] for r in self.rows]][:,None]
            self.slip=self.f['slip'][:,None]
            p=self.f['phase'][0];dy=np.diff(self.f['y'])
            E0=A*np.sum(dy*(p[:-1]+p[1:])/2)+A*ELL**2*np.sum(np.diff(p)**2/dy)
            self.energy=np.r_[E0,[r['bulk']+r['cohesive']+r['fracture'] for r in self.rows]]
            d=self.f['driving']@pullback(self.f['y'])
            self.driving=np.stack([d[:,0]+d[:,1],d[:,2]],axis=1)
            self.onset=self.ref['summary']['onset']
        self.t=self.f['time']

    def sample(self,k,x,y):
        if not self.native:
            f=self.f;p=np.interp(y,f['y'],f['phase'][k]);chi=h(p)/self.I[k]
            n=len(f['y'])-1;c=np.minimum((y*n).astype(int),n-1)
            grad=np.stack([np.zeros_like(y),np.diff(f['phase'][k])[c]*n],axis=-1)
            nm=self.ref['summary']['mechanical_cells']
            if nm:
                c=np.minimum((y*nm).astype(int),nm-1);xi=2*(y*nm-c)-1
                T=f['coefficients'][k,c,0]+xi*f['coefficients'][k,c,1]
                shear=T-f['Z'][k]*chi
            else:
                shear=np.full_like(y,0. if k==0 else self.rows[k-1]['tau_mean'])
                T=shear+f['Z'][k]*chi
            tau=np.zeros((len(y),3));tau[:,2]=shear
            return p,chi,tau,grad
        f=self.f;ny,nx=f['cell_grid'].shape
        c=f['cell_grid'][np.minimum((y*ny).astype(int),ny-1),np.minimum((x*nx).astype(int),nx-1)]
        r=(x-f['cells']['x0'][c])/f['cells']['dx'][c];s=(y-f['cells']['y0'][c])/f['cells']['dy'][c]
        nodal=f['phase'][k,f['indices'][c]];p=np.sum(nodal*q1(r,s),axis=1)
        grad=np.stack([np.sum(nodal*np.array([-(1-s),1-s,-s,s]).T,axis=1)/f['cells']['dx'][c],
            np.sum(nodal*np.array([-(1-r),-r,1-r,r]).T,axis=1)/f['cells']['dy'][c]],axis=-1)
        T=np.einsum('qi,qj,qijk->qk',lagrange2(r),lagrange2(s),f['tensor'][k,c])
        surface_x=np.linspace(0,1,f['Z'].shape[1]);I=np.interp(x,surface_x,f['I'][k]);Z=np.interp(x,surface_x,f['Z'][k])
        chi=h(p)/I;tau=T.copy();tau[:,2]-=Z*chi
        return p,chi,tau,grad


def physical_comparison(a,b,scales):
    """Unaligned full plane-strain differences, common times and interval rates."""
    z,w=gauss(4);yy=((np.arange(1024)[:,None]+z)/1024).ravel();wy=np.tile(w/1024,1024)
    zx,wx=gauss(3);xx=((np.arange(2)[:,None]+zx)/2).ravel();wx=np.tile(wx/2,2)
    x=np.repeat(xx,len(yy));y=np.tile(yy,len(xx));weights=np.repeat(wx,len(yy))*np.tile(wy,len(xx))
    xv=np.repeat(np.array([0.,.5,1.]),1025);yv=np.tile(np.linspace(0,1,1025),3)
    rows=[];indices=np.searchsorted(b.t,a.t);assert max(abs(b.t[indices]-a.t))<1e-14
    for k,j in enumerate(indices):
        pa,ca,sa,da=a.sample(k,x,y);pb,cb,sb,db=b.sample(j,x,y)
        whole=np.sqrt(weights@norm2(sa-sb));local=np.sqrt((weights*cb)@norm2(sa-sb)/(weights@cb))
        rate_a=a.V[k];rate_b=b.V[j]
        if k:
            start=indices[k-1];dt=a.t[k]-a.t[k-1]
            rate_b=np.sum(b.V[start+1:j+1]*np.diff(b.t[start:j+1])[:,None],axis=0)/dt
        q=float(max(abs(a.q[k]-b.q[j])));slip=float(max(abs(a.slip[k]-b.slip[j])))
        phase_Linf=float(max(abs(a.sample(k,xv,yv)[0]-b.sample(j,xv,yv)[0])))
        row=dict(time=float(a.t[k]),whole_Pa=float(whole),localized_Pa=float(local),
            phase_Linf=phase_Linf,phase_gradient_L2=float(np.sqrt(weights@np.sum((da-db)**2,axis=1))),
            whole_stress=float(whole/scales['stress']),localized_stress=float(local/scales['stress']),
            phase_L2=float(np.sqrt(weights@((pa-pb)**2))/scales['phase']),
            q_Pa=q,q=q/scales['q'],slip_m=slip,slip=slip/scales['slip'],
            V_m_per_s=float(max(abs(rate_a-rate_b))),rate=float(max(abs(rate_a-rate_b)))/scales['rate'],
            energy_J_per_m=float(abs(a.energy[k]-b.energy[j])),energy=float(abs(a.energy[k]-b.energy[j])/scales['energy']),
            driving=float(max(abs(a.driving[k,0]-b.driving[j,0]))/scales['driving']),
            fracture_driving=float(max(abs(a.driving[k,1]-b.driving[j,1]))))
        rows.append(row)
    names=[k for k in rows[0] if k!='time']
    return dict(maximum={n:max(r[n] for r in rows) for n in names},
        worst_time={n:max(rows,key=lambda r:r[n])['time'] for n in names},by_time=rows,
        onset_a=a.onset,onset_b=b.onset,onset_difference=abs(a.onset-b.onset),branch_alignment='none')


def energy_summary(case):
    result={}
    for name,select in [('loading',lambda r:r['time']<=.5625),('hold',lambda r:r['time']>.5625)]:
        work=[r for r in case.report['work'] if select(r)]
        names=['storage_change','boundary_work','pressure_work','bulk_viscous','cohesive_viscous','friction',
            'free_mechanical_work','contact_residual_work','phase_work','external_remainder']
        result[name]=dict(signed_sum={n:sum(r[n] for r in work) for n in names},
            absolute_interval_sum={n:sum(abs(r[n]) for r in work) for n in names},
            maximum_external_remainder=max(abs(r['external_remainder']) for r in work),
            maximum_external_time=max(work,key=lambda r:abs(r['external_remainder']))['time'],
            maximum_identity_remainder=max(abs(r['identity_remainder']) for r in work))
        scale=max(max(case.energy),result[name]['absolute_interval_sum']['boundary_work'])
        result[name]['scale_J_per_m']=float(scale)
        result[name]['external_fraction']=result[name]['absolute_interval_sum']['external_remainder']/scale
    return result


def hold_summary(case):
    f=case.f;k=int(np.flatnonzero(case.t==.5625)[0]);beta=np.exp(-(.59375-.5625)/100)
    return dict(beta=beta,phase_Linf=float(max(abs(f['phase'][-1]-f['phase'][k]))),
        Z_Maxwell_error=float(max(abs(f['Z'][-1]-beta*f['Z'][k]))),
        T_Maxwell_error=float(np.max(abs(f['tensor'][-1]-beta*f['tensor'][k]))),
        T_Maxwell_scaled_error=float(np.max(abs(f['tensor'][-1]-beta*f['tensor'][k]))/max(1.,np.max(abs(f['tensor'][k])))),
        cohesive_traction_ratio=float(np.mean(case.q[-1])/np.mean(case.q[k])),
        slip_increment=float(max(abs(case.slip[-1]-case.slip[k]))),
        maximum_rate=float(np.max(case.V[k+1:])))


def centroid(case):
    if case.native:
        p=case.f['phase'][-1,case.f['node_grid']]
        nx=p.shape[1]-1;mass=np.full(nx+1,1/nx);mass[[0,-1]]*=.5
        p=p@mass;y=np.linspace(0,1,len(p))
    else:p=case.f['phase'][-1];y=case.f['y']
    dy=np.diff(y)
    numerator=np.sum(dy*(y[:-1]*(p[:-1]+p[1:])/2+dy*(p[:-1]/6+p[1:]/3)))
    return float(numerator/np.sum(dy*(p[:-1]+p[1:])/2))


def initialization():
    from scipy.interpolate import PPoly
    f=np.load(PROP/'matched_initialization_evidence/reference.npz')
    poly=PPoly.construct_fast(f['coefficients'],f['knots'],axis=1)
    z,w=gauss(8);y=((np.arange(4096)[:,None]+z)/4096).ravel();w=np.tile(w/4096,4096)
    d=abs(y-.5);cut=float(f['cut']);inside=d<=cut;de=np.empty_like(y)
    de[inside]=poly.derivative()(d[inside]/cut)[0]/cut
    de[~inside]=poly.derivative()((d[~inside]-cut)/(.5-cut))[2]/(.5-cut)
    denominator=float(np.sqrt(w@(de**2)))
    table=np.loadtxt(SEED);assert np.array_equal(evaluate_reference(table[:,0]),table[:,1])
    return dict(exact_gradient_L2=denominator,native_table_equals_continuum=True,
        minimum_phase=float(min(table[:,1])),maximum_phase=float(max(table[:,1])),
        meshes=[dict(r,relative_gradient_L2=r['gradient_L2_error']/denominator) for r in seed_errors() if r['cells'] in [512,1024]])


def compare(root,output):
    cases={n:PhysicalCase(root/n,n.startswith('native')) for n in ['native-primary','native-half','ref512-d128','ref512-d256','ref1024-continuum-d256']}
    ref=cases['ref1024-continuum-d256'];load=ref.t<=.5625
    z,w=gauss(8);y=((np.arange(1024)[:,None]+z)/1024).ravel();w=np.tile(w/1024,1024)
    phase_scale=max(np.sqrt(w@(np.interp(y,ref.f['y'],p)**2)) for p in ref.f['phase'][load])
    scales=dict(stress=float(np.sqrt(2)*max(abs(r['tau_mean']) for r in ref.rows if r['time']<=.5625)),
        phase=float(phase_scale),q=float(np.max(ref.q[load])),slip=float(np.max(ref.slip[load])),rate=float(np.max(ref.V[load])),
        energy=float(max(ref.energy[load])),driving=float(np.max(abs(ref.driving[load,0]))))
    comparisons={}
    for name,a,b in [('temporal_native','native-primary','native-half'),('native_reference_primary','native-primary','ref512-d128'),
        ('native_reference_half','native-half','ref512-d256'),('spatial_reference','ref512-d256','ref1024-continuum-d256'),
        ('temporal_reference','ref512-d128','ref512-d256')]:
        comparisons[name]=physical_comparison(cases[a],cases[b],scales);print(name,comparisons[name]['maximum'],flush=True)
    temporal=dict(whole_stress=.002,localized_stress=.004,phase_L2=.002,phase_Linf=.001,q_Pa=1.,q=.002,
        slip=.001,rate=.005,energy=.002,driving=.004)
    spatial=dict(whole_stress=.01,localized_stress=.02,phase_L2=.01,phase_Linf=.005,q=.01,slip=.01,energy=.01,driving=.02)
    passes={}
    for name,limits in [('temporal_native',temporal),('spatial_reference',spatial),('native_reference_primary',spatial),('native_reference_half',spatial)]:
        passes[name]={k:comparisons[name]['maximum'][k]<=v for k,v in limits.items()}
        passes[name]['onset']=comparisons[name]['onset_difference']<=1/128
    energy={n:energy_summary(cases[n]) for n in ['native-primary','native-half']}
    passes['physical_work']=dict(loading_bound=energy['native-primary']['loading']['external_fraction']<=.005,
        halving_bound=energy['native-half']['loading']['external_fraction']<=.005,
        decreases=energy['native-half']['loading']['absolute_interval_sum']['external_remainder']<energy['native-primary']['loading']['absolute_interval_sum']['external_remainder'])
    result=dict(initialization=initialization(),reference_scales=scales,comparisons=comparisons,energy=energy,
        hold={n:hold_summary(cases[n]) for n in ['native-primary','native-half']},passes=passes,
        final_phase_centroid_y={n:centroid(c) for n,c in cases.items()},
        strict_reference_match_passed=False,all_practical_comparisons_passed=all(all(p.values()) for p in passes.values()))
    write(output,result)
    return result


def snapshots(run,output):
    """Retained independent stress, source and weak-load checks at dense exports."""
    folder=run/'run';metrics=np.atleast_1d(csv(folder/'metrics.csv'));checks=[]
    for t in [.5,.53125,.5625,.59375]:
        row=metrics[np.flatnonzero(metrics['time']==t)[0]];step=int(row['step'])
        result=state(folder/f'state-{step}',folder/f'state-{step-1}',float(row['dt']))
        # Keep the unprojected reaction, and evaluate contact complementarity
        # using the actual native lower rate rather than calling it an error.
        path=folder/f'state-{step}';surface=csv(path/'accepted-surface.csv')
        loads=csv(path/'native-loads.csv');residual=loads['residual']
        pressure=csv(path/'velocity.csv')['p']
        lower=surface['V']<=1.000000000001e-20
        projected=np.where(lower,np.minimum(-residual,0.),-residual)
        result.update(time=t,step=step,raw_contact_load_Linf=float(max(abs(residual))),
            projected_contact_load_Linf=float(max(abs(projected))),
            Z_mean=float(np.mean(surface['Z'])),I_mean=float(np.mean(surface['I'])),
            q_mean=float(np.mean(surface['Z']/surface['I'])),V_mean=float(np.mean(surface['V'])),
            slip_mean=float(np.mean(surface['slip'])),
            pressure_dynamic_min=float(min(pressure)),pressure_dynamic_max=float(max(pressure)),
            weak_loads_constant_test={name:float(sum(loads[name])) for name in ['shear','normal','cohesive','friction']})
        checks.append(result);print(run.name,'independent snapshot',t,flush=True)
    write(output,checks)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='compare':
        p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path)
        a=p.parse_args(sys.argv[2:]);result=compare(a.root,a.output)
        print(json.dumps(result['passes'],indent=2));return
    if len(sys.argv)>1 and sys.argv[1]=='snapshots':
        p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path)
        a=p.parse_args(sys.argv[2:]);snapshots(a.run,a.output);return
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);p.add_argument('--full-work-checks',action='store_true');a=p.parse_args()
    result=extract(a.run,a.output,a.full_work_checks)
    print(json.dumps(dict(label=result['label'],phase_growth=result['phase_growth'],onset=result['onset'],max_x_spread=result['max_x_spread'])))

if __name__=='__main__':main()
