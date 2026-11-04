#!/usr/bin/env python3
"""Independent coupled endpoint/refinement and material-energy reductions.

Reconstructs native Q2 tensors/velocities and Q1 phase from diagnostic exports.
No constitutive implementation is called. Energy is the mature physical bulk
branch only: auxiliary Z has no cohesive energy. A continuum balance residual
is measured, not defined away as a fitted flux or numerical dissipation.
"""
import argparse
import hashlib
import json
import os
import re
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','DEAL_II_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from assess_material_native import read, RUNS, ROOT, norm3, basis
from isolate_profile_projection import ClosedState, grid, q1, compact
from isolate_tensor_defect import MemoryProjection

OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/bounded_profile'

def dot3(a,b):
    return a[:,0]*b[:,0]+a[:,1]*b[:,1]+(a[:,0]+a[:,1])*(b[:,0]+b[:,1])+2*a[:,2]*b[:,2]

def quadrature(n,order=8):
    g,w=np.polynomial.legendre.leggauss(order);h=4/n
    x=(-2+h*(np.arange(n)[:,None]+.5*(g+1))).reshape(-1)
    w=np.tile(.5*h*w,n)
    return np.stack(np.meshgrid(x,x,indexing='ij'),axis=-1).reshape(-1,2),np.outer(w,w).ravel()

def tensor(state,x):
    v=state.at(x)
    return v[:,3:6]-v[:,6:9]-state.memory(x,v[:,0])

def work_traction(state):
    """Independent common-volume weak shear load, projected with its own mass."""
    if hasattr(state,'traction_cache'):return state.traction_cache
    x,w=quadrature(state.n);v=state.at(x);tau=tensor(state,x)
    X=state.surface[:,1:3];width=state.surface[:,7];I=state.surface[:,5]
    owner=np.full(len(x),-1);xi_values=np.zeros(len(x));shear=np.zeros(len(x));chi=np.zeros(len(x));best=np.full(len(x),np.inf)
    full_chi=np.zeros(len(x));full_best=best.copy()
    for j in range(len(X)-1):
        d=X[j+1]-X[j];t=d/np.linalg.norm(d);n=np.array([-t[1],t[0]])
        q=((x-X[j])@d)/(d@d);dist=abs((x-X[j])@n)
        full=(q>=0)&(q<=1)&(dist<full_best);u=q[full];phi=v[full,0]
        full_chi[full]=state.m*phi*(1+phi)/(1-phi)**2/((1-u)*I[j]+u*I[j+1]);full_best[full]=dist[full]
        select=(q>=0)&(q<=1)&(dist<best)&(dist<=(1-q)*width[j]+q*width[j+1])
        u=q[select];phi=v[select,0];a=tau[select]
        chi[select]=state.m*phi*(1+phi)/(1-phi)**2/((1-u)*I[j]+u*I[j+1])
        shear[select]=t[0]*(a[:,0]*n[0]+a[:,2]*n[1])+t[1]*(a[:,2]*n[0]+a[:,1]*n[1])
        owner[select]=j;xi_values[select]=u;best[select]=dist[select]
    mask=owner>=0;j=owner[mask];q=xi_values[mask];weight=w[mask]*chi[mask];a=shear[mask];size=len(X)
    add=lambda z:np.bincount(j,weights=z,minlength=size-1)
    diagonal=np.zeros(size);diagonal[:-1]+=add(weight*(1-q)**2);diagonal[1:]+=add(weight*q*q)
    off=add(weight*q*(1-q));M=np.diag(diagonal)+np.diag(off,1)+np.diag(off,-1)
    rhs=np.zeros(size);rhs[:-1]+=add(weight*(1-q)*a);rhs[1:]+=add(weight*q*a)
    state.traction_cache=(np.linalg.solve(M,rhs),M)
    measure=float(w@chi);unclipped=float(w@full_chi);length=float(np.linalg.norm(np.diff(X,axis=0),axis=1).sum())
    state.normalization_diagnostics=dict(localization_measure=measure,unclipped_localization_measure=unclipped,
        curve_length=length,normalization_integral_relative_error=measure/length-1,
        normal_support_fraction_lost=(unclipped-measure)/unclipped)
    return state.traction_cache

def energy(state,G,eta_expression,order=8):
    assert eta_expression in ['1','1+.5*tanh(4*x)'],eta_expression
    x,w=quadrature(state.n,order);v=state.at(x);tau=tensor(state,x)
    tr=v[:,12]+v[:,15]
    D=np.stack([v[:,12]-tr/3,v[:,15]-tr/3,.5*(v[:,13]+v[:,14])],axis=1)
    L=.5*state.memory(x,v[:,0],6)
    eta=10*(1+.5*np.tanh(4*x[:,0])) if eta_expression!='1' else 10*np.ones(len(x))
    psi=dot3(tau,tau)/(4*G)
    result=dict(storage=float(w@psi),mechanical_power=float(w@dot3(tau,D)),
                slip_power=float(w@dot3(tau,L)),maxwell_loss=float(w@(dot3(tau,tau)/(2*eta))),
                dilatation_power=float(w@(psi*tr)))
    g,w1=np.polynomial.legendre.leggauss(order);h=state.h
    line=(-2+h*(np.arange(state.n)[:,None]+.5*(g+1))).ravel();weights=np.tile(.5*h*w1,state.n)
    flux=0.;incoming=0.;outgoing=0.
    for axis in [0,1]:
        for side in [-1,1]:
            points=np.zeros((len(line),2));points[:,axis]=2*side;points[:,1-axis]=line
            a=state.at(points);vn=side*a[:,axis+1];p=dot3(tensor(state,points),tensor(state,points))/(4*G)
            flux+=weights@(vn*p);incoming+=weights@(np.minimum(vn,0)*p);outgoing+=weights@(np.maximum(vn,0)*p)
    result.update(flux=float(flux),incoming_flux=float(incoming),outgoing_flux=float(outgoing))
    return result

def replay(old,new,dt,method):
    nodes=grid(old.n).reshape(-1,2);v=old.remote_at(nodes)[:,1:3];foot=nodes-dt*v
    forward=old.remote_at(foot)[:,0];expected=forward.copy()
    lo=np.zeros(len(nodes));hi=np.zeros(len(nodes));count=np.zeros(len(nodes),int)
    base=np.floor((foot+2)/old.h).astype(int)
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            ij=base+[dx,dy];u=(foot+2)/old.h-ij
            valid=np.all((ij>=0)&(ij<old.n)&(u>=-1e-6)&(u<=1+1e-6),axis=1)
            i,j=ij[valid].T
            vertices=np.array([[-1,-1],[1,-1],[-1,1],[1,1]])
            values=np.einsum('vk,nkf->nvf',basis(vertices),old.coeff[i,j])[:,:,0]
            lower=values.min(axis=1);upper=values.max(axis=1)
            lo[valid]=np.where(count[valid]==0,lower,np.minimum(lo[valid],lower))
            hi[valid]=np.maximum(hi[valid],upper);count[valid]+=1
    if method=='bounded MacCormack':
        # Reconstruct Q1 predictor in a copy of the old polynomial grid, so the
        # return lookup has the native tolerance/averaging convention as well.
        import copy
        pred=copy.copy(old);pred.coeff=old.coeff.copy()
        f=forward.reshape(old.n+1,old.n+1)
        a,b,c,d=f[:-1,:-1],f[1:,:-1],f[:-1,1:],f[1:,1:]
        pred.coeff[:,:,:,0]=0
        pred.coeff[:,:,0,0]=(a+b+c+d)/4
        pred.coeff[:,:,3,0]=(-a+b-c+d)/4
        pred.coeff[:,:,1,0]=(-a-b+c+d)/4
        pred.coeff[:,:,4,0]=(a-b-c+d)/4
        reverse=pred.remote_at(nodes+dt*v)[:,0]
        covered=np.all((nodes+dt*v>=-2-1e-6*old.h)&(nodes+dt*v<=2+1e-6*old.h),axis=1)
        expected=np.clip(np.where(covered,forward+.5*(old.at(nodes)[:,0]-reverse),forward),lo,hi)
    actual=new.at(nodes)[:,0]
    return dict(profile_replay_Linf=float(np.max(abs(actual-expected))),
                donor_bound_violation=float(max(0.,np.max(lo-actual),np.max(actual-hi))),
                min_phi=float(actual.min()),max_phi=float(actual.max()))

def load(name,final_only=False):
    folder=RUNS/name;command=json.loads((folder/'command.json').read_text());assert command['exit_code']==0,name
    raw=read(folder/'output/material-summary.csv')
    rows=[r for r in raw if int(r['step'])==0 or float(r['dt'])>64*np.finfo(float).eps*max(abs(float(r['time'])),abs(command['arguments']['dt']))]
    command['analysis_excluded_terminal_steps']=[r for r in raw if r not in rows]
    meta={k:float(v) for k,v in read(folder/'output/material-model.csv')[0].items()}
    selected=rows[-1:] if final_only else rows
    states={int(r['step']):ClosedState(folder/'output',int(r['step']),command['arguments']['level'],meta['m']) for r in selected}
    return command,rows,meta,states

def reduce(name):
    command,rows,meta,states=load(name);steps=[];previous=None
    cumulative_slip=np.zeros(len(states[min(states)].surface))
    keys=['mechanical_power','slip_power','maxwell_loss','dilatation_power','flux','incoming_flux','outgoing_flux']
    endpoint_integrals={k:0. for k in keys};trapezoid_integrals=endpoint_integrals.copy()
    endpoint_sum=trap_sum=endpoint_abs=trap_abs=scale=0.
    for row in rows:
        step=int(row['step']);state=states[step];dt=float(row['dt']);e=energy(state,meta['G'],command['arguments']['eta'])
        e.update(step=step,time=float(row['time']),dt=dt)
        if previous is not None:
            cumulative_slip+=dt*state.surface[:,6]
            p=previous
            for k in keys:
                endpoint_integrals[k]+=dt*e[k]
                trapezoid_integrals[k]+=.5*dt*(e[k]+p[k])
            rhs=lambda z:z['mechanical_power']-z['slip_power']-z['maxwell_loss']+z['dilatation_power']-z['flux']
            endpoint=e['storage']-p['storage']-dt*rhs(e)
            trap=e['storage']-p['storage']-.5*dt*(rhs(e)+rhs(p))
            endpoint_sum+=endpoint;trap_sum+=trap;endpoint_abs+=abs(endpoint);trap_abs+=abs(trap)
            scale+=.5*dt*sum(abs(z[k]) for z in [p,e] for k in ['mechanical_power','slip_power','maxwell_loss','dilatation_power','flux'])
            e.update(endpoint_remainder=endpoint,trapezoid_remainder=trap,
                     **replay(states[step-1],state,dt,command['arguments']['profile_method']))
            old=states[step-1];V=state.surface[:,6];eta=state.surface[:,8]
            age=old.surface[:,3]*np.exp(-V*dt/meta['Dc'])-meta['Dc']/V*np.expm1(-V*dt/meta['Dc'])
            Z=old.surface[:,4]*np.exp(-meta['G']*dt/eta)-eta*np.expm1(-meta['G']*dt/eta)*V
            e['age_source_Linf']=float(abs(age-state.surface[:,3]).max())
            e['memory_source_Linf']=float(abs(Z-state.surface[:,4]).max())
            e['crossed_labels']=int(np.any(np.floor((old.surface[:,1:3]+2)/state.h)!=np.floor((state.surface[:,1:3]+2)/state.h),axis=1).sum())
            assert e['profile_replay_Linf']<2e-11,(name,step,e)
            assert e['donor_bound_violation']<2e-11,(name,step,e)
            assert e['age_source_Linf']<1e-11 and e['memory_source_Linf']<1e-11,(name,step,e)
        steps.append(e);previous=e
    final=states[max(states)];x,w=quadrature(final.n);phi=final.at(x)[:,0];mass=w@phi;center=np.sum((w*phi)[:,None]*x,axis=0)/mass
    variance=np.sum((w*phi)[:,None]*(x-center)**2,axis=0)/mass
    tangent=final.surface[-1,1:3]-final.surface[0,1:3];tangent/=np.linalg.norm(tangent)
    normal=np.array([-tangent[1],tangent[0]])
    normal_variance=float((w*phi)@((x-center)@normal)**2/mass)
    initial=states[min(states)];initial_mass=float(w@initial.at(x)[:,0])
    delta=MemoryProjection(final).at(x)-final.memory(x,phi)
    tau=tensor(final,x);delta2=float(w@dot3(delta,delta))
    projected_storage_difference=float(w@(-2*dot3(tau,delta)+dot3(delta,delta))/(4*meta['G']))
    projection_identity=projected_storage_difference+delta2/(4*meta['G'])
    assert abs(projection_identity)<1e-15,(name,projection_identity)
    work_traction(final)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [RUNS/name/'command.json',*(RUNS/name/'output').glob('material-*.csv')]}
    log=(RUNS/name/'run.log').read_text();iterations=[int(v) for v in re.findall(r'Fault linear solve: iterations=(\d+)',log)]
    return dict(case=name,arguments=command['arguments'],wall_s=command['wall_s'],steps=steps,
                excluded_terminal_steps=command['analysis_excluded_terminal_steps'],
                linear_solves=len(iterations),linear_iterations=sum(iterations),max_linear_iterations=max(iterations,default=0),
                profile_mass=float(mass),initial_mass=initial_mass,centroid=center.tolist(),variance=variance.tolist(),
                normal_variance=normal_variance,peak=float(final.at(grid(final.n).reshape(-1,2))[:,0].max()),
                endpoint_cumulative_slip=cumulative_slip.tolist(),
                localization_projection_L2=float(np.sqrt(delta2)),
                projected_minus_physical_storage=projected_storage_difference,
                projection_orthogonality_defect=projection_identity,
                **final.normalization_diagnostics,
                storage_change=steps[-1]['storage']-steps[0]['storage'],
                endpoint_integrals=endpoint_integrals,trapezoid_integrals=trapezoid_integrals,
                endpoint_signed=endpoint_sum,endpoint_absolute=endpoint_abs,trapezoid_signed=trap_sum,trapezoid_absolute=trap_abs,
                power_scale=scale,endpoint_relative=endpoint_abs/max(scale,1e-300),trapezoid_relative=trap_abs/max(scale,1e-300),
                input_sha256=hashes),final

def compare(state,ref):
    # Common quadrature on the finer nested mesh. Also compare surface labels,
    # normalization and direct point traction, independently of Newton residual.
    x,w=quadrature(max(state.n,ref.n),4);a=state.at(x);b=ref.at(x)
    tau=tensor(state,x);reference=tensor(ref,x)
    relative=lambda err,base:float(np.sqrt(w@(err*err)/max(w@(base*base),1e-300)))
    result=dict(profile_relative_L2=relative(a[:,0]-b[:,0],b[:,0]),
                velocity_relative_L2=relative(np.linalg.norm(a[:,1:3]-b[:,1:3],axis=1),np.linalg.norm(b[:,1:3],axis=1)),
                stress_relative_L2=relative(norm3(tau-reference),norm3(reference)))
    for col,label in [(3,'age'),(4,'memory'),(5,'normalization'),(6,'slip_rate')]:
        result[label+'_relative_L2']=float(np.linalg.norm(state.surface[:,col]-ref.surface[:,col])/max(np.linalg.norm(ref.surface[:,col]),1e-300))
    result['geometry_Linf']=float(np.linalg.norm(state.surface[:,1:3]-ref.surface[:,1:3],axis=1).max())
    def traction(s):
        X=s.surface[:,1:3];d=np.diff(X,axis=0);length=np.linalg.norm(d,axis=1);t=d/length[:,None]
        g,weights=np.polynomial.legendre.leggauss(3);q=.5*(g+1)
        points=(X[:-1,None,:]+q[None,:,None]*d[:,None,:]).reshape(-1,2)
        t=np.repeat(t,3,axis=0);n=np.stack([-t[:,1],t[:,0]],axis=1);tau=tensor(s,points)
        shear=t[:,0]*(tau[:,0]*n[:,0]+tau[:,2]*n[:,1])+t[:,1]*(tau[:,2]*n[:,0]+tau[:,1]*n[:,1])
        return shear,(.5*length[:,None]*weights).ravel()
    tr,tw=traction(ref);ts,_=traction(state)
    result['point_traction_relative_L2']=float(np.sqrt(tw@((ts-tr)**2)/max(tw@(tr*tr),1e-300)))
    a,_=work_traction(state);b,M=work_traction(ref);d=a-b
    result['work_traction_relative_L2']=float(np.sqrt(d@M@d/max(b@M@b,1e-300)))
    # Retain the original vertex diagnostic, including its endpoint sampling
    # sensitivity, so the refined quadrature metric does not hide that evidence.
    def vertices(s):
        X=s.surface[:,1:3];t=np.gradient(X,axis=0);t/=np.linalg.norm(t,axis=1)[:,None];n=np.stack([-t[:,1],t[:,0]],axis=1)
        a=tensor(s,X);return t[:,0]*(a[:,0]*n[:,0]+a[:,2]*n[:,1])+t[:,1]*(a[:,2]*n[:,0]+a[:,1]*n[:,1])
    b=vertices(ref);result['vertex_traction_relative_l2']=float(np.linalg.norm(vertices(state)-b)/max(np.linalg.norm(b),1e-300))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cases',nargs='+');p.add_argument('--output',default='comparison.json');p.add_argument('--reference',action='append',default=[],help='eta-key:direction:case, e.g. constant:positive:bp-c-p-m-ref')
    options=p.parse_args();results=[];states={};refs={};ref_names={}
    for specification in options.reference:
        eta,direction,name=specification.split(':');refs[(eta,direction)]=load(name,final_only=True)[3];ref_names[(eta,direction)]=name
    for name in options.cases:
        result,state=reduce(name);states[name]=state
        key=('constant' if result['arguments']['eta']=='1' else 'layered','positive' if result['arguments']['speed']>0 else 'negative')
        if key in refs:
            result['fine_reference']=compare(state,refs[key][max(refs[key])]);result['reference_case']=ref_names[key]
        results.append(result)
        print(name, json.dumps({k:result[k] for k in ['wall_s','endpoint_relative','trapezoid_relative']}),flush=True)
    by_case={r['case']:r for r in results}
    for r in results:
        if r.get('reference_case') in by_case:
            ref=by_case[r['reference_case']];a=np.array(r['endpoint_cumulative_slip']);b=np.array(ref['endpoint_cumulative_slip'])
            r['fine_reference']['cumulative_slip_relative_L2']=float(np.linalg.norm(a-b)/np.linalg.norm(b))
            r['fine_reference']['final_storage_relative']=abs(r['steps'][-1]['storage']-ref['steps'][-1]['storage'])/ref['steps'][-1]['storage']
            flux=r['trapezoid_integrals']['flux']-ref['trapezoid_integrals']['flux']
            r['fine_reference']['flux_integral_error']=flux
            r['fine_reference']['flux_error_over_active_scale']=abs(flux)/ref['power_scale']
    OUT.mkdir(exist_ok=True,parents=True);(OUT/options.output).write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
