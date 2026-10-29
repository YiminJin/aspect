#!/usr/bin/env python3
"""Original-QP, common-physical-kernel diagnosis; never updates a history."""
import argparse,json
from pathlib import Path
import numpy as np
from assess_native import csv,gauss,KF,ELL,G,M,SEED_R,SEED_P

BASE=Path('build-particle-free/bulk-orientation')
REFERENCE=Path('build-particle-free/bulk-sensitivity/runs/right-distance')


def geometry(path,mesh_path,phase_step=0):
    mesh=csv(mesh_path);nodes=csv(path/f'phase-{phase_step:04d}.csv');nodes.sort(order='dof')
    lookup={(x,y):i for i,(x,y) in enumerate(zip(nodes['x'],nodes['y']))}
    vertices=np.stack([np.stack([mesh[f'x{i}'],mesh[f'y{i}']],axis=-1) for i in range(4)],axis=1)
    indices=np.array([[lookup[tuple(v)] for v in cell] for cell in vertices])
    return mesh,nodes,vertices,indices


def rule(vertices,order):
    z,w=gauss(order);u,v=np.meshgrid(z,z,indexing='xy');u=u.ravel();v=v.ravel();weights=(w[:,None]*w).ravel()
    N=np.stack([(1-u)*(1-v),u*(1-v),(1-u)*v,u*v],axis=-1)
    dN=np.stack([np.stack([-(1-v),1-v,-v,v],axis=-1),np.stack([-(1-u),-u,1-u,u],axis=-1)],axis=-1)
    points=np.einsum('qi,cid->cqd',N,vertices)
    J=np.einsum('qia,cib->cqba',dN,vertices);det=np.linalg.det(J)
    assert np.all(det>0)
    gradients=np.einsum('qia,cqab->cqib',dN,np.linalg.inv(J))
    return points,det*weights,N,gradients


def state(path,mesh_path,step):
    mesh,nodes,vertices,indices=geometry(path,mesh_path,step)
    points,weight,N,gradN=rule(vertices,8)
    phase=nodes['phi'][indices]
    phi=np.einsum('qi,ci->cq',N,phase);gradient=np.einsum('cqib,ci->cqb',gradN,phase)
    # Persistent local histories use numeric CellId ordering, not cell iteration.
    raw=np.fromfile(path/f'state-{step:04d}.bin');metrics=csv(path/'fixture.csv')
    ndg=int(metrics['DG_DoFs'][0]);nq=weight.size
    order=np.argsort([int(c.split('_')[0]) for c in mesh['cell']]);inverse=np.argsort(order)
    local=raw[ndg:ndg+5*nq].reshape(len(mesh),64,5)[inverse]
    assert np.array_equal(raw[ndg+5*nq:-3],nodes['phi'])
    assert raw[-3]==step and raw[-1]==step
    q=local[:,:,3];Y=M*(1+3*phi)/(1-phi)**3*q*q/(2*G)
    # Check decoding against an actual dense export, including all local modes.
    dense=csv(path/('bulk-0000.csv' if path!=REFERENCE else 'bulk-0052.csv'))
    dense_indices={cell:np.flatnonzero(dense['cell']==cell) for cell in np.unique(dense['cell'])}
    geo_error=max(float(np.max(abs(points[i]-np.stack([dense['x'][dense_indices[cell]],dense['y'][dense_indices[cell]]],axis=-1)))) for i,cell in enumerate(mesh['cell']))
    assert geo_error<1e-13
    if step>0:
        force=csv(path/f'phase-forces-{step:04d}.csv');force.sort(order='dof')
        gf=np.zeros(len(nodes));gm=np.zeros(len(nodes))
        for i in range(4):
            np.add.at(gf,indices[:,i],np.sum(weight*KF*(N[None,:,i]+2*ELL**2*np.sum(gradient*gradN[:,:,i],axis=-1)),axis=1))
            np.add.at(gm,indices[:,i],np.sum(-weight*N[None,:,i]*Y,axis=1))
        weak_error=max(float(np.max(abs(gf-force['fracture_gradient']))),float(np.max(abs(gm-force['mechanical_gradient']))))
        weak_normalized=max(float(np.max(abs(gf-force['fracture_gradient'])/force['mass'])),
                            float(np.max(abs(gm-force['mechanical_gradient'])/force['mass'])))/KF
        assert weak_normalized<1e-10
    else:weak_error=weak_normalized=0.
    return dict(points=points.reshape(-1,2),weight=weight.ravel(),phi=phi.ravel(),gradient=gradient.reshape(-1,2),q=q.ravel(),Y=Y.ravel(),
                vertices=vertices,indices=indices,phase_nodes=nodes['phi'],node_xy=np.stack([nodes['x'],nodes['y']],axis=-1),
                geometry_error=geo_error,weak_gradient_error=weak_error,weak_gradient_normalized_error=weak_normalized,time=float(raw[-2]),step=step)


def kernel(s,centers,radius=.0625):
    result=[]
    for z in centers:
        d=s['points']-z;r2=np.sum(d*d,axis=-1)/radius**2;inside=r2<1
        r2=r2[inside];d=d[inside];w=s['weight'][inside]
        K=(1-r2)**3;gradK=-6*d/radius**2*(1-r2[:,None])**2
        mass=np.sum(w*K)
        F=KF*np.sum(w*(K+2*ELL**2*np.sum(s['gradient'][inside]*gradK,axis=-1)))/mass
        values=[np.sum(w*K*s[k][inside])/mass for k in ['phi','q','Y']]
        result.append([*z,*values,F,F-values[-1],mass])
    return np.array(result)


def initial_error(s):
    points,w,N,_=rule(s['vertices'],24)
    p=np.einsum('qi,ci->cq',N,s['phase_nodes'][s['indices']])
    exact=np.interp(np.hypot(np.maximum(points[:,:,0]-.5,0),points[:,:,1]-.5),SEED_R,SEED_P,right=0.)
    cap=(points[:,:,0]>.5)&(np.linalg.norm(points-.5,axis=-1)<.35)
    return dict(whole_L2=float(np.sqrt(np.sum(w*(p-exact)**2)/np.sum(w*exact**2))),
                cap_L2=float(np.sqrt(np.sum(w[cap]*(p[cap]-exact[cap])**2)/np.sum(w[cap]*exact[cap]**2))))


def phase_only_kernel(s,centers,order):
    points,w,N,gradN=rule(s['vertices'],order)
    p=s['phase_nodes'][s['indices']]
    phi=np.einsum('qi,ci->cq',N,p);gradient=np.einsum('cqib,ci->cqb',gradN,p)
    # Only the Q1 field/fracture test is reintegrated. No history or return map
    # is evaluated at these extra quadrature points.
    data=dict(points=points.reshape(-1,2),weight=w.ravel(),phi=phi.ravel(),gradient=gradient.reshape(-1,2),
              q=np.zeros(w.size),Y=np.zeros(w.size))
    return kernel(data,centers)


def independent_source(vertices):
    # Tensor-product quadratic Lagrange basis; this independent ordering differs
    # from FE_Q's vertex/edge/interior ordering. All nine basis modes are tested.
    def data(n):
        z,w=gauss(n);u,v=np.meshgrid(z,z,indexing='xy');u=u.ravel();v=v.ravel()
        def poly(t):return np.stack([2*t*t-3*t+1,4*t-4*t*t,2*t*t-t],axis=-1)
        def deriv(t):return np.stack([4*t-3,4-8*t,4*t-1],axis=-1)
        P=np.einsum('qi,qj->qji',poly(u),poly(v)).reshape(-1,9)
        du=np.einsum('qi,qj->qji',deriv(u),poly(v)).reshape(-1,9)
        dv=np.einsum('qi,qj->qji',poly(u),deriv(v)).reshape(-1,9)
        _,weight,_,_=rule(vertices[None],n)
        dN=np.stack([np.stack([-(1-v),1-v,-v,v],axis=-1),np.stack([-(1-u),-u,1-u,u],axis=-1)],axis=-1)
        J=np.einsum('qia,ib->qba',dN,vertices)
        gradients=np.einsum('qia,qab->qib',np.stack([du,dv],axis=-1),np.linalg.inv(J))
        return P,gradients,weight[0]
    P3,g3,w3=data(3);P8,g8,w8=data(8)
    def coefficients(P,g,w):
        rhs=np.einsum('qi,qjd,q->ijd',P,g,w)
        return np.linalg.solve(P.T@(w[:,None]*P),rhs.reshape(9,-1)).reshape(9,9,2)
    c3=coefficients(P3,g3,w3);c8=coefficients(P8,g8,w8)
    error=float(np.max(abs(np.einsum('qi,ijd->qjd',P8,c3)-g8)))
    return dict(source3_error=error,source8_error=float(np.max(abs(np.einsum('qi,ijd->qjd',P8,c8)-g8))),
                coefficient3_8_difference=float(np.max(abs(c3-c8))),gradient_max=float(np.max(abs(g8))),
                relative_error=error/float(np.max(abs(g8))))


def native_checks(path,initial):
    ledger=csv(path/'bulk-accepted.csv');fields=csv(path/'bulk-0040.csv');Txy=0.
    for row in ledger[ledger['step']<=40]:
        rate=.018 if .05+1e-12<row['time']<=.15+1e-12 else 0.
        Txy=(Txy+G*rate*row['dt'])/(1+row['dt']*.01)
    error=np.sqrt(fields['Txx']**2+fields['Tyy']**2+(fields['Txx']+fields['Tyy'])**2+2*(fields['Txy']-Txy)**2)
    fixed=True
    for file in path.glob('phase-[0-9][0-9][0-9][0-9].csv'):
        nodes=csv(file);nodes.sort(order='dof');fixed=fixed and np.array_equal(nodes['phi'],initial['phase_nodes'])
    gates=csv(path/'gates.csv');last=int(ledger['step'][-1])
    result=dict(affine_control_step=40,analytic_Txy=Txy,max_full_tensor_error=float(error.max()),
        analytic_bulk_energy=1.5*Txy*Txy/(2*G),native_bulk_energy=float(ledger['bulk'][ledger['step']==40][0]),
        all_accepted_phase_bitwise_unchanged=bool(fixed),min_phase=float(initial['phase_nodes'].min()),max_phase=float(initial['phase_nodes'].max()),
        all_diagnostic_gates_passed=bool(np.all(gates['passed']==1)),accepted_steps=len(ledger),publications=int(ledger['publications'][-1]),
        maximum_accepted_projection_error=float(ledger['projection_error'].max()),maximum_accepted_mechanical_residual=float(ledger['mechanical_residual'].max()),
        maximum_accepted_phase_KKT=float(ledger['phase_KKT'].max()),failed_step_not_published=not (path/f'state-{last+1:04d}.bin').exists())
    assert fixed and result['all_diagnostic_gates_passed'] and result['failed_step_not_published']
    assert np.array_equal(ledger['publications'],np.arange(1,len(ledger)+1))
    assert result['maximum_accepted_projection_error']<=1e-12 and result['maximum_accepted_mechanical_residual']<=1e-9 and result['maximum_accepted_phase_KKT']<=1e-7
    _,w,N,_=rule(initial['vertices'],8);mass=np.zeros(len(initial['phase_nodes']))
    for i in range(4):np.add.at(mass,initial['indices'][:,i],np.sum(w*N[None,:,i],axis=1))
    f=csv(path/f'phase-forces-{last:04d}.csv');f.sort(order='dof')
    result['native_KKT_mass_diagnostic']=dict(minimum_integral_N_to_native_lumped_mass=float(np.min(mass/f['mass'])),maximum_integral_N_to_native_lumped_mass=float(np.max(mass/f['mass'])))
    return result


def main(out):
    out.mkdir(exist_ok=True)
    run=BASE/'runs/rotated';metrics=csv(run/'fixture.csv');last=int(metrics['step'][-1])
    meshes=dict(reference=BASE/'runs/identity/mesh-qualification.csv',rotated=run/'mesh-qualification.csv')
    paths=dict(reference=REFERENCE,rotated=run);summary={'last_common_step':last,'last_common_time':float(metrics['time'][-1]),'cases':{}}
    angles=np.linspace(-90,90,181);radii=[.225,.2881107642904027,.29851520585223124,.325]
    arc_centers=np.concatenate([.5+r*np.stack([np.cos(np.deg2rad(angles)),np.sin(np.deg2rad(angles))],axis=-1) for r in radii])
    for name in paths:
        initial=state(paths[name],meshes[name],0);final=state(paths[name],meshes[name],last)
        if name=='rotated':summary['native_checks']=native_checks(paths[name],initial)
        error=initial_error(initial);record={'initial_representation':error,'independent_geometry_error':final['geometry_error'],
            'independent_weak_gradient_error':final['weak_gradient_error'],'independent_weak_gradient_normalized_error':final['weak_gradient_normalized_error']}
        sample_angles=np.deg2rad([-75,-60,-45,-30,-15,0,15,30,45,60,75])
        sample_centers=np.concatenate([.5+r*np.stack([np.cos(sample_angles),np.sin(sample_angles)],axis=-1) for r in radii])
        k8=kernel(initial,sample_centers);k16=phase_only_kernel(initial,sample_centers,16)
        record['initial_kernel_8_16_check']=dict(phase_max_error=float(np.max(abs(k8[:,2]-k16[:,2]))),
            fracture_max_error=float(np.max(abs(k8[:,5]-k16[:,5]))))
        delta=final['phase_nodes']-initial['phase_nodes'];r=np.linalg.norm(final['node_xy']-.5,axis=-1)
        cap=(final['node_xy'][:,0]>.5)&(r<.35)
        record.update(max_cap_increment=float(np.max(delta[cap])),max_increment=float(np.max(delta)))
        growing=np.flatnonzero(cap&(delta>1e-12));record['growing_cap_nodes']=[dict(x=float(final['node_xy'][i,0]),y=float(final['node_xy'][i,1]),delta=float(delta[i])) for i in growing]
        summary['cases'][name]=record
        arrays={}
        for label,s in [('initial',initial),('common',final)]:
            a=kernel(s,arc_centers)
            a=np.column_stack([np.repeat(radii,len(angles)),np.tile(angles,len(radii)),a])
            np.savetxt(out/f'{name}-{label}-arcs.csv',a,delimiter=',',header='radius,angle,x,y,phi,q,Y,fracture,total,kernel_mass',comments='')
            arrays.update({f'{label}_{k}':v for k,v in s.items() if isinstance(v,np.ndarray)})
        np.savez_compressed(out/f'{name}-fields.npz',**arrays)
    m=csv(meshes['rotated']);worst=int(np.argmax(m['source3_error']))
    vertices=np.array([[m[f'x{i}'][worst],m[f'y{i}'][worst]] for i in range(4)])
    summary['independent_worst_source']=dict(cell=m['cell'][worst],native_source3_error=float(m['source3_error'][worst]),**independent_source(vertices))
    assert abs(summary['independent_worst_source']['source3_error']-float(m['source3_error'][worst]))<1e-8
    affine=int(np.argmin(m['affine_defect']))
    vertices=np.array([[m[f'x{i}'][affine],m[f'y{i}'][affine]] for i in range(4)])
    summary['independent_affine_source']=independent_source(vertices)
    assert summary['independent_affine_source']['relative_error']<1e-12
    summary['common_kernel_initial_comparison']=[]
    ref=np.genfromtxt(out/'reference-initial-arcs.csv',delimiter=',',names=True)
    rot=np.genfromtxt(out/'rotated-initial-arcs.csv',delimiter=',',names=True)
    for r in radii:
        a=ref[ref['radius']==r];b=rot[rot['radius']==r]
        summary['common_kernel_initial_comparison'].append(dict(radius=r,
            reference_F_min=float(min(a['fracture'])),reference_F_max=float(max(a['fracture'])),
            rotated_F_min=float(min(b['fracture'])),rotated_F_max=float(max(b['fracture'])),
            maximum_F_difference=float(max(abs(a['fracture']-b['fracture']))),
            maximum_phase_difference=float(max(abs(a['phi']-b['phi'])))))
    log=Path('benchmarks/reconstructed_fault/refactoring_boundary/evidence/bulk-orientation-native.log').read_text()
    summary.update(native_source_guard_failed='Bulk native DG source projection fails its independent QP check.' in log,
                   strict_reference_match_passed=False,orientation_onset_qualified=False)
    assert summary['native_source_guard_failed']
    (out/'assessment.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);main(p.parse_args().output)
