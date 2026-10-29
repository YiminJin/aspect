#!/usr/bin/env python3
"""Independent Q2/Jacobian history, source, stress and lifecycle checks."""
import argparse,json
from pathlib import Path
import numpy as np
from assess_native import csv,gauss,norm3,G,KF,ELL,M
from assess_orientation import geometry,rule


def basis(points,nodes):
    def factor(t,k,derivative=False):
        if derivative:
            return np.where(k==0,4*t-3,np.where(k==.5,4-8*t,4*t-1))
        return np.where(k==0,2*t*t-3*t+1,np.where(k==.5,4*t-4*t*t,2*t*t-t))
    u=points[:,0,None];v=points[:,1,None];a=nodes[None,:,0];b=nodes[None,:,1]
    N=factor(u,a)*factor(v,b)
    dN=np.stack([factor(u,a,True)*factor(v,b),factor(u,a)*factor(v,b,True)],axis=-1)
    return N,dN


def physical_gradient(vertices,points,values,nodes):
    u,v=points.T
    d1=np.stack([np.stack([-(1-v),1-v,-v,v],axis=-1),np.stack([-(1-u),-u,1-u,u],axis=-1)],axis=-1)
    J=np.einsum('qia,cib->cqba',d1,vertices)
    _,dN=basis(points,nodes)
    gradN=np.einsum('qia,cqab->cqib',dN,np.linalg.inv(J))
    return np.einsum('cqib,cid->cqdb',gradN,values),np.linalg.det(J)


def strain_tensor(gradient):
    trace=gradient[:,:,0,0]+gradient[:,:,1,1]
    return np.stack([gradient[:,:,0,0]-trace/3,gradient[:,:,1,1]-trace/3,
                     .5*(gradient[:,:,0,1]+gradient[:,:,1,0])],axis=-1)


def coefficients(table):
    return {name:np.stack([table[c] for c in columns],axis=-1).reshape(-1,9,len(columns)) for name,columns in
            [('A',['Ax','Ay']),('DG',['DGxx','DGyy','DGxy']),('P',['Pxx','Pyy','Pxy'])]}


def assess(path,recovery_path=None):
    ledger=csv(path/'bulk-accepted.csv');metrics=csv(path/'fixture.csv');gates=csv(path/'gates.csv')
    assert np.all(gates['passed']==1)
    assert np.array_equal(ledger['publications'],np.arange(1,len(ledger)+1))
    assert max(ledger['projection_error'])<=1e-12 and max(ledger['mechanical_residual'])<=1e-9 and max(ledger['phase_KKT'])<=1e-7
    recovery=['converged trial does not publish','outer rejection restores native solution and all bulk histories',
              'identical retry native solution','repeated rejection leaves accepted histories unchanged']
    recovery_gates=csv((recovery_path or path)/'gates.csv')
    assert np.all(recovery_gates['passed']==1)
    for gate in recovery:assert np.sum(recovery_gates['gate']==gate)==1
    mesh,nodes,vertices,indices=geometry(path,path/'mesh-qualification.csv')
    mesh_index={cell:i for i,cell in enumerate(mesh['cell'])}
    original=nodes['phi'];errors=dict(point_tensor=0.,DG_source=0.,native_projection=0.,physical_stress=0.,FE_recurrence=0.,FE_roundoff_ratio=0.,
        bulk_energy=0.,cohesive_energy=0.,fracture_energy=0.,cohesive_relation=0.,friction_interval=0.,driving=0.)
    peaks=dict(remainder=0.,mixed_Q2_coefficient=0.,normal_stress=0.,initial_hold_phase_increment=0.)
    previous=None;Fprevious=None
    z,w=gauss(3);u,v=np.meshgrid(z,z);source_points=np.stack([u.ravel(),v.ravel()],axis=-1);weights=(w[:,None]*w).ravel()
    for row in metrics:
        step=int(row['step']);label=f'{step:04d}'
        table=csv(path/f'cell-history-{label}.csv');point=csv(path/f'point-history-{label}.csv')
        assert np.all(table['cell'].reshape(-1,9)==table['cell'][::9,None])
        cell_order=[mesh_index[c] for c in table['cell'][::9]];verts=vertices[cell_order]
        c=coefficients(table);node_points=np.stack([table['u'][:9],table['v'][:9]],axis=-1)
        sample_points=np.stack([point['u'][:7],point['v'][:7]],axis=-1)
        N,_=basis(sample_points,node_points);grad,_=physical_gradient(verts,sample_points,c['A'],node_points)
        expected=np.einsum('qi,cid->cqd',N,c['DG']-c['P'])+2*G*strain_tensor(grad)
        actual=np.stack([point[k] for k in ['Txx','Tyy','Txy']],axis=-1).reshape(-1,7,3)
        error=np.sqrt(norm3(expected-actual))/np.maximum(1000.,np.sqrt(norm3(actual)))
        errors['point_tensor']=max(errors['point_tensor'],float(error.max()))
        gradient,det=physical_gradient(verts,source_points,c['A'],node_points)
        N3,_=basis(source_points,node_points);weighted=det*weights
        mass=np.einsum('qi,qj,cq->cij',N3,N3,weighted)
        rhs=np.einsum('qi,cqd,cq->cid',N3,2*G*strain_tensor(gradient),weighted)
        projected=np.linalg.solve(mass,rhs)
        error=np.sqrt(norm3(projected-c['P']))/np.maximum(1000.,np.sqrt(norm3(c['P'])))
        errors['native_projection']=max(errors['native_projection'],float(error.max()))
        F=np.fromfile(path/f'native-history-{label}.bin');solution=np.fromfile(path/f'solution-{label}.bin')
        if step:
            beta=1/(1+row['dt']*.01);velocity=solution[:len(F)]
            target=beta*(Fprevious+row['dt']*velocity);difference=abs(F-target)
            scale=np.maximum(abs(Fprevious)+abs(row['dt']*velocity),1e-300)
            errors['FE_recurrence']=max(errors['FE_recurrence'],float(difference.max()))
            errors['FE_roundoff_ratio']=max(errors['FE_roundoff_ratio'],float(np.max(difference/(np.finfo(float).eps*scale))))
            # Separate check of the unchanged nine-mode source3 DG update.
            du=c['A']/beta-previous['A'];gradient,det=physical_gradient(verts,source_points,du,node_points)
            N3,_=basis(source_points,node_points)
            target=beta*(np.einsum('qi,cid->cqd',N3,previous['DG'])+2*G*strain_tensor(gradient))
            weighted=det*weights
            mass=np.einsum('qi,qj,cq->cij',N3,N3,weighted)
            rhs=np.einsum('qi,cqd,cq->cid',N3,target,weighted)
            projected=np.linalg.solve(mass,rhs)
            error=np.sqrt(norm3(projected-c['DG']))/np.maximum(1000.,np.sqrt(norm3(c['DG'])))
            errors['DG_source']=max(errors['DG_source'],float(error.max()))
        previous=c;Fprevious=F
        # Polynomial coefficient u^2 v^2, all nine modes retained.
        u0,v0=node_points.T;V=np.stack([np.ones(9),u0,v0,u0*u0,u0*v0,v0*v0,u0*u0*v0,u0*v0*v0,u0*u0*v0*v0],axis=-1)
        modes=np.linalg.solve(V,c['DG'].transpose(1,0,2).reshape(9,-1)).reshape(9,-1,3)
        peaks['mixed_Q2_coefficient']=max(peaks['mixed_Q2_coefficient'],float(abs(modes[-1]).max()))
        current=csv(path/f'phase-{label}.csv');current.sort(order='dof')
        if row['time']<=.05+1e-12:peaks['initial_hold_phase_increment']=max(peaks['initial_hold_phase_increment'],float(max(abs(current['phi']-original))))
        dense=path/f'bulk-{label}.csv'
        if not dense.exists():continue
        d=csv(dense)
        T=np.stack([d[k] for k in ['Txx','Tyy','Txy']],axis=-1).astype(float);tau=T.copy();tau[:,2]-=G*d['a']
        exported=np.stack([d[k] for k in ['tauxx','tauyy','tauxy']],axis=-1).astype(float)
        error=np.sqrt(norm3(exported-tau))/np.maximum(1000.,np.sqrt(norm3(tau)))
        errors['physical_stress']=max(errors['physical_stress'],float(error.max()))
        h=M*d['phi']*(1+d['phi'])/(1-d['phi'])**2;hp=M*(1+3*d['phi'])/(1-d['phi'])**3
        bulk=norm3(tau)/(4*G);cohesive=h*d['q']**2/(2*G)
        errors['cohesive_relation']=max(errors['cohesive_relation'],float(max(abs(G*d['a']-h*d['q'])))/1000)
        errors['driving']=max(errors['driving'],float(max(abs(d['driving']-hp*d['q']**2/(2*G))))/KF)
        errors['friction_interval']=max(errors['friction_interval'],float(max(np.maximum(abs(tau[:,2]-d['q'])-1000,0)))/1000)
        R=np.stack([d[k] for k in ['Rxx','Ryy','Rxy']],axis=-1)
        peaks['remainder']=max(peaks['remainder'],float(np.sqrt(norm3(R)).max()))
        peaks['normal_stress']=max(peaks['normal_stress'],float(max(abs(T[:,0]).max(),abs(T[:,1]).max())))
        if step:
            l=ledger[ledger['step']==step][0]
            errors['bulk_energy']=max(errors['bulk_energy'],abs(float(np.sum(d['JxW']*bulk))-l['bulk']))
            errors['cohesive_energy']=max(errors['cohesive_energy'],abs(float(np.sum(d['JxW']*cohesive))-l['cohesive']))
            _,weight,N,gradN=rule(vertices,8);p=current['phi'][indices]
            phi=np.einsum('qi,ci->cq',N,p);grad=np.einsum('cqib,ci->cqb',gradN,p)
            fracture=KF*np.sum(weight*(phi+ELL**2*np.sum(grad*grad,axis=-1)))
            errors['fracture_energy']=max(errors['fracture_energy'],abs(fracture-l['fracture']))
    assert max(errors[k] for k in ['point_tensor','DG_source','native_projection','physical_stress'])<1e-10
    assert errors['FE_roundoff_ratio']<=8
    assert max(errors[k] for k in ['bulk_energy','cohesive_energy','fracture_energy','cohesive_relation','driving','friction_interval'])<1e-10
    tangent=csv(path/'bulk-tangent.csv');assert max(tangent['maximum_FD_error'])<1e-6
    deliberate=csv(path/'deliberate-source-consistency-failure.csv');assert deliberate['error'][0]>deliberate['bound'][0]
    return dict(errors=errors,peaks=peaks,maximum_native_source_error=float(max(ledger['projection_error'])),
        maximum_mechanical_residual=float(max(ledger['mechanical_residual'])),maximum_phase_KKT=float(max(ledger['phase_KKT'])),
        maximum_tangent_FD_error=float(max(tangent['maximum_FD_error'])),accepted_publications=len(ledger),
        deliberate_inconsistency={k:(str(deliberate[k][0]) if deliberate[k].dtype.kind in 'US' else float(deliberate[k][0])) for k in deliberate.dtype.names},
        recovery_gates=recovery,recovery_run=str(recovery_path or path),strict_reference_match_passed=False)


def restart(primary,before,after):
    full=csv(primary/'bulk-accepted.csv');prefix=csv(before/'bulk-accepted.csv');tail=csv(after/'bulk-accepted.csv')
    assert np.array_equal(np.r_[prefix,tail],full)
    count=0
    for path in [before,after]:
        for file in path.iterdir():
            if not file.is_file() or file.suffix not in ['.csv','.bin']:continue
            if not file.stem.rsplit('-',1)[-1].isdigit():continue
            assert (primary/file.name).read_bytes()==file.read_bytes(),file
            count+=1
    return dict(ledgers_bitwise_equal=True,number_of_binary_and_field_files_bitwise_equal=count,
                prefix_steps=len(prefix),resumed_steps=len(tail))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--before',type=Path);p.add_argument('--after',type=Path);a=p.parse_args()
    result=assess(a.run,a.after)
    if a.before:result['restart']=restart(a.run,a.before,a.after)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
