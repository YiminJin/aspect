#!/usr/bin/env python3
"""Read-only native-cell reconstruction of the saved affine loading states.

Retains the DG polynomial and the independently projected native FE complement.
No constitutive history is interpolated or advanced by this diagnostic.
"""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from assess_native import csv,G,gauss,KF,SEED_R
from assess_spatial import layout,q1
from assess_source_history import basis,strain_tensor

RUN=Path('build-particle-free/bulk-continued/runs/continued')
MAP=Path('build-particle-free/bulk-parallel/runs/layout')
STEPS=[120,121,123,124,128,139,140,148]

def load_layout(map_path):
    L=layout(RUN); initial=csv(map_path/'phase-0000.csv');initial.sort(order='dof')
    assert np.array_equal(initial,L['nodes']), 'Fresh native numbering/physical phase differs from saved serial layout'
    # Actual cell/FE shape-number associations from the identical fixed mesh.
    cellnames=np.genfromtxt(RUN/'bulk-0000.csv',delimiter=',',skip_header=1,usecols=0,dtype=str)[::64]
    nm=csv(map_path/'native-cell-dofs.csv');dg=csv(map_path/'dg-cell-dofs.csv')
    nc=len(cellnames); ix={c:i for i,c in enumerate(cellnames)}
    native=np.empty((nc,2,9),dtype=int);native_nodes=np.empty((2,9,2));dgids=np.empty((nc,9),dtype=int);dg_nodes=None
    native_rows={r['cell'][0]:r for r in nm.reshape(nc,18)}
    dg_rows={r['cell'][0]:r for r in dg.reshape(nc,9)}
    for c in cellnames:
        i=ix[c];rows=native_rows[c]
        for d in [0,1]:
            r=rows[rows['component']==d];assert len(r)==9
            native[i,d]=r['dof'];native_nodes[d]=np.stack([r['u'],r['v']],axis=-1)
        r=dg_rows[c];assert len(r)==9;dgids[i]=r['dof'];dg_nodes=np.stack([r['u'],r['v']],axis=-1)
    L.update(cellnames=cellnames,native=native,native_nodes=native_nodes,dgids=dgids,dg_nodes=dg_nodes)
    return L

def strain(L,memory,points):
    # Same harmless constant-translation removal as the native guard. Extended
    # summation reduces differentiation cancellation in accumulated FE memory.
    gradient=np.empty((len(L['cellnames']),len(points),2,2))
    for d in [0,1]:
        values=memory[L['native'][:,d]].astype(np.longdouble)
        values-=values[:,0,None]
        _,dN=basis(points,L['native_nodes'][d])
        for j,h in enumerate([L['dx'],L['dy']]):
            gradient[:,:,d,j]=np.einsum('ci,qi->cq',values,(dN[:,:,j]/h).astype(np.longdouble))
    return strain_tensor(gradient)

def reconstruct(L,step):
    nc=len(L['cellnames']);ndg=nc*9
    raw=np.fromfile(RUN/f'state-{step:04d}.bin');memory=np.fromfile(RUN/f'native-history-{step:04d}.bin')
    Tdg=raw[:3*ndg].reshape(3,ndg)[:,L['dgids']].transpose(1,2,0)
    native_order=np.argsort([int(c.split('_')[0]) for c in L['cellnames']]);inverse=np.argsort(native_order)
    history=raw[3*ndg:3*ndg+nc*64*5].reshape(nc,64,5)[inverse]
    z,w=gauss(3);u,v=np.meshgrid(z,z);source=np.stack([u.ravel(),v.ravel()],axis=-1)
    N3,_=basis(source,L['dg_nodes']);weights=(w[:,None]*w).ravel()*L['dx']*L['dy']
    mass=np.einsum('qi,qj,q->ij',N3,N3,weights)
    projected=np.linalg.solve(mass,np.einsum('qi,cqd,q->icd',N3,2*G*strain(L,memory,source),weights).reshape(9,-1)).reshape(9,nc,3).transpose(1,0,2)
    z,_=gauss(8);u,v=np.meshgrid(z,z);points=np.stack([u.ravel(),v.ravel()],axis=-1);N8,_=basis(points,L['dg_nodes'])
    R=2*G*strain(L,memory,points)-np.einsum('qi,cid->cqd',N8,projected)
    T=np.einsum('qi,cid->cqd',N8,Tdg)+R;tau=T.copy();tau[:,:,2]-=G*history[:,:,0]
    phase=csv(RUN/f'phase-{step:04d}.csv');phase.sort(order='dof');phi=q1(L,phase['phi'],L['points'])[0]
    hp=128.00000000016*(1+3*phi)/(1-phi)**3
    Y=hp*history[:,:,3]**2/(2*G)
    return T,tau,Y,history,R

def kernels(points,weights,centers,radius=.0625):
    tree=cKDTree(points);rows=[];cols=[];values=[]
    for j,idx in enumerate(tree.query_ball_point(centers,radius)):
        idx=np.asarray(idx);r=points[idx]-centers[j]
        w=(1-np.sum(r*r,axis=1)/radius**2)**3*weights[idx];w/=sum(w)
        rows.extend([j]*len(idx));cols.extend(idx);values.extend(w)
    return csr_matrix((values,(rows,cols)),shape=(len(centers),len(points)))

def main(out,map_path):
    out.mkdir(exist_ok=True,parents=True);L=load_layout(map_path);assessment={}
    xx=np.linspace(0,1.5,193);rr=np.linspace(0,.69,121)
    lines=np.concatenate([np.stack([xx,xx*0+.5],axis=1),np.stack([.5+rr/np.sqrt(2),.5+rr/np.sqrt(2)],axis=1),np.stack([.5+rr/np.sqrt(2),.5-rr/np.sqrt(2)],axis=1)])
    K=kernels(L['points'].reshape(-1,2),L['weights'].ravel(),lines)
    nodal_points=L['node_xy'];force=csv(RUN/'phase-forces-0120.csv');force.sort(order='dof')
    Kn=kernels(nodal_points,force['mass'],lines)
    fields={};profiles=[]
    for step in STEPS:
        T,tau,Y,h,R=reconstruct(L,step)
        d=RUN/f'bulk-{step:04d}.csv';check={}
        if d.exists():
            native=csv(d);actual=np.stack([native[k] for k in ['Txx','Tyy','Txy']],axis=-1).reshape(T.shape)
            check['max_full_T_difference_Pa']=float(abs(T-actual).max());assert check['max_full_T_difference_Pa']<1e-8
            actual=np.stack([native[k] for k in ['tauxx','tauyy','tauxy']],axis=-1).reshape(tau.shape)
            check['max_physical_stress_difference_Pa']=float(abs(tau-actual).max());assert check['max_physical_stress_difference_Pa']<1e-8
            check['max_Y_difference_Pa']=float(abs(Y-native['driving'].reshape(Y.shape)).max());assert check['max_Y_difference_Pa']<1e-8
        force=csv(RUN/f'phase-forces-{step:04d}.csv');force.sort(order='dof')
        D=-force['mechanical_gradient']/force['mass'];F=force['fracture_gradient']/force['mass'];balance=D-F
        profiles.append(np.stack([K@tau[:,:,2].ravel(),K@Y.ravel(),Kn@D,Kn@F,Kn@balance],axis=-1))
        fields[str(step)]=dict(tau=tau[:,:,2],Y=Y,drive=D,fracture=F,balance=balance,
            phase=force['phi'],increment=force['phi']-L['nodes']['phi'],step_increment=force['phi']-force['obstacle'])
        assessment[str(step)]=dict(**check,max_complement_Pa=float(abs(R).max()),max_tau=float(tau[:,:,2].max()),
            max_Y=float(Y.max()),growing_nodes=int(np.sum(force['phi']>force['obstacle']+1e-12)),
            growing_balance_Pa=float(abs(balance[force['phi']>force['obstacle']+1e-12]).max(initial=0)),
            max_D_minus_F_at_unchanged_nodes_Pa=float(balance[force['phi']<=force['obstacle']+1e-12].max(initial=0)))
        print(step,assessment[str(step)],flush=True)
    np.savez_compressed(out/'remote-fields.npz',points=L['points'],weights=L['weights'],x=L['x'],y=L['y'],
        grid_ids=L['grid_ids'],nodes=L['node_xy'],phase0=L['nodes']['phi'],steps=STEPS,
        profiles=np.stack(profiles),line_points=lines,xx=xx,rr=rr,
        **{f'{k}_{name}':a for k,v in fields.items() for name,a in v.items()})
    (out/'remote-reconstruction.json').write_text(json.dumps(assessment,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--map',type=Path,default=MAP);a=p.parse_args();main(a.output,a.map)
