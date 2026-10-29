#!/usr/bin/env python3
"""Mechanically relaxed critical-space check of exported native tangents.

Rows already have physical factors and native C^T H C; never condense again.
The return map is eliminated in the actual native tangent. The mixed mechanical
block includes pressure and its native constraints, not a frozen-strain proxy.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu,LinearOperator,eigsh

def analyze(run,out):
    started=time.monotonic();meta=json.load(open(run/'continuation-state.json'))
    c=np.genfromtxt(run/'continuation-coordinates.csv',delimiter=',',names=True)
    with (run/'continuation-tangent.bin').open('rb') as f:
        n,nnz=np.fromfile(f,dtype='<u8',count=2)
        a=np.fromfile(f,dtype=[('i','<u4'),('j','<u4'),('v','<f8')],count=int(nnz))
        assert f.read()==b'' and len(c)==n
    H=coo_matrix((a['v'],(a['i'],a['j'])),shape=(int(n),int(n))).tocsr();del a
    mech=np.flatnonzero(c['phase_dof']<0);phase=np.flatnonzero(c['phase_dof']>=0)
    A=H[mech][:,mech].tocsc();B=H[mech][:,phase].tocsr();D=H[phase][:,phase].tocsr()
    cross=H[phase][:,mech];difference=cross-B.T
    result=dict(meta=meta,size=int(n),nnz=int(nnz),mechanical_dofs=len(mech),phase_dofs=len(phase),
      symmetry_relative_max=float(abs(H-H.T).max()/max(1.,abs(H).max())),
      cross_symmetry_relative_max=float(abs(difference).max()/max(1.,abs(B).max())))
    assert result['cross_symmetry_relative_max']<1e-11
    cp=c[phase];mass=cp['mass'];gap=cp['phi']-cp['obstacle'];density=cp['physical_gradient']/(mass*meta['Kf'])
    peak=float(cp['phi'].max());hmax=4*peak*(1+peak)/(1-peak)**2
    result['mechanical_engineering_shear_coercivity_lower_bound_Pa']=1e6*meta['beta']**2/(1+hmax)
    result['min_phase_gap_density']=float(density.min());result['min_increment']=float(gap.min());result['max_increment']=float(gap.max())
    inverse=None;cache={}
    mode_directory=run.parent.parent/'stability';mode_directory.mkdir(exist_ok=True)
    def check(ids,label):
        nonlocal inverse
        key=ids.tobytes()
        if key in cache:return dict(cache[key],same_subspace_as=cache[key]['label'])
        if not len(ids):return dict(dofs=0,reason='Strict positive obstacle multipliers exclude all phase directions. Mechanics is positive on the incompressible velocity space.')
        if inverse is None:inverse=splu(A)
        b=B[:,ids].tocsc();d=D[ids][:,ids].tocsr();sc=1/np.sqrt(mass[ids]*meta['Kf'])
        def apply(x):
            z=sc*x
            return sc*(d@z-b.T@inverse.solve(b@z))
        frozen=LinearOperator((len(ids),len(ids)),matvec=lambda z:sc*(d@(sc*z)))
        if len(ids)<4:
            frozen_min=float(np.linalg.eigvalsh(np.column_stack([frozen@z for z in np.eye(len(ids))]))[0])
        else:
            frozen_min=float(eigsh(frozen,k=1,which='SA',v0=np.cos(np.arange(len(ids))+.17),tol=1e-8,return_eigenvectors=False)[0])
        op=LinearOperator((len(ids),len(ids)),matvec=apply)
        if len(ids)<4:
            h=np.column_stack([apply(v) for v in np.eye(len(ids))]);values,vectors=np.linalg.eigh(h)
        else:values,vectors=eigsh(op,k=min(3,len(ids)-1),which='SA',v0=np.cos(np.arange(len(ids))+.17),tol=1e-8,maxiter=1000)
        errors=[float(np.linalg.norm(apply(vectors[:,j])-values[j]*vectors[:,j])) for j in range(len(values))]
        rng=np.random.default_rng(19);u=rng.standard_normal(len(ids));v=rng.standard_normal(len(ids));hu=apply(u);hv=apply(v)
        symmetry=abs(u@hv-v@hu)/max(1.,abs(u@hv),abs(v@hu))
        assert symmetry<1e-10
        v=sc*vectors[:,0];response=-inverse.solve(b@v)
        scaled_v=v/max(abs(v));negative=scaled_v<0;positive=scaled_v>0
        feasible_positive=min(float(np.min(gap[ids][negative]/(-scaled_v[negative]))) if np.any(negative) else float('inf'),float(np.min((1-cp['phi'][ids][positive])/scaled_v[positive])) if np.any(positive) else float('inf'))
        feasible_negative=min(float(np.min(gap[ids][positive]/scaled_v[positive])) if np.any(positive) else float('inf'),float(np.min((1-cp['phi'][ids][negative])/(-scaled_v[negative]))) if np.any(negative) else float('inf'))
        full=np.zeros(len(c));full[mech]=response;full[phase[ids]]=v
        np.savez_compressed(mode_directory/(out.stem+'-'+label+'-mode.npz'),coordinates=c,phase_indices=ids,mode=full,eigenvalues=values)
        result=dict(label=label,dofs=len(ids),frozen_mechanics_minimum_eigenvalue=frozen_min,phase_peak_normalized_feasible_amplitudes=[feasible_positive,feasible_negative],eigenvalues=values.tolist(),eigen_residuals=errors,symmetry_relative=float(symmetry),
          relaxed_mechanical_residual=float(np.linalg.norm(A@response+b@v)/max(1.,np.linalg.norm(b@v))),
          phase_multiplier_range=[float(density[ids].min()),float(density[ids].max())])
        cache[key]=result
        return result
    # Interior directions are two-sided feasible for sufficiently small steps.
    # Enlarging with near-zero-multiplier obstacle nodes provides a sufficient
    # positivity test; a negative vector there is not automatically cone-feasible.
    interior=np.flatnonzero(gap>1e-9)
    result['interior']=check(interior,'interior')
    for multiplier in (1.,10.):
        tol=multiplier*1e-7
        ids=np.flatnonzero((gap>1e-9)|(density<=tol))
        result[f'critical_superset_{tol:g}']=check(ids,f'critical-{tol:g}')
    result['incremental_potential']='E_bulk+E_cohesive+E_fracture+beta*dt*D_friction; incompressibility constrained'
    result['seconds']=time.monotonic()-started
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('out',type=Path);a=p.parse_args();analyze(a.run,a.out)
