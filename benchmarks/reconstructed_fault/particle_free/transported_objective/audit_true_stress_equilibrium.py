#!/usr/bin/env python3
"""Independent Q2/Q1 weak equilibrium and strong-trace audit from accepted exports.

No mechanics solve. Dirichlet velocity: left/right/bottom, natural top. Continuous
Q1 pressure with no normalization. Force residuals use Pa m (per out-of-plane m),
continuity residuals m^2/s. Relative norms divide by assembled absolute element
contributions; no fitted force or power correction. Native quadrature is Gauss8.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json
import numpy as np
from assess_material_native import basis
from compare_bounded_profile import load,tensor

def lagrange(x):
    return np.stack([x*(x-1)/2,1-x*x,x*(x+1)/2],-1),np.stack([x-.5,-2*x,x+.5],-1)

def audit(state,order=8):
    n,h=state.n,state.h;g,w=np.polynomial.legendre.leggauss(order)
    uv=np.stack(np.meshgrid(g,g,indexing='ij'),-1).reshape(-1,2);weights=np.outer(w,w).ravel()*h*h/4
    a,da=lagrange(uv[:,0]);b,db=lagrange(uv[:,1])
    grad=np.stack([(da[:,:,None]*b[:,None,:]).reshape(-1,9),(a[:,:,None]*db[:,None,:]).reshape(-1,9)],-1)*2/h
    pressure_basis=(((1+uv[:,0,None]*np.array([-1,1]))/2)[:,:,None]*((1+uv[:,1,None]*np.array([-1,1]))/2)[:,None,:]).reshape(-1,4)
    force=np.zeros((2*n+1,2*n+1,2));force_abs=force.copy();div=np.zeros((n+1,n+1));div_abs=div.copy()
    for i in range(n):
      for j in range(n):
        x=-2+h*(np.array([i,j])+.5*(uv+1));v=state.at(x);t=tensor(state,x);t[:,0]-=v[:,17];t[:,1]-=v[:,17]
        stress=np.stack([t[:,0],t[:,2],t[:,2],t[:,1]],-1).reshape(-1,2,2)
        integrand=np.einsum('qab,qib->qia',stress,grad)*weights[:,None,None]
        r=integrand.sum(0).reshape(3,3,2);scale=abs(integrand).sum(0).reshape(3,3,2)
        force[2*i:2*i+3,2*j:2*j+3]+=r;force_abs[2*i:2*i+3,2*j:2*j+3]+=scale
        z=weights[:,None]*pressure_basis*(v[:,12]+v[:,15])[:,None]
        div[i:i+2,j:j+2]+=z.sum(0).reshape(2,2);div_abs[i:i+2,j:j+2]+=abs(z).sum(0).reshape(2,2)
    free=np.ones((2*n+1,2*n+1),bool);free[0]=False;free[-1]=False;free[:,0]=False
    # Per-face boundary stresses and interior jumps evaluated using each cell's
    # polynomial trace. Avoid choosing a neighboring cell at an interface.
    def trace(i,j,u):
        v=basis(u)@state.coeff[i,j];x=-2+h*(np.array([i,j])+.5*(u+1))
        t=v[:,3:6].copy()
        if not getattr(state,'direct',False):t-=v[:,6:9]+state.memory(x,v[:,0])
        t[:,0]-=v[:,17];t[:,1]-=v[:,17]
        return t
    face={};jump_integral=np.zeros(2);jump_abs=0.;jump_sq=0.
    for axis in [0,1]:
      for side in [-1,1]:
        total=np.zeros(2)
        for k in range(n):
          ij=[k,k];ij[axis]=0 if side<0 else n-1
          u=np.zeros((order,2));u[:,axis]=side;u[:,1-axis]=g
          t=trace(*ij,u);traction=t[:,[0,2] if axis==0 else [2,1]]
          total+=side*.5*h*(w@traction)
        face[f'{"xy"[axis]}{side:+}']=total.tolist()
      for edge in range(1,n):
        for k in range(n):
          ij=[k,k];ij[axis]=edge-1;u=np.zeros((order,2));u[:,axis]=1;u[:,1-axis]=g
          left=trace(*ij,u);ij[axis]=edge;u[:,axis]=-1;right=trace(*ij,u)
          jump=(left-right)[:,[0,2] if axis==0 else [2,1]]
          jump_integral+=.5*h*(w@jump);jump_abs+=.5*h*(w@np.linalg.norm(jump,axis=1));jump_sq+=.5*h*(w@np.sum(jump*jump,axis=1))
    return dict(order=order,mechanical_free_l2=float(np.linalg.norm(force[free])),mechanical_free_linf=float(abs(force[free]).max()),
      mechanical_absolute_assembly_scale=float(np.linalg.norm(force_abs[free])),mechanical_relative=float(np.linalg.norm(force[free])/np.linalg.norm(force_abs[free])),
      continuity_l2=float(np.linalg.norm(div)),continuity_linf=float(abs(div).max()),continuity_absolute_assembly_scale=float(np.linalg.norm(div_abs)),
      net_weak_force=force.sum((0,1)).tolist(),constrained_reaction=force[~free].sum(0).tolist(),constrained_reaction_l2=float(np.linalg.norm(force[~free],axis=0).sum()),
      constrained_component_l2=np.linalg.norm(force[~free],axis=0).tolist(),raw_faces=face,raw_net_force=np.array(list(face.values())).sum(0).tolist(),
      interior_jump_signed=jump_integral.tolist(),interior_jump_L1=float(jump_abs),interior_jump_L2=float(np.sqrt(jump_sq)))

def main():
    p=argparse.ArgumentParser();p.add_argument('cases',nargs='+');p.add_argument('--output',required=True);args=p.parse_args();result={}
    for name in args.cases:
      cmd,rows,meta,states=load(name,True);s=states[max(states)]
      result[name]=audit(s);print(name,json.dumps(result[name]),flush=True)
    from pathlib import Path
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
