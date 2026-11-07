#!/usr/bin/env python3
"""Preserve the entry state and reconstruct the saved Q2 velocity (no new solve)."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
from assess_material_native import State,basis
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/corner_replay'
RUN=ROOT/'build-particle-free/transported-objective/stage-c/material-runs/shear32-dt025'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=False)
    old=json.loads((OUT.parent/'corner_isolation/final-provenance.json').read_text())
    protected={p:h for group in old['preserved'].values() for p,h in group.items()}
    # Preserve every prior experiment file, excluding status files intentionally updated here.
    for p,h in old['files_sha256'].items():
        if '/corner_isolation/' in p or '/corner_inflow/' in p or '/corner-inflow/' in p or '/material-runs/' in p:protected[p]=h
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    entry=dict(root=str(ROOT),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
               revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
               tracked_changes=subprocess.check_output(['git','status','--short','--untracked-files=no'],text=True),
               preserved=protected,source_run=str(RUN.relative_to(ROOT)),source_step=16)
    (OUT/'entry.json').write_text(json.dumps(entry,indent=2)+'\n')
    state=State(RUN/'output',16,5,50)
    coeff=state.coeff[:,:,:,1:3];h=state.h;n=state.n
    ij=np.floor((state.x+2)/h).astype(int);u=2*((state.x+2)/h-ij)-1
    fitted=np.einsum('ni,nij->nj',basis(u),coeff[ij[:,0],ij[:,1]])
    grad=[]
    for axis in (0,1):
        a=np.stack([np.ones(len(u)),u[:,0],.5*(3*u[:,0]**2-1)],axis=1)
        b=np.stack([np.ones(len(u)),u[:,1],.5*(3*u[:,1]**2-1)],axis=1)
        derivative=np.stack([np.zeros(len(u)),np.ones(len(u)),3*u[:,axis]],axis=1)*2/h
        if axis==0:a=derivative
        else:b=derivative
        grad.append(np.einsum('ni,nij->nj',(a[:,:,None]*b[:,None,:]).reshape(-1,9),coeff[ij[:,0],ij[:,1]]))
    L=np.stack(grad,axis=2)
    gradient_error=float(np.max(np.abs(L-state.values[:,12:16].reshape(-1,2,2))))
    q=np.linspace(-1,1,17);jump=0.
    for axis in (0,1):
        for i in range(n-1):
            for j in range(n):
                left=(i,j) if axis==0 else (j,i);right=(i+1,j) if axis==0 else (j,i+1)
                ul=np.column_stack([np.ones(17),q]) if axis==0 else np.column_stack([q,np.ones(17)])
                ur=ul.copy();ur[:,axis]=-1
                jump=max(jump,float(np.max(np.abs(basis(ul)@coeff[left]-basis(ur)@coeff[right]))))
    metrics=dict(velocity_reconstruction_max=float(np.max(abs(fitted-state.values[:,1:3]))),
                 gradient_reconstruction_max=gradient_error,face_continuity_max=jump,
                 saved_gauss_speed_max=float(np.linalg.norm(fitted,axis=1).max()),
                 divergence_max=float(np.max(abs(np.trace(L,axis1=1,axis2=2)))),
                 nonaffine_velocity_max=float(np.linalg.norm(fitted-np.column_stack([.4+.5*state.x[:,1],np.zeros(len(u))]),axis=1).max()))
    assert metrics['velocity_reconstruction_max']<1e-12 and gradient_error<1e-10 and jump<1e-12,metrics
    with (OUT/'velocity.txt').open('w') as f:
        f.write(f'{n}\n')
        for i in range(n):
            for j in range(n):f.write(' '.join(format(v,'.17g') for v in coeff[i,j].ravel())+'\n')
    (OUT/'velocity-audit.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
