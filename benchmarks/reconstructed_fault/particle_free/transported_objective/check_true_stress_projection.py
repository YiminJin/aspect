#!/usr/bin/env python3
"""Independent Legendre projections of the native frozen source, at q8/q12/q16."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,copy,json
from pathlib import Path
import numpy as np
from compare_bounded_profile import load,quadrature,dot3,ROOT,RUNS
from assess_material_native import basis,read
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/direct_true_stress'

def check(name):
    cmd,rows,meta,states=load(name);s=copy.deepcopy(states[0]);s.surface[:,6]=.002*(1+.2*np.cos(s.surface[:,1]))
    eta=lambda x:10*(1+.5*np.tanh(4*x[:,0])) if cmd['arguments']['eta']!='1' else np.full(len(x),10.)
    def local(x):
        k=-eta(x)*np.expm1(-10*cmd['arguments']['initial_dt']/eta(x));D=np.array([.03,-.03,.01])
        return 2*k[:,None]*D-k[:,None]*s.memory(x,s.at(x)[:,0],6)
    def project(q):
        g,w=np.polynomial.legendre.leggauss(q);u=np.stack(np.meshgrid(g,g,indexing='ij'),-1).reshape(-1,2)
        B=basis(u);weight=np.outer(w,w).ravel();M=B.T@(weight[:,None]*B)
        coeff=np.zeros((s.n,s.n,9,3))
        for i in range(s.n):
          for j in range(s.n):
            x=-2+s.h*(np.array([i,j])+.5*(u+1));coeff[i,j]=np.linalg.solve(M,B.T@(weight[:,None]*local(x)))
        return coeff
    def evaluate(coeff,x):
        ij=np.floor((x+2)/s.h).astype(int);u=2*((x+2)/s.h-ij)-1
        return np.einsum('ni,nij->nj',basis(u),coeff[ij[:,0],ij[:,1]])
    p8,p12,p16=project(8),project(12),project(16);result={}
    for q in [8,12,16]:
        x,w=quadrature(s.n,q);src=local(x);a=evaluate(p8,x);b=evaluate(p12,x);c=evaluate(p16,x)
        result[q]=dict(candidate_L2=float(np.sqrt(w@dot3(a,a))),projection_remainder_L2=float(np.sqrt(w@dot3(a-src,a-src))),
          P12_minus_P8_L2=float(np.sqrt(w@dot3(a-b,a-b))),P16_minus_P12_L2=float(np.sqrt(w@dot3(c-b,c-b))),
          P8_energy_loss=float(w@(dot3(src,src)-dot3(a,a))/40))
    native=read(RUNS/name/'output/true-stress-source.csv')
    error=abs(result[8]['candidate_L2']-float(native[0]['candidate_L2']))
    assert error<1e-12,(name,error)
    return dict(quadrature=result,native_candidate_norm_discrepancy=error)

def main():
    p=argparse.ArgumentParser();p.add_argument('cases',nargs='+');a=p.parse_args();result={n:check(n) for n in a.cases}
    (OUT/'independent-source-projection.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
