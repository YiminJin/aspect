#!/usr/bin/env python3
"""Check one-sided endpoint traces and unchanged uncut cells from saved results."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_frozen_split_q2 import OUT,REFERENCE,GRIDS,METRIC,SplitQ2,load_fixture,basis,index_basis,mass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--reference',type=Path,default=REFERENCE)
    args=parser.parse_args()
    assert not (args.output/'audit.json').exists(), 'Preserve completed audit.'
    f=load_fixture(args.reference/'frozen-input.npz')
    coeff=np.load(args.output/'split-coefficients.npz')
    saved=np.load(args.reference/'projection-coefficients.npz')
    results=[]
    for n in GRIDS:
        space=SplitQ2(f,n)
        c=coeff[f'n{n}']
        same=~space.cut
        delta=c[same]-saved[f'n{n}_cut'][space.parents[same]]
        error=np.sqrt(np.einsum('ckl,k->l',np.einsum('...i,ij,...j->...',delta,METRIC,delta),mass(n)))
        assert np.all(error<1e-12), (n,'uncut cell invariance',error)
        ends=[]
        for node,region in [(0,0),(-1,2)]:
            x=f.X[node]
            cell,B=index_basis(x[None],f.state.n)
            phi=float(np.einsum('pk,pk->p',B,f.phi[cell])[0])
            chi=f.state.m*phi*(1+phi)/(1-phi)**2/f.I[node]
            eta=np.array([10.,10*(1+.5*np.tanh(4*x[0]))])
            kappa=-eta*np.expm1(-f.G*f.dt/eta)
            exact=-2*kappa[:,None]*chi*f.V[node]*f.S
            parent=int(index_basis(x[None],n)[0][0])
            pieces=space.lookup[parent,[1,region]]
            assert np.all(pieces>=0)
            # Evaluate both polynomial limits at the same point, selecting
            # material side explicitly rather than testing endpoint equality.
            B=basis(2*(x-space.lower[pieces])/space.width[pieces]-1)
            tau=np.einsum('pk,pklc->plc',B,c[pieces])
            jump=tau[0]-tau[1]
            exact_shear=exact@(METRIC@f.S)
            measured=jump@(METRIC@f.S)
            ends.append(dict(endpoint='left' if node==0 else 'right',
                exact_shear_jump=exact_shear.tolist(),split_shear_jump=measured.tolist(),
                signed_jump_error=(measured-exact_shear).tolist(),
                relative_jump_error=((measured-exact_shear)/exact_shear).tolist()))
        results.append(dict(n=n,uncut_cell_invariance_L2=error.tolist(),ends=ends))
    data=dict(scope='Frozen side limits only; no physical-tip law or transport qualification',cases=results,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'audit.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(max_uncut_change=max(max(r['uncut_cell_invariance_L2']) for r in results),
        max_jump_relative_error=max(abs(v) for r in results for e in r['ends'] for v in e['relative_jump_error']))))


if __name__=='__main__':main()
