#!/usr/bin/env python3
"""Recover common complete native fields and their equivalent FE history."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from assess_pf6a import Ref, csv, grids, strains, norm2


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    record=json.loads((source/'record.json').read_text());case=record['PF6a_case']
    assert case['normal_cells']==32 and case['divisor']==4 and case['quadrature']==12 and case['end_time']==.5625
    ref=Ref(source/'run/state-72');nx=round(1/ref.cells['dx'][0]);ny=round(1/ref.cells['dy'][0])
    displacement=np.zeros((len(ref.cells),3,3,2)); saved={}; compatibility=[]
    for step in range(1,73):
        d=source/'run'/f'state-{step}';vel=grids(csv(d/'velocity.csv'),['ux','uy'],len(ref.cells))
        dt=float(csv(d/'clock.csv')['dt']) if (d/'clock.csv').exists() else 1/128
        beta=np.exp(-dt/100);kappa=-1e8*np.expm1(-dt/100)
        displacement=beta*displacement+kappa/1e6*vel
        if step in [68,71,72]:
            r=Ref(d);g=np.polynomial.legendre.leggauss(3)[0]/2+.5;rr=np.repeat(g,3);ss=np.tile(g,3)
            e,_=strains(r,displacement,rr,ss);t=r.tensor_values(rr,ss)
            compatibility.append(dict(step=step,unisolvent_points=9,max_absolute_Pa=float(np.max(abs(2e6*e-t))),relative=float(np.max(abs(2e6*e-t))/max(1.,np.max(abs(t))))))
            saved[step]=displacement.copy()
    assert max(x['relative'] for x in compatibility)<1e-11,compatibility
    phase=[];R=[];T=[];Z=[]
    for step,pstep in [(72,72),(71,72),(68,68)]:
        r=Ref(source/'run'/f'state-{step}');ph=Ref(source/'run'/f'state-{pstep}')
        phi=np.full((ny+1,nx+1),np.nan);disp=np.full((2*ny+1,2*nx+1,2),np.nan)
        for row in ph.nodes:phi[round(row['y']*ny),round(row['x']*nx)]=row['phase']
        tensor=np.zeros((ny,nx,3,3,3))
        tt=grids(r.tensor,['xx','yy','xy'],len(r.cells))
        for c,cell in enumerate(r.cells):
            ix,iy=round(cell['x0']*nx),round(cell['y0']*ny)
            tensor[iy,ix]=tt[c]
            for i in range(3):
                for j in range(3):
                    y,x=2*iy+j,2*ix+i
                    if np.isfinite(disp[y,x]).all():assert max(abs(disp[y,x]-saved[step][c,i,j]))<1e-13
                    disp[y,x]=saved[step][c,i,j]
        phase.append(phi);R.append(disp);T.append(tensor);Z.append(r.surface['Z'])
    arrays=dict(phase=np.array(phase),R=np.array(R),T=np.array(T),Z=np.array(Z),nx=nx,ny=ny)
    np.savez_compressed(output/'common.npz',**arrays)
    with (output/'common.txt').open('w') as f:
        f.write(f'{nx} {ny} {len(Z[0])}\n')
        for k in range(3):
            for a in [phase[k],R[k],T[k],Z[k]]:np.savetxt(f,np.asarray(a).reshape(-1,1),fmt='%.17g')
    metadata=dict(source=str(source),snapshot_steps=[72,71,68],phase_steps=[72,72,68],compatibility=compatibility,
                  source_record_sha256=hashlib.sha256((source/'record.json').read_bytes()).hexdigest(),
                  common_txt_sha256=hashlib.sha256((output/'common.txt').read_bytes()).hexdigest(),
                  meaning='Physical final fields, frozen-mechanics incoming histories, and conditional fresh event entry. Complete 2D Q2 coefficients, no shear-only replacement.')
    (output/'preparation.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();prepare(a.source,a.output)
