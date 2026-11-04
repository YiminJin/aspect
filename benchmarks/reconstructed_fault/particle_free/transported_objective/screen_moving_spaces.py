#!/usr/bin/env python3
"""Best-space screen using independent exact histories, not transport evolution."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import moving_model as m
from stage_b_fixed import norm3_squared,save_csv
from stage_b_moving import mesh,project

ROOT=Path(__file__).resolve().parents[4]
DEFAULT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_b_moving'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args();out=args.output.resolve();rows=[]
    for n in (8,16,32):
        points,w,phi,_=mesh(n,12);flat=points.reshape(-1,2)
        exact=m.reference(flat,m.END,.24,48).reshape(n*n,-1,9)
        T=project(exact[...,3:6],phi,w);R=project(exact[...,6:9],phi,w)
        trial=np.einsum('qi,cik->cqk',phi,T,optimize=True)
        f=m.kinematics(flat,m.END,.24);B=(2*m.G*(f['chi']*f['s'])[:,None]*f['S']).reshape(n*n,-1,3)
        average=lambda x:float(np.mean(np.sum(x*w,axis=1)))
        for degree in (1,2):
            r=R.copy()
            if degree==1:r[:,[2,5,6,7,8],:]=0
            residual=np.einsum('qi,cik->cqk',phi,r,optimize=True)
            physical=trial-B-residual;delta=physical-exact[...,:3]
            shear,normal=m.tractions(delta,m.END)
            E=average(norm3_squared(physical));Er=average(norm3_squared(exact[...,:3]))
            rows.append(dict(n=n,residual_degree=degree,best_R_projection_scaled=np.sqrt(average(norm3_squared(residual-exact[...,6:9])))/m.G,
                full_tensor_projection_scaled=np.sqrt(average(norm3_squared(delta)))/m.G,
                normal_scaled=np.sqrt(average(normal**2))/m.G,shear_scaled=np.sqrt(average(shear**2))/m.G,
                energy_relative_error=abs(E-Er)/Er,scalar_coefficients=(27+3*(degree+1)**2)*n*n))
    record=dict(scope='Best independently projected T/R histories only; not a P1 transported solve',rows=rows,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.mkdir(parents=True,exist_ok=True);save_csv(out/'space_screen.csv',rows)
    (out/'space_screen.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))


if __name__=='__main__':main()
