#!/usr/bin/env python3
"""Quantify the surface information used by the analytic-B comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import moving_model as m
from stage_b_moving import mesh
from stage_b_fixed import norm3_squared

ROOT=Path(__file__).resolve().parents[4]
DEFAULT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_b_moving'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args();out=args.output.resolve()
    p,w,_,_=mesh(16,12);f=m.kinematics(p.reshape(-1,2),m.END,.24)
    rows=[]
    for intervals in (16,32,64):
        alpha=np.linspace(-m.ALPHA_LIMIT,m.ALPHA_LIMIT,intervals+1)
        points=m.centre(m.END,.24)+alpha[:,None]*(m.T0+np.array([m.SHEAR*m.END*m.T0[1],0.]))
        values=m.kinematics(points,m.END,.24)['s']
        s=np.interp(f['alpha'],alpha,values)
        difference=s-f['s']
        B=2*m.G*(f['chi']*difference)[:,None]*f['S']
        error=float(np.sqrt(np.mean(np.sum(norm3_squared(B).reshape(16*16,-1)*w,axis=1)))/m.G)
        rows.append(dict(material_intervals=intervals,stored_scalars=intervals+1,
                         scalar_max_error_m=float(np.max(np.abs(difference))),B_extension_scaled=error))
    ratios=[rows[i]['B_extension_scaled']/rows[i+1]['B_extension_scaled'] for i in (0,1)]
    # h halves for a second-order scalar interpolation, so error ratios are4.
    orders=[float(np.log2(r)) for r in ratios]
    result=dict(scope='Snapshot Q1 surface-memory representation; no numerical ALE/evolution claim',rows=rows,
                orders=orders,status='passed' if all(1.8<o<2.2 for o in orders) and rows[-1]['B_extension_scaled']<1e-3 else 'failed',
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.mkdir(parents=True,exist_ok=True);(out/'surface_screen.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
