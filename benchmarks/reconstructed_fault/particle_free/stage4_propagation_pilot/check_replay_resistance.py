"""Resolve local resistance differences against the midpoint section law at 80 us."""
import argparse
import json
from pathlib import Path
import numpy as np
from phase import read, coefficients, H

def predicted(v, age):
    dt, dc, v0 = 4e-5, 3.2211126126299625e-7, .001
    travel = abs(v)*dt/dc
    average = np.divide(-np.expm1(-travel),travel,out=np.ones_like(v),where=travel!=0)
    stage = age*np.exp(-travel)+dt*average
    return 3*np.arcsinh(v*np.exp((.6+.02*np.log(stage*v0/dc))/.015)/(2*v0))+1000*v

def audit(old, new, step):
    data = [read(d/f'tip-sections-{step:04d}.csv') for d in (old,new)]
    planes = [read(d/f'tip-planes-{step:04d}.csv') for d in (old,new)]
    x = planes[0]['x']
    dv = np.maximum(abs(data[0]['V']),abs(data[1]['V']))
    df = abs(data[0]['force']-data[1]['force'])
    mask = df>1
    phase_diff = abs(coefficients(old,step)-coefficients(new,step))
    return dict(step=step,sections=np.flatnonzero(mask).tolist(),x=x[mask].tolist(),
                largest_local_speed=float(dv[mask].max()),max_resistance_difference=float(df.max()),
                max_difference_above_1nm_s=float(df[dv>1e-9].max()),
                max_difference_above_1mm_s=float(df[dv>.001].max()),
                max_phase_difference_in_affected_columns=float(phase_diff[:,np.unique((x[mask]/H).astype(int))].max()),
                max_law_errors=[float(abs(predicted(s['V'],p['parent_age'])-s['force'])[mask].max()) for s,p in zip(data,planes)],
                tail_work=[float(8e-5*np.sum(abs(s['V']*s['force']*s['ds'])[mask])) for s in data],
                total_work=[float(8e-5*np.sum(abs(s['V']*s['force']*s['ds']))) for s in data])
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ['original','repaired','destination']:
        parser.add_argument(arg,type=Path)
    args=parser.parse_args()
    rows=[audit(args.original,args.repaired,i) for i in (69,70)]
    args.destination.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))
