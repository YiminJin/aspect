#!/usr/bin/env python3
"""Measured fixed-FF, temporal and physical-ownership mature DG comparisons."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_pf6_mature import read
from analyze_pf6_mature_resolution import tensors,surface,UPPER,NORMAL
from sample_pf6_mature_dg import sample


def merged(run,name,keys):
    paths=list(run.glob('rank-*-'+name)) or [run/name]
    ds=[read(p) for p in paths];d={k:np.concatenate([p[k] for p in ds]) for k in ds[0]}
    order=np.lexsort(tuple(d[k] for k in reversed(keys)))
    return {k:v[order] for k,v in d.items()}


def replicated(run,name):
    paths=list(run.glob('rank-*-'+name)) or [run/name]
    first=read(paths[0])
    for path in paths[1:]:
        data=read(path)
        for key in first:assert np.array_equal(first[key],data[key]),(name,key)
    return first


def compare(run_a,step_a,run_b,step_b,profile,independent=False):
    a=merged(run_a,f'authoritative-stress-{step_a}.csv',('cell','qp'))
    if independent:
        x=np.column_stack((a['x'],a['y']));b=sample(run_b,step_b,x)
        b.update({k:a[k] for k in ('qp','x','y','JxW')})
    else:
        b=merged(run_b,f'authoritative-stress-{step_b}.csv',('cell','qp'))
        assert np.array_equal(a['cell'],b['cell'])
    r=(np.column_stack((a['x'],a['y']))-UPPER)@NORMAL
    h=np.interp(abs(r),profile['r_m'],profile['h'],right=0.)
    stress=tensors(a,b,h)
    pa=read(run_a/f'coupled-profile-{step_a}.csv');pb=read(run_b/f'coupled-profile-{step_b}.csv')
    wa=replicated(run_a,f'surface-weak-{step_a}.csv');wb=replicated(run_b,f'surface-weak-{step_b}.csv')
    return dict(stress=stress,surface=surface(pa,pb,wa,wb))


def physical_states(run_a,step_a,run_b,step_b):
    out={}
    for name,keys,identity in [(f'dg-coefficients-{{}}.csv',('cell','mode'),('cell','mode')),
                               (f'physical-fe-{{}}.csv',('component','x','y'),('component','x','y'))]:
        a=merged(run_a,name.format(step_a),keys);b=merged(run_b,name.format(step_b),keys)
        assert len(a[keys[0]])==len(b[keys[0]])
        for k in identity:assert np.array_equal(a[k],b[k]),(name,k)
        out[name]={k:float(np.max(abs(a[k]-b[k]))) for k in a if k not in identity}
    for prefix in ('coupled-profile','surface-weak'):
        a=replicated(run_a,f'{prefix}-{step_a}.csv');b=replicated(run_b,f'{prefix}-{step_b}.csv')
        out[prefix]={k:float(np.max(abs(a[k]-b[k]))) for k in a}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('first',type=Path);p.add_argument('second',type=Path)
    p.add_argument('--steps',type=int,nargs=2,default=[3,3]);p.add_argument('--independent',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[3]
    profile=read(root/'build-particle-free/mature-resolution/inputs-lf/physical-profile.csv')
    result=compare(a.first,a.steps[0],a.second,a.steps[1],profile,a.independent)
    if not a.independent:result['persistent']=physical_states(a.first,a.steps[0],a.second,a.steps[1])
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k]['all'] for k in ('stress','surface')},indent=2))
if __name__=='__main__':main()
