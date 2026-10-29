#!/usr/bin/env python3
"""Canonical physical-field, accepted-owner and lifecycle comparisons."""
import argparse,json,hashlib
import csv as standard_csv
from pathlib import Path
import numpy as np
from assess_native import csv,norm3
ROOT=Path('build-particle-free/bulk-parallel/runs')


def compare(a,b,steps=None):
    phases=sorted(p for p in a.glob('phase-*.csv') if p.stem[6:].isdigit())
    if steps is not None:phases=[p for p in phases if int(p.stem.split('-')[-1]) in steps]
    maxima={k:0. for k in ['phase','growth','DG','native_projection','native_memory','a','signed_shear','absolute_shear','q','rate','physical_stress','Y','storage']}
    bitwise=True;count=0
    p0=csv(a/'phase-0000.csv');p0.sort(order=['x','y'])
    for f in phases:
        step=int(f.stem.split('-')[-1]);other=b/f.name
        if not other.exists():continue
        p=csv(f);q=csv(other);p.sort(order=['x','y']);q.sort(order=['x','y'])
        assert np.array_equal(p[['x','y']],q[['x','y']]);assert len(p)==len(p0)
        def rel(k,x,y,floor=1e-12):maxima[k]=max(maxima[k],float(np.linalg.norm(x-y)/max(floor,np.linalg.norm(x))))
        rel('phase',p['phi'],q['phi']);rel('growth',p['phi']-p0['phi'],q['phi']-p0['phi'])
        bitwise &= np.array_equal(p['phi'],q['phi']);count+=1
        fa=a/f'record-{step:04d}.bin';fb=b/fa.name
        if fa.exists() and fb.exists() and fa.stat().st_size and fb.stat().st_size:
            x=np.fromfile(fa).reshape(-1,392);y=np.fromfile(fb).reshape(-1,392);assert x.shape==y.shape
            bitwise &= np.array_equal(x,y)
            for k,sl in [('DG',slice(0,27)),('native_projection',slice(27,54)),('native_memory',slice(54,72))]:rel(k,x[:,sl],y[:,sl],(1000. if k!='native_memory' else .001/32)*np.sqrt(x[:,sl].size))
            xx=x[:,72:].reshape(-1,64,5);yy=y[:,72:].reshape(-1,64,5)
            for j,k in enumerate(['a','signed_shear','absolute_shear','q','rate']):rel(k,xx[:,:,j],yy[:,:,j])
        fa=a/f'bulk-{step:04d}.csv';fb=b/fa.name
        if fa.exists() and fb.exists():
            x=csv(fa);y=csv(fb);x.sort(order=['cell','x','y']);y.sort(order=['cell','x','y'])
            t=np.stack([x[k] for k in ['tauxx','tauyy','tauxy']],axis=-1);u=np.stack([y[k] for k in ['tauxx','tauyy','tauxy']],axis=-1)
            e=np.sqrt(np.sum(x['JxW']*norm3(t-u))/max(1000.**2*np.sum(x['JxW']),np.sum(x['JxW']*norm3(t))))
            maxima['physical_stress']=max(maxima['physical_stress'],float(e));rel('Y',x['driving'],y['driving'])
    la=csv(a/'bulk-accepted.csv');lb=csv(b/'bulk-accepted.csv')
    for row in la:
        other=lb[lb['step']==row['step']]
        if not len(other):continue
        r=other[-1]
        if steps is not None and int(row['step']) not in steps:continue
        for k in ['bulk','cohesive','fracture','Wext','Db','Dc','Df','energy_change']:
            maxima['storage']=max(maxima['storage'],float(abs(row[k]-r[k])/max(1.,abs(row[k]))))
    assert max(maxima.values())<=1e-10,maxima
    return dict(maxima=maxima,compared_states=count,bitwise_physical_owners=bool(bitwise))


def native(path,recovery=False):
    l=csv(path/'bulk-accepted.csv');g=csv(path/'gates.csv');assert np.all(g['passed']==1)
    assert np.array_equal(l['step'],np.arange(1,len(l)+1)) and np.array_equal(l['publications'],l['step'])
    assert max(l['mechanical_residual'])<=1e-9 and max(l['phase_KKT'])<=1e-7 and max(l['projection_error'])<=1e-12
    p0=csv(path/'phase-0000.csv');p0.sort(order=['x','y']);p=csv(path/f"phase-{int(l['step'][-1]):04d}.csv");p.sort(order=['x','y'])
    growth=float(np.max(p['phi']-p0['phi']));assert growth>1e-8
    if recovery:
        for label in ['trial does not publish','rejection restores all accepted owners','deterministic retry','second rejection unchanged']:assert np.sum(g['gate']==label)==1,label
    r=csv(path/'resources.csv');resources=list(standard_csv.DictReader((path/'resources.csv').open()));fingerprints={row['particle_fingerprint'] for row in resources};assert len(fingerprints)==1 and all(row['freeze']=='1' for row in resources)
    profile=csv(path/'bulk-profile.csv');loaded=profile[(profile['time']>.12+1e-12)&(profile['time']<=.16+1e-12)]
    # Rejected trials are intentionally included in recovery cost; controls have
    # one solve per accepted step. Report both rather than hide repeated work.
    return dict(steps=len(l),phase_growth=growth,maximum_mechanics=float(max(l['mechanical_residual'])),maximum_KKT=float(max(l['phase_KKT'])),maximum_source=float(max(l['projection_error'])),maximum_ledger_J_m=float(max(abs(l['ledger_error']))),particle_fingerprint=int(next(iter(fingerprints))),particles=int(r['particles'][0]),max_rank_MiB=float(max(r['max_rank_KiB'])/1024),sum_rank_peaks_MiB=float(max(r['sum_rank_peaks_KiB'])/1024),loaded_profile={k:float(np.sum(loaded[k])) for k in ['assembly','communication','linear','history','total']},loaded_evaluations=int(np.sum(loaded['evaluations'])),loaded_linear_solves=int(np.sum(loaded['linear_solves'])),output_seconds=float(np.sum(csv(path/'output-profile.csv')['seconds'])))


def main(out):
    result={}
    result['baseline_serial']=compare(ROOT/'baseline',ROOT/'r1')
    for n in [1,2,4]:
        path=ROOT/f'r{n}';result[f'r{n}']=native(path)
        if n>1:result[f'comparison_1_{n}']=compare(ROOT/'r1',path)
    (out/'parallel-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
