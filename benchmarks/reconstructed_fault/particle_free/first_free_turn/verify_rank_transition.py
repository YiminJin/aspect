#!/usr/bin/env python3
"""Check nonuniform 2-to-4-rank restart and the same first-append endpoint."""
from analyze import ROOT,RUNS,OUT,read,gather,verify_parent
import numpy as np,json
reference='first-turn-append-admissible';candidate='first-turn-four-rank-recovery'
result={'restore':verify_parent(reference,candidate,14)}
a=gather(RUNS/reference/'output','tip-points-0015-r',['cell','q'])
b=gather(RUNS/candidate/'output','tip-points-0015-r',['cell','q'])
assert np.array_equal(a.cell,b.cell) and np.array_equal(a.q,b.q)
result['phase_max']=float(max(abs(a.phi-b.phi)));assert result['phase_max']<1e-7
for prefix in ('tau','A'):
    error=sum(k*np.sum(a.weight*(a[prefix+n]-b[prefix+n])**2) for n,k in [('xx',1),('yy',1),('zz',1),('xy',2)])
    norm=sum(k*np.sum(a.weight*a[prefix+n]**2) for n,k in [('xx',1),('yy',1),('zz',1),('xy',2)])
    result[prefix+'_relative_L2']=float(np.sqrt(error/norm));assert result[prefix+'_relative_L2']<1e-7
sa=read(RUNS/reference/'output/tip-sections-0015.csv');sb=read(RUNS/candidate/'output/tip-sections-0015.csv')
result['sections']={n:float(max(abs(sa[n]-sb[n]))) for n in ['age','slip','V','force','ds']}
assert result['sections']['age']<1e-7 and result['sections']['slip']<1e-10
pa=read(RUNS/reference/'output/tip-planes-0015.csv');pb=read(RUNS/candidate/'output/tip-planes-0015.csv')
assert np.array_equal(pa.formed,pb.formed)
result['angle_max']=float(max(abs(pa.angle-pb.angle)));assert result['angle_max']<2e-7
result['criterion']='Restored parent <=2e-13 max(1, field maximum); endpoint phase/relative tensor/age <=1e-7, slip <=1e-10 m, angle <=2e-7 rad; identical formed flags. No solver criterion changed.'
(OUT/'four-rank-recovery-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='restore'},indent=2))
