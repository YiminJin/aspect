#!/usr/bin/env python3
"""Independent sensitivity to initial material-label spacing, without resampling."""
import os
for key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS"]:os.environ[key]="1"
import json
import hashlib
import numpy as np
from compare_bounded_profile import OUT,RUNS,ROOT,load,tensor,quadrature,dot3,work_traction

def main():
    results=[]
    for c in ['c','l']:
        names=[f'bpf-{c}-p-m-joint',f'bpf-{c}-p-m-surface']
        states=[load(n,final_only=True)[3] for n in names];a,b=[s[max(s)] for s in states]
        x,w=quadrature(max(a.n,b.n));ta,tb=tensor(a,x),tensor(b,x)
        va,vb=a.at(x),b.at(x)
        result=dict(cases=names,initial_spacing=[.1,.05],
                    stress_relative_L2=float(np.sqrt((w@dot3(ta-tb,ta-tb))/(w@dot3(tb,tb)))),
                    profile_relative_L2=float(np.sqrt((w@(va[:,0]-vb[:,0])**2)/(w@vb[:,0]**2))))
        # Comparisons use the original fixed material parameter; neither source
        # state nor its physical localization is altered by this interpolation.
        la=np.linspace(0,1,len(a.surface));lb=np.linspace(0,1,len(b.surface))
        for col,key in [(3,'age'),(4,'memory'),(5,'normalization'),(6,'slip_rate')]:
            aa=np.interp(lb,la,a.surface[:,col]);bb=b.surface[:,col]
            result[key+'_relative_L2']=float(np.linalg.norm(aa-bb)/np.linalg.norm(bb))
        wa,_=work_traction(a);wb,M=work_traction(b);delta=np.interp(lb,la,wa)-wb
        result['work_traction_relative_L2']=float(np.sqrt((delta@M@delta)/(wb@M@wb)))
        result['input_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for n in names for p in (RUNS/n/'output').glob('material-*.csv')}
        result['normalization']=[a.normalization_diagnostics,b.normalization_diagnostics]
        results.append(result)
    (OUT/'surface-sensitivity.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
