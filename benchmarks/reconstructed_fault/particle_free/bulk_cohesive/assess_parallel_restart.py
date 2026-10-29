#!/usr/bin/env python3
"""Compare actual native restart/rejection branches with uninterrupted owners."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
from assess_native import csv as table
BASE=Path('build-particle-free/bulk-parallel/runs')
OUT=Path('benchmarks/reconstructed_fault/particle_free/evidence/bulk_parallel')

def main():
    result={}
    for rank in [1,2,4]:
        reference=BASE/f'r{rank}';branch=BASE/f'resume-r{rank}'
        if not (branch/'bulk-accepted.csv').exists():continue
        rows=table(branch/'bulk-accepted.csv')
        if rows['step'][-1]!=64:continue
        assert np.array_equal(rows['step'],np.arange(48,65)) and np.array_equal(rows['step'],rows['publications'])
        gates=list(csv.DictReader((branch/'gates.csv').open()));assert all(r['passed']=='1' for r in gates)
        for name in ['trial does not publish','rejection restores all accepted owners','deterministic retry','second rejection unchanged']:
            assert sum(r['gate']==name for r in gates)==1
        bitwise={};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        for kind in ['phase','record','means','bulk','phase-forces']:
            matched=[]
            for f in branch.glob(kind+'-*'):
                if not f.stem[len(kind)+1:].isdigit() or not (reference/f.name).exists():continue
                equal=sha(f)==sha(reference/f.name);assert equal,(rank,f)
                matched.append(f.name)
            assert len(matched)>0,kind;bitwise[kind]=matched
        source=table(reference/'bulk-accepted.csv');source=source[source['step']>=48]
        assert np.array_equal(rows,source),rank
        a=list(csv.DictReader((reference/'resources.csv').open()));b=list(csv.DictReader((branch/'resources.csv').open()))
        assert {r['particle_fingerprint'] for r in a+b}=={a[0]['particle_fingerprint']}
        assert all(r['freeze']=='1' for r in a+b)
        result[str(rank)]=dict(accepted_steps=17,rejected_converged_trials=2,publications=list(map(int,rows['publications'])),
            bitwise_files=bitwise,bitwise_accepted_ledger=True,particles_fixed_with_restart=True,
            maximum_source=float(max(rows['projection_error'])))
    (OUT/'restart-verification.json').write_text(json.dumps(result,indent=2)+'\n');print({n:sum(map(len,r['bitwise_files'].values())) for n,r in result.items()})
if __name__=='__main__':main()
