"""Compare post-cleanup A/B replays with preserved small inclined evidence."""
import json
import re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'bp5/interpolation-inclined'

def array(out,stem,step,n):
    a=np.concatenate([np.fromfile(p,dtype='float64').reshape(-1,n)
                      for p in out.glob(f'{stem}_{step}_rank*.bin')])
    return a[np.lexsort((a[:,1],a[:,0]))]

def main():
    result={}
    for case in 'AB':
        before=ROOT/f'{case}-one';after=ROOT/f'cleanup-final-{case}'
        assert (before/'initial_hash.txt').read_bytes()==(after/'initial_hash.txt').read_bytes()
        assert (before/'finite_elements.txt').read_bytes()==(after/'finite_elements.txt').read_bytes()
        summary=np.genfromtxt(after/'summary.csv',delimiter=',',names=True,dtype=None,encoding=None)
        assert np.array_equal(summary['step'],np.arange(5)) and np.all(summary['relative']<1e-8)
        checks=re.findall(r'Fault linear solve: iterations=(\d+), fresh=([^, ]+), target=([^, ]+)',
                          (ROOT/f'cleanup-final-{case}.log').read_text())
        assert len(checks)==5 and all(float(a)<=float(b) for _,a,b in checks)
        rows=[]
        for step in range(5):
            row={'step':step}
            for stem,n in (('fields',15),('histories',9),('coefficients',7)):
                a=array(before,stem,step,n);b=array(after,stem,step,n)
                np.testing.assert_array_equal(a[:,:2],b[:,:2])
                error=float(np.max(abs(a-b)))
                assert error<=1e-8*max(np.max(abs(a)),1e-8)+1e-8
                row[stem+'_max_abs_difference']=error
            a=np.genfromtxt(before/f'clean_after_{step}_rank0.csv',delimiter=',',names=True,dtype=None,encoding=None)
            b=np.genfromtxt(after/f'clean_after_{step}_rank0.csv',delimiter=',',names=True,dtype=None,encoding=None)
            a=np.sort(a,order='id');b=np.sort(b,order='id')
            np.testing.assert_array_equal(a['id'],b['id'])
            error=max(float(np.max(abs(a[c]-b[c]))) for c in ('xx','yy','xy'))
            assert error<=1e-8*max(np.max(abs(a[c])) for c in ('xx','yy','xy'))+1e-8
            row['published_particle_stress_max_abs_difference']=error
            rows.append(row)
        result[case]=rows
    (ROOT/'cleanup_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
