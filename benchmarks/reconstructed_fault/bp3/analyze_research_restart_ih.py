"""Offline cold-restart Ih comparison; no altered solve or tolerance."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve_banded

HERE=Path(__file__).resolve().parent
REF=HERE/'fully-frictional-cleanup-local4'
RUN=HERE/'fully-frictional-restart-qualified-local4'


def profiles(root):
    rows=np.concatenate([np.genfromtxt(p,names=True,delimiter=',')
                         for p in sorted(root.glob('ih_bottom_completion_rank*.csv'))])
    rows.sort(order='id')
    return rows


def main():
    a,b=profiles(REF),profiles(RUN)
    for key in ('id','segment','xi','x','y','surface_weight','outside'):
        np.testing.assert_array_equal(a[key],b[key])
    delta=b['completed']-a['completed']
    n=int(a['segment'].max())+2
    # The linear projected difference is computed independently, keeping the
    # small RHS separate instead of subtracting two O(1e4 m) nodal solutions.
    diagonal=np.zeros(n);upper=np.zeros(n-1);rhs=np.zeros(n)
    for q,d in zip(a,delta):
        j=int(q['segment']);x=q['xi'];w=q['surface_weight']
        diagonal[j]+=w*(1-x)**2;diagonal[j+1]+=w*x*x
        upper[j]+=w*x*(1-x)
        rhs[j]+=w*(1-x)*d;rhs[j+1]+=w*x*d
    band=np.zeros((3,n));band[0,1:]=upper;band[1]=diagonal;band[2,:-1]=upper
    projected=solve_banded((1,1),band,rhs)
    index=int(np.argmax(np.abs(delta)))
    result=dict(profiles=len(a),changed_profiles=int(np.count_nonzero(delta)),
                largest_profile_difference_m=float(np.max(np.abs(delta))),
                scaled_profile_difference=float(np.max(np.abs(delta))/np.max(a['completed'])),
                largest_difference_profile=int(a['id'][index]),
                largest_difference_segment=int(a['segment'][index]),
                maximum_linear_projected_difference_m=float(np.max(np.abs(projected))),
                note='Projected difference is an offline linear impact, not an export of the failed constitutive call.')
    np.savetxt(RUN/'restart_ih_profile_difference.csv',
               np.column_stack([a['id'],a['segment'],a['x'],a['y'],a['completed'],b['completed'],delta]),
               delimiter=',',header='id,segment,x,y,reference,restarted,difference',comments='')
    np.savetxt(RUN/'restart_ih_linear_projected_difference.csv',
               np.column_stack([np.arange(n),projected]),delimiter=',',
               header='node,linear_projected_difference_m',comments='')
    (RUN/'restart_ih_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
