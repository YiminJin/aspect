"""Compare unreplaced physical weak rows, not projected nodal residual density."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
def rows(path):
    with path.open() as f:return list(csv.DictReader(f))
files=sorted(a.directory.glob('history_surface_rank*.csv'))
assert len(files)==4
parts=[rows(f) for f in files]
numeric=list(parts[0][0])[5:]
total=[]
for i,r in enumerate(parts[0]):
    out={k:float(r[k]) for k in list(r)[:5]}
    for part in parts:assert all(float(part[i][k])==out[k] for k in list(r)[:5])
    out.update({k:sum(float(part[i][k]) for part in parts) for k in numeric})
    out['xd']=(.5e5*(1+1/np.sqrt(3))-out['x'])*.5+(1e5-out['y'])*np.sqrt(3)/2
    for mode in ['particle','fe']:
        for k in ['q','sigma','C','friction','damping','R','Kdiag','Koff_sum']:
            out[mode+'_'+k+'_density']=out[mode+'_'+k]/out['weight']
        terms=out[mode+'_q']-out[mode+'_C']-out[mode+'_friction']-out[mode+'_damping']
        assert abs(terms-out[mode+'_R'])<1e-12*max(abs(out[mode+'_q']),abs(out[mode+'_friction']))
    for k in ['q','sigma','friction','R','Kdiag','Koff_sum']:
        out['delta_'+k+'_density']=out['fe_'+k+'_density']-out['particle_'+k+'_density']
    total.append(out)
with (a.directory/'surface_comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(total[0]));w.writeheader();w.writerows(total)
selected=[r for r in total if 39699<r['xd']<40101 or abs(r['xd']-25000)<1]
print('xd, V, q, C, friction, R (Pa test-weight mean), delta_q, delta_friction, delta_R, delta_R/|R|')
for r in selected:
    print(*(r[k] for k in ['xd','V','particle_q_density','particle_C_density','particle_friction_density','particle_R_density','delta_q_density','delta_friction_density','delta_R_density']),
          r['delta_R_density']/max(abs(r['particle_R_density']),1e-30),sep=', ')
bound=total[756]
summary={'bound':bound,'neighbors_and_control':selected,
         'interpretation':'R=q-C-friction-damping; lower-bound resisting reaction is -R for R<=0. Physical weak rows divided by positive test weight, not M^-1-projected nodal values.'}
if 'frozen' in a.directory.name:
    baseline=rows(Path(__file__).resolve().parent/'normal-stress-complete-local4/fault_12.csv')
    error=max(abs(r['particle_R']-float(baseline[i]['weak_residual'])) for i,r in enumerate(total))
    scale=max(abs(r['particle_q']) for r in total)
    summary['baseline_R_max_error']=error
    summary['baseline_R_error_relative_term_scale']=error/scale
    assert error/scale<1e-10
fd=a.directory/'history_tangent_fd.csv'
if fd.exists():
    data=rows(fd);summary['tangent_fd']=[dict(mode=int(r['mode']),h=float(r['h']),node=int(r['node']),
      relative_error=abs(float(r['finite_difference'])-float(r['minus_K']))/abs(float(r['minus_K']))) for r in data]
(a.directory/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
