"""Compare the single disposable FE-history solve to the accepted particle baseline."""
import csv
import json
import re
from pathlib import Path
import numpy as np
from scipy.linalg import solve_banded
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here=Path(__file__).resolve().parent
frozen=here/'history-mechanics-frozen-verified-local4'
out=here/'history-mechanics-solve-local4'
def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
base=rows(frozen/'surface_comparison.csv');new=rows(out/'surface_comparison.csv')
mass=rows(out/'noncommitting_surface.csv');original=rows(here/'normal-stress-complete-local4/fault_12.csv')
assert len(base)==len(new)==len(mass)==len(original)
d=np.array([float(r['mass_diagonal']) for r in mass]);u=np.array([float(r['mass_upper']) for r in mass])[:-1]
assert max(abs(d-np.array([float(r['mass_diagonal']) for r in original])))<1e-7
matrix=np.zeros((3,len(d)));matrix[1]=d;matrix[0,1:]=u;matrix[2,:-1]=u
arrays={}
for label,source,mode in [('baseline',base,'particle'),('fe_solve_fe',new,'fe'),('fe_solve_particle',new,'particle')]:
    arrays[label]={k:solve_banded((1,1),matrix,np.array([float(r[mode+'_'+k]) for r in source])) for k in ['q','sigma','R']}
    arrays[label]['p']=solve_banded((1,1),matrix,np.array([float(r['p']) for r in source]))
    arrays[label]['minus_tauN']=arrays[label]['sigma']-50e6-arrays[label]['p']
    arrays[label]['V']=np.array([float(r['V']) for r in source])
x=np.array([float(r['xd']) for r in base]);sel=(x>=37000)&(x<=43000)
summary={'nodes':[],'profiles':{},'raw_extrema':{}}
profile=[]
for i,r in enumerate(new):
    profile.append(dict(node=i,xd=x[i],**{label+'_'+k:float(v[i]) for label,a in arrays.items() for k,v in a.items()}))
    if 39699<x[i]<40101 or abs(x[i]-25000)<1:
        summary['nodes'].append(dict(node=i,xd=x[i],V_baseline=float(base[i]['V']),V_fe=float(r['V']),
           relative_rate_change=float(r['V'])/float(base[i]['V'])-1,
           baseline_R_density=float(base[i]['particle_R_density']),fe_R_density=float(r['fe_R_density']),
           particle_R_at_fe_solution=float(r['particle_R_density']),lower_active=int(mass[i]['lower_active'])))
with (out/'re_equilibrated_profiles.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(profile[0]));w.writeheader();w.writerows(profile)
for label,a in arrays.items():
    summary['profiles'][label]={k:dict(min=float(min(v[sel])),max=float(max(v[sel])),ptp=float(np.ptp(v[sel]))) for k,v in a.items()}
    summary['profiles'][label]['max_V_change_relative_global_peak']=float(max(abs(a['V']-arrays['baseline']['V']))/max(arrays['baseline']['V']))
for mode in ['particle','fe']:
    summary[mode+'_tensile_weight_at_fe_solution']={name:sum(float(r[mode+'_tensile_weight']) for r in new if lo<=float(r['xd'])<=hi)
      for name,(lo,hi) in {'junction':(39000,41000),'bottom':(114000,116000),'all':(-1,116000)}.items()}

# Both tensors at the new solution: exact linear history change, validated by
# the production dual-response weak fields. Selection is by the FE-mode extrema.
parent={}
for f in (here/'normal-stress-history-load-local4').glob('history_load_parents_rank*.csv'):
    for r in rows(f):
        diff=[float(r[c+'_working'])-float(r[c+'_particle']) for c in ['xx','yy','xy']]
        parent[int(r['id'])]=(diff,.9999998794215765)
raw=[]
for f in out.glob('stress_samples_12_rank*.csv'):
    for r in rows(f):
        diff,beta=parent[int(r['particle'])]
        deltaN=.75*diff[0]+.25*diff[1]-np.sqrt(3)/2*diff[2]
        deltaS=np.sqrt(3)/4*(diff[1]-diff[0])-.5*diff[2]
        xd=(.5e5*(1+1/np.sqrt(3))-float(r['surface_x']))*.5+(1e5-float(r['surface_y']))*np.sqrt(3)/2
        raw.append(dict(particle=int(r['particle']),xd=xd,xi=float(r['xi']),weight=float(r['weight']),
          p=float(r['delta_p']),fe_sigma=float(r['sigma_n']),particle_sigma=float(r['sigma_n'])+beta*deltaN,
          fe_q=float(r['q']),particle_q=float(r['q'])-beta*deltaS))
with (out/'dual_history_selected_samples.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
for name,(lo,hi) in {'junction':(39000,41000),'bottom':(114000,116000)}.items():
    select=[r for r in raw if lo<=r['xd']<=hi]
    summary['raw_extrema'][name]={k:dict(min=min(r[k] for r in select),max=max(r[k] for r in select)) for k in ['p','fe_sigma','particle_sigma','fe_q','particle_q']}
log=(out/'run.log').read_text()
checks=[]
for line in log.splitlines():
    if 'Fault linear solve:' in line:
        get=lambda k:float(re.search(r'\b'+k+r'=([^,\s]+)',line)[1])
        checks.append(dict(iterations=int(get('iterations')),estimated=get('estimated'),fresh=get('fresh'),target=get('target')))
assert checks and all(c['fresh']<=c['target'] for c in checks)
assert 'BP3 noncommitting rollback verified:' in log
summary['linear']={'directions':len(checks),'iterations':sum(r['iterations'] for r in checks),
                   'worst_fresh_target_ratio':max(r['fresh']/r['target'] for r in checks),'checks':checks}
summary['rollback_verified']=True
fig,axes=plt.subplots(4,1,figsize=(9,11),sharex=True)
for ax,k in zip(axes,['p','minus_tauN','sigma','V']):
    for label,style in [('baseline','-'),('fe_solve_fe','--'),('fe_solve_particle',':')]:
        if k=='V' and label=='fe_solve_particle':continue
        ax.plot(x[sel]/1000,arrays[label][k][sel]/(1 if k=='V' else 1e6),style,label=label)
    ax.axvline(40,color='gray',lw=.6);ax.grid(alpha=.3);ax.set_ylabel(k+(' (m/s)' if k=='V' else ' (MPa)'))
axes[0].legend(fontsize=8);axes[-1].set_xlabel('down-dip distance (km)')
fig.tight_layout();fig.savefig(out/'re_equilibrated_profiles.png',dpi=150)
(out/'solve_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['profiles','linear']},indent=2))
