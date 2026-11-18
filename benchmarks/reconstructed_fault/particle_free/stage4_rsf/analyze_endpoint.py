#!/usr/bin/env python3
"""Matched Stage 4 endpoint, retaining the old registration selection exactly."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,csv,hashlib
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'overnight_free_turn'))
from observations import *
OUT=HERE.parent/'evidence/stage4_rsf'
label='stage4-free-turn-fine-endpoint';path=RUNS/label/'output'
manifest=json.loads((path.parent/'command.json').read_text())
assert 'exit_code' in manifest,'Do not analyze open native output'
base=HERE.parent/'evidence/overnight_free_turn'
registration=json.loads((base/'C-registration-comparison.json').read_text())
changed=registration['first_registration_time_changed']['sections']
keys=sorted(r['section'] for r in changed);assert len(keys)==12
accepted=read(path/'bulk-accepted.csv') if (path/'bulk-accepted.csv').exists() else []
reached=bool(len(accepted))
if reached:
 assert len(accepted)==1 and abs(accepted[0].time-.08)<1e-12
step=17 if reached else 16
fine_label=label if reached else 'overnight-free-turn-c-half'
physical_path=RUNS/fine_label/'output'
common_time=.08 if reached else .0775
initial={'label':'overnight-free-turn-c-pilot','field':seed('overnight-free-turn-c-pilot',1,.015625)}
initial_c={'label':'first-turn-positive-prefix','field':seed('first-turn-positive-prefix',1)}
fine,fp=endpoint(fine_label,step,.015625,initial)
coarse,cp=endpoint('recover-free-turn-positive-half',step,.03125,initial_c)
parent=field('overnight-free-turn-c-half',16,.015625)
assert np.min(fp.coefficients-parent.coefficients)>=-1e-12
s=read(physical_path/f'tip-sections-{step:04d}.csv');c=read(RUNS/'recover-free-turn-positive-half/output'/f'tip-sections-{step:04d}.csv')
f=read(physical_path/f'tip-planes-{step:04d}.csv');p=read(RUNS/'recover-free-turn-positive-half/output'/f'tip-planes-{step:04d}.csv')
pf=read(RUNS/'overnight-free-turn-c-half/output/tip-planes-0016.csv')
assert np.max(abs(f.angle[pf.formed>0]-pf.angle[pf.formed>0]))<1e-12
assert np.max(abs(f.ds-f.reference_dx/np.cos(f.angle)))<1e-12
assert min(s.age)>0 and fine['phase_KKT']<=1e-7 and fine['mechanical']<=1e-9
clock=read(path/'restore-clock.csv')[0]
assert clock.accepted_step==16 and abs(clock.accepted_time-.0775)<1e-12 and abs(clock.pending_dt-.0025)<1e-12
initialplanes=read(RUNS/'first-turn-positive-prefix/output/tip-planes-0001.csv')
old=initialplanes.parent_formed>0
rows=[]
for k in keys:
 rows.append(dict(section=k,x=float(s.s[k]),fine_slip=float(s.slip[k]),coarse_slip=float(c.slip[k]),fine_age=float(s.age[k]),coarse_age=float(c.age[k]),fine_V=float(s.V[k]),coarse_V=float(c.V[k]),fine_angle=float(f.angle[k]),coarse_angle=float(p.angle[k]),fine_ds=float(s.ds[k]),coarse_ds=float(c.ds[k])))
with (OUT/'affected-registration-sections.csv').open('w') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
shares={}
for name,sections in [('fine',s),('coarse',c)]:
 weighted=abs(sections.slip)*sections.ds
 shares[name]=dict(selected_integrated_absolute_slip=float(sum(weighted[keys])),fraction_of_new_section_slip=float(sum(weighted[keys])/sum(weighted[~old])),fraction_of_all_slip=float(sum(weighted[keys])/sum(weighted)))
# Also retain the exact historical paired-registration weighting convention.
paired=[r['section'] for r in registration['all_common']]
new_paired=[r['section'] for r in registration['all_common'] if not (r['coarse']['initial_seed'] and r['fine']['initial_seed'])]
averaged=.5*(abs(s.slip)*s.ds+abs(c.slip)*c.ds)
shares['paired_historical_convention']=dict(fraction_of_new_section_slip=float(sum(averaged[keys])/sum(averaged[new_paired])),fraction_of_all_slip=float(sum(averaged[keys])/sum(averaged[paired])))
common=[]
for a,b in zip(fine['centerlines'],coarse['centerlines']):
 if 'center' in a and 'center' in b:common.append(dict(x=a['x'],fine_center=a['center'],coarse_center=b['center'],difference=a['center']-b['center'],fine_width=a['width'],coarse_width=b['width']))
result=dict(endpoint_reached=reached,run_exit_code=manifest['exit_code'],seconds=manifest['seconds'],common_time=common_time,fine=fine,coarse=coarse,
 front_difference={key:fine['front'][key]-coarse['front'][key] for key in ['dx','dy']},common_centerlines=common,
 phase_error=l2(cp,fp,parent=initial_c['field']),affected_sections=rows,affected_slip_shares=shares,
 section_error={key:float(max(abs(s[key]-c[key]))) for key in ['age','slip','V']},
 fine_work=work(['overnight-free-turn-c-pilot','overnight-free-turn-c-prefix','overnight-free-turn-c-half']+([label] if reached else [])),
 coarse_work=work(['first-turn-positive-prefix','first-turn-positive-end','first-turn-positive-fourcore-end','recover-free-turn-positive-half']))
for name in ['fine_work','coarse_work']:
 rows=[r for r in result[name]['steps'] if r['time']<=common_time+1e-12]
 result[name]=dict(steps=rows,signed=sum(r['error'] for r in rows),absolute=sum(abs(r['error']) for r in rows))
import re
trials=re.findall(r'Bulk tip coupled Newton (\d+): mechanics=([^,]+), phase KKT=([^,]+)',(path.parent/'run.log').read_text())
result['last_trial']=None if not trials else dict(iteration=int(trials[-1][0]),mechanics=float(trials[-1][1]),phase_KKT=float(trials[-1][2]))
result['physical_data_note']='New accepted .080 endpoint' if reached else 'No new accepted endpoint; retained .0775 data reused. Trial fields are excluded.'
(OUT/'endpoint-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
fig,axs=plt.subplots(1,3,figsize=(13,4.2))
for obj,color,name in [(initial_c['field'],'0.6','seed'),(cp,'C0','128×128'),(fp,'C1','128×256')]:
 seg,_=contour_segments(obj,32);axs[0].add_collection(LineCollection(seg,colors=color,linewidths=1.4,label=name))
axs[0].set(xlim=(5.8,6.4),ylim=(1.75,2.25),xlabel='x (m)',ylabel='y (m)',title=f'Absolute phase=.05 at {common_time:.4f} s');axs[0].set_aspect('equal');axs[0].legend()
right=[r for r in result['affected_sections'] if r['x']>6]
for prefix,color in [('coarse','C0'),('fine','C1')]:
 axs[1].plot([r['x'] for r in right],[r[prefix+'_slip']*1e6 for r in right],'o-',color=color,label=prefix)
 axs[2].plot([r['x'] for r in right],[r[prefix+'_angle']*180/np.pi for r in right],'o-',color=color,label=prefix)
axs[1].set(xlabel='Section x (m)',ylabel='Cumulative slip (µm)',title='Earlier-registration labels: right end')
axs[2].set(xlabel='Section x (m)',ylabel='Frozen angle (degrees)',title='Preserved orientation memory')
for ax in axs[1:]:ax.legend();ax.grid(alpha=.2)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(OUT/('endpoint-comparison.'+ext),dpi=180)
print({key:result[key] for key in ['endpoint_reached','seconds','front_difference','affected_slip_shares','section_error']})
