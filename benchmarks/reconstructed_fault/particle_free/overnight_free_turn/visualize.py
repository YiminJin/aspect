#!/usr/bin/env python3
"""One accepted contour overlay, centerline/width chart and compact table."""
from observations import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
panels=[];table=[]
for case in ['A','B','C']:
 path=OUT/(case+'-comparison.json')
 if not path.exists():continue
 data=json.loads(path.read_text())
 if not data['comparisons']:continue
 last=data['comparisons'][-1]
 if case=='A':pairs=[(last['half'],'dt=0.0025 s','#4c72b0'),(last['quarter'],'dt=0.00125 s','#dd8452')]
 else:pairs=[(last['coarse_free'],'Free planes,128x128','#4c72b0'),(last['candidate'],'Horizontal planes' if case=='B' else 'Free planes,128x256','#55a868' if case=='B' else '#8172b3')]
 panels.append((case,last,pairs))
 for r,name,color in pairs:
  if case=='A':
   workkey='half_work_to_common_time' if r is last['half'] else 'quarter_work_to_common_time'
   errors=last[workkey];error_start=.075;difference=last['phase_error']['relative_increment']
  else:
   workkey='coarse_work_to_time' if r is last['coarse_free'] else 'candidate_work_to_time'
   errors=last[workkey];error_start=0.;difference=last['increment_error']['relative_field']
  table.append(dict(comparison=case,label=r['label'],description=name,time=r['time'],hy=r['hy'],dt=r['dt'],stage_time=r['stage_time'],front_x=r['front']['point'][0],front_y=r['front']['point'][1],advance_m=r['front']['dx'],lateral_m=r['front']['dy'],advance_cells=r['front']['dx_cells'],lateral_cells=r['front']['dy_cells'],verified_connection=r['front']['verified_seed_connection'],added_fracture=r['added_fracture'],central_W_over_hy=r['width'][0]['W_over_hn'],phase_error_reference_start=.075 if case=='A' else 0.,pair_relative_phase_increment_difference=difference,pair_late_endpoint_relative_error=last.get('late_endpoint_error',{}).get('relative_increment'),pair_late_increment_relative_error=last.get('late_increment_error',{}).get('relative_field'),integrated_slip=r['integrated_slip'],phase_KKT=r['phase_KKT'],work_interval_start=error_start,work_signed=errors['signed'],work_absolute=errors['absolute'],last_interval_physical_work_error=r['physical_work_error'],mixed_sign_sections=r['cancellation']['mixed_sections'],negative_local_rate=r['local_rate_negative']))
assert panels,'No completed common-time comparison'
fig,axes=plt.subplots(1,len(panels),figsize=(5.2*len(panels),4.8),squeeze=False,layout='constrained')
original=seed('first-turn-positive-prefix',1)
xmax=max(6.3,max(r['front']['point'][0]+.03 for _,_,pairs in panels for r,_,_ in pairs))
for ax,(case,last,pairs) in zip(axes.flat,panels):
 seg,_=contour_segments(original,32);ax.add_collection(LineCollection(seg,colors='black',linewidths=.9,linestyles=':',label='Original seed'))
 for r,name,color in pairs:
  f=field(r['label'],r['step'],r['hy']);seg,_=contour_segments(f,48)
  ax.add_collection(LineCollection(seg,colors=color,linewidths=1.4,label=name))
  ax.plot(*r['front']['point'],'o',ms=3,color=color)
 for x in np.arange(5.75,6.6,.0625):ax.axvline(x,color='.88',lw=.5,zorder=-2)
 for y in np.arange(1.75,2.3,.03125):ax.axhline(y,color='.88',lw=.5,zorder=-2)
 if case=='C':
  for y in np.arange(1.75+.015625,2.3,.03125):ax.axhline(y,color='.93',lw=.4,zorder=-2)
 ax.axvline(6,color='.4',ls='--',lw=.7);ax.axhline(2,color='.6',ls=':',lw=.6)
 ax.set(xlim=(5.8,xmax),ylim=(1.8,2.2),xlabel='x (m)',ylabel='y (m)',title=f'{case}: accepted t={last["time"]:g} s')
 ax.set_aspect('equal');ax.legend(loc='lower left',fontsize=7)
fig.suptitle(r'Connected absolute $\phi=0.05$ contours; native DG traces and physical mesh')
for ext in ['png','pdf']:fig.savefig(OUT/('contour-comparisons.'+ext),dpi=180)
plt.close(fig)
fig,axes=plt.subplots(2,len(panels),figsize=(5.2*len(panels),7),squeeze=False,layout='constrained')
for j,(case,last,pairs) in enumerate(panels):
 for r,name,color in pairs:
  valid=[p for p in r['centerlines'] if 'center' in p]
  axes[0,j].plot([p['x'] for p in valid],[p['lateral'] for p in valid],'o-',ms=3,color=color,label=name)
  axes[1,j].plot([p['x'] for p in valid],[p['width'] for p in valid],'o-',ms=3,color=color)
 axes[0,j].axhline(0,color='.5',ls=':',lw=.7)
 axes[0,j].set(title=f'{case}: t={last["time"]:g} s',ylabel='Contour midpoint minus 2 (m)')
 axes[1,j].set(xlabel='x beyond original endpoint (m)',ylabel='Upper/lower contour width (m)')
 for ax in axes[:,j]:ax.set_xlim(6,6.3);ax.grid(alpha=.2)
 axes[0,j].legend(fontsize=7)
 if not last.get('common_centerlines'):
  axes[0,j].text(.5,.5,'No common station beyond x=6',transform=axes[0,j].transAxes,ha='center',fontsize=9)
for ext in ['png','pdf']:fig.savefig(OUT/('centerline-width-comparisons.'+ext),dpi=180)
plt.close(fig)
with (OUT/'comparison-table.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
print('Saved',len(panels),'common-time panels and',len(table),'table rows.')
