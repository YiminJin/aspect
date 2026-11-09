#!/usr/bin/env python3
"""Plot the retained fixed-parent cost and common-history time comparison."""
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-tip-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from analyze_efficiency import ROOT,OUT,read
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
summary=json.loads((OUT/'time-comparison.json').read_text())['cases']
control=summary['efficiency-control02'];full=summary['efficiency-final01'];half=summary['efficiency-half03']
# These are disjoint outer scopes. Stokes/AMG inner timers are already included.
accounted=sum(control['timers'][k]['seconds'] for k in ('Tip exact search','Tip shifted search','Tip tangent evaluation','Tip trial evaluation','Assemble Stokes system','Build Stokes preconditioner'))
fig,axes=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
axes[0].bar(['Control\naccounted work','Retained\nstep timer'],[accounted,full['tip-profile']['step_seconds']],color=['C0','C1'])
axes[0].set(ylabel='seconds',title='Difficult .0575 → .06 step')
for x,y in enumerate([accounted,full['tip-profile']['step_seconds']]):axes[0].text(x,y+2,f'{y:.1f}',ha='center')
xx=np.arange(2)
for offset,key,title in [(-.18,'krylov_iterations','Coupled FGMRES'),(.18,'stokes_solves','Stokes calls')]:
 axes[1].bar(xx+offset,[control['tip-solver'][key],full['tip-solver'][key]],width=.35,label=title)
axes[1].set_xticks(xx,['Control','Retained']);axes[1].set(ylabel='count',title='Expensive inverse applications');axes[1].legend(fontsize=8)
axes[2].bar(xx-.18,[full['energy']['remainder']['signed'],half['energy']['remainder']['signed']],width=.35,label='Work remainder')
axes[2].bar(xx+.18,[full['energy']['added_fracture'],half['energy']['added_fracture']],width=.35,label='Added fracture')
axes[2].set_xticks(xx,['One full step','Two half steps']);axes[2].set(ylabel='J/m',title='Same accepted history and endpoint');axes[2].legend(fontsize=8)
fig.savefig(OUT/'cost-and-time.png',dpi=160);plt.close(fig)
planning=dict(control_accounted_seconds=accounted,retained_step_seconds=full['tip-profile']['step_seconds'],accounted_speedup=accounted/full['tip-profile']['step_seconds'],
             full_remainder_to_added_fracture=full['energy']['remainder']['signed']/full['energy']['added_fracture'],
             half_remainder_to_added_fracture=half['energy']['remainder']['signed']/half['energy']['added_fracture'],
             measured_work_error_ratio=full['energy']['remainder']['signed']/half['energy']['remainder']['signed'],
             active_work_definition='sum dt ds f V = friction + radiation (positive for this admitted law)',
             full_active_work=full['energy']['friction']['signed']+full['energy']['radiation']['signed'],
             half_active_work=half['energy']['friction']['signed']+half['energy']['radiation']['signed'])
planning['hypothetical_substeps_for_ten_percent_fracture']=2*planning['half_remainder_to_added_fracture']/.1
planning['hypothetical_dt']=.0025/planning['hypothetical_substeps_for_ten_percent_fracture']
planning['constant_measured_cost_hours']=planning['hypothetical_substeps_for_ten_percent_fracture']*half['tip-profile']['step_seconds']/2/3600
planning['warning']='Planning extrapolation only: two levels support local first-order error, not a 50-fold extrapolation or constant solver cost at much smaller dt.'
(OUT/'cost-planning.json').write_text(json.dumps(planning,indent=2)+'\n');print(json.dumps(planning,indent=2))
