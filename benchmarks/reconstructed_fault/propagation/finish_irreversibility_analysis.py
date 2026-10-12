#!/usr/bin/env python3
"""Common physical times and slips, separate lag interpretation and old intervals."""
from pathlib import Path
import csv,json,os,re
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-irrev-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[3];b=root/'build-propagation-integration';e=Path(__file__).parent/'irreversibility_evidence'
def rows(p):return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(p.open())]
def field(p):
 v=np.zeros((33,33))
 for r in rows(p):v[round(32*r['x']),round(32*r['y'])]=r['phi']
 return v
new={};old={};comparisons={};energetics={}
for tag,dt in [('125',.125),('0625',.0625),('03125',.03125)]:
 p=b/('output-irreversibility_dt'+tag);o=b/('output-coupling_evolving_dt'+tag)
 f=rows(p/'feedback.csv');loc=rows(p/'material-local.csv');hist=rows(p/'material-particles.csv');en=rows(p/'recoverable-energy.csv');work=rows(p/'coupling-work.csv');of=rows(o/'feedback.csv');ol=rows(o/'material-local.csv')
 assert f[-1]['time']==1 and len(f)==round(1/dt)+1
 assert (p/'material-phase-accepted-0.csv').read_bytes()==(o/'material-phase-accepted-0.csv').read_bytes()
 new[tag]=(p,f,loc,hist,en,work);old[tag]=(o,of,ol)
 differences=[]
 for z in of:
  k=int(z['step']);v=field(p/f'material-phase-accepted-{k}.csv');ov=field(o/f'material-phase-accepted-{k}.csv');difference=v-ov
  nl=[r for r in loc if r['step']==k];olo=[r for r in ol if r['step']==k]
  differences.append(dict(time=z['time'],phase_inf=float(abs(difference).max()),core_phase_inf=float(abs(difference[:,16]).max()),q_relative=float(max(abs(r['q']-s['q']) for r,s in zip(nl,olo))/max(1,max(abs(s['q']) for s in olo))),slip_inf=float(max(abs(r['slip']-s['slip']) for r,s in zip(nl,olo)))))
 comparisons[tag]=dict(old_last_accepted_time=of[-1]['time'],common_interval_differences=differences)
 energetics[tag]=dict(final_cohesive_storage=en[-1]['fault_stored'],final_bulk_parent_storage=en[-1]['bulk_stored'],fracture_expenditure=en[-1]['fracture_change'],bulk_viscous=en[-1]['bulk_viscous'],cohesive_viscous=en[-1]['fault_viscous'],friction=en[-1]['friction'],cumulative_profile_work=sum(r['profile_work'] for r in work[1:]),cumulative_profile_integral=sum(r['profile_integral'] for r in work[1:]),cumulative_fault_storage_profile_jump=sum(r['fault_phase_jump'] for r in work[1:]),cumulative_FE_to_particle_energy=sum(r['mapped_particle_energy']-r['native_QP_energy'] for r in work[1:]),max_stress_transfer_relative=max(r['stress_transfer_relative'] for r in work))
common=[];lag=[]
for t in np.arange(.125,1.0001,.125):
 cases={}
 for tag,dt in [('125',.125),('0625',.0625),('03125',.03125)]:
  p,f,loc,hist,en,work=new[tag];z=next(r for r in f if r['time']==t);k=int(z['step']);
  cases[tag]=dict(phi_max=z['phi_max'],phase=field(p/f'material-phase-accepted-{k}.csv'),local=[r for r in loc if r['time']==t and r['x'] in [0,.5,1]],particle=[r for r in hist if r['time']==t],H_gain=z['H_cumulative_integral'])
 pairs=[]
 for a,c in [('125','0625'),('0625','03125')]:
  va,vc=cases[a]['phase'],cases[c]['phase'];la,lc=cases[a]['local'],cases[c]['local']
  pairs.append(dict(pair=[a,c],phase_inf=float(abs(va-vc).max()),core_phase_inf=float(abs(va[:,16]-vc[:,16]).max()),q_inf=max(abs(x['q']-y['q']) for x,y in zip(la,lc)),slip_inf=max(abs(x['slip']-y['slip']) for x,y in zip(la,lc)),H_gain_difference=abs(cases[a]['H_gain']-cases[c]['H_gain'])))
 for z in cases.values():del z['phase']
 common.append(dict(time=float(t),cases=cases,pair_differences=pairs))
# Interpret the sequential lag by comparing consumption through one common H time.
# These are explicitly NOT common-current-time convergence evidence.
for t in np.arange(.5,.8751,.125):
 cases={}
 for tag,dt in [('125',.125),('0625',.0625),('03125',.03125)]:
  p,f,loc,hist,en,work=new[tag];r=next(r for r in f if r['time']==t+dt);k=int(r['step']);cases[tag]=dict(phase_time=t+dt,consumed_history_time=t,phi_max=r['phi_max'],phase=field(p/f'material-phase-accepted-{k}.csv'))
 differences=[dict(pair=[a,c],phase_inf=float(abs(cases[a]['phase']-cases[c]['phase']).max()),maximum_difference=abs(cases[a]['phi_max']-cases[c]['phi_max'])) for a,c in [('125','0625'),('0625','03125')]]
 for z in cases.values():del z['phase']
 lag.append(dict(history_time=float(t),cases=cases,pair_differences=differences))
# Paired representation at common local slip over their overlapping ranges.
slip_comparison={}
for x in [0,.5,1]:
 histories={tag:[r for r in d[2] if r['x']==x] for tag,d in new.items()};end=min(rs[-1]['slip'] for rs in histories.values());slip_comparison[str(x)]=[]
 for s in np.linspace(0,end,17):
  cases={tag:{key:float(np.interp(s,[r['slip'] for r in rs],[r[key] for r in rs])) for key in ['time','phi','q','V']} for tag,rs in histories.items()}
  slip_comparison[str(x)].append(dict(slip=float(s),cases=cases,pair_phase_differences=[abs(cases[a]['phi']-cases[c]['phi']) for a,c in [('125','0625'),('0625','03125')]],pair_q_differences=[abs(cases[a]['q']-cases[c]['q']) for a,c in [('125','0625'),('0625','03125')]]))
(e/'refinement.json').write_text(json.dumps(dict(common_times=common,lag_interpretation=lag,common_local_slips=slip_comparison,old_accepted_interval_comparison=comparisons,energetics=energetics),indent=2)+'\n')
fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
for col,x in enumerate([0,.5,1]):
 for tag in new:
  f=new[tag][1];rs=[r for r in new[tag][2] if r['x']==x];oo=[r for r in old[tag][2] if r['x']==x];dt=f[1]['dt'];label=f'dt={dt}'
  line=axes[0,col].plot([r['slip'] for r in rs],[r['phi'] for r in rs],'.-',label=label)[0]
  axes[0,col].plot([r['slip'] for r in oo],[r['phi'] for r in oo],'--',alpha=.5,color=line.get_color())
  line=axes[1,col].plot([r['slip'] for r in rs],[r['q'] for r in rs],'.-',label=label)[0]
  axes[1,col].plot([r['slip'] for r in oo],[r['q'] for r in oo],'--',alpha=.5,color=line.get_color())
 for row,y in [(0,'Phase'),(1,'q (Pa)')]:axes[row,col].set(title=f'x={x} m; dashed=box accepted interval',xlabel='Local slip (m)',ylabel=y);axes[row,col].legend(fontsize=7);axes[row,col].grid(alpha=.2)
fig.savefig(e/'constraint-response.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
for tag,(_,f,loc,hist,en,work) in new.items():
 label=f"dt={f[1]['dt']}"
 for ax,key in [(axes[0],'fault_stored'),(axes[1],'fracture_change')]:ax.plot([r['time'] for r in en],[r[key] for r in en],'.-',label=label)
 axes[2].plot([r['time'] for r in work],np.cumsum([0]+[r['profile_work'] for r in work[1:]]),'.-',label=label+' bulk profile work')
 axes[2].plot([r['time'] for r in work],np.cumsum([0]+[r['fault_phase_jump'] for r in work[1:]]),'--',label=label+' fault storage profile jump')
for ax,title in zip(axes,['Cohesive storage','Fracture expenditure','Changing-profile terms (separate)']):ax.set(title=title,xlabel='Time (s)',ylabel='J/m out of plane');ax.legend(fontsize=6);ax.grid(alpha=.2)
fig.savefig(e/'energy-profile.png',dpi=170);plt.close(fig)
print(json.dumps(dict(final_common_time=common[-1],energetics=energetics,old_comparison={t:z['common_interval_differences'][-1] for t,z in comparisons.items()}),indent=2))

# Record the actual unchanged stopping scales and accuracy diagnostics.
allowances={}
for tag,(p,f,loc,hist,en,work) in new.items():
 log=(e/('irreversibility_dt'+tag+'.log')).read_text()
 targets=[float(x) for x in re.findall(r'Phase box target: initial=[^\n]* target=([^\n]+)',log)]
 roundoff=[float(x) for x in re.findall(r'Phase box target: initial=[^\n]* roundoff=([^ ]+)',log)]
 last=int(f[-1]['step']);m=rows(p/f'material-phase-metrics-accepted-{last}.csv')[0]
 normfiles=list(p.glob('material-normalization-*.csv'));source=max(abs(z['normalized_source_integral']-1) for file in normfiles for z in rows(file))
 raw=rows(p/f'material-normalization-accepted-{last}.csv');worst=max(raw,key=lambda z:abs(z['normalized_source_integral']-1))
 allowances[tag]=dict(target_min=min(targets),target_max=max(targets),roundoff_min=min(roundoff),roundoff_max=max(roundoff),cpdi_min=m['cpdi_min'],cpdi_max=m['cpdi_max'],max_normalization_source_error_all_states=source,final_worst_normalization_location=worst['x'],offband_H_increases=sum(z['offband_H_increases'] for z in f),offband_phi_final=f[-1]['offband_phi_max'],H_publication_error=max(z['candidate_commit_error'] for z in f))
(e/'numerical-allowances.json').write_text(json.dumps(allowances,indent=2)+'\n')
fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
for tag,(p,f,loc,hist,en,work) in new.items():
 v=field(p/f'material-phase-accepted-{int(f[-1]["step"])}.csv');initial=field(p/'material-phase-accepted-0.csv');xx=np.arange(33)/32;label=f"dt={f[1]['dt']}"
 axes[0].plot(xx,v[0],label=label);axes[1].plot(xx,v[16],label=label);axes[2].plot(xx,v[:,16],label=label)
for ax,title,x in zip(axes,['Transverse profile at side endpoint','Transverse profile at center','Fault-core phase along x'],['y (m)','y (m)','x (m)']):ax.set(title=title,xlabel=x,ylabel='Phase at t=1 s');ax.legend(fontsize=7);ax.grid(alpha=.2)
axes[0].plot(xx,initial[0],'k--',label='Initial relaxed');axes[1].plot(xx,initial[16],'k--');axes[2].plot(xx,initial[:,16],'k--');axes[0].legend(fontsize=7)
fig.savefig(e/'spatial-profiles.png',dpi=170);plt.close(fig)
