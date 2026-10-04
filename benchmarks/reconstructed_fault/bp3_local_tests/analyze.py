#!/usr/bin/env python3
"""Matched physical-coordinate comparisons; no numerical acceptance gate is relaxed."""
from pathlib import Path
import json,re,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
R=Path(__file__).resolve().parent;O=R/'results';O.mkdir(exist_ok=True)
V0=1e-3
checks={}
def read(path):return np.atleast_1d(np.genfromtxt(path,delimiter=',',names=True))
def joined(folder,pattern):return np.concatenate([read(p) for p in sorted(folder.glob(pattern))])
def probes(folder,step):
 a=joined(folder,f'velocity_{step}_rank*.csv');return a[np.lexsort((a['x'],a['y']))]
def difference(a,b):
 d=np.asarray(b)-np.asarray(a)
 return dict(rms=float(np.sqrt(np.mean(d*d))),maximum=float(np.max(abs(d))))
def rough(a):
 vals=[]
 for y in [250,500,750]:
  t=a[a['y']==y];u=np.column_stack((t['ux'],t['uy']))
  vals.extend(np.linalg.norm(u[4:-4]-(u[:-8]+u[8:])/2,axis=1))
 return dict(rms=float(np.sqrt(np.mean(np.array(vals)**2))),maximum=float(max(vals)))
summary={'scale_m_per_s':V0,'roughness':'center minus mean at x+-8m on the three raw 2m-spaced transects; not an exact error','coupled':{},'mesh_comparison':{},'transport':{}}
for label in ['A','B']:
 folder=R/f'output-{label}';accepted=read(folder/'accepted_steps.csv');motion=read(folder/'local_summary.csv');pop=read(folder/'particle_summary.csv')
 log=(R/f'evidence/{label}-np2.log').read_text();fresh=[tuple(map(float,x)) for x in re.findall(r'fresh=([\deE+.\-]+), target=([\deE+.\-]+)',log)]
 checks[label+'_20_steps']=len(accepted)==21 and int(accepted[-1]['step'])==20
 checks[label+'_fixed_dt']=bool(np.allclose(accepted['dt'][1:],.05,rtol=1e-12,atol=0))
 checks[label+'_fresh_residuals']=bool(fresh) and all(a<=b for a,b in fresh)
 checks[label+'_no_cutbacks']='Repeating the current time step' not in log and 'nonlinear solver failures occurred' not in log and np.all(accepted['min_alpha']==1)
 checks[label+'_H_positive']=bool(min(pop['H_min'])>0)
 checks[label+'_nonzero_history']=bool(accepted[-1]['max_committed_stress_Pa']>0)
 stats=json.loads((R/f'evidence/{label}-np2.json').read_text())
 dofs=re.search(r'Number of degrees of freedom: ([\d,]+) \(([\d,]+)\+([\d,]+)',log)
 summary['coupled'][label]=dict(cells=int(re.search(r'BP3 mesh geometry: cells=(\d+)',log)[1]),stokes_dofs=int(dofs[2].replace(',',''))+int(dofs[3].replace(',','')),particles=int(pop[0]['particles']),seconds=stats['seconds'],max_rank_child_RSS_KiB=stats['child_max_rss_kib'],max_V=float(max(accepted['max_V'])),max_sampled_bulk_speed=float(max(motion['max_probe_u'])),max_Cp=float(max(motion['Cp_max'])),max_Ap=float(max(motion['Ap_max'])),max_log_state_change=float(max(motion['max_delta_log_Theta'])),max_S=float(max(accepted['max_step_slip_over_Dc'])),max_stress=float(max(accepted['max_committed_stress_Pa'])),max_nonlinear_residual=float(max(accepted['normalized_nonlinear_residual'])),max_fresh_over_target=max(a/b for a,b in fresh),krylov_total=int(sum(accepted['krylov_iterations'])),births=int(sum(pop['births_since_backup'])),losses=int(sum(pop['removed_or_exited_since_backup'])),population_range=[int(min(motion['pop_min'])),int(max(motion['pop_max']))])
 fault_states=joined(folder,'fault_*.csv')
 actual_samples=joined(folder,'friction_samples_*_rank*.csv')
 summary['coupled'][label].update(Theta_range=[float(min(fault_states['Theta'])),float(max(fault_states['Theta']))],Omega_range=[float(min(fault_states['Omega'])),float(max(fault_states['Omega']))],raw_normal_sample_range=[float(min(actual_samples['raw_normal'])),float(max(actual_samples['raw_normal']))],actual_normal_sample_range=[float(min(actual_samples['actual_normal'])),float(max(actual_samples['actual_normal']))],active_dt_limiter='imposed maximum time step .05 s; final end-time floating-point truncation only')
 for name in ['accepted_steps.csv','local_summary.csv','particle_summary.csv','restored_growth.csv']:
  shutil.copy2(folder/name,O/f'{label}-{name}')
 for step in [0,1,20]:
  shutil.copy2(folder/f'fault_{step}.csv',O/f'{label}-fault_{step}.csv')
for step in [0,1,20]:
 a=read(R/f'output-A/fault_{step}.csv');b=read(R/f'output-B/fault_{step}.csv');assert np.max(abs(a['s']-b['s']))<1e-8
 result={}
 for name in ['V','Theta','raw_normal_Q1','friction_normal_Q1']:
  result[name]=difference(a[name],b[name])
 result['V']['rms_over_Vtest']=result['V']['rms']/V0
 result['active_nodes']=int(sum((a['V']>1e-5)&(b['V']>1e-5)))
 for region,mask in [('endpoints',(a['s']<200)|(a['s']>a['s'].max()-200)),('interior',(a['s']>=200)&(a['s']<=a['s'].max()-200))]:result[region]=difference(a['V'][mask],b['V'][mask])
 # Local constitutive sensitivity only; Q1 weak tractions are not pointwise
 # partners of nodal V and this is NOT a discrete velocity-difference prediction.
 susceptibility=a['mu']/(a['friction_normal_Q1']*a['mu_V']+4624440.)
 result['frozen_local_normal_sensitivity_m_s_Pa_max']=float(max(abs(susceptibility)))
 result['normal_only_local_sensitivity_bound_m_s']=float(max(abs(susceptibility*(b['friction_normal_Q1']-a['friction_normal_Q1']))))
 pa,pb=probes(R/'output-A',step),probes(R/'output-B',step)
 checks[f'matched_probes_{step}']=bool(np.max(abs(pa['x']-pb['x']))<1e-8 and np.max(abs(pa['y']-pb['y']))<1e-8)
 result['bulk_u']=difference(np.column_stack([pa['ux'],pa['uy']]),np.column_stack([pb['ux'],pb['uy']]))
 result['roughness_A']=rough(pa);result['roughness_B']=rough(pb)
 for label,field in [('A',a),('B',b)]:
  c=field['V'][1:-1]-.5*(field['V'][:-2]+field['V'][2:]);inside=(field['s'][1:-1]>200)&(field['s'][1:-1]<field['s'].max()-200)
  result['fault_interior_chord_'+label]=dict(rms=float(np.sqrt(np.mean(c[inside]**2))),maximum=float(max(abs(c[inside]))))
 for label,p in [('A',pa),('B',pb)]:
  uniform=[];interface=[]
  for y in [250,500,750]:
   z=p[p['y']==y];u=np.column_stack([z['ux'],z['uy']]);v=np.linalg.norm(u[4:-4]-.5*(u[:-8]+u[8:]),axis=1)
   r=np.sqrt(3)/2*z['x'][4:-4]+.5*(y-500)
   # Exclude the physical loading transition from this exterior roughness.
   for j,value in enumerate(v):
    if abs(r[j])<60:continue
    (uniform if np.all(z['h'][j:j+9]==z['h'][j+4]) else interface).append(value)
  result['bulk_exterior_roughness_'+label]={name:dict(rms=float(np.sqrt(np.mean(np.array(vals)**2))),maximum=float(max(vals))) for name,vals in [('uniform',uniform),('interface',interface)]}
 af=read(R/f'output-A/profiles/fault_{step}.csv');bf=read(R/f'output-B/profiles/fault_{step}.csv')
 for name in ['slip_m','q_weak_Pa','sigma_n_weak_Pa']:result[name]=difference(af[name],bf[name])
 summary['mesh_comparison'][str(step)]=result
# Compact matched-node table; tractions here are the weak Q1 projection.
import csv
with (O/'matched_locations.csv').open('w') as stream:
 writer=csv.writer(stream);writer.writerow(['step','s_m','V_A','V_B','B_minus_A_V','B_minus_A_Theta','B_minus_A_shear_Q1','B_minus_A_friction_normal_Q1','B_minus_A_slip'])
 for step in [0,1,20]:
  a=read(R/f'output-A/fault_{step}.csv');b=read(R/f'output-B/fault_{step}.csv');fa=read(R/f'output-A/profiles/fault_{step}.csv');fb=read(R/f'output-B/profiles/fault_{step}.csv')
  indices=sorted(set([int(np.argmax(abs(b['V']-a['V']))),0,len(a)//4,len(a)//2,3*len(a)//4,len(a)-1]))
  for i in indices:writer.writerow([step,a['s'][i],a['V'][i],b['V'][i],b['V'][i]-a['V'][i],b['Theta'][i]-a['Theta'][i],fb['q_weak_Pa'][i]-fa['q_weak_Pa'][i],b['friction_normal_Q1'][i]-a['friction_normal_Q1'][i],fb['slip_m'][i]-fa['slip_m'][i]])
  qa=joined(R/'output-A',f'friction_samples_{step}_rank*.csv');qb=joined(R/'output-B',f'friction_samples_{step}_rank*.csv')
  qa=qa[np.lexsort((qa['x'],qa['y']))];qb=qb[np.lexsort((qb['x'],qb['y']))]
  checks[f'matched_actual_friction_points_{step}']=len(qa)==len(qb) and np.max(abs(qa['x']-qb['x']))<1e-8 and np.max(abs(qa['y']-qb['y']))<1e-8
  summary['mesh_comparison'][str(step)]['actual_friction_normal_at_matched_QPs']=difference(qa['actual_normal'],qb['actual_normal'])
  checks[f'actual_friction_compression_{step}']=min(qa['actual_normal'])>0 and min(qb['actual_normal'])>0
checks['provisional_1percent_RMS_screen']=summary['mesh_comparison']['20']['V']['rms_over_Vtest']<.01
# Native restart qualification and abrupt change from identical checkpoint.
base=R/'output-A-replay';small=R/'output-A-small-dt'
for step in [10,11]:
 for name in [f'fault_{step}.csv',f'velocity_{step}_rank0.csv',f'velocity_{step}_rank1.csv']:
  checks['replay_'+name]=(base/name).read_bytes()==(R/'output-A'/name).read_bytes()
for rank in [0,1]:
 for kind in ['bulk.csv','particles.csv','rng.txt']:
  name=f'common_restart_rank{rank}_{kind}';checks['identical_'+name]=(base/name).read_bytes()==(small/name).read_bytes()
a=read(base/'fault_11.csv');b=read(small/'fault_17.csv')
pa,pb=probes(base,11),probes(small,17)
summary['timestep_change']={name:difference(a[name],b[name]) for name in ['V','Theta','raw_normal_Q1','friction_normal_Q1']}
summary['timestep_change'].update(bulk_u=difference(np.column_stack([pa['ux'],pa['uy']]),np.column_stack([pb['ux'],pb['uy']])),roughness_baseline=rough(pa),roughness_small=rough(pb))
# Integrate the recorded right-endpoint step velocities over the same interval;
# compare dt*u sums, not the unequal last-step increments alone.
base_displacement=sum(np.column_stack([probes(base,k)['dt_ux'],probes(base,k)['dt_uy']]) for k in [10,11])
small_displacement=sum(np.column_stack([probes(small,k)['dt_ux'],probes(small,k)['dt_uy']]) for k in range(10,18))
summary['timestep_change']['integrated_probe_displacement']=difference(base_displacement,small_displacement)
fa=read(base/'profiles/fault_11.csv');fb=read(small/'profiles/fault_17.csv')
summary['timestep_change']['cumulative_slip']=difference(fa['slip_m'],fb['slip_m'])
summary['timestep_change']['early_roughness']={str(k):rough(probes(small,k)) for k in range(10,18)}
for label,folder in [('base',base),('small',small)]:
 logfiles=list((R/'evidence').glob(('A-replay-v2' if label=='base' else 'A-small-dt')+'-np2.log'))
 log=logfiles[0].read_text();fresh=[tuple(map(float,x)) for x in re.findall(r'fresh=([\deE+.\-]+), target=([\deE+.\-]+)',log)]
 checks[label+'_fresh_checks']=bool(fresh) and all(x<=y for x,y in fresh)
 checks[label+'_no_cutbacks']='Repeating the current time step' not in log and 'nonlinear solver failures occurred' not in log
 samples=joined(folder,'friction_samples_*_rank*.csv')
 summary['timestep_change'][label+'_constitutive']=dict(eta_ve_range=[float(min(samples['eta_ve'])),float(max(samples['eta_ve']))],beta_range=[float(min(samples['beta'])),float(max(samples['beta']))],actual_normal_range=[float(min(samples['actual_normal'])),float(max(samples['actual_normal']))])
 checks[label+'_positive_actual_normal']=bool(min(samples['actual_normal'])>0)
for label,folder,step in [('base',base,11),('small',small,17)]:
 ac=read(folder/'accepted_steps.csv');mo=read(folder/'local_summary.csv')
 summary['timestep_change'][label]=dict(last_time=float(ac[-1]['time']),max_Cp=float(max(mo['Cp_max'])),max_Ap=float(max(mo['Ap_max'])),max_delta_stress=float(max(mo['max_delta_stress'])),last_dt=float(ac[-1]['dt']))
 for name in ['accepted_steps.csv','local_summary.csv','restored_growth.csv']:shutil.copy2(folder/name,O/f'{label}-{name}')
 for filename in folder.glob('fault_*.csv'):shutil.copy2(filename,O/f'{label}-{filename.name}')
checks['eight_small_steps']=len(read(small/'local_summary.csv'))==8
checks['common_end_time']=abs(summary['timestep_change']['base']['last_time']-summary['timestep_change']['small']['last_time'])<1e-12
for step in [10,11]:
 for name in [f'fault_{step}.csv',f'velocity_{step}_rank0.csv',f'velocity_{step}_rank1.csv']:
  checks['final_build_'+name]=(R/'output-A-replay-final'/name).read_bytes()==(R/'output-A'/name).read_bytes()
# All bulk fields at the same DoFs, same mesh/partition. Particle membership and
# property/position differences are reported rather than interpolated away.
for kind in ['bulk','particles']:
 va=[];vb=[]
 for rank in [0,1]:
  kwargs={'delimiter':',','skiprows':1 if kind=='bulk' else 0}
  va.extend(np.loadtxt(base/f'common_end_rank{rank}_{kind}.csv',**kwargs));vb.extend(np.loadtxt(small/f'common_end_rank{rank}_{kind}.csv',**kwargs))
 va=np.array(va);vb=np.array(vb);va=va[np.argsort(va[:,0])];vb=vb[np.argsort(vb[:,0])]
 checks['common_'+kind+'_membership']=bool(np.array_equal(va[:,0],vb[:,0]))
 summary['timestep_change'][kind+'_column_max_abs_differences']=np.max(abs(va-vb),axis=0).tolist()
# Mesh / raw FE velocity and fault plots, same scales and coordinates.
fig,axes=plt.subplots(2,2,figsize=(10,7))
for j,label in enumerate(['A','B']):
 mesh=joined(R/f'output-{label}','mesh_rank*.csv');edges=[]
 for x,y,h in zip(mesh['x'],mesh['y'],mesh['h']):
  if abs(y-500)<160 and abs(x)<300:edges.extend([[(x-h/2,y-h/2),(x+h/2,y-h/2)],[(x-h/2,y-h/2),(x-h/2,y+h/2)]])
 axes[0,j].add_collection(LineCollection(edges,colors='k',linewidths=.2));axes[0,j].set(xlim=(-300,300),ylim=(350,650),title=f'{label}: mesh, central interface window',aspect='equal')
 p=probes(R/f'output-{label}',20);m=p['y']==500
 axes[1,0].plot(p['x'][m],p['ux'][m]/V0,label=label)
 f=read(R/f'output-{label}/fault_20.csv');axes[1,1].plot(f['s'],f['V']/V0,label=label)
axes[1,0].set(xlabel='x (m), y=500 m',ylabel='FE ux / Vtest');axes[1,1].set(xlabel='fault s (m)',ylabel='fault V / Vtest')
for ax in axes[1]:ax.legend();ax.grid(alpha=.3)
fig.tight_layout();fig.savefig(O/'mesh_velocity.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(8,6))
for label,folder,step in [('constant dt',base,11),('dt/4',small,17)]:
 p=probes(folder,step);m=p['y']==500;axes[0].plot(p['x'][m],p['ux'][m]/V0,label=label)
 f=read(folder/f'fault_{step}.csv');axes[1].plot(f['s'],f['V']/V0,label=label)
for ax in axes:ax.legend();ax.grid(alpha=.3)
axes[0].set(xlabel='x (m), y=500 m',ylabel='FE ux / Vtest at common time');axes[1].set(xlabel='fault s (m)',ylabel='V / Vtest at common time')
fig.tight_layout();fig.savefig(O/'timestep_change.png',dpi=150);plt.close(fig)
# Updated transport diagnostics distinguish ID reuse from true motion.
for label,dirname in [('A','output-transport-A-final'),('B','output-transport-B-final'),('A-serial','output-transport-A-serial-final')]:
 folder=R/dirname;rows=[]
 for step in range(5):
  a=joined(folder,f'transport_{step}_rank*.csv')
  checks[f'{label}_shadow_exact_{step}']=bool(max(a['shadow_particle_max_error'])==0)
  checks[f'{label}_newborn_H_{step}']=bool(max(a['newborn_H_max_error'])==0 and min(a['H_min'])>0)
  checks[f'{label}_survivor_H_{step}']=bool(max(a['survivor_H_max_change'])==0)
  checks[f'{label}_tensor_constraints_{step}']=bool(max(a['tensor_constraint_max'])<1e-6)
  entry=dict(step=step,particles=int(sum(a['n'])),births=int(sum(a['born'])),reused_id_births=int(sum(a['reused_id_births'])),crossings=int(sum(a['crossings'])),max_Cp=float(max(a['Cp_max'])),max_Ap=float(max(a['Ap_max'])),newborn_stress_max_error=float(max(a['newborn_stress_max_error'])),retained_stress_max_error=float(max(a['retained_stress_max_error'])),population_min=int(min(a['n'])),population_max=int(max(a['n'])))
  for region,mask in [('boundary',a['boundary']>0),('interface_interior',(a['interface']>0)&(a['boundary']==0)),('uniform_fine_interior',(a['interface']==0)&(a['boundary']==0)&(a['h']==3.90625)),('uniform_coarse_interior',(a['interface']==0)&(a['boundary']==0)&(a['h']>3.90625))]:
   z=a[mask];entry[region]={name:float(np.sqrt(np.average(z[name]**2,weights=z['h']**2))) for name in ['LLS_stored_rms','LLS_shadow_rms','Q2_stored_rms','Q2_shadow_rms']};entry[region]['shadow_max']=float(max(z['Q2_shadow_max']))
  entry['losses_or_outflow']=0 if not rows else rows[-1]['particles']+entry['births']-entry['particles']
  rows.append(entry)
 summary['transport'][label]=rows
 checks[label+'_interface_crossing']=sum(x['crossings'] for x in rows)>0
 checks[label+'_fractional_cell_motion']=.25<rows[-1]['max_Ap']<.51
 for f in folder.glob('reused_ids_*'):shutil.copy2(f,O/f'{label}-{f.name}')
fig,ax=plt.subplots(figsize=(8,4))
for label in ['A','B']:
 for region in ['uniform_fine_interior','interface_interior']:
  entries=summary['transport'][label];ax.plot([e['max_Ap'] for e in entries],[e[region]['Q2_shadow_rms']/1e8 for e in entries],marker='o',label=label+' '+region)
ax.set(xlabel='maximum surviving accumulated path / incoming cell edge',ylabel='exact-shadow Q2 RMS / 1e8 Pa',yscale='log');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(O/'transport.png',dpi=150);plt.close(fig)
summary['simulation_seconds']=sum(json.loads(p.read_text())['seconds'] for p in (R/'evidence').glob('*-np*.json'))
summary['checks_passed']=sum(bool(v) for v in checks.values());summary['checks_total']=len(checks)
(O/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n');(O/'checks.json').write_text(json.dumps({k:bool(v) for k,v in checks.items()},indent=2)+'\n')
print(json.dumps(summary,indent=2))
assert all(checks.values()),[k for k,v in checks.items() if not v]
