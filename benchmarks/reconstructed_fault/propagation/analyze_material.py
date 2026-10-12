#!/usr/bin/env python3
"""Explicit numerical checks, compact comparisons and scientific plots for P2 material cases."""
import csv,json,math,os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-material-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
here=Path(__file__).resolve().parent;build=here.parents[2]/'build-propagation-integration';e=here/'material_evidence'
labels=['material_reference_a004','material_reference_a008','material_reference_a016','material_primary_a016','material_reference_a016_half']
def read(path):
 with path.open() as f: return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
def phase_grid(path):
 a=np.zeros((33,33))
 for r in read(path):a[round(32*r['x']),round(32*r['y'])]=r['phi']
 return a
def phi_at(field,x,y):
 i=np.minimum((32*np.asarray(x)).astype(int),31);j=np.minimum((32*np.asarray(y)).astype(int),31)
 s=32*np.asarray(x)-i;t=32*np.asarray(y)-j
 return (1-s)*((1-t)*field[i,j]+t*field[i,j+1])+s*((1-t)*field[i+1,j]+t*field[i+1,j+1])
def g(phi,m):return (1-phi)**2/((1-phi)**2+m*phi*(1+phi))
summary={};all_data={}
for label in labels:
 d=build/('output-'+label);record=json.loads((e/(label+'.json')).read_text());assert record['code']==0
 f=read(d/'feedback.csv');local=read(d/'material-local.csv');particles=read(d/'material-particles.csv');cal=read(d/'material-calibration.csv')
 assert len(f)==round(1/record['dt'])+1 and f[-1]['time']==1
 for r in f:
  assert 0<=r['phi_min']<=r['phi_max']<1 and r['phase_kkt']<=1e-11
  assert r['phase_healing']<=4.922e-12 and r['mechanical_residual']<=1e-8 and r['Ih_min']>0
  assert r['H_increment_integral']>=0 and r['offband_H_increases']==0
 all_data[label]=(f,local,particles)
 # The most damaged accepted state is selected from actual maxima, not assumed final.
 peak=max(range(len(f)),key=lambda i:f[i]['phi_max'])
 out=dict(case=record['case'],Gc=record['Gc'],a=record['amplitude'],dt=record['dt'],accepted_steps=len(f)-1,
          phi_initial=f[0]['phi_max'],phi_final=f[-1]['phi_max'],phase_max_gain=f[-1]['phi_max']-f[0]['phi_max'],
          max_phase_increment=max(r['phase_change'] for r in f),max_healing=max(r['phase_healing'] for r in f),
          max_KKT=max(r['phase_kkt'] for r in f),max_mechanical_residual=max(r['mechanical_residual'] for r in f),
          H_cumulative_integral=f[-1]['H_cumulative_integral'],phase_changing_steps=[int(r['step']) for r in f if r['phase_change']>4.922e-12],
          H_changing_steps=[int(r['step']) for r in f if r['H_increases']>0],calibration=cal,
          locations=read(d/f'material-phase-metrics-accepted-{peak}.csv')[0],normalization={},local={},candidate_probes=[])
 for target in [0.,.1,.5,.9,1.]:
  rows=[r for r in local if abs(r['x']-target)<1e-12]
  out['local'][str(target)]=dict(phi0=rows[0]['phi'],phi_end=rows[-1]['phi'],gain=rows[-1]['phi']-rows[0]['phi'],q0=rows[0]['q'],q_end=rows[-1]['q'],V_end=rows[-1]['V'],slip_end=rows[-1]['slip'])
 # Accuracy at initialization and maximally damaged accepted state. If no evolution,
 # also retain the final-state check rather than implying it changed.
 for step in sorted(set([0,peak,len(f)-1])):
  norm=read(d/f'material-normalization-accepted-{step}.csv');x=np.array([r['x'] for r in norm]);direct=np.array([r['direct'] for r in norm]);I=np.array([r['projected'] for r in norm])
  assert np.isfinite(direct).all() and min(direct)>0 and min(I)>0
  raw=read(d/f'material-phase-accepted-{step}.csv');m=cal[0]['m'];assert max(abs(g(r['phi'],m)-r['g']) for r in raw)<4e-16
  field=phase_grid(d/f'material-phase-accepted-{step}.csv')
  # Independent Gauss Q16/Q32 on exact bulk intervals, including side profiles.
  errors=[];quad_values=[]
  for order in [16,32]:
   z,w=leggauss(order);ys=(np.arange(32)[:,None]+(z+1)/2)/32
   ph=phi_at(field,x[:,None,None],ys[None,:,:]);integrand=m*ph*(1+ph)/(1-ph)**2
   values=np.sum(integrand*w[None,None,:]/64,axis=(1,2));quad_values.append(values)
   errors.append(float(max(abs(values/direct-1))))
  out['normalization'][str(step)]=dict(max_relative_I_error=float(max(abs(I/direct-1))),max_source_error=float(max(abs(direct/I-1))),
    relative_L1=float(np.trapezoid(abs(I-direct),x)/np.trapezoid(direct,x)),endpoint_source_ratios=[float(direct[0]/I[0]),float(direct[-1]/I[-1])],
    Q16_direct_max_relative=errors[0],Q32_direct_max_relative=errors[1],Q16_Q32_max_relative=float(max(abs(quad_values[0]/quad_values[1]-1))))
  assert errors[1]<1e-10
 # Replay native candidate at the three fixed parent probes, including the
 # incremental factor; parent observer already checks native replay for all parents.
 for target in [0.,.5,1.]:
  ps=[r for r in particles if r['target']==target]
  for row,oldrow in zip(ps[1:],ps[:-1]):
   step=int(row['step']);dt=f[step]['dt'];beta=math.exp(-dt*1e6/1e8);eta_ve=-1e8*math.expm1(-dt*1e6/1e8)
   fault=read(d/f'feedback-fault-{step}.csv');xs=[r['x'] for r in fault]
   q=float(np.interp(row['x'],xs,[r['q_committed'] for r in fault]));oldq=float(np.interp(row['x'],xs,[r['q_incoming'] for r in fault]))
   current_g=g(row['phi_point'],cal[0]['m']);old_g=g(oldrow['phi_point'],cal[0]['m'])
   b=beta*(1/old_g-1)*oldq/(1-current_g);a=q/current_g
   candidate=dt*(a-b)*(a+b)/(2*eta_ve)
   assert abs(max(row['H_in'],candidate)-row['H_new'])<=1e-10*max(1,row['H_new'])
   out['candidate_probes'].append(dict(time=row['time'],target=target,q=q,q_old=oldq,H0=row['H0'],H_in=row['H_in'],H_new=row['H_new'],candidate=candidate,candidate_over_incoming=candidate/row['H_in'],candidate_over_initial=candidate/row['H0']))
 sampling=read(d/'material-sampling.csv');out['max_cpdi_point_difference']=max(r['max_cpdi_point_difference'] for r in sampling)
 summary[label]=out
comparison={}
for target in [0.,.1,.5,.9,1.]:
 trajectories=[]
 for label in ['material_reference_a016','material_primary_a016']:
  rows=[r for r in all_data[label][1] if abs(r['x']-target)<1e-12];trajectories.append(rows)
 common=min(r[-1]['slip'] for r in trajectories)
 values=[]
 for rows in trajectories:
  values.append(dict(phase_gain=float(np.interp(common,[r['slip'] for r in rows],[r['phi']-rows[0]['phi'] for r in rows])),q=float(np.interp(common,[r['slip'] for r in rows],[r['q'] for r in rows]))))
 comparison[str(target)]=dict(common_slip=common,resolved_slip_range=common>1e-12,reference=values[0] if common>1e-12 else None,primary=values[1] if common>1e-12 else None)
coarse=summary['material_reference_a016'];fine=summary['material_reference_a016_half']
refinement=dict(coarse_phi_gain=coarse['phase_max_gain'],fine_phi_gain=fine['phase_max_gain'],coarse_H_increase=coarse['H_cumulative_integral'],fine_H_increase=fine['H_cumulative_integral'],common_times=[])
for r in all_data['material_reference_a016'][0]:
 s=next(v for v in all_data['material_reference_a016_half'][0] if v['time']==r['time'])
 refinement['common_times'].append(dict(time=r['time'],phi_coarse=r['phi_max'],phi_fine=s['phi_max'],H_change_coarse=r['H_cumulative_integral'],H_change_fine=s['H_cumulative_integral']))
(e/'summary.json').write_text(json.dumps(dict(cases=summary,common_slip=comparison,refinement=refinement,numerical_checks_pass=True,gradual_feedback_qualified=False,post_onset_broadening='not demonstrated; primary inactive',timestep_feedback_acceptable=False,optional_case_run=False),indent=2)+'\n')
# Compact exportable plots. Solid curves use accepted complete steps only.
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25})
selected=['material_reference_a016','material_primary_a016','material_reference_a016_half']
names=['Gc=26.67, dt=0.125','Gc=106.67, dt=0.125','Gc=26.67, dt=0.0625']
colors=['tab:blue','tab:orange','tab:green']
fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for label,name,color in zip(selected,names,colors):
 f,local,_=all_data[label]
 axes[0,0].plot([r['time'] for r in f],[r['phi_max']-f[0]['phi_max'] for r in f],'.-',label=name,color=color)
 for target,style,desc in [(0.,'-','endpoint'),(.5,'--','center')]:
  rows=[r for r in local if abs(r['x']-target)<1e-12]
  axes[0,1].plot([r['slip'] for r in rows],[r['phi']-rows[0]['phi'] for r in rows],style,color=color,label=name+', '+desc)
  axes[1,0].plot([r['slip'] for r in rows],[r['q'] for r in rows],style,color=color)
  axes[1,1].plot([r['time'] for r in rows],[r['V'] for r in rows],style,color=color)
axes[0,0].set(xlabel='Time (s)',ylabel='Increase of phase maximum');axes[0,0].legend(fontsize=8)
axes[0,1].set(xlabel='Accumulated local slip (m)',ylabel='Local phase increase');axes[0,1].legend(fontsize=6)
axes[1,0].set(xlabel='Accumulated local slip (m)',ylabel='Cohesive traction q (Pa)')
axes[1,1].set(xlabel='Time (s)',ylabel='Slip rate (m/s)')
fig.suptitle('Accepted P2 material trajectories; a=0.016 s$^{-2}$, nominal core=0.6')
fig.savefig(e/'material-feedback.png',dpi=180);fig.savefig(e/'material-feedback.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
for label,name,color in zip(selected,names,colors):
 f=all_data[label][0];step=int(f[-1]['step']);d=build/('output-'+label)
 initial=phase_grid(d/'material-phase-accepted-0.csv');final=phase_grid(d/f'material-phase-accepted-{step}.csv');x=np.linspace(0,1,33)
 axes[0].plot(x,final[:,16],'.-',color=color,label=name)
 axes[0].plot(x,initial[:,16],':',color=color)
 axes[1].plot(x,final[0,:],color=color,label=name+', endpoint')
 axes[1].plot(x,final[16,:],'--',color=color)
 norm=read(d/f'material-normalization-accepted-{step}.csv')
 axes[2].plot([r['x'] for r in norm],[r['normalized_source_integral']-1 for r in norm],color=color)
axes[0].set(xlabel='Along-fault x (m)',ylabel='Phase at y=0.5');axes[0].legend(fontsize=7)
axes[1].set(xlabel='Normal coordinate y (m)',ylabel='Final phase (solid endpoint; dashed center)')
axes[2].set(xlabel='Along-fault x (m)',ylabel='Integral of h/I projected minus 1')
fig.suptitle('Accepted final spatial fields; dotted along-fault curves show initialization')
fig.savefig(e/'material-spatial.png',dpi=180);fig.savefig(e/'material-spatial.pdf');plt.close(fig)
print(json.dumps({k:{p:v for p,v in s.items() if p in ['phase_max_gain','max_phase_increment','phase_changing_steps','H_cumulative_integral','normalization','max_cpdi_point_difference']} for k,s in summary.items()},indent=2))
