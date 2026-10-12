#!/usr/bin/env python3
"""Qualification and separate energy/transfer diagnostics; never retune a trajectory."""
import csv,json,math,os
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-recoverable-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
here=Path(__file__).resolve().parent;build=here.parents[2]/'build-propagation-integration';e=here/'recoverable_evidence'
def rows(p):
 with p.open() as f:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
def field(p):
 v=np.zeros((33,33))
 for r in rows(p):v[round(32*r['x']),round(32*r['y'])]=r['phi']
 return v
def at(v,x,y):
 x=np.asarray(x);y=np.asarray(y);i=np.minimum((x*32).astype(int),31);j=np.minimum((y*32).astype(int),31);s=x*32-i;t=y*32-j
 return (1-s)*((1-t)*v[i,j]+t*v[i,j+1])+s*((1-t)*v[i+1,j]+t*v[i+1,j+1])
def integral(v,x,m):
 a=1-at(v,np.asarray(x)[...,None],np.arange(32)/32);b=1-at(v,np.asarray(x)[...,None],np.arange(1,33)/32);assert np.min(a)>0 and np.min(b)>0
 z=(b-a)/a;ratio=np.ones_like(z);np.divide(np.log1p(z),z,out=ratio,where=z!=0)
 return np.sum(m/32*(2/(a*b)-3*ratio/a+1),axis=-1)
def norm_check(v,norm,m):
 x=np.array([r['x'] for r in norm]);direct=integral(v,x,m);I=np.array([r['projected'] for r in norm])
 assert np.max(abs(direct/np.array([r['direct'] for r in norm])-1))<1e-11
 errors=[]
 for n in [16,32]:
  z,w=leggauss(n);y=(np.arange(32)[:,None]+(z+1)/2)/32
  ph=at(v,x[:,None,None],y[None,:,:]);h=m*ph*(1+ph)/(1-ph)**2
  q=np.sum(h*w[None,None,:]/64,axis=(1,2));errors.append(float(max(abs(q/direct-1))))
 assert max(errors)<1e-10
 return dict(max_source_error=float(max(abs(direct/I-1))),max_I_error=float(max(abs(I/direct-1))),Q16_Q32_errors=errors)
# Independent derivative of the projected surface energy, with Z frozen at
# surface quadrature. Use native default 3-point Gauss per 0.1 m element.
def projected_derivative(v,qnodes,m):
 z,w=leggauss(3);N=np.zeros((30,11));x=[];W=[]
 for j in range(10):
  for k in range(3):
   a=(z[k]+1)/2;N[3*j+k,j]=1-a;N[3*j+k,j+1]=a;x.append((j+a)/10);W.append(w[k]/20)
 x=np.array(x);W=np.array(W);M=N.T@(W[:,None]*N)
 project=lambda values:N@np.linalg.solve(M,N.T@(W*values))
 I=project(integral(v,x,m));q=N@qnodes;Z=I*q;G=1e6
 # Fixed Q1 perturbation with normal and along-fault variation. No H mutation.
 direction=v*(1+.3*np.cos(2*np.pi*np.arange(33)[:,None]/32))
 eps=1e-5
 Iplus=integral(v+eps*direction,x,m);Iminus=integral(v-eps*direction,x,m)
 dI=(Iplus-Iminus)/(2*eps);dIh=project(dI)
 fd=np.sum(W*Z*Z/(2*G)*(1/project(Iplus)-1/project(Iminus)))/(2*eps)
 exact=-np.sum(W*q*q/(2*G)*dIh);continuum=-np.sum(W*q*q/(2*G)*dI)
 assert abs(fd-exact)<1e-7*max(1,abs(exact))
 return dict(finite_difference=float(fd),discrete_derivative=float(exact),continuum_driver_derivative=float(continuum),
    relative_projection_discrepancy=float(abs(exact-continuum)/max(1e-30,abs(exact))))
labels=['recoverable_dt125','recoverable_dt0625','recoverable_dt03125'];data={};summary={}
R=np.zeros((11,11))
for r in rows(build/'output-recoverable_projection/recoverable-q-transfer.csv'):R[int(r['row']),int(r['column'])]=r['value']
eig=np.linalg.eigvals(R);assert np.max(abs(R@np.ones(11)-1))<1e-12
projection=dict(identity_error=float(np.max(abs(R-np.eye(11)))),constant_error=float(np.max(abs(R@np.ones(11)-1))),
 linear_error=float(np.max(abs(R@np.linspace(0,1,11)-np.linspace(0,1,11)))),eigenvalues=[float(v.real) for v in eig],
 fastest_mode_decay_rate={str(dt):float(-np.log(min(eig.real))/dt) for dt in [.125,.0625,.03125]})
for label in labels:
 d=build/('output-'+label);run=json.loads((e/(label+'.json')).read_text());assert run['code']==0
 f=rows(d/'feedback.csv');local=rows(d/'material-local.csv');energy=rows(d/'recoverable-energy.csv');p=rows(d/'material-particles.csv');m=rows(d/'material-calibration.csv')[0]['m']
 assert len(f)==round(1/run['dt'])+1
 for r in f:
  assert 0<=r['phi_min']<=r['phi_max']<1 and r['phase_kkt']<=1e-11 and r['phase_healing']<=4.922e-12
  assert r['mechanical_residual']<=1e-8 and r['Ih_min']>0 and r['H_increment_integral']>=0
 assert (d/'material-phase-accepted-0.csv').read_bytes()==(build/'output-material_reference_a016/material-phase-accepted-0.csv').read_bytes()
 v0=field(d/'material-phase-accepted-0.csv');vf=field(d/f'material-phase-accepted-{len(f)-1}.csv');change=vf-v0;ix=np.unravel_index(np.argmax(change),change.shape)
 norm={}
 for step in [0,len(f)-1]:norm[str(step)]=norm_check(field(d/f'material-phase-accepted-{step}.csv'),rows(d/f'material-normalization-accepted-{step}.csv'),m)
 peak_step=len(f)-1;fault=rows(d/f'feedback-fault-{peak_step}.csv');qnodes=np.array([r['q_committed'] for r in fault])
 pred_error=0;fixed_error=0
 for row in f[1:]:
  rs=rows(d/f"feedback-fault-{int(row['step'])}.csv");vec=lambda name:np.array([v[name] for v in rs]);dt=row['dt'];beta=math.exp(-dt/100);kappa=-1e8*math.expm1(-dt/100)
  constitutive=(kappa*vec('V')+beta*vec('Ih_incoming')*vec('q_incoming'))/vec('Ih_current');actual=vec('q_committed')
  error=float(np.max(abs(R@constitutive-actual))/max(1,max(abs(actual))));pred_error=max(pred_error,error)
  if row['phase_cumulative_integral']==0:fixed_error=max(fixed_error,error)
 assert fixed_error<1e-12
 last=energy[-1];scale=last['bulk_stored']+last['fault_stored']+last['bulk_viscous']+last['fault_viscous']+last['friction']
 summary[label]=dict(dt=run['dt'],accepted_steps=len(f)-1,phi_max_gain=f[-1]['phi_max']-f[0]['phi_max'],max_nodal_gain=float(np.max(change)),
  max_gain_location=([ix[0]/32,ix[1]/32] if np.max(change)>0 else None),max_step_increment=max(r['phase_change'] for r in f),
  phase_changing_steps=[int(r['step']) for r in f if r['phase_change']>4.922e-12],H_gain=f[-1]['H_cumulative_integral'],
  max_KKT=max(r['phase_kkt'] for r in f),max_healing=max(r['phase_healing'] for r in f),mechanical_residual=max(r['mechanical_residual'] for r in f),
  offband_H_increases=sum(r['offband_H_increases'] for r in f),normalization=norm,projected_energy_derivative=projected_derivative(vf,qnodes,m),
  q_transfer_prediction_error=pred_error,fixed_phase_transfer_prediction_error=fixed_error,energy=last,relative_energy_defect=abs(last['energy_defect'])/scale)
 data[label]=(f,local,energy,p,vf,v0)
# Complete common-time and interpolated common-slip comparisons.
common_times=[]
for t in [i/8 for i in range(9)]:
 entries=[]
 for label in labels:
  f,loc,en,p,*_=data[label];r=next(v for v in f if v['time']==t);ee=next(v for v in en if v['time']==t)
  entries.append(dict(label=label,phi_max=r['phi_max'],H_gain=r['H_cumulative_integral'],Ymax=ee['Ymax'],q_max=r['q_max'],
    local=[v for v in loc if v['time']==t and v['x'] in [0,.5,1]]))
 common_times.append(dict(time=t,values=entries))
common_slip={}
for target in [0.,.5,1.]:
 locs=[[v for v in data[l][1] if abs(v['x']-target)<1e-12] for l in labels];slip=min(r[-1]['slip'] for r in locs);values=[]
 for label,rs in zip(labels,locs):
  val={k:float(np.interp(slip,[v['slip'] for v in rs],[v[k] for v in rs])) for k in ['time','phi','q','V']}
  ps=[v for v in data[label][3] if v['target']==target]
  val['H']=float(np.interp(slip,[v['slip'] for v in rs],[v['H_new'] for v in ps]));val['label']=label;values.append(val)
 common_slip[str(target)]=dict(slip=slip,values=values)
zero=rows(build/'output-recoverable_zero/feedback.csv');assert all(r['H_cumulative_integral']==0 and r['phase_change']==0 for r in zero)
# Historical default still reproduces all old diagnostic files exactly.
old=build/'output-material_reference_a016';new=build/'output-recoverable_legacy_check';files=list(old.glob('*.csv'));assert all(p.read_bytes()==(new/p.name).read_bytes() for p in files)
result=dict(cases=summary,traction_transfer=projection,common_times=common_times,common_slip=common_slip,zero_growth=False,legacy_identical_csvs=len(files),
 local_recoverable_law_pass=True,numerical_gates_pass=True,coupled_energy_balance_pass=False,feedback_convergence_established=False,stress_scaling_run=False,
 loading_amplitude=.016,additional_loading_pilot_run=False)
(e/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25})
fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for label in labels:
 f,loc,en,p,*_=data[label];name=f"dt={summary[label]['dt']} s"
 axes[0,0].plot([r['time'] for r in f],[r['phi_max']-f[0]['phi_max'] for r in f],'.-',label=name)
 rs=[r for r in loc if r['x']==0]
 axes[0,1].plot([r['slip'] for r in rs],[r['phi']-rs[0]['phi'] for r in rs],'.-',label=name)
 axes[1,0].plot([r['time'] for r in rs],[r['q'] for r in rs],'.-',label=name)
 axes[1,1].plot([r['slip'] for r in rs],[r['q'] for r in rs],'.-',label=name)
for ax,title,x,y in zip(axes.flat,['Phase maximum','Left endpoint phase','Left endpoint traction','Left endpoint traction'],['Time (s)','Local slip (m)','Time (s)','Local slip (m)'],['Phase increase','Phase increase','q (Pa)','q (Pa)']):ax.set(title=title,xlabel=x,ylabel=y);ax.legend()
fig.suptitle('Unloaded recoverable P2, a=0.016 s$^{-2}$; accepted states, coupling unqualified')
fig.savefig(e/'recoverable-trajectories.png',dpi=180);fig.savefig(e/'recoverable-trajectories.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
for ax,label in zip(axes,labels):
 vf,v0=data[label][-2:];v=(vf-v0).T;im=ax.imshow(v,origin='lower',extent=[0,1,0,1],vmin=0,vmax=max(summary[l]['max_nodal_gain'] for l in labels));ax.set(title=f"dt={summary[label]['dt']} s",xlabel='x (m)',ylabel='y (m)');fig.colorbar(im,ax=ax,label='Accepted phase change')
fig.savefig(e/'recoverable-spatial.png',dpi=180);fig.savefig(e/'recoverable-spatial.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
for label,color in zip(labels,['tab:blue','tab:orange','tab:green']):
 en=data[label][2];name=f"dt={summary[label]['dt']}"
 axes[0].plot([r['time'] for r in en],[r['bulk_stored']+r['fault_stored'] for r in en],label=name,color=color)
 axes[0].plot([r['time'] for r in en],[r['bulk_input_work'] for r in en],'--',color=color)
 axes[1].plot([r['time'] for r in en],[r['bulk_viscous']+r['fault_viscous'] for r in en],label=name)
 axes[2].plot([r['time'] for r in en],[r['fracture_change'] for r in en],label=name)
axes[0].set(title='Stored energy (solid), bulk work (dashed)',ylabel='J/m',xlabel='Time (s)');axes[1].set(title='Cumulative viscous loss',ylabel='J/m',xlabel='Time (s)');axes[2].set(title='Fracture expenditure above initial seed',ylabel='J/m',xlabel='Time (s)')
for ax in axes:ax.legend(fontsize=8)
fig.savefig(e/'recoverable-energy.png',dpi=180);fig.savefig(e/'recoverable-energy.pdf');plt.close(fig)
print(json.dumps(result,indent=2))
