#!/usr/bin/env python3
"""Frozen-state variations and matched native-step energy accounting.
No phase solve, mechanical trajectory, or committed history is advanced.
"""
import csv,json,math,os
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
HERE=Path(__file__).resolve().parent;BUILD=HERE.parents[2]/'build-propagation-integration';OUT=HERE/'accuracy_energy_evidence';OUT.mkdir(exist_ok=True)
ns={'__file__':str(HERE/'analyze_irreversibility.py')};exec((HERE/'analyze_irreversibility.py').read_text().split('import argparse')[0],ns)
field,at,integral,rows=(ns[x] for x in ['field','at','integral','rows'])
G=1e6;eta=1e8;m=128.00000000016;lam=G/eta
z,w=leggauss(3)
coord=(np.arange(32)[:,None]+(z+1)/2)/32
X,Y=np.meshgrid(coord.ravel(),coord.ravel(),indexing='ij');WB=np.outer(np.tile(w/64,32),np.tile(w/64,32))
# Tensor convention S=sym(t tensor n), S:S=1/2, tau:S=tau_xy.

def surface(ds):
 n=round(1/ds);nodes=np.linspace(0,1,n+1);x=(nodes[:-1,None]+ds*(z+1)/2).ravel();W=np.tile(w*ds/2,n)
 N=np.zeros((len(x),n+1));j=np.repeat(np.arange(n),3);t=np.tile((z+1)/2,n);N[np.arange(len(x)),j]=1-t;N[np.arange(len(x)),j+1]=t
 M=N.T@(W[:,None]*N)
 return nodes,x,W,N,M

def state(v,nodes,x,W,N,M):
 I=np.linalg.solve(M,N.T@(W*integral(v,x,m)))
 ph=at(v,X,Y);h=m*ph*(1+ph)/(1-ph)**2
 return I,h/np.interp(X,nodes,I)

folder=BUILD/'output-irreversibility_dt03125';vf=field(folder/'material-phase-accepted-32.csv');vp=field(folder/'material-phase-accepted-31.csv');v0=field(folder/'material-phase-accepted-0.csv')
fault=rows(folder/'feedback-fault-32.csv');qnodes=np.array([r['q_committed'] for r in fault]);input_nodes=np.array([r['x'] for r in fault]);Znodes=np.array([r['Ih_current']*r['q_committed'] for r in fault])
records=[];variation=[]
for ds in [.1,.025]:
 nodes,x,W,N,M=surface(ds)
 If,chi=state(vf,nodes,x,W,N,M);I0,chi0=state(v0,nodes,x,W,N,M)
 # Preserve the existing representable Z function when changing diagnostic spacing.
 Zn=np.interp(nodes,input_nodes,Znodes);Z=np.interp(X,nodes,Zn);Zs=N@Zn
 for pattern in ['uniform','normal-gradient','along-normal-gradient']:
  tau=np.full_like(X,1500.)
  if pattern!='uniform':tau+=8000*np.abs(Y-.5)
  if pattern=='along-normal-gradient':tau*=1+.25*np.cos(2*np.pi*X)
  direction=vf-vp
  eps=1e-4
  Ip,chip=state(vf+eps*direction,nodes,x,W,N,M);Im,chim=state(vf-eps*direction,nodes,x,W,N,M)
  dchi=(chip-chim)/(2*eps);dI=N@((Ip-Im)/(2*eps))
  # Fixed total strain/displacement, Maxwell viscous strain and elastic slip Z/G.
  tp=tau-Z*(chip-chi);tm=tau-Z*(chim-chi)
  fd_bulk=np.sum(WB*(tp**2-tm**2)/(2*G))/(2*eps)
  bulk=-np.sum(WB*Z/G*tau*dchi)
  fault_fd=np.sum(W*Zs**2/(2*G)*(1/(N@Ip)-1/(N@Im)))/(2*eps)
  fault_derivative=-np.sum(W*(Zs/(N@If))**2/(2*G)*dI)
  assert abs(fd_bulk-bulk)<1e-7*max(1,abs(bulk))
  assert abs(fault_fd-fault_derivative)<1e-7*max(1,abs(fault_derivative))
  z8,w8=leggauss(8);coords8=(np.arange(32)[:,None]+(z8+1)/2)/32
  X8,Y8=np.meshgrid(coords8.ravel(),coords8.ravel(),indexing='ij');WB8=np.outer(np.tile(w8/64,32),np.tile(w8/64,32))
  phip8=at(vf+eps*direction,X8,Y8);phim8=at(vf-eps*direction,X8,Y8)
  chip8=m*phip8*(1+phip8)/(1-phip8)**2/np.interp(X8,nodes,Ip)
  chim8=m*phim8*(1+phim8)/(1-phim8)**2/np.interp(X8,nodes,Im)
  tau8=np.full_like(X8,1500.)
  if pattern!='uniform':tau8+=8000*np.abs(Y8-.5)
  if pattern=='along-normal-gradient':tau8*=1+.25*np.cos(2*np.pi*X8)
  bulk8=-np.sum(WB8*np.interp(X8,nodes,Zn)/G*tau8*(chip8-chim8)/(2*eps))
  variation.append(dict(spacing=ds,stress_pattern=pattern,bulk_derivative=float(bulk),bulk_Q8_comparator=float(bulk8),bulk_Q3_Q8_difference=float(abs(bulk-bulk8)),bulk_finite_difference=float(fd_bulk),fault_derivative=float(fault_derivative),fault_finite_difference=float(fault_fd),total_derivative=float(bulk+fault_derivative),bulk_over_fault=float(bulk/fault_derivative),direction='retained accepted step31 to32',Z='fixed accepted step32 Q1 Z',stress_units='Pa'))
  for dt in [.125,.0625,.03125]:
   beta=math.exp(-lam*dt);kappa=-eta*math.expm1(-lam*dt)
   C=beta*Z/kappa*(chi-chi0)
   # Native piecewise-constant rate during a step, V=0, eps_dot=0.
   rate_xy=-C/2;steady=2*eta*rate_xy;transient=tau-steady
   end=beta*tau+2*kappa*rate_xy
   integral_stress=steady*dt+transient*(-math.expm1(-lam*dt))/lam
   square=steady**2*dt+2*steady*transient*(-math.expm1(-lam*dt))/lam+transient**2*(-math.expm1(-2*lam*dt))/(2*lam)
   Eold=np.sum(WB*tau**2/(2*G));Enew=np.sum(WB*end**2/(2*G));D=np.sum(WB*square/eta)
   profile=np.sum(WB*integral_stress*C)
   identity=(Enew-Eold)+D+profile
   assert abs(identity)<1e-8*max(1,abs(profile))
   fault_old=np.sum(W*Zs**2/(2*G)/(N@I0));fault_profile=np.sum(W*Zs**2/(2*G)*(1/(N@If)-1/(N@I0)))
   fault_after=np.sum(W*(beta*Zs)**2/(2*G)/(N@If));Df=(1-beta**2)*np.sum(W*Zs**2/(2*G)/(N@If))
   assert abs(fault_after-fault_old+Df-fault_profile)<1e-10
   # Distinct exact endpoint path: instantaneous elastic redistribution then relaxation.
   jump=tau-Z*(chi-chi0);jump_delta=np.sum(WB*(jump**2-tau**2)/(2*G));jump_D=np.sum(WB*(1-beta**2)*jump**2/(2*G))
   assert np.max(abs(beta*jump-end))<1e-10
   records.append(dict(spacing=ds,stress_pattern=pattern,dt=dt,ordinary_slip_work=0,bulk_external_work=0,bulk_profile_work=float(profile),bulk_storage_change=float(Enew-Eold),bulk_viscous=float(D),bulk_identity_defect=float(identity),cohesive_storage_change=float(fault_after-fault_old),fault_profile_storage_change=float(fault_profile),fault_viscous=float(Df),friction=0,transfer_energy_change=0,instantaneous_bulk_profile_storage_change=float(jump_delta),instantaneous_bulk_viscous=float(jump_D),native_minus_jump_bulk_dissipation=float(D-jump_D)))

# Retained native trajectories: one bulk QP measure, one native surface Q3 measure.
trajectories={}
for tag in ['125','0625','03125']:
 label='irreversibility_dt'+tag;d=BUILD/('output-'+label);bulk=rows(d/'coupling-work.csv');energy=rows(d/'recoverable-energy.csv')
 matched=rows(HERE/'irreversibility_evidence'/f'{label}-matched-work.csv');f=rows(d/'feedback.csv');storage=viscous=fault_slip=fault_visc=0;fault_profile=0;recs=[]
 for r,s in zip(bulk[1:],matched):
  k=int(r['step']);dt=f[k]['dt'];b=math.exp(-lam*dt);one=-math.expm1(-lam*dt);two=-math.expm1(-2*lam*dt);fa=rows(d/f'feedback-fault-{k}.csv');Eold=Enew=D=0
  for j in range(10):
   for zz,ww in zip(z,w):
    xi=(zz+1)/2;wt=ww/20
    interp=lambda key:(1-xi)*fa[j][key]+xi*fa[j+1][key]
    I=interp('Ih_current');oldI=interp('Ih_incoming');V=interp('V');Z=(1-xi)*fa[j]['Ih_incoming']*fa[j]['q_incoming']+xi*fa[j+1]['Ih_incoming']*fa[j+1]['q_incoming']
    Znew=(b*Z-eta*math.expm1(-lam*dt)*V)
    Eold+=wt*Z*Z/(2*G*oldI);Enew+=wt*Znew*Znew/(2*G*I)
    steady=eta*V;tr=Z-steady
    D+=wt*(steady*steady*dt+2*steady*tr*one/lam+tr*tr*two/(2*lam))/(eta*I)
  err=Enew-Eold+D-s['Q3_cohesive_work']-s['Q3_fault_phase_jump']
  assert abs(err)<1e-8
  fault_visc+=D;fault_profile+=s['Q3_fault_phase_jump'];fault_slip+=s['Q3_cohesive_work']
  recs.append(dict(time=r['time'],ordinary_bulk_slip_work=r['bulk_crack_work']-r['profile_work'],bulk_profile_work=r['profile_work'],bulk_storage_change=r['native_QP_energy']-r['old_FE_energy'],bulk_viscous=r['bulk_viscous'],bulk_external_work=r['bulk_work'],bulk_identity_defect=r['bulk_identity_defect'],fault_storage_change=Enew-Eold,fault_viscous=D,fault_profile_storage_change=s['Q3_fault_phase_jump'],ordinary_fault_slip_work=s['Q3_cohesive_work'],friction=s['Q3_friction_work'],fault_identity_defect=err,particle_mapping_energy_change=r['mapped_particle_energy']-r['native_QP_energy'],time_integrated_exchange_defect=s['native_Q3_integrated_exchange_defect'],endpoint_exchange_defect=s['native_Q3_endpoint_exchange_defect']))
 reset=sum(bulk[k]['old_FE_energy']-bulk[k-1]['native_QP_energy'] for k in range(1,len(bulk)))
 gap=sum(a['bulk_profile_work']-a['fault_profile_storage_change']+a['time_integrated_exchange_defect'] for a in recs)-energy[-1]['fracture_change']
 trajectories[label]=dict(final_time=1,fracture_expenditure=energy[-1]['fracture_change'],FE_reset_energy_change=reset,bulk_native_final=bulk[-1]['native_QP_energy'],bulk_particle_final=energy[-1]['bulk_stored'],mechanical_plus_fracture_gap_excluding_transfer=gap,summed={key:sum(a[key] for a in recs) for key in recs[0] if key!='time'},final_fault_energy_Q3=Enew,max_bulk_identity_defect=max(abs(a['bulk_identity_defect']) for a in recs),max_fault_identity_defect=max(abs(a['fault_identity_defect']) for a in recs),records=recs)

(OUT/'profile-energy.json').write_text(json.dumps(dict(controlled_steps=records,phase_variations=variation,retained_trajectories=trajectories),indent=2)+'\n')
for r in variation:print('variation',r['spacing'],r['stress_pattern'],r['bulk_derivative'],r['fault_derivative'],r['bulk_over_fault'])
for l,r in trajectories.items():print(l,r['summed'])

# Compact plots keep unrelated measures and signed terms separate.
os.environ.setdefault('MPLCONFIGDIR','/tmp/aspect-accuracy-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
for label,r in trajectories.items():
 rs=r['records'];t=[a['time'] for a in rs];name=label.split('dt')[1]
 for ax,key in zip(axes,['bulk_profile_work','particle_mapping_energy_change','time_integrated_exchange_defect']):
  ax.plot(t,np.cumsum([a[key] for a in rs]),label='dt '+name)
for ax,title in zip(axes,['Bulk profile work','Native QP to particle-interpolated energy','Integrated ordinary exchange defect']):
 ax.set(title=title,xlabel='Time (s)',ylabel='Cumulative J/m');ax.legend();ax.grid(alpha=.25)
fig.savefig(OUT/'energy-terms.png',dpi=160);fig.savefig(OUT/'energy-terms.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(6,3.5),constrained_layout=True)
a=[r for r in variation if r['spacing']==.025];x=np.arange(len(a))
ax.bar(x-.15,[r['bulk_derivative'] for r in a],.3,label='Bulk profile derivative')
ax.bar(x+.15,[r['fault_derivative'] for r in a],.3,label='Fault derivative')
ax.set(xticks=x,xticklabels=[r['stress_pattern'] for r in a],ylabel='Directional derivative (J/m)',title='Retained step 31–32 direction, fixed Z and displacement');ax.legend();ax.grid(axis='y',alpha=.25)
fig.savefig(OUT/'profile-derivative.png',dpi=160);plt.close(fig)
