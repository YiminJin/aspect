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
here=Path(__file__).resolve().parent;build=here.parents[2]/'build-propagation-integration';e=here/'coupling_evidence'
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
 Inodes=np.linalg.solve(M,N.T@(W*integral(v,x,m)));I=N@Inodes;Z=N@(Inodes*qnodes);q=Z/I;G=1e6
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

import argparse
parser=argparse.ArgumentParser();parser.add_argument('labels',nargs='+');parser.add_argument('--name',default='comparison');args=parser.parse_args()
summary={};data={}
for label in args.labels:
 d=build/('output-'+label);run=json.loads((e/(label+'.json')).read_text());f=rows(d/'feedback.csv');loc=rows(d/'material-local.csv');work=rows(d/'coupling-work.csv');energy=rows(d/'recoverable-energy.csv')
 for r in f:
  assert 0<=r['phi_min']<=r['phi_max']<1 and r['phase_kkt']<=1e-11 and r['phase_healing']<=4.922e-12
  assert r['mechanical_residual']<=1e-8 and r['Ih_min']>0 and r['H_increment_integral']>=0
 last=int(f[-1]['step']);v0=field(d/'material-phase-accepted-0.csv');vf=field(d/f'material-phase-accepted-{last}.csv');delta=vf-v0;ix=np.unravel_index(np.argmax(delta),delta.shape)
 norm={str(k):norm_check(field(d/f'material-phase-accepted-{k}.csv'),rows(d/f'material-normalization-accepted-{k}.csv'),128.00000000016) for k in [0,last]}
 qnodes=np.array([r['q_committed'] for r in rows(d/f'feedback-fault-{last}.csv')]);derivative=projected_derivative(vf,qnodes,128.00000000016)
 summary[label]=dict(dt=run['dt'],exit_code=run['code'],accepted_time=f[-1]['time'],accepted_steps=last,phase_max_gain=f[-1]['phi_max']-f[0]['phi_max'],max_nodal_gain=float(delta.max()),max_gain_location=[ix[0]/32,ix[1]/32],max_step_increment=max(r['phase_change'] for r in f),phase_changing_steps=[int(r['step']) for r in f if r['phase_change']>4.922e-12],max_healing=max(r['phase_healing'] for r in f),max_KKT=max(r['phase_kkt'] for r in f),max_mechanical_residual=max(r['mechanical_residual'] for r in f),H_gain=f[-1]['H_cumulative_integral'],normalization=norm,phase_energy_derivative=derivative,final_work=work[-1],max_operator_error=max(r['operator_adjoint_error'] for r in work),max_publication_error=max(r['q_publication_error'] for r in work),max_bulk_identity_defect=max(abs(r['bulk_identity_defect']) for r in work),cumulative_exchange_defect=sum(r['time_integrated_exchange_defect'] for r in work[1:]),cumulative_stress_transfer_energy=sum(r['mapped_particle_energy']-r['native_QP_energy'] for r in work[1:]),final_energy=energy[-1],local_final=[r for r in loc if r['step']==last and r['x'] in [0,.5,1.]])
 data[label]=(f,loc,work,energy,vf,v0)
common=[]
last_time=min(d[0][-1]['time'] for d in data.values())
for t in np.arange(0,last_time+1e-10,.125):
 common.append(dict(time=float(t),cases={l:dict(feedback=next(r for r in data[l][0] if abs(r['time']-t)<1e-12),local=[r for r in data[l][1] if abs(r['time']-t)<1e-12 and r['x'] in [0,.5,1.]]) for l in args.labels}))
slips={}
for x in [0,.5,1.]:
 histories={l:[r for r in data[l][1] if r['x']==x] for l in args.labels};end=min(rs[-1]['slip'] for rs in histories.values())
 slips[str(x)]=[dict(slip=float(s),cases={l:{k:float(np.interp(s,[r['slip'] for r in rs],[r[k] for r in rs])) for k in ['time','phi','q','V']} for l,rs in histories.items()}) for s in np.linspace(0,end,9)]
(e/(args.name+'.json')).write_text(json.dumps(dict(cases=summary,common_times=common,common_slips=slips),indent=2)+'\n')
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25})
fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for label in args.labels:
 f,loc,work,en,vf,v0=data[label];name=f"dt={summary[label]['dt']} s";rs=[r for r in loc if r['x']==0]
 axes[0,0].plot([r['time'] for r in f],[r['phi_max']-f[0]['phi_max'] for r in f],'.-',label=name)
 axes[0,1].plot([r['slip'] for r in rs],[r['phi']-rs[0]['phi'] for r in rs],'.-',label=name)
 axes[1,0].plot([r['time'] for r in rs],[r['q'] for r in rs],'.-',label=name)
 axes[1,1].plot([r['slip'] for r in rs],[r['q'] for r in rs],'.-',label=name)
for ax,title,x,y in zip(axes.flat,['Phase maximum','Endpoint phase','Endpoint traction','Endpoint traction'],['Time (s)','Local slip (m)','Time (s)','Local slip (m)'],['Phase increase','Phase increase','q (Pa)','q (Pa)']):ax.set(title=title,xlabel=x,ylabel=y);ax.legend()
fig.savefig(e/(args.name+'-trajectories.png'),dpi=160);fig.savefig(e/(args.name+'-trajectories.pdf'));plt.close(fig)
fig,axes=plt.subplots(1,len(args.labels),figsize=(4*len(args.labels),3.5),squeeze=False,constrained_layout=True)
for ax,label in zip(axes[0],args.labels):
 f,loc,work,en,vf,v0=data[label];im=ax.imshow((vf-v0).T,origin='lower',extent=[0,1,0,1],vmin=0,vmax=max(1e-12,max(x['max_nodal_gain'] for x in summary.values())));ax.set(title=f"dt={summary[label]['dt']} s",xlabel='x (m)',ylabel='y (m)');fig.colorbar(im,ax=ax,label='Accepted phase change')
fig.savefig(e/(args.name+'-spatial.png'),dpi=160);plt.close(fig)
for l,r in summary.items():print(l,r['accepted_time'],r['max_nodal_gain'],r['max_operator_error'],r['cumulative_exchange_defect'])
