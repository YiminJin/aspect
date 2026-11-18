#!/usr/bin/env python3
"""Check sensitivity to the already measured cell-scale restoring action.
This inexpensive diagnostic cannot qualify the missing full native spectrum.
"""
from stability import *
from scipy.linalg import eig
raw=np.genfromtxt(RUN/'restoring-modes.csv',delimiter=',',names=True)
s=np.genfromtxt(RUN/'restoring-sections.csv',delimiter=',',names=True)
w=s['ds'];n=len(w);P=raw['force_basis_per_sqrt_m'].reshape(7,n).T;C=raw['slip_response_per_force_coefficient'].reshape(7,n).T
Cp=P.T@(w[:,None]*C);K=np.linalg.inv(.5*(Cp+Cp.T));Psi=C@K
active=s['cohesive_compliance_bound_m_Pa']>0
Psi=Psi[active];w=w[active];x=s['x'][active]
data=np.load(OUT/'reduced-operator.npz');smooth=data['Psi'];v=data['V'];theta=data['theta'];dc=float(data['Dc']);r=float(data['r'])
B=smooth.T@(w[:,None]*smooth)
record={}
for name,V,age,D,b,mu in [('baseline_observed',s['V'][active],s['stage_age'][active],DC,EVOLUTION,MU),('selected_weakening',v,theta,dc,EVOLUTION,MU),('selected_neutral',v,theta,dc,DIRECT,data['mu_neutral'])]:
 spec,A,M=matrix_spectrum(K,Psi,w,V,age,D,r,b,mu)
 values,vectors=eig(A);j=int(np.argmax(values.real));y=vectors[:,j]
 delta=Psi@(A[:7]@y+r*y[:7]);weight=w*abs(delta)**2
 fit=smooth@np.linalg.solve(B,smooth.T@(w*delta))
 center=float(sum(weight*x)/sum(weight));width=float(np.sqrt(sum(weight*(x-center)**2)/sum(weight)))
 record[name]=dict(max_real=float(values[j].real),imag=float(values[j].imag),rms_width_m=width,rms_width_cells=width/.0625,
   fraction_of_velocity_norm_squared_outside_smooth_space=float(sum(w*abs(delta-fit)**2)/sum(weight)))
 if name=='selected_weakening':
  np.savetxt(OUT/'grid-probe-fastest-mode.csv',np.c_[s['section'][active],x,delta.real/max(abs(delta)),delta.imag/max(abs(delta))],delimiter=',',header='section,x,relative_rate_real,relative_rate_imag',comments='')
record['scope']='Same smooth initial state and full responsive age set, with one additional cell-scale mechanical response. This is a sensitivity warning, not spatial convergence or a full native spectrum.'
(OUT/'probe-stability.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
mode=np.genfromtxt(OUT/'grid-probe-fastest-mode.csv',delimiter=',',names=True)
fig,ax=plt.subplots(1,2,figsize=(11,3.8))
ax[0].plot(x,v/max(abs(v)),label='Smooth initial slip rate',color='0.4')
ax[0].plot(mode['x'],mode['relative_rate_real'],lw=.8,label='Fastest augmented mode: real')
ax[0].plot(mode['x'],mode['relative_rate_imag'],lw=.8,label='Imaginary',alpha=.7)
ax[0].set(xlim=(2.5,5.5),xlabel='Section x (m)',ylabel='Normalized rate',title='Broad envelope hides cell-scale oscillations');ax[0].legend(fontsize=7)
initial=json.loads((OUT/'stability.json').read_text())['pilot']['initial_frozen_coefficient_growth']
ax[1].bar(['Six smooth modes','Plus one cell-scale mode'],[initial,record['selected_weakening']['max_real']],color=['C0','C3'])
ax[1].set(ylabel='Largest initial growth rate (1/s)',title='Selected Dc: unresolved spatial sensitivity')
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(OUT/('mode-resolution.'+ext),dpi=180)
