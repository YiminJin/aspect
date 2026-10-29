#!/usr/bin/env python3
"""Compact figures from actual native exports and independent comparisons."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-interface-native')
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_cohesive_interface_native import OUT,read

def a(rows,key):return np.array([r[key] for r in rows])
def save(fig,name,title):
 fig.suptitle(title,fontsize=12);fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/name,dpi=160);plt.close(fig)

def main():
 plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
 summary=json.load(open(OUT/'summary.json'))
 fig,axes=plt.subplots(2,2,figsize=(11,7))
 for name,color in [('formation','#d88022'),('formation-half','#236b99')]:
  r=read(OUT/name/'states.csv');times=np.unique(a(r,'time_s'));means=[np.mean([x['damage'] for x in r if x['time_s']==t]) for t in times]
  axes[0,0].plot(times,means,label=name,color=color)
 axes[0,0].axvline(.3757048877712288,color='k',ls=':',label='analytic onset');axes[0,0].set(xlim=(.36,.42),xlabel='Time (s)',ylabel='Surface damage a');axes[0,0].legend(fontsize=8)
 for name,ax in [('mature',axes[0,1]),('nonuniform',axes[1,0])]:
  r=read(OUT/name/'states.csv');times=np.unique(a(r,'time_s'));means=[np.mean([x['shear_Pa'] for x in r if x['time_s']==t]) for t in times];ax.plot(times,np.array(means)/1000,label=name);ax.set(xlabel='Time (s)',ylabel='Mean physical shear traction (kPa)');ax.legend()
 r=read(OUT/'nonuniform/states.csv');last=[x for x in r if x['time_s']==max(a(r,'time_s'))]
 axes[1,1].plot(a(last,'x_m'),a(last,'pn_Pa')/1000,label='current normal pressure');axes[1,1].axhline(10,color='k',ls=':',label='background only');axes[1,1].set(xlabel='Along-fault x (m)',ylabel='Normal pressure (kPa)');axes[1,1].legend(fontsize=8)
 save(fig,'native_response.png','Native mixed FE / geometric Γ components: intact admission, mature reversal and current pressure')
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 modes=summary['operator_modes'];full=[r for r in modes if r['width_m']==.025]
 axes[0].plot([r['n'] for r in full],[r['K_mode_Pa_m']/1e6 for r in full],'o-',label='finite-width native')
 axes[0].plot([r['n'] for r in full],[r['sharp_Q1_K_Pa_m']/1e6 for r in full],'s--',label='sharp, same Q1 space')
 axes[0].set(xlabel='Surface segments',ylabel='1 m mode stiffness (MPa/m)');axes[0].legend()
 for n in [16,32]:
  r=[r for r in modes if r['n']==n];r.sort(key=lambda x:x['width_m']);axes[1].plot([x['width_m'] for x in r],[-100*x['K_width_difference'] for x in r],'o-',label=f'{n} segments')
 axes[1].set(xlabel='Representation half-width w (m)',ylabel='Stiffness deficit against sharp (%)');axes[1].legend()
 save(fig,'representation_error.png','FE refinement and finite-width error are different: no recalibration was applied')
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 old=read(OUT/'development/nodal_instability.csv');new=read(OUT/'formation/history.csv');new=[r for r in new if r['min_curvature_Pa_m']>0]
 axes[0].plot(a(old,'time_s'),a(old,'min_curvature_Pa_m')/1e6,label='failed nodal integration');axes[0].plot(a(new,'time_s'),a(new,'min_curvature_Pa_m')/1e6,label='consistent surface Gauss3');axes[0].axhline(0,color='k',ls=':');axes[0].set(xlabel='Time (s)',ylabel='Minimum relaxed curvature (MPa/m)');axes[0].legend(fontsize=8)
 for name in ['formation','formation-half','mature','nonuniform']:
  r=read(OUT/name/'history.csv');oldenergy=0;absolute=[];total=0
  for row in r:
   E=row['bulk_J_m']+row['bond_J_m'];D=row['Maxwell_J_m']+row['friction_J_m']+row['radiation_J_m'];total+=abs(row['trapezoid_work_J_m']-(E-oldenergy+row['fracture_increment_J_m']+D));absolute.append(total);oldenergy=E
  axes[1].plot(a(r,'time_s'),absolute,label=name)
 axes[1].set(xlabel='Time (s)',ylabel='Σ |physical work remainder| (J/m)');axes[1].legend(fontsize=8)
 save(fig,'stability_and_work.png','The failed alternating mode is retained; consistent integration and prepublication checks replace it')
 r=read(OUT/'operator-32/frozen_fields.csv');x=a(r,'x_m');y=a(r,'y_m');fig,axes=plt.subplots(3,1,figsize=(11,5.5))
 for ax,key,label in zip(axes,['tau_xy_Pa','p_perturbation_Pa','pn_Pa'],['Physical shear τxy (Pa)','Solved p′ (Pa)','Physical pn = p0+p′−τyy (Pa)']):
  im=ax.scatter(x,y,c=a(r,key),s=5,marker='s',cmap='coolwarm');ax.axhline(.05,color='k',lw=.8);ax.axhline(.025,color='.3',ls=':',lw=.6);ax.axhline(.075,color='.3',ls=':',lw=.6);ax.set(xlim=(0,1),ylim=(0,.125),ylabel='y (m)',aspect='equal');fig.colorbar(im,ax=ax,label=label,fraction=.02,pad=.02)
 axes[-1].set_xlabel('x (m)')
 save(fig,'frozen_native_fields.png','Fixed nonuniform history probe, not an evolved specimen; actual QPs, equal physical axes')

if __name__=='__main__':main()
