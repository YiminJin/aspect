#!/usr/bin/env python3
"""Draw only accepted histories, with physical axes and distinct surface damage."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation,PillowWriter
from analyze_cohesive_interface_evolution import OUT,case,at,read

def main():
 plt.rcParams.update({'font.size':10,'figure.dpi':140,'axes.grid':True})
 cases=[case('straight-hold-coarse',64),case('straight-long',128),case('straight-half-window',128)];fine=cases[1]
 fig,ax=plt.subplots(2,1,figsize=(12,5),constrained_layout=True,gridspec_kw={'height_ratios':[1,1.1]})
 a=ax[0];a.add_patch(Rectangle((0,0),8,1,facecolor='#edf3f7',edgecolor='k'));a.hlines(.4,.5,7.5,color='#888',lw=4,label='Installed tied candidate');a.hlines(.4,2.5,5.5,color='#6a3d9a',lw=6,label='Initial mature seed: a=1');a.plot([.5,7.5],[.4,.4],'ks');a.axvline(1,ls=':',color='#777');a.axvline(7,ls=':',color='#777')
 for x in np.arange(1,8):a.arrow(x,1,.22,0,head_width=.06,head_length=.08,color='#0072b2',length_includes_head=True)
 a.text(4,1.15,'Top: relative speed 0.001 m/s to 2.45 s, then displacement hold',ha='center');a.text(4,-.15,'Bottom fixed; vertical wall motion zero; background pressure 10 kPa',ha='center');a.text(.08,.7,'Periodic',rotation=90);a.text(7.9,.7,'Periodic',rotation=90,ha='right');a.set(xlim=(-.1,8.1),ylim=(-.25,1.35),aspect='equal',xlabel='x [m]',ylabel='y [m]');a.legend(loc='lower right',fontsize=8)
 a=ax[1];t=2.47;i=at(fine,t);x=fine['s']['x_m'][i];damage=fine['s']['damage'][i];installed=(x>=.5)&(x<=7.5);phase_cmap=LinearSegmentedColormap.from_list('damage',['#bcbcbc','#fdae61','#6a3d9a']);phase_image=a.scatter(x[installed],np.full(sum(installed),.4),c=damage[installed],vmin=0,vmax=1,cmap=phase_cmap,s=14);fig.colorbar(phase_image,ax=a,label='Surface damage a',shrink=.7,pad=.01);a.axvline(2.5,color='#6a3d9a',ls='--',label='Original seed tip');a.axvline(.5,color='k',ls=':');a.axvline(.75,color='#c44',ls=':',label='Candidate buffer');
 for xx in np.arange(.5,3.001,.0625):a.axvline(xx,color='#ddd',lw=.6,zorder=0)
 for yy in np.arange(.3,.501,.0125):a.axhline(yy,color='#ddd',lw=.6,zorder=0)
 a.text(1.55,.44,'Surface damage a at 2.47 s; physical bulk mesh',ha='center');a.set(xlim=(.4,3),ylim=(.27,.53),aspect='equal',xlabel='x [m]',ylabel='y [m]');a.legend(fontsize=8,loc='lower right');fig.savefig(OUT/'specimen.png');plt.close(fig)
 colors=['#999','#0072b2','#009e73'];labels=['64×40, Δt=.0025 s','128×80, Δt=.0025 s','128×80, half-step event replay']
 fig,ax=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
 for c,color,label in zip(cases,colors,labels):
  r=np.genfromtxt(OUT/(c['name']+'-compact.csv'),delimiter=',',names=True,dtype=None,encoding=None);t=r['time_s'];mask=t<=2.47+1e-10
  ax[0,0].plot(t[mask],np.maximum(0,2.5-r['a100_left'][mask]),color=color,label=label)
  ax[0,0].fill_between(t[mask],np.maximum(0,2.5-r['a100_left'][mask]),np.maximum(0,2.5-r['a100_left_lower'][mask]),color=color,alpha=.13)
  ax[0,1].plot(t[mask],r['integrated_new_damage_m'][mask],color=color,label=label)
  ax[1,0].plot(.001*np.minimum(t[mask],2.45)*1000,r['reaction_N_m'][mask]/1000,color=color,label=label)
  ax[1,1].semilogy(t[mask],r['absolute_remainder_to_date_J_m'][mask],color=color,label=label)
 ax[0,0].set(xlim=(2.3,2.47),ylabel='Left full-decohesion advance [m]',title='Full-decohesion front; shading shows Gauss brackets')
 ax[0,1].set(xlim=(1.5,2.47),ylabel='∫(a−a_initial) ds [m]',title='Added fracture / Gc; initial seed excluded')
 ax[1,0].set(xlabel='Imposed relative displacement [mm]',ylabel='Reaction [kN/m]',title='Vertical segment: displacement hold')
 ax[1,1].set(xlim=(0,2.47),xlabel='Time [s]',ylabel='Cumulative absolute remainder [J/m]',title='Raw trapezoidal physical-work error')
 for a in [ax[0,0],ax[0,1]]:a.axvline(2.45,color='k',ls=':');a.set_xlabel('Time [s]')
 for a in ax.flat:a.legend(fontsize=8)
 ax[0,1].get_legend().remove()
 inset=ax[0,1].inset_axes([.12,.49,.43,.32])
 for c,color in zip(cases[:2],colors[:2]):
  r=np.genfromtxt(OUT/(c['name']+'-compact.csv'),delimiter=',',names=True,dtype=None,encoding=None);inset.plot(r['time_s'],r['integrated_new_damage_m'],color=color)
 inset.set(xlim=(1.5,2.05),ylim=(0,.045));inset.set_title('Early spatial difference: 65.4% at 1.8 s',fontsize=8);inset.tick_params(labelsize=7)
 ax[0,1].text(.03,.24,'At 2.47 s: spatial Δgrowth = .00207 m (.136%)\nHalf-step Δgrowth = .00504 m (.330%)',transform=ax[0,1].transAxes,fontsize=8)
 fig.savefig(OUT/'comparison.png');plt.close(fig)
 fig,ax=plt.subplots(3,1,figsize=(12,8),constrained_layout=True)
 for t in [2.3,2.38,2.42,2.45,2.47]:
  i=at(fine,t);x=fine['s']['x_m'][i];ax[0].plot(x,fine['s']['damage'][i],label=f'{t:g} s');ax[1].plot(x,fine['s']['jump_m'][i]*1000,label=f'{t:g} s');ax[2].plot(x,fine['s']['V_m_s'][i]*1000,label=f'{t:g} s')
 for a in ax:
  for xx in [2.5,5.5]:a.axvline(xx,ls='--',color='#6a3d9a',lw=.8)
  a.set(xlim=(.5,7.5),xlabel='x along fixed Γ [m]');a.legend(ncol=5,fontsize=8)
 ax[0].set(ylabel='Surface damage a',ylim=(-.03,1.03),title='Cohesive-zone expansion and full decohesion are distinct')
 ax[1].set_ylabel('Signed physical jump [mm]');ax[2].set_ylabel('Signed slip velocity [mm/s]');fig.savefig(OUT/'profiles.png');plt.close(fig)
 c=fine;h=c['h'];t=h['time_s'];fig,ax=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
 for value,label in [(h['bulk_J_m'],'Bulk recoverable'),(h['bond_J_m'],'Cohesive recoverable'),(np.cumsum(h['fracture_increment_J_m']),'New fracture'),(np.cumsum(h['friction_J_m']),'Friction loss'),(np.cumsum(h['radiation_J_m']),'Radiation loss'),(np.cumsum(h['Maxwell_J_m']),'Maxwell loss')]:ax[0].plot(t,value,label=label)
 ax[0].plot(t,np.cumsum(h['trapezoid_work_J_m']),'k--',label='Physical trapezoidal work');ax[0].set(ylabel='Energy per thickness [J/m]');ax[0].legend(ncol=3,fontsize=8)
 r=np.genfromtxt(OUT/(c['name']+'-compact.csv'),delimiter=',',names=True,dtype=None,encoding=None);ax[1].plot(t,np.cumsum(r['physical_remainder_J_m']),label='Signed physical remainder');ax[1].plot(t,r['absolute_remainder_to_date_J_m'],label='Sum of absolute per-step remainders');ax[1].plot(t,np.cumsum(h['ledger_error_J_m']),label='Separate BE ledger');ax[1].set(xlabel='Time [s]',ylabel='Remainder [J/m]');ax[1].legend()
 for a in ax:a.axvline(2.45,color='k',ls=':')
 fig.savefig(OUT/'energy.png');plt.close(fig)
 # Animate actual accepted Gauss records; do not synthesize constitutive states.
 frame_times=np.r_[np.arange(1.5,2.3,.05),np.arange(2.3,2.47001,.005)]
 indices=sorted(set(int(np.argmin(abs(t-v))) for v in frame_times))
 fig,ax=plt.subplots(3,1,figsize=(11,6),constrained_layout=True);x=c['s']['x_m'][0]
 lines=[]
 for a,key,lab,scale in zip(ax,['damage','jump_m','V_m_s'],['Surface damage a','Physical jump [mm]','Slip velocity [mm/s]'],[1,1000,1000]):
  line,=a.plot([],[],color='#0072b2');lines.append((line,key,scale));a.set(xlim=(.5,7.5),ylabel=lab);a.axvspan(2.5,5.5,color='#6a3d9a',alpha=.08)
  for xx in [2.5,5.5]:a.axvline(xx,color='#6a3d9a',ls='--',lw=.8)
 ax[0].set_ylim(-.03,1.03);ax[1].set_ylim(0,1.05*np.max(c['s']['jump_m'])*1000);ax[2].set_ylim(min(0,np.min(c['s']['V_m_s'])*1100),np.max(c['s']['V_m_s'])*1100);ax[-1].set_xlabel('x along the fixed straight interface [m]')
 title=fig.suptitle('')
 def update(k):
  i=indices[k]
  for line,key,scale in lines:line.set_data(x,c['s'][key][i]*scale)
  title.set_text(f't={t[i]:.4f} s, imposed displacement={min(t[i],2.45):.4f} mm'+(' — displacement hold' if t[i]>=2.45 else ' — loading'))
 ani=FuncAnimation(fig,update,frames=len(indices));ani.save(OUT/'evolution.gif',writer=PillowWriter(fps=8));plt.close(fig)
if __name__=='__main__':main()
