#!/usr/bin/env python3
"""Actual native Gauss6 stress/storage, with explicit physical averaging."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from analyze_cohesive_interface_evolution import OUT,read,table,case,at

def main():
 fine=case('straight-long',128);rows=[];profiles=[];checks=[];fields=[]
 for t in [2.3,2.45,2.47]:
  _,raw=read(OUT/'physical-fields'/f'physical-fields-{round(t*1e6)}.csv')
  x,y,w=raw[:,0],raw[:,1],raw[:,2];xx,yy,zz,xy,pp,pn=raw[:,4],raw[:,5],raw[:,6],raw[:,7],raw[:,8],raw[:,9]
  maximum=.5*np.hypot(xx-yy,2*xy);remote=(x<.5)|(x>7.5);index=np.flatnonzero(remote)[np.argmax(maximum[remote])];energy=float(w@(xx*xx+yy*yy+zz*zz+2*xy*xy)/(4e6));i=at(fine,t)
  checks.append(dict(time_s=t,points=len(x),volume_m2=float(sum(w)),bulk_energy_J_m=energy,energy_vs_accepted_J_m=energy-fine['h']['bulk_J_m'][i],pressure_perturbation_mean_Pa=float(w@pp/sum(w)),max_remote_inplane_shear_Pa=float(maximum[index]),remote_max_x_m=float(x[index]),remote_max_y_m=float(y[index]),min_bulk_horizontal_normal_pressure_Pa=float(min(pn)),max_bulk_inplane_shear_Pa=float(max(maximum))))
  # Cell averages are for rendering. Extrema above use unaveraged Gauss values.
  cell=raw.reshape(-1,36,10);weight=cell[:,:,2];volume=weight.sum(axis=1);mean=(cell*weight[:,:,None]).sum(axis=1)/volume[:,None]
  fields.append(mean)
  for r,v in zip(mean,volume):rows.append(dict(time_s=t,x_m=r[0],y_m=r[1],cell_area_m2=v,tau_xx_Pa=r[4],tau_yy_Pa=r[5],tau_zz_Pa=r[6],tau_xy_Pa=r[7],pprime_Pa=r[8],horizontal_pn_Pa=r[9]))
  bins=np.minimum(127,(x/.0625).astype(int))
  for side,mask in [('lower',(y>=.375)&(y<.4)),('upper',(y>.4)&(y<=.425))]:
   for col in range(128):
    selected=mask&(bins==col);norm=sum(w[selected]);assert norm>0
    profiles.append(dict(time_s=t,x_m=(col+.5)*.0625,side=side,strip_width_m=.025,tau_xy_Pa=float(w[selected]@xy[selected]/norm),tau_xx_Pa=float(w[selected]@xx[selected]/norm),tau_yy_Pa=float(w[selected]@yy[selected]/norm),pprime_Pa=float(w[selected]@pp[selected]/norm),horizontal_pn_Pa=float(w[selected]@pn[selected]/norm)))
 table(OUT/'physical-cells.csv',rows);table(OUT/'physical-profiles.csv',profiles);table(OUT/'physical-field-checks.csv',checks)
 (OUT/'physical-field-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
 plt.rcParams.update({'font.size':10,'figure.dpi':150})
 fig,ax=plt.subplots(3,2,figsize=(12,6.5),constrained_layout=True)
 shear_scale=max(np.max(abs(v[:,7])) for v in fields)/1000;pressure_scale=max(np.max(abs(v[:,8])) for v in fields)/1000
 for row,(t,v) in enumerate(zip([2.3,2.45,2.47],fields)):
  for col,(key,scale,title) in enumerate([(7,shear_scale,'Physical shear τxy [kPa]'),(8,pressure_scale,'Solved pressure perturbation p′ [kPa]')]):
   a=ax[row,col];image=np.empty((80,128));image[np.floor(v[:,1]/.0125).astype(int),np.floor(v[:,0]/.0625).astype(int)]=v[:,key]/1000;im=a.pcolormesh(np.linspace(0,8,129),np.linspace(0,1,81),image,cmap='RdBu_r',vmin=-scale,vmax=scale,shading='flat',rasterized=True)
   i=at(fine,t);x=fine['s']['x_m'][i];damage=fine['s']['damage'][i];a.plot([.5,7.5],[.4,.4],color='#777',lw=.7)
   a.scatter(x[damage>=.05],np.full(sum(damage>=.05),.4),s=3,color='#fdae61',label='a≥.05')
   a.scatter(x[damage==1],np.full(sum(damage==1),.4),s=3,color='k',label='a=1')
   for tip in [2.5,5.5]:a.axvline(tip,ls=':',color='k',lw=.7)
   a.set(xlim=(0,8),ylim=(0,1),aspect='equal',xlabel='x [m]',ylabel='y [m]',title=f'{t:g} s — '+title)
   fig.colorbar(im,ax=a,shrink=.65)
 ax[0,0].legend(fontsize=8,loc='upper right');fig.savefig(OUT/'physical-stress.png');plt.close(fig)
 fig,ax=plt.subplots(2,1,figsize=(11,6),constrained_layout=True)
 for t in [2.3,2.45,2.47]:
  for side,ls in [('lower','-'),('upper','--')]:
   r=[p for p in profiles if p['time_s']==t and p['side']==side]
   ax[0].plot([p['x_m'] for p in r],[p['tau_xy_Pa']/1000 for p in r],ls,label=f'{t:g} s, {side}')
   ax[1].plot([p['x_m'] for p in r],[p['horizontal_pn_Pa']/1000 for p in r],ls,label=f'{t:g} s, {side}')
 for a in ax:
  for tip in [2.5,5.5]:a.axvline(tip,ls=':',color='k',lw=.7)
  a.set(xlim=(0,8),xlabel='x [m]');a.grid();a.legend(ncol=3,fontsize=8)
 ax[0].set(ylabel='Physical τxy [kPa]',title='Actual bulk stress: 0.025 m strip means on each side of Γ')
 ax[1].set(ylabel='Horizontal normal pressure [kPa]');fig.savefig(OUT/'physical-profiles.png');plt.close(fig)
 print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
