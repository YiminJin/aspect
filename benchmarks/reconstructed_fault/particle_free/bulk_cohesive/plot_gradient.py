#!/usr/bin/env python3
"""Compact physical axes, native data and independently labeled strip control."""
from gradient_native_analysis import ROOT,WORK,OUT,read,col
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.rcParams.update({'font.size':10,'axes.titlesize':11})
cases=['n4-checked','n8','n16-final'];labels=['4 × 4 cells','8 × 8 cells','16 × 16 cells']
data=[read(WORK/'runs'/case/'coulomb-0020.csv') for case in cases]
fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
for j,(rows,label) in enumerate(zip(data,labels)):
    x=col(rows,'x_m');y=col(rows,'y_m');rate=2*np.hypot(col(rows,'Wxx_s-1'),col(rows,'Wxy_s-1'))
    for i,(v,norm,cmap,title) in enumerate([(rate,Normalize(0,.08),'magma','Physical tensor slip rate √(2 W:W) [s⁻¹]'),
                                          (col(rows,'tauxy_Pa'),Normalize(-1100,1100),'coolwarm','Physical deviator τxy [Pa]')]):
        ax=axes[i,j];im=ax.tripcolor(x,y,v,shading='gouraud',norm=norm,cmap=cmap)
        ax.set(xlim=(0,.25),ylim=(0,.25),xlabel='x [m]',ylabel='y [m]',aspect='equal')
        if i==0:ax.set_title(label)
        if j==2:fig.colorbar(im,ax=axes[i,:],shrink=.85,label=title)
fig.suptitle('Native mixed mechanics + slip-gradient field, t = 0.125 s\nFixed φ = 0.8; ℓs = 6 mm; microfree boundaries; Gauss-point samples')
fig.savefig(OUT/'native-fields.png',dpi=170);plt.close(fig)

profiles=read(OUT/'native-profiles.csv');h=read(OUT/'native-history.csv');joint=read(OUT/'joint-history.csv')
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for case,label,style in [('local','Local law, 4 × 4','--'),('n4-checked','6 mm, 4 × 4','-'),('n8','6 mm, 8 × 8','-'),('n16-final','6 mm, 16 × 16','-'),('n8-l3','3 mm, 8 × 8',':')]:
    r=[v for v in profiles if v['case']==case and abs(float(v['time_s'])-.125)<1e-12 and float(v['kernel_mass_m2'])>1e-5]
    for ax,key in [(axes[0,0],'slip_rate_s1'),(axes[0,1],'stress_xy_Pa')]:ax.plot(col(r,'normal_coordinate_m'),col(r,key),style,label=label)
axes[0,0].set(title='Native cross-band slip profile',xlabel='Physical normal coordinate [m]',ylabel='Slip rate [s⁻¹]');axes[0,0].legend(fontsize=8)
axes[0,1].set(title='Native physical shear stress',xlabel='Physical normal coordinate [m]',ylabel='τxy [Pa]')
r=[v for v in h if v['case']=='hold'];axes[1,0].plot(col(r,'time_s'),col(r,'gradient_energy_J_m'),label='Gradient storage')
axes[1,0].axvline(.125,color='.4',linestyle=':',label='Displacement hold starts')
axes[1,0].set(title='Native load then hold: K does not relax by itself',xlabel='Time [s]',ylabel='Gradient storage [J/m]');axes[1,0].legend(fontsize=8)
for case,label in [('grad-n65','6 mm'),('grad3-n65','3 mm')]:
    r=[v for v in joint if v['case']==case];axes[1,1].plot(col(r,'time_s'),col(r,'gradient_storage_J_m-2'),label=label)
for t,label in [(.125,'hold'),(.15,'reverse shear')]:axes[1,1].axvline(t,color='.5',linestyle=':');axes[1,1].text(t,.97,label,transform=axes[1,1].get_xaxis_transform(),ha='right',va='top',fontsize=8,rotation=90)
axes[1,1].set(title='Prescribed-strain strip: recoverable backstress memory',xlabel='Time [s]',ylabel='Gradient storage [J/m²]');axes[1,1].legend(fontsize=8)
fig.suptitle('Uncalibrated slip lengths; pressure-dependent friction and zero dilation retained\nNative profile kernel: Gaussian σ = 23.4375 mm; strip is a separate prescribed-strain problem')
fig.savefig(OUT/'gradient-response.png',dpi=170);plt.close(fig)
