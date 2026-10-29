#!/usr/bin/env python3
"""Gauss6 physical-stress maxima and compact quadrature-weighted cell means."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_cohesive_interface_insertion import write
from run_cohesive_interface_insertion import OUT,WORK

def main():
    rows=[];summary=[];maps=[]
    for time in [2.,2.42]:
        path=WORK/f'runs/physical-fields/output/physical-fields-{round(time*1e6)}.csv'
        a=np.loadtxt(path,delimiter=',',skiprows=1);names=path.open().readline().strip().split(',');v={k:a[:,i] for i,k in enumerate(names)}
        maximum_shear=np.hypot(.5*(v['tau_xx_Pa']-v['tau_yy_Pa']),v['tau_xy_Pa']);remote=(v['x_m']<1)|(v['x_m']>7)
        selected=np.flatnonzero(remote)[np.argmax(maximum_shear[remote])]
        summary.append(dict(time_s=time,Gauss_points=len(a),max_inplane_shear_all_Pa=float(max(maximum_shear)),max_inplane_shear_remote_Pa=float(max(maximum_shear[remote])),remote_x_m=float(v['x_m'][selected]),remote_y_m=float(v['y_m'][selected]),min_physical_pn_Pa=float(min(v['pn_Pa'])),remote_below_strength=bool(max(maximum_shear[remote])<3000)))
        weight=v['JxW_m2'].reshape(-1,36);area=weight.sum(axis=1)
        mean={key:(values.reshape(-1,36)*weight).sum(axis=1)/area for key,values in v.items() if key!='JxW_m2'}
        mx=maximum_shear.reshape(-1,36).max(axis=1)
        for i in range(len(area)):
            rows.append(dict(time_s=time,cell=i,x_m=float(mean['x_m'][i]),y_m=float(mean['y_m'][i]),area_m2=float(area[i]),tau_xx_Pa=float(mean['tau_xx_Pa'][i]),tau_yy_Pa=float(mean['tau_yy_Pa'][i]),tau_zz_Pa=float(mean['tau_zz_Pa'][i]),tau_xy_Pa=float(mean['tau_xy_Pa'][i]),p_perturbation_Pa=float(mean['p_perturbation_Pa'][i]),max_inplane_shear_at_Gauss_Pa=float(mx[i])))
        maps.append(mean)
    write('bulk_fields.csv',rows);(OUT/'bulk_field_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(2,1,figsize=(12,4.5),sharex=True,sharey=True,layout='constrained');maximum=max(max(m['tau_xy_Pa']) for m in maps)/1000
    for ax,m,time in zip(axes,maps,[2.,2.42]):
        ix=np.rint(m['x_m']/.0625-.5).astype(int);iy=np.rint(m['y_m']/.0125-.5).astype(int);grid=np.zeros((80,128));grid[iy,ix]=m['tau_xy_Pa']/1000
        image=ax.pcolormesh(np.linspace(0,8,129),np.linspace(0,1,81),grid,vmin=0,vmax=maximum,cmap='viridis',rasterized=True)
        ax.plot([1,7],[.4,.4],color='white',ls=':',lw=1);ax.plot([2.5,5.5],[.4,.4],color='white',lw=2)
        ax.set_aspect('equal');ax.set(ylabel='y (m)',title=f'Physical τxy at t={time:.2f} s — quadrature-weighted cell means')
    axes[-1].set_xlabel('x (m)');fig.colorbar(image,ax=axes,label='physical shear stress τxy (kPa)',shrink=.85);fig.savefig(OUT/'physical_stress.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
