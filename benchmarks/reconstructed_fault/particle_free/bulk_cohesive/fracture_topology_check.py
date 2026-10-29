#!/usr/bin/env python3
"""Verify geometric component classification on one finer diagnostic raster."""
import numpy as np
from scipy.ndimage import label
from fracture_analysis import OUT,BUILD,LEVELS,grid_value,read,save_csv

def main():
    rows=[]
    for name,steps in [('released',[117,118,120,124,148]),('candidate',[108,116,139,140,148])]:
        d=np.load(BUILD/f'{name}-plot-data.npz');old=read(OUT/f'{name}-contours.csv')
        x=np.linspace(0,1.5,(len(d['x'])-1)*8+1);y=np.linspace(0,1,(len(d['y'])-1)*8+1);X,Y=np.meshgrid(x,y)
        for n in steps:
            phase=grid_value(d['x'],d['y'],d['phase'][n],X,Y)
            for level_ in LEVELS:
                components,_=label(phase>=level_);seed=components[len(y)//2,0];assert seed!=0
                ids=np.unique(components);detached=len(ids[(ids!=0)&(ids!=seed)])
                row=old[(old['step']==n)&(old['phase_threshold']==level_)][0]
                assert detached==int(row['detached_components'])
                right=float(np.max(X[components==seed]));error=abs(right-row['seed_component_rightmost_raster_x_m'])
                assert error<=(d['x'][1]-d['x'][0])/4+1e-14
                rows.append([name,n,float(d['time'][n]),level_,int(row['detached_components']),detached,float(row['seed_component_rightmost_raster_x_m']),right,error])
    save_csv(OUT/'topology-check.csv',['case','step','time_s','phase_threshold','detached_at_4','detached_at_8','rightmost_at_4_m','rightmost_at_8_m','extent_difference_m'],rows)
    print('60 component/extent comparisons pass; no constitutive sampling or evolution changed.')

if __name__=='__main__':main()
