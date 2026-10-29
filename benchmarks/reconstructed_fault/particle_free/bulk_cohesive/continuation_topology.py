#!/usr/bin/env python3
"""Seed-connected contour support on the retained 4/8-subdivision diagnostic."""
import argparse,json
import numpy as np
from scipy.ndimage import label
from continuation_analysis import BASE,OUT,LEVELS,grid_value,save_csv

def run(name):
    with np.load(BASE/(name+'-plot-data.npz')) as a:d={k:a[k] for k in ['x','y','phase','times','delta']}
    selected=set([int(np.argmax(np.max(d['delta'],axis=(1,2)))),int(np.argmin(abs(d['times']-.174375))),int(np.argmin(abs(d['times']-.175))),len(d['times'])-1]);rows=[]
    for subdivision in [4,8]:
        x=np.linspace(d['x'][0],d['x'][-1],(len(d['x'])-1)*subdivision+1);y=np.linspace(d['y'][0],d['y'][-1],(len(d['y'])-1)*subdivision+1);X,Y=np.meshgrid(x,y)
        for k,t in enumerate(d['times']):
            if t<.12-1e-12 or t>.17875+1e-12 or (subdivision==8 and k not in selected):continue
            p=grid_value(d['x'],d['y'],d['phase'][k],X,Y)
            for level in LEVELS:
                components,count=label(p>=level);seed=components[np.argmin(abs(y-.5)),0];assert seed>0
                ids=np.unique(components);detached=len(ids[(ids>0)&(ids!=seed)]);area=float(np.count_nonzero((components>0)&(components!=seed))*(x[1]-x[0])*(y[1]-y[0]));extent=float(np.max(X[components==seed]));rows.append([t,level,subdivision,detached,area,extent])
    save_csv(OUT/(name+'-topology.csv'),['time_s','phi_contour','subdivisions_per_cell','detached_components','detached_raster_area_m2','seed_component_rightmost_raster_x_m'],rows)
    summary=dict(case=name,definition='Q1 field sampled on a uniform 4/8-subdivision raster; 4-connected components, seed at left mid-height. Raster extent is separate from the exact Q1 centerline crossing.',maximum_detached_components=int(max(r[3] for r in rows)),checked_states_at_8=len(selected))
    (OUT/(name+'-topology.json')).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('names',nargs='+');a=p.parse_args()
    for name in a.names:run(name)
