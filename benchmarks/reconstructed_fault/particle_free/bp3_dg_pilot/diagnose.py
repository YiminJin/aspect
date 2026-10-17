#!/usr/bin/env python3
"""Read actual weak loads and authoritative tensors without filtering/clipping."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sigma=50e6;Vp=1e-9;Vr=1e-6
shear=sigma*.025*math.asinh(Vp/(2*Vr)*math.exp((.6+.015*math.log(Vr/Vp))/.025))+4624440*Vp
result={}
for file in sorted(a.run.glob('coupled-profile-*.csv')):
    step=int(file.stem.rsplit('-',1)[1]);rows=list(csv.DictReader(file.open()));s=np.array([float(row['s_m']) for row in rows]);L=max(s)-min(s)
    weak=a.run/f'surface-weak-{step}.csv'
    if not weak.exists():weak=a.run/f'rank-0-surface-weak-{step}.csv'
    wrows=list(csv.DictReader(weak.open()));w={key:np.array([float(row[key]) for row in wrows]) for key in wrows[0]}
    mass=w['mass_diagonal']+w['mass_edge']+np.r_[0.,w['mass_edge'][:-1]]
    entry=dict(time_s=float(rows[0]['time_s']),V_range=[min(float(row['V_m_s']) for row in rows),max(float(row['V_m_s']) for row in rows)],
               theta_range=[min(float(row['theta_s']) for row in rows),max(float(row['theta_s']) for row in rows)],
               I_range_m=[min(float(row['I_h_m']) for row in rows),max(float(row['I_h_m']) for row in rows)],weak={})
    for region,mask in [('all',np.ones(len(s),dtype=bool)),('endpoints',(s<1000)|(s>L-1000)),('interior',(s>=1000)&(s<=L-1000))]:
        entry['weak'][region]={key+'_perturbation_max_Pa_m':float(max(abs((w[key]-background*mass)[mask]))) for key,background in [('shear',shear),('raw_normal',sigma),('filtered_normal',sigma)]}
    paths=list(a.run.glob(f'rank-*-authoritative-stress-{step}.csv')) or list(a.run.glob(f'authoritative-stress-{step}.csv'))
    volume=quadratic=0.;max_norm=max_normal=max_shear=0.;points=0
    for path in paths:
        with path.open() as f:
            for row in csv.DictReader(f):
                xx,yy,xy,pressure,weight=(float(row[key]) for key in ('tau_xx','tau_yy','tau_xy','p','JxW'))
                norm=xx*xx+yy*yy+(xx+yy)**2+2*xy*xy
                volume+=weight;quadratic+=weight*norm;points+=1;max_norm=max(max_norm,math.sqrt(norm))
                max_normal=max(max_normal,abs(pressure-.75*xx-.25*yy-math.sqrt(3)/2*xy))
                max_shear=max(max_shear,abs(-math.sqrt(3)/4*xx+math.sqrt(3)/4*yy+.5*xy))
    entry['authoritative']=dict(points=points,tau_RMS_Pa=math.sqrt(quadratic/volume) if volume else None,tau_max_Pa=max_norm,
        raw_point_normal_perturbation_max_Pa=max_normal,raw_point_shear_perturbation_max_Pa=max_shear,
        trace='native matching CellId; Gauss3 diagnostic rule; no component bounds')
    result[step]=entry
a.output.write_text(json.dumps(result,indent=2)+'\n')
