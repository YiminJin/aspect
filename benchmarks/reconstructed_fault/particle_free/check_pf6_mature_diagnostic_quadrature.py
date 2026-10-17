#!/usr/bin/env python3
"""Fixed Gauss6 control of diagnostic norms; does not change Gauss6 mechanics."""
import argparse,json,time
from pathlib import Path
import numpy as np
from prepare_pf6_mature_resolution import mesh
from sample_pf6_mature_dg import sample
from analyze_pf6_mature import read,norm3
from analyze_pf6_mature_resolution import UPPER,NORMAL
p=argparse.ArgumentParser(description=__doc__);p.add_argument('fine',type=Path);p.add_argument('ff',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
start=time.monotonic();grid=mesh(a.fine);assert len(grid)<=40000
q,w=np.polynomial.legendre.leggauss(6);q=(q+1)/2;w=w/2
points=np.concatenate([lo+h*np.array([[x,y] for y in q for x in q]) for cell,level,lo,h in grid]);weights=np.concatenate([h*h*np.array([x*y for y in w for x in w]) for cell,level,lo,h in grid])
first=sample(a.fine,3,points);second=sample(a.ff,3,points)
physical=read(Path(__file__).resolve().parents[3]/'build-particle-free/mature-resolution/inputs-lf/physical-profile.csv')
h=np.interp(abs((points-UPPER)@NORMAL),physical['r_m'],physical['h'],right=0.)
difference={key:first[key]-second[key] for key in ('tau_xx','tau_yy','tau_zz','tau_xy')}
result=dict(points=len(points),diagnostic_rule='Gauss6 per coordinate on fine mesh; saved full DG/Q1 fields; native mechanics unchanged',seconds=time.monotonic()-start)
for label,weight in [('whole',weights),('localized',weights*h)]:
    result[label+'_difference_RMS_Pa']=float(np.sqrt(np.sum(weight*norm3(difference))/np.sum(weight)))
    result[label+'_fine_RMS_Pa']=float(np.sqrt(np.sum(weight*norm3(first))/np.sum(weight)))
    result[label+'_FF_RMS_Pa']=float(np.sqrt(np.sum(weight*norm3(second))/np.sum(weight)))
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
