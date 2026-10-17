#!/usr/bin/env python3
"""Unfiltered point tractions from the bounded 60-degree authoritative stress export.

These are common Gauss3 diagnostic points, independent of the selected mechanical
integration rule. No filter, component clipping, or second Maxwell update occurs.
"""
import argparse
import csv
import math
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('stress',type=Path);p.add_argument('output',type=Path)
a=p.parse_args()
Vp,Vref,normal0,damping=1e-9,1e-6,50e6,4624440.
a_rsf,b_rsf,mu0=.025,.015,.6
shear0=normal0*a_rsf*math.asinh(Vp/(2*Vref)*math.exp((mu0+b_rsf*math.log(Vref/Vp))/a_rsf))+damping*Vp
with a.stress.open() as source,a.output.open('x') as target:
    reader=csv.DictReader(source);writer=csv.writer(target)
    writer.writerow(['cell','qp','x','y','JxW','chi','pressure_Pa','tauN_Pa','tauS_Pa','raw_normal_Pa','raw_shear_Pa'])
    for row in reader:
        xx,yy,xy,pressure=(float(row[k]) for k in ('tau_xx','tau_yy','tau_xy','p'))
        tensor_normal=.75*xx+.25*yy+math.sqrt(3)/2*xy
        tensor_shear=-math.sqrt(3)/4*xx+math.sqrt(3)/4*yy+.5*xy
        writer.writerow([row[k] for k in ('cell','qp','x','y','JxW','chi')]+
            [format(v,'.17g') for v in (pressure,tensor_normal,tensor_shear,
                                       normal0+pressure-tensor_normal,shear0+tensor_shear)])
