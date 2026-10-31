#!/usr/bin/env python3
"""Frozen common surface memory only; never writes/feeds native histories.
Compare native bulk responses through common work-conjugate test functions.
"""
import argparse,json,csv
from pathlib import Path
import numpy as np
from scipy.linalg import solve
p=argparse.ArgumentParser(__doc__);p.add_argument('coarse');p.add_argument('fine');p.add_argument('surface',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
Q=np.loadtxt(a.fine+'-Q.csv',delimiter=',');pts=np.genfromtxt(a.fine+'-points.csv',delimiter=',',names=True);w=pts['weight_m'];M=Q.T@(w[:,None]*Q)
rows=list(csv.DictReader(a.surface.open()));memory=np.array([float(r['relaxed_jump_m']) for r in rows]);assert len(memory)==len(w)
u=solve(M,Q.T@(w*memory));error=float(max(abs(Q@u-memory)));assert error<1e-12,error
Kc=np.loadtxt(a.coarse+'-K.csv',delimiter=',')[1:,1:];Kf=np.loadtxt(a.fine+'-K.csv',delimiter=',')[1:,1:];fc=Kc@u;ff=Kf@u;diff=fc-ff
x=np.linspace(4,10.125,Q.shape[1]+2)[1:-1];result=[]
for center in np.linspace(4.5,9.5,81):
 test=np.exp(-.5*((x-center)/.375)**2);measure=float(w@(Q@test))
 result.append([center,float(test@fc/measure),float(test@ff/measure),float(test@diff/measure)])
np.savetxt(a.output/'work-loads.csv',result,delimiter=',',header='kernel_center_x_m,coarse_memory_resistance_Pa,fine_memory_resistance_Pa,difference_Pa',comments='')
r=dict(surface=str(a.surface),memory_reproduction_m=error,kernel_standard_deviation_m=.375,kernel='Q1 interpolant of physical Gaussian, normalized by actual surface quadrature',fine_slip_response_storage_J_m=float(.5*u@ff),storage_difference_J_m=float(.5*u@diff),storage_relative=float(u@diff/(u@ff)),unfiltered_weak_load_mass_norm_relative=float(np.sqrt(diff@solve(M,diff)/(ff@solve(M,ff)))),maximum_averaged_load_difference_Pa=float(np.max(abs(np.array(result)[:,3]))))
(a.output/'common-loads.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
