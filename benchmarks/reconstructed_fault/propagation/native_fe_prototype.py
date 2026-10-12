#!/usr/bin/env python3
"""Endpoint comparator with the fresh native Q1 seed; prototype physics unchanged.

The prototype retains its single free bulk mode. It is a comparator, not a
full-Stokes reference. Old prototype seeds and evidence remain untouched.
"""
import csv
from pathlib import Path
import numpy as np
import total_energy_time_comparison as comparison
from total_energy_step import state,N,ROOT
source=ROOT/'build-propagation-integration/output-native-fe-zero/fe-phase-0.csv'
def native_seed(energy):
    with source.open() as stream:
        values={(float(r['x']),float(r['y'])):float(r['phi']) for r in csv.DictReader(stream)}
    phi=np.array([values[(i/N,j/N)] for i in range(N+1) for j in range(N+1)])
    return state(energy,phi,np.zeros_like(energy.x),np.zeros(energy.ns+1)),source
comparison.initial_state=native_seed
out=Path(__file__).resolve().parent/'native_fe_evidence/prototype64'
out.mkdir(exist_ok=False)
comparison.run(out,64,'endpoint',True,.256,1.,True)
