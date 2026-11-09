#!/usr/bin/env python3
"""Same-fixture native-basis comparison of physical layered exports."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import numpy as np
from layered_fields import State,surface,powers,NAMES
ROOT=Path(__file__).resolve().parents[4]
RUNS=ROOT/'build-particle-free/transported-objective/stage-c/material-runs'
labels=('layered-phase-integration-tip-efficiency02','layered-phase-integration-bernstein03')
records={};states={}
for label in labels:
 p=RUNS/label/'output';initial=State(p,0,.5);s=State(p,1,.5)
 states[label]=s
 records[label]=dict(energy=powers(s,initial,initial),initial_phase=initial.values[:,0].tolist(),
                     physical_surface={k:v.tolist() for k,v in surface(s,(np.arange(80)+.5)/80).items()})
a,b=(states[k] for k in labels);assert np.max(abs(a.x-b.x))==0
result=dict(bulk_max={k:float(np.max(abs(b.values[:,j]-a.values[:,j]))) for j,k in enumerate(NAMES+['pressure'])},
            surface_max={k:float(np.max(abs(np.asarray(records[labels[1]]['physical_surface'][k])-np.asarray(v)))) for k,v in records[labels[0]]['physical_surface'].items()},
            initial_phase_max=float(np.max(abs(np.array(records[labels[1]]['initial_phase'])-records[labels[0]]['initial_phase']))),
            energy_difference={k:float(records[labels[1]]['energy'][k]-v) for k,v in records[labels[0]]['energy'].items() if isinstance(v,(float,int))})
for r in records.values():del r['initial_phase'];del r['physical_surface']
out=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/native_bernstein/layered-comparison.json'
out.write_text(json.dumps(dict(cases=records,comparison=result),indent=2)+'\n')
print(json.dumps(result,indent=2))
