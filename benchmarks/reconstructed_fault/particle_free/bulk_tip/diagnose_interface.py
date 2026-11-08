#!/usr/bin/env python3
"""Decompose the retained four accepted interface steps without replaying them."""
import csv,json
from pathlib import Path
root=Path(__file__).resolve().parents[4]
source=root/'build-particle-free/propagation-recovery/runs/episode-short/output'
rows=[]
for step in range(749,753):
    with (source/f'cohesive-{step:06d}-diagnostics.csv').open() as f:
        d={r['quantity']:float(r['value']) for r in csv.DictReader(f)}
    r=dict(step=step,dt=d['dt_s'],fracture=d['fracture_increment_J_m'],physical=d['physical_remainder_J_m'],
           bulk_BE=d['BE_bulk_J_m'],surface_endpoint=d['surface_remainder_J_m'],
           endpoint_minus_trapezoid=d['endpoint_work_J_m']-d['trapezoid_work_J_m'],
           quadrature=d['quadrature_repartition_J_m'],ledger=d['ledger_error_J_m'],
           transfer_stress_Pa=d['transfer_stress_Pa'],source_error_Pa=d['source_error_Pa'])
    r['decomposition_error']=r['physical']-r['bulk_BE']-r['surface_endpoint']+r['endpoint_minus_trapezoid']-r['ledger']
    rows.append(r)
keys=[k for k in rows[0] if k not in ('step','dt','transfer_stress_Pa','source_error_Pa')]
result=dict(rows=rows,signed={k:sum(r[k] for r in rows) for k in keys},
            absolute={k:sum(abs(r[k]) for r in rows) for k in keys},
            note='The exported physical remainder uses trapezoidal work: BE + surface endpoint - (endpoint minus trapezoid) + ledger. Same dt and changing states do not establish timestep order. No projection/transfer occurred; source and ledger defects are reported separately.')
out=root/'benchmarks/reconstructed_fault/particle_free/evidence/bulk_tip/interface_decomposition.json'
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['signed'],indent=2))
