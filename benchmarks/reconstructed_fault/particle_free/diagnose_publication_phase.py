#!/usr/bin/env python3
"""Read-only endpoint branch diagnostic; never align the primary comparison."""
import argparse,json
from pathlib import Path
import numpy as np
from assess_phase_energy import csv
from pf6a_spatial_reference import Endpoint,A
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('reference',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
f=np.load(a.reference/'fields.npz');history=json.loads((a.reference/'history.json').read_text());nodes=csv(a.run/'run/state-72/nodes.csv')
y=f['y'];values=np.array([[nodes['phase'][(nodes['y']==yy)&(nodes['x']==xx)][0]for xx in [0.,.5,1.]]for yy in y]);native=values@np.array([.25,.5,.25]);reference=f['phase'][-1]
model=Endpoint(512,512,12);old=history[-2];model.incoming(f['phase'][-2],old['meanT'],old['Z'],old['slip'],.5625,1/128)
rows={}
for name,phase in [('native_x_average',native),('selected_reference',reference)]:
 potential,g=model.evaluate(phase);H=model.hessian(phase);direction=np.zeros_like(phase);direction[1:-1]=-(phase[2:]-phase[:-2])/(2/512);direction[phase<=model.lower+1e-13]=0.;direction/=max(abs(direction))
 sampled=model.B@phase;mass=float(model.w@sampled);centroid=float(model.w@(model.yq*sampled)/mass)
 projected=np.where(phase<=model.lower+1e-13,np.minimum(g,0.),g)
 rows[name]=dict(phase_centroid=centroid,potential_J_per_m=float(A*potential),KKT_dimensionless=float(max(abs(projected))),
                 minimum_gap_to_reference_incoming_phase=float(min(phase-model.lower)),translation_curvature_J_per_m=float(A*(direction@(H@direction))))
result=dict(primary_unregistered_errors_retained=True,no_native_state_or_reference_modified=True,
 native_along_fault_phase_spread=float(max(np.ptp(values,axis=1))),phase_centroid_difference=float(rows['native_x_average']['phase_centroid']-rows['selected_reference']['phase_centroid']),
 mirrored_profile_Linf_diagnostic_only=float(max(abs(native-reference[::-1]))),unregistered_profile_Linf=float(max(abs(native-reference))),
 energy_difference_same_incoming_J_per_m=float(rows['native_x_average']['potential_J_per_m']-rows['selected_reference']['potential_J_per_m']),states=rows,
 comparison_rules=dict(native='second candidate must improve current best by the retained arithmetic allowance',independent_reference='candidate must improve entry by the allowance; best-candidate comparison is strict without that allowance'),
 limitation='The x-averaged reduced 1D curvature is a branch diagnostic, not a full 2D stability qualification; reflection is never used in primary errors.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
