#!/usr/bin/env python3
"""Compact measured campaign tables; no automatic qualification thresholds."""
import argparse,csv,json
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from assess_layered_campaign import OUT

def summarize(name):
 c=json.loads((OUT/(name+'.json')).read_text());r=c['records'];a=r[0];b=r[-1];d=b['spin_diagnostics']
 expose=np.array(d['fault_spin_exposure']);idx=int(np.argmax(abs(expose)))
 tendency=np.zeros(len(expose))
 for before,after in zip(r[:-1],r[1:]):tendency+=after['dt']*np.array(before['spin_diagnostics']['stress_weighted_tendency'])
 actual=[];predicted=[];slip_step=[]
 for before,after in zip(r[:-1],r[1:]):
  theta=np.array(before['age_vertices']);v=np.array(before['slip_vertices']);dt=after['dt'];f=v*dt/.008
  pred=theta*np.exp(-f)+dt*(-np.expm1(-f))/f
  predicted.append(float(np.max(abs(np.log(pred/theta)))))
  actual.append(float(np.max(abs(np.log(np.array(after['age_vertices'])/theta)))))
  slip_step.append(float(np.max(np.array(after['slip_vertices'])*dt/.008)))
 row=dict(max_predicted_log_age_change=max(predicted),max_actual_log_age_change=max(actual),max_endpoint_Vdt_over_Dc=max(slip_step),case=name,steps=len(r)-1,T=b['time'],h=4/(2**c['arguments']['level']*c['arguments'].get('depth_repetitions',1)),U=c['arguments']['translation'],law=c['arguments']['spin'],
 dt_min=min(z['dt'] for z in r[1:]),dt_max=max(z['dt'] for z in r[1:]),wall_s=c['wall_s'],linear_solves=c['cost']['stokes_solves'],linear_iterations=c['cost']['stokes_iterations'],
 max_V=max(max(z['slip_vertices']) for z in r),max_slip=max(b['cumulative_vertices']),min_age=min(min(z['age_vertices']) for z in r),
 envelope_rad=d['sampled_max_envelope'],max_label_exposure_rad=float(abs(expose[idx])),exposure_label=idx,
 max_label_integrated_tendency_Pa=float(tendency.max()),
 min_phase=min(z['energy']['nodal_phase_min'] for z in r),max_phase=max(z['energy']['nodal_phase_max'] for z in r),
 final_phase_peak=b['energy']['peak_phase'],phase_mass_change=b['energy']['profile_mass']/a['energy']['profile_mass']-1,
 normalization_Linf=max(abs(z['energy']['normalization_relative']) for z in r),boundary_phase_max=max(z['energy']['boundary_phase_max'] for z in r),
 curve_clearance=c['checks']['minimum_curve_clearance'],mean_x_displacement=float(np.mean(np.array(b['curve'])[:,0]-np.array(a['curve'])[:,0])),
 endpoint_signed=c['energy']['signed'],endpoint_absolute=c['energy']['absolute'],endpoint_relative=c['energy']['relative'],
 trapezoid_signed=c['energy']['trapezoid_signed'],trapezoid_absolute=c['energy']['trapezoid_absolute'],
 energy_active=c['energy']['active'],geometry_replay=c['checks']['geometry_replay_Linf'],age_replay=c['checks']['age_replay_Linf'])
 return row

def main():
 p=argparse.ArgumentParser();p.add_argument('cases',nargs='+');a=p.parse_args();rows=[summarize(n) for n in a.cases]
 (OUT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (OUT/'summary.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 print(json.dumps(rows,indent=2),flush=True)
if __name__=='__main__':main()
