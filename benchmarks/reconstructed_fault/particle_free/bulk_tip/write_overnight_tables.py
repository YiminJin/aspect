#!/usr/bin/env python3
"""Flat numerical evidence tables for completed overnight histories and retained BE."""
import argparse,csv,json,os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from analyze_material_overnight import OUT,growth,read

def write(name,rows):
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('labels',nargs='+');args=parser.parse_args()
    energies=[];summary=[];regions=[];regional_windows=[];thresholds=[];onsets=[];costs=[]
    for label in args.labels:
        d=growth.RUNS/label/'output';rows=read(d/'bulk-accepted.csv');power=read(d/'tip-power.csv')
        assert np.array_equal(rows.step,power.step)
        path=OUT/f'{label}-analysis.json'
        if not path.exists():path=growth.ROOT/f'benchmarks/reconstructed_fault/particle_free/evidence/synthetic_growth/{label}.json'
        result=json.loads(path.read_text());command=result['command'];initial=result['initial_fracture'];dt=command['arguments']['dt'];h=.125/2**command['arguments']['refinement'];method=command['arguments']['time_update'] or 'backward Euler'
        prefix=dict(label=label,method=method,h=h,dt=dt,accepted_end=float(rows.time[-1]))
        for window,start in [('full',0.),('late',.04)]:
            mask=rows.time>start+1e-10
            for field in ['Wext','energy_change','Db','Dc','Df','stress_loss','traction_loss','phase_remainder','fracture_loss','phase_work','mechanical_work','ledger_error']:
                v=rows[field][mask];energies.append(dict(**prefix,window=window,term=field,signed=float(sum(v)),absolute=float(sum(abs(v)))))
            for field in ['friction','radiation']:
                v=power[field][mask];energies.append(dict(**prefix,window=window,term=field,signed=float(sum(v)),absolute=float(sum(abs(v)))))
            v=(rows.Wext-rows.energy_change-rows.Db-rows.Dc-rows.Df)[mask]
            energies.append(dict(**prefix,window=window,term='R_phys',signed=float(sum(v)),absolute=float(sum(abs(v)))))
            parent=initial if start==0 else rows.fracture[np.argmin(abs(rows.time-start))]
            added=float(rows.fracture[-1]-parent)
            energies.append(dict(**prefix,window=window,term='added_fracture',signed=added,absolute=float(sum(abs(np.diff(np.r_[parent,rows.fracture[mask]]))))))
            summary.append(dict(**prefix,window=window,added_fracture=added,R_signed=float(sum(v)),R_absolute=float(sum(abs(v))),absolute_ratio=float(sum(abs(v))/added),channel_closure_max=float(max(abs(rows.Df[mask]-power.friction[mask]-power.radiation[mask]))),wall_seconds=command['seconds']))
        last=result['steps'][-1]
        for region,data in last['regions'].items():regions.append(dict(**prefix,region=region,**data))
        at_four=min(result['steps'],key=lambda item:abs(item['time']-.04))
        assert abs(at_four['time']-.04)<1e-10
        for window,parent in [('full',None),('late',at_four)]:
            for region,data in last['regions'].items():
                previous=parent['regions'][region] if parent else {}
                regional_windows.append(dict(**prefix,window=window,region=region,
                    added_fracture=data['added_fracture']-previous.get('added_fracture',0.),
                    volume_fracture_change=data['volume_fracture_change']-previous.get('volume_fracture_change',0.)))
        profile=read(d/'tip-profile.csv');solver=read(d/'tip-solver.csv')
        costs.append(dict(**prefix,ranks=command['arguments']['ranks'],intervals=len(rows),
            wall_seconds=command['seconds'],max_interval_seconds=float(max(profile.step_seconds)),
            nonlinear_iterations=int(sum(solver.nonlinear_iterations)),
            max_nonlinear_iterations=int(max(solver.nonlinear_iterations)),
            krylov_iterations=int(sum(solver.krylov_iterations)),stokes_solves=int(sum(solver.stokes_solves)),
            line_trials=int(sum(profile.line_trials)),evaluations=int(sum(profile.evaluations)),
            tangent_assemblies=int(sum(profile.tangent_assemblies))))
        for axis,levels in last['polynomial_sections']['extents'].items():
            for level,positions in levels.items():
                before=result['initial_sections']['extents'][axis][level]
                thresholds.append(dict(**prefix,axis=axis,level=level,initial_left=before[0],initial_right=before[1],final_left=positions[0],final_right=positions[1],left_displacement_over_h=(before[0]-positions[0])/h,right_displacement_over_h=(positions[1]-before[1])/h))
        for event,t in result['onset'].items():onsets.append(dict(**prefix,event=event,first_endpoint=t,previous_endpoint=max(0.,t-dt),stage_time=t-(.5*dt if method=='midpoint' else 0)))
    for name,data in [('energy-components.csv',energies),('energy-summary.csv',summary),('regional-fracture.csv',regions),('regional-fracture-windows.csv',regional_windows),('threshold-positions.csv',thresholds),('onset-brackets.csv',onsets),('costs.csv',costs)]:write(name,data)
    print('Wrote full/late energy channels, regions, thresholds and onset tables.')

if __name__=='__main__':main()
