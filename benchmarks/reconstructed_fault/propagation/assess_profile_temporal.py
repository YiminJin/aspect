#!/usr/bin/env python3
"""Compare identical times, lag/scaling, controls, and physical reaction work."""
import json
import numpy as np
import assess_time_resolution as temporal
from profile_temporal_diagnosis import E, OLD, directory, Reference, load


def main():
    ds=[64,128,256,512,1024,2048,4096]
    temporal.directory=directory
    paths={d:temporal.trajectory(d) for d in ds}
    r=Reference(256,1.,24)
    cfg=load(OLD/'baseline.json');ref=paths[4096]
    comparisons={str(d):temporal.difference(a,ref,r) for d,a in paths.items()}
    pairs=[]
    for da,db in zip(ds[:-1],ds[1:]):
        pairs.append(dict(coarse=da,fine=db,common=temporal.difference(paths[da],paths[db],r,True),
                          all_coarse_times=temporal.difference(paths[da],paths[db],r)))
    orders={key:float(np.log2(pairs[-2]['common'][key]/pairs[-1]['common'][key]))
            for key in cfg['targets'] if min(pairs[-2]['common'][key],pairs[-1]['common'][key])>1e-12}
    rows={};controls={}
    for d in ds:
        s=load(E/f'diagnosis-{d}/summary.json')
        hist=load(E/f'diagnosis-{d}/history.json')
        regions=np.load(E/f'diagnosis-{d}/diagnostic-fields.npz')
        yq=regions['yq'];spatial=regions['accumulated_profile_work'];spatial_abs=regions['accumulated_absolute_profile_work']
        phase=paths[d]['phi']
        energy=paths[d]['energy']
        top=sorted(hist,key=lambda v:abs(v['actual_work_difference']),reverse=True)[:5]
        rows[str(d)]=dict(maxima=s['maxima'],endpoint_phase_final=s['endpoint_phase_final'],
                          incoming_phase_KKT=s['maximum_incoming_phase_KKT'],relative_velocity_max=s['relative_velocity_max'],
                          maximum_phase_increment=float(np.max(np.diff(phase,axis=0))),
                          signed_release=sum(v['D_relax'] for v in energy),
                          release_absolute=sum(abs(v['D_relax']) for v in energy),
                          final_phase_max=float(max(phase[-1])),onset=temporal.onset(paths[d]),
                          totals=s['totals'],top_steps=top,
                          absolute_work_fraction_after_point6=float(sum(abs(v['actual_work_difference']) for v in hist if v['time']>.6)/s['totals']['actual_work_difference']['absolute']),
                          center_work_fraction=dict(signed=float(sum(spatial[abs(yq-.5)<.1])),
                                                   absolute=float(sum(spatial_abs[abs(yq-.5)<.1])/sum(spatial_abs))),
                          quadrature_region_decomposition_error=float(abs(sum(spatial)-s['totals']['actual_work_difference']['signed'])))
    for d in [64,256,1024]:
        s=load(E/f'smooth-diagnosis-{d}/summary.json')
        controls[str(d)]=dict(maxima=s['maxima'],relative_velocity_max=s['relative_velocity_max'],totals=s['totals'])
    last=rows['4096']
    # Physical fields and a consistent reference are necessary, but persistent
    # endpoint phase lag / path-work discrepancy blocks native adoption.
    result=dict(partitions=ds,comparison_to_4096=comparisons,successive_halvings=pairs,
                last_common_orders=orders,targets=cfg['targets'],evolving=rows,smooth_controls=controls,
                selected_native_timestep=None,native_gate_pass=False,
                reason='Persistent endpoint phase lag and a failed energetic saved-entry correction require prototype/model-path review first.',
                core_pressure_and_material_unchanged=True)
    (E/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(last_comparison=pairs[-1],last_orders=orders,scaling={d:dict(chi_rate=v['maxima']['delta_chi_rate_max'],force=v['maxima']['redistributed_interior_force_max'],lag=v['maxima']['endpoint_phase_KKT'],work=v['totals']['actual_work_difference']['signed']) for d,v in rows.items()},native_gate=False),indent=2))


if __name__=='__main__':main()
