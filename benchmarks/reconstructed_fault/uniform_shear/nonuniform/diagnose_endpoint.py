#!/usr/bin/env python3
"""Bounded saved-data history-transfer audit, with no changes to mechanics.

Particle histories are joined by ID across accepted times. The native bulk
polynomial supplies the published FE transfer; replay exports additionally
identify the constrained history actually used by assembly.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from compare_cases import bulk_at, interpolate_q2
from measure_case import friction, project, read


def align_history(current, previous):
    current = np.sort(current, order='id')
    previous = np.sort(previous, order='id')
    if not np.array_equal(current['id'], previous['id']):
        raise ValueError('A particle was added or removed; ID history join is incomplete')
    return current, previous


def cell_average_transfer(current, previous, bulk, nx):
    """Reconstruct this fixture's actual cell-average / shared-node writes.

    The bulk export and production transfer traverse the same active cells.
    Last-cell assignment is reproduced, not replaced by a nodal averaging rule.
    """
    ny = 4*nx
    ix = np.clip((current['x']*nx/.25).astype(int),0,nx-1)
    iy = np.clip(((current['y']+.5)*ny).astype(int),0,ny-1)
    cell = ix*ny+iy
    counts = np.bincount(cell,minlength=nx*ny)
    if min(counts)==0:
        raise ValueError('Empty-cell fallback needs a separate audit')
    means = np.column_stack([np.bincount(cell,weights=previous[name],minlength=nx*ny)/counts
                            for name in ('tau_xx','tau_yy','tau_xy')])
    node = np.empty((2*nx+1,2*ny+1,3))
    winner = np.empty((2*nx+1,2*ny+1),dtype=int)
    # The exact nine-QP cell grouping is checked before relying on row order.
    bx = np.clip((bulk['x']*nx/.25).astype(int),0,nx-1)
    by = np.clip(((bulk['y']+.5)*ny).astype(int),0,ny-1)
    ids = (bx*ny+by).reshape(-1,9)
    if not np.all(ids==ids[:,0,None]) or len(np.unique(ids[:,0]))!=nx*ny:
        raise ValueError('Unexpected native quadrature/cell traversal')
    for cid in ids[:,0]:
        i,j = divmod(cid,ny)
        node[2*i:2*i+3,2*j:2*j+3] = means[cid]
        winner[2*i:2*i+3,2*j:2*j+3] = cid
    return node, winner, counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case',type=Path)
    p.add_argument('output',type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    get = lambda name,k: read(args.case,name,k,())
    initial = get('surface',0)
    nx = (len(np.unique(get('bulk',0)['x'])))//3
    ny = 4*nx
    node_x,node_y = np.meshgrid(np.linspace(0,.25,2*nx+1),np.linspace(-.5,.5,2*ny+1),indexing='ij')
    nodes = np.column_stack((node_x.ravel(),node_y.ravel()))
    report = dict(case=str(args.case),nx=nx,initial=dict(C0_min=float(min(initial['C'])),
        C0_max=float(max(initial['C'])),Theta0_min=float(min(initial['Theta'])),
        Theta0_max=float(max(initial['Theta']))),steps=[])
    for k in (1,2):
        current,previous = align_history(get('particles',k),get('particles',k-1))
        bulk,surface = get('bulk',k),get('surface',k)
        dt = float(get('time',k)[0]['dt']); beta=math.exp(-dt/100)
        selected = current['active']==1
        a,old = current[selected],previous[selected]
        points = np.column_stack((a['x'],a['y']))
        fe = interpolate_q2(bulk,bulk['old_tau_xy'],points)
        delta = beta*(fe-old['tau_xy'])
        predicted = bulk_at(bulk,surface,.3088215939070757,dt,points)['q']
        identity = float(max(abs(predicted-a['tau_xy']-delta)))
        if identity>2e-8:
            raise ValueError('Stress decomposition / correct ID history does not close')
        segment = a['segment'].astype(int)
        samples = np.column_stack((a['tau_xy'],beta*old['tau_xy'],
                                   a['tau_xy']-beta*old['tau_xy'],delta))
        mass,rhs,q = project(segment,a['xi'],a['volume'],samples,len(surface))
        old_mass,old_rhs,old_volume_q = project(segment,a['xi'],old['volume'],samples,len(surface))
        if not np.all(old['active']==1):
            raise ValueError('Association membership changed; geometry audit needs new samples')
        _,_,previous_geometry_q = project(old['segment'].astype(int),old['xi'],old['volume'],
                                          samples,len(surface))
        interp = lambda nodal: (1-a['xi'])*nodal[segment]+a['xi']*nodal[segment+1]
        old_surface=get('surface',k-1)
        v=interp(surface['V'])
        cohesive=-1e8*math.expm1(-dt/100)*v/interp(surface['Ih'])+beta*interp(old_surface['C'])
        F=a['tau_xy']-cohesive-1000*friction(v,interp(old_surface['Theta']))-1e5*v
        _,_,volume_F=project(segment,a['xi'],old['volume'],F[:,None],len(surface))
        _,_,current_F=project(segment,a['xi'],a['volume'],F[:,None],len(surface))
        sample_count = np.bincount(segment,minlength=len(surface))+np.bincount(segment+1,minlength=len(surface))
        rows = mass.sum(axis=1)
        probe = np.array([0,1,len(surface)//2,len(surface)-2,len(surface)-1])
        row = dict(time_s=float(get('time',k)[0]['time']),history_identity_error_Pa=identity,
            particle_count=len(current),active_count=len(a),
            changed_active_count=int(np.count_nonzero(current['active']!=previous['active'])),
            changed_segment_count=int(np.count_nonzero(a['segment']!=old['segment'])),
            max_coordinate_error_m=float(max(abs(a['x']-(segment+a['xi'])*(.25/(len(surface)-1))))),
            max_displacement_m=float(max(np.hypot(current['x']-previous['x'],current['y']-previous['y']))),
            probe_nodes=probe.tolist(),probe_row_mass_m2=rows[probe].tolist(),
            probe_sample_count=sample_count[probe].tolist(),
            probe_projected_transfer_error_Pa=q[probe,3].tolist(),
            probe_volume_only_q_change_Pa=(q-old_volume_q)[probe,0].tolist(),regions={})
        row['probe_geometry_only_old_history_change_Pa']=(q-previous_geometry_q)[probe,1].tolist()
        row['probe_geometry_only_current_q_change_Pa']=(q-previous_geometry_q)[probe,0].tolist()
        row['probe_previous_volume_residual_Pa']=volume_F[probe,0].tolist()
        row['probe_current_volume_residual_Pa']=current_F[probe,0].tolist()
        for name,mask in (('left',a['x']<.015625),
                          ('interior',(a['x']>.109375)&(a['x']<.140625)),
                          ('right',a['x']>.234375)):
            w=a['volume'][mask]
            row['regions'][name]=dict(count=int(sum(mask)),volume_m2=float(sum(w)),
                min_volume_m2=float(min(w)),volume_relative_change_max=float(max(abs(w/old['volume'][mask]-1))),
                transfer_error_mean_Pa=float(np.average(delta[mask],weights=w)),
                transfer_error_rms_Pa=float(np.sqrt(np.average(delta[mask]**2,weights=w))),
                old_stress_mean_Pa=float(np.average(old['tau_xy'][mask],weights=w)))
        nodal,winner,counts = cell_average_transfer(current,previous,bulk,nx)
        observed = interpolate_q2(bulk,bulk['old_tau_xy'],nodes).reshape(2*nx+1,2*ny+1)
        row['published_last_cell_assignment_error_Pa']=float(max(abs(observed-nodal[:,:,2]).ravel()))
        row['particles_per_cell_min_max']=[int(min(counts)),int(max(counts))]
        row['published_periodic_trace_jump_max_Pa']={name:float(max(abs(nodal[0,:,c]-nodal[-1,:,c])))
                                                   for c,name in enumerate(('xx','yy','xy'))}
        if (args.case/f'history_transfer_{k}.csv').exists():
            transfer=get('history_transfer',k)
            if not np.array_equal(transfer['row'],np.arange(len(bulk))):
                raise ValueError('Replay transfer rows do not match bulk quadrature')
            row['assembly_history']={}
            for c,name in enumerate(('xx','yy','xy')):
                pub,actual=transfer['published_'+name],transfer['assembly_'+name]
                pub_nodes=interpolate_q2(bulk,pub,nodes).reshape(2*nx+1,2*ny+1)
                actual_nodes=interpolate_q2(bulk,actual,nodes).reshape(2*nx+1,2*ny+1)
                row['assembly_history'][name]=dict(
                    published_last_cell_error_Pa=float(max(abs(pub_nodes-nodal[:,:,c]).ravel())),
                    assembly_minus_published_qp_max_Pa=float(max(abs(actual-pub))),
                    assembly_periodic_trace_jump_max_Pa=float(max(abs(actual_nodes[0]-actual_nodes[-1]))),
                    left_trace_change_max_Pa=float(max(abs(actual_nodes[0]-pub_nodes[0]))),
                    right_trace_change_max_Pa=float(max(abs(actual_nodes[-1]-pub_nodes[-1]))))
                if name=='xy':
                    actual_particle=interpolate_q2(bulk,actual,points)
                    _,_,actual_projection=project(segment,a['xi'],a['volume'],
                        (beta*(actual_particle-old['tau_xy']))[:,None],len(surface))
                    row['assembly_history'][name]['probe_projected_transfer_error_Pa']=actual_projection[probe,0].tolist()
        np.savetxt(args.output/f'weak_terms_{k}.csv',np.column_stack((surface['x'],rows,q,rhs/rows[:,None],q[:,0]-old_volume_q[:,0],
            q[:,1]-previous_geometry_q[:,1],q[:,0]-previous_geometry_q[:,0])),
            delimiter=',',header='s,mass,q,old_particle,rate,published_transfer_error,weak_q,weak_old_particle,weak_rate,weak_transfer_error,volume_only_q_change,geometry_only_old_history_change,geometry_only_current_q_change',comments='')
        # Preserve a cell-ID-independent structured source index for each node;
        # native cell ID/QP provenance remains in the original bulk_cell_ids CSV.
        np.savetxt(args.output/f'transfer_nodes_{k}.csv',np.column_stack((nodes,winner.ravel(),nodal.reshape(-1,3))),
            delimiter=',',header='x,y,source_structured_cell,published_xx,published_yy,published_xy',comments='')
        report['steps'].append(row)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
