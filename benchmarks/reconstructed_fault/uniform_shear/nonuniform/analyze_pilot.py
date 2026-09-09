#!/usr/bin/env python3
"""K2.1 measurements against a homogeneous *coupled* run, not local K1 roots."""
import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import boundary_error, read
from analyze_convergence import profile_primitive


def initial_theta(x):
    return 200*(1+.05*np.where(abs(x-.125)<.0625,
                              ((1+np.cos(np.pi*(x-.125)/.0625))/2)**2, 0))


def stress(bulk, dt):
    # Complete evaluated Maxwell stress; tau_xy composition alone is history.
    return (bulk['kappa']*(bulk['ux_y']+bulk['uy_x']
                           -bulk['chi']*bulk['V']-bulk['history'])
            + math.exp(-dt/100)*bulk['old_tau_xy'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('homogeneous', type=Path)
    args = parser.parse_args()
    output = args.directory.parent/'measurements'
    output.mkdir(exist_ok=True)
    get = lambda name, step: read(args.directory, name, step, ())
    baseline = lambda name, step: read(args.homogeneous, name, step, ())
    phase, surface, particles = get('phase', 0), get('surface', 0), get('particles', 0)
    segments = get('segments', 0)
    for name in ('phase', 'segments'):
        if not np.array_equal(get(name, 0), baseline(name, 0)):
            raise ValueError('K2 geometry/profile changed from validated K1: '+name)
    if max(abs(surface['y']))>1e-13 or max(abs(segments['nx']))>1e-13:
        raise ValueError('Horizontal-profile diagnostic is not applicable')
    x = surface['x']
    weights = np.r_[np.diff(x)/2, 0]+np.r_[0, np.diff(x)/2]
    rms = lambda values: float(np.sqrt(np.dot(weights, values**2)/sum(weights)))
    # Independently reconstruct the consistent Q1 initial-state projection
    # from actual production particle associations and domain volumes.
    mass, rhs = np.zeros((len(x), len(x))), np.zeros(len(x))
    for p in particles[particles['active']==1]:
        s, shape = int(p['segment']), np.array([1-p['xi'], p['xi']])
        mass[s:s+2,s:s+2] += p['volume']*np.outer(shape, shape)
        rhs[s:s+2] += p['volume']*shape*initial_theta(p['x'])
    theta = np.linalg.solve(mass, rhs)
    projection_error = float(max(abs(theta-surface['Theta'])))
    if projection_error>1e-8:
        raise ValueError('Supplied projected Theta0 was not retained')
    profile = np.sort(phase[phase['x']==np.unique(phase['x'])[len(np.unique(phase['x']))//2]], order='y')
    width = segments['half_width_plus'][0]
    primitive, integral = profile_primitive(profile['y'], profile['phi'], np.array([-width,width]))
    omitted = float(1-(primitive[1]-primitive[0])/integral)
    report = dict(
        reference='homogeneous fully coupled ASPECT baseline; no K2 numerical reference yet',
        initial=dict(Theta_min_s=float(min(theta)), Theta_max_s=float(max(theta)),
                     projection_error_s=projection_error,
                     nodal_difference_from_analytic_s=float(max(abs(theta-initial_theta(x)))),
                     endpoint_Theta_s=theta[[0,-1]].tolist(),
                     remote_Theta0_difference_s=float(max(abs(theta[abs(x-.125)>=.09375]-200))),
                     endpoint_projection_mass_m2=np.diag(mass)[[0,-1]].tolist(),
                     max_C0_difference_Pa=float(max(abs(surface['C']-baseline('surface',0)['C']))),
                     max_particle_stress0_error_Pa=float(max(abs(particles['tau_xy']-1500))),
                     max_H0_difference_Pa=float(max(abs(particles['H']-baseline('particles',0)['H'])))),
        support=dict(half_width_m=float(width), independent_Ih_m=float(integral),
                     omitted_fraction=omitted, original_containment_target=1e-6,
                     containment_pass=bool(omitted<=1e-6), K1_allowance_applied=False),
        steps=[])
    slip, base_slip = np.zeros(len(x)), np.zeros(len(x))
    fig, axes = plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for step in range(5):
        time = get('time',step)[0]
        if not np.array_equal(get('time',step),baseline('time',step)):
            raise ValueError('Different accepted loading/time sequence')
        if not np.array_equal(get('phase',step),phase):
            raise ValueError('Phase field was not frozen')
        bulk, base_bulk = get('bulk',step), baseline('bulk',step)
        if not np.array_equal(bulk[['x','y','weight']],base_bulk[['x','y','weight']]):
            raise ValueError('Bulk quadrature changed')
        state, base_state, pp = get('surface',step),baseline('surface',step),get('particles',step)
        dt = float(time['dt'])
        if step:
            theta = theta*np.exp(-state['V']*dt/.001)+.001/state['V']*(-np.expm1(-state['V']*dt/.001))
            slip += dt*state['V']; base_slip += dt*base_state['V']
        q, base_q = stress(bulk,dt),stress(base_bulk,dt)
        dv = state['V']-base_state['V']
        outside = abs(x-.125)>.0625
        # Integrate actual exported chi*V+history over each full normal column.
        # Column weights sum to its tangential Gauss weight times W=1 m.
        bx, index = np.unique(bulk['x'],return_inverse=True)
        sums = np.bincount(index,weights=bulk['weight'])
        normal = lambda values: np.bincount(index,weights=bulk['weight']*values)/sums
        normalization = normal(bulk['chi']*bulk['V']+bulk['history'])/np.interp(bx,x,state['V'])
        global_norm = float(np.dot(bulk['weight'],bulk['chi']*bulk['V']+bulk['history'])/np.dot(weights,state['V']))
        normal_q = normal(q)
        row = dict(time_s=float(time['time']), dt_s=dt,
                   V_min_m_s=float(min(state['V'])),V_max_m_s=float(max(state['V'])),
                   delta_V_rms_m_s=rms(dv),delta_V_min_m_s=float(min(dv)),delta_V_max_m_s=float(max(dv)),
                   outside_delta_V_max_m_s=float(max(abs(dv[outside]))),
                   outside_delta_V_rms_m_s=float(np.sqrt(np.average(dv[outside]**2,weights=weights[outside]))),
                   remote_delta_V_max_m_s=float(max(abs(dv[abs(x-.125)>=.09375]))),
                   delta_ux_max_m_s=float(max(abs(bulk['ux']-base_bulk['ux']))),
                   delta_p_max_Pa=float(max(abs(bulk['p']-base_bulk['p']))),
                   delta_q_max_Pa=float(max(abs(q-base_q))),
                   bulk_normal_average_q_range_Pa=[float(min(normal_q)),float(max(normal_q))],
                   Theta_update_error_s=float(max(abs(theta-state['Theta']))),
                   slip_difference_rms_m=rms(slip-base_slip),
                   global_normalization=global_norm,
                   max_column_normalization_error=float(max(abs(normalization-1))),
                   normalization_pass=bool(max(abs(normalization-1))<=1e-4 and abs(global_norm-1)<=1e-4),
                   max_Ih_relative_error=float(max(abs(state['Ih']/integral-1))),
                   boundary_velocity_error_m_s=boundary_error(bulk,float(time['U'])),
                   particle_volume_sum_m2=float(sum(pp['volume'])),
                   particle_volume_relative_error=float(sum(pp['volume'])/.25-1),
                   particle_volume_min_m2=float(min(pp['volume'])),
                   H_change_Pa=float(max(abs(np.sort(pp,order='id')['H']-np.sort(particles,order='id')['H']))))
        report['steps'].append(row)
        np.savetxt(output/f'surface_{step}.csv',np.column_stack((x,state['V'],base_state['V'],dv,state['Theta'],theta,
                    state['C'],state['Ih'],slip,slip-base_slip,np.full(len(x),1000))),delimiter=',',
                   header='s,V,homogeneous_V,delta_V,Theta,Theta_update_check,C_retained,Ih,slip,delta_slip,sigma_prescribed_Pa',comments='')
        np.savetxt(output/f'normal_columns_{step}.csv',np.column_stack((bx,normal_q,normal(q-base_q),normalization)),
                   delimiter=',',header='x,bulk_normal_average_q_Pa,delta_bulk_normal_average_q_Pa,actual_slip_normalization',comments='')
        np.savetxt(output/f'raw_qps_{step}.csv',np.column_stack((bulk['x'],bulk['y'],bulk['weight'],q,q-base_q,
                    bulk['ux'],bulk['uy'],bulk['p'])),delimiter=',',header='x,y,weight,q_Pa,delta_q_Pa,ux,uy,p',comments='')
        near_y = min(bulk['y'][bulk['y']>0])
        near = np.flatnonzero(bulk['y']==near_y)
        near = near[np.argsort(bulk['x'][near])]
        np.savetxt(output/f'near_fault_qps_{step}.csv',np.column_stack((bulk['x'][near],bulk['y'][near],q[near],(q-base_q)[near])),
                   delimiter=',',header='x,y,q_Pa,delta_q_Pa',comments='')
        for location in (.03125,.125):
            sample_x = bx[np.argmin(abs(bx-location))]
            mask = bulk['x']==sample_x
            column = np.column_stack((bulk['x'][mask],bulk['y'][mask],q[mask],(q-base_q)[mask],
                                      bulk['ux'][mask],bulk['uy'][mask],bulk['p'][mask]))
            np.savetxt(output/f'transverse_x{location}_{step}.csv',column[np.argsort(column[:,1])],delimiter=',',
                       header='x,y,q_Pa,delta_q_Pa,ux,uy,p',comments='')
        for axis,values,label in zip(axes,(dv,state['Theta'],(q-base_q)[near]),
                                     ('delta V [m/s]','Theta [s]',f'raw delta q at y={near_y:.3g} m [Pa]')):
            axis.plot(bulk['x'][near] if axis is axes[2] else x,values,label=f'{time["time"]:g} s')
            axis.set(xlabel='x [m]',ylabel=label)
    axes[0].legend(); fig.savefig(output/'coupled_profiles.png',dpi=160); plt.close(fig)
    report['completed_to_2_seconds'] = report['steps'][-1]['time_s']==2
    report['lifecycle_and_normalization_pass'] = all(r['normalization_pass'] and r['Theta_update_error_s']<1e-8
        and r['boundary_velocity_error_m_s']<1e-13 and r['H_change_Pa']==0
        and r['particle_volume_min_m2']>0 for r in report['steps'])
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
