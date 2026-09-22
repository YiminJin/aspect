"""Bounded offline onset/timing audit of the first long BP3 run.

Deep friction attribution uses projected nodal tractions as a labelled proxy,
not as a reconstruction of the unexported production quadrature residual.
"""
import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from plot_fault_evolution import read_profiles, YEAR
from plot_cumulative_slip import read_profiles as read_slip

VP, VMIN, DC, VREF, B, A, MU0, ETA = 1e-9, 1e-20, .008, 1e-6, .015, .025, .6, 4624440.


def records(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mu(v, theta):
    return A*np.arcsinh(v/(2*VREF)*np.exp((MU0+B*np.log(theta*VREF/DC))/A))


def extrema(v, x):
    # The neighbour minimum avoids counting a smooth monotonic front as ringing.
    neighbours = np.minimum(v[:-2], v[2:])
    eligible = (x[1:-1] >= 5000) & (x[1:-1] <= 18000) & (neighbours > 1e-12)
    relative = np.zeros(len(v)-2)
    relative[eligible] = 1-v[1:-1][eligible]/neighbours[eligible]
    dips = np.flatnonzero(eligible & (relative > .1))+1
    strongest = int(np.argmax(relative))+1
    return dips, strongest, float(relative[strongest-1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    out = args.output or args.run/'bounded_diagnosis'
    out.mkdir(exist_ok=True)
    geometry, times, steps, data, _, missing = read_profiles(args.run, skip_missing=True)
    x = geometry[:, 1]
    order = geometry[:, 0].astype(int)
    by_step = {int(step): i for i, step in enumerate(steps)}
    with (args.run/'accepted_steps.csv').open() as stream:
        accepted = {int(r['step']): r for r in csv.DictReader(stream)}
    controls = [int(np.argmin(np.abs(x-target))) for target in (25000., 40000., 80000.)]
    theta = data['Theta_s'][0].copy()
    initial_theta = theta.copy()
    profile_metrics, history_metrics, deep, node_records = [], [], [], []
    first = None
    previous_slip = None
    max_theta_error = 0.
    max_control_theta_error = 0.
    snapshots = {}
    for slip_profile in read_slip(args.run/'cumulative_slip.csv'):
        step, time = slip_profile.step, slip_profile.time
        slip = slip_profile.slip[order]
        index = by_step.get(step)
        incoming = theta.copy()
        inferred = index is None
        if previous_slip is None:
            v = data['V_m_per_s'][0]
            dt = 0.
        else:
            dt = time-previous_time
            v = np.maximum((slip-previous_slip)/dt, VMIN)
            if index is not None:
                v = data['V_m_per_s'][index]
            z = v*dt/DC
            factor = np.ones_like(z)
            np.divide(-np.expm1(-z), z, out=factor, where=z != 0)
            theta = theta*np.exp(-z)+dt*factor
        dips, strongest, depth = extrema(v, x)
        metric = dict(step=step, time_yr=time/YEAR, dt_s=dt,
                      rate_source='delta_slip/dt' if inferred else 'saved_profile_V',
                      significant_dips=len(dips), deepest_relative_dip=depth,
                      strongest_node=int(order[strongest]), strongest_xd_m=x[strongest],
                      shallow_at_floor=int(np.count_nonzero((v <= VMIN*(1+1e-8)) & (x<=18000))),
                      floor_count_is_inferred=inferred,
                      solver_lower_active_count=int(accepted[step]['lower_active']) if step in accepted else '')
        profile_metrics.append(metric)
        if first is None and len(dips):
            first = metric.copy()
            onset_step = step
            onset_x = x[strongest]
        if index is not None:
            error = np.abs(theta/data['Theta_s'][index]-1)
            max_theta_error = max(max_theta_error, float(error.max()))
            max_control_theta_error = max(max_control_theta_error, float(error[controls].max()))
            history_metrics.append(dict(step=step, time_yr=time/YEAR,
                                        max_theta_relative_error=float(error.max()),
                                        deep_control_theta_relative_error=float(error[controls].max())))
            for c, target in zip(controls, (25, 40, 80)):
                q, sigma = data['q_weak_Pa'][index, c], data['sigma_n_weak_Pa'][index, c]
                q0, sigma0 = data['q_weak_Pa'][0, c], data['sigma_n_weak_Pa'][0, c]
                # Exact additive decomposition of projected q/sigma (symmetric
                # two-factor split); the friction interpretation remains a proxy.
                shear_log = (q-q0)*.5*(1/sigma+1/sigma0)/A
                normal_log = .5*(q+q0)*(1/sigma-1/sigma0)/A
                state_log = -B/A*np.log(incoming[c]/initial_theta[c])
                damping_log = -ETA*(v[c]/sigma-data['V_m_per_s'][0, c]/sigma0)/A
                proxy = q-sigma*mu(v[c], incoming[c])-ETA*v[c]
                deep.append(dict(step=step, time_yr=time/YEAR, dt_s=dt, target_km=target,
                                 actual_xd_m=x[c], node=int(order[c]), V_over_Vp=v[c]/VP,
                                 theta_incoming_s=incoming[c], theta_committed_s=data['Theta_s'][index,c],
                                 q_projected_MPa=q/1e6, sigma_projected_MPa=sigma/1e6,
                                 friction_proxy_MPa=sigma*mu(v[c],incoming[c])/1e6,
                                 damping_Pa=ETA*v[c], nodal_proxy_imbalance_Pa=proxy,
                                 use_updated_theta_friction_error_Pa=sigma*(mu(v[c],data['Theta_s'][index,c])-mu(v[c],incoming[c])),
                                 shear_log_V_contribution=shear_log,
                                 normal_log_V_contribution=normal_log,
                                 state_log_V_contribution=state_log,
                                 damping_log_V_contribution=damping_log,
                                 observed_log_V_change=np.log(v[c]/data['V_m_per_s'][0,c]),
                                 proxy_predicted_log_V_change=shear_log+normal_log+state_log+damping_log))
        # Preserve only a small set of frozen data for the onset close-up.
        if step < 50 or (first is not None and step <= onset_step+8):
            snapshots[step] = dict(V=v.copy(), theta_in=incoming.copy(), theta_out=theta.copy(),
                                   time=time, slip=slip.copy(), inferred=inferred)
        previous_slip, previous_time = slip, time
    if first is not None:
        for step in sorted(snapshots):
            if step < onset_step-4 or step > onset_step+6:
                continue
            snap = snapshots[step]
            for i in np.flatnonzero(np.abs(x-onset_x) <= 800):
                node_records.append(dict(step=step, time_yr=snap['time']/YEAR, node=int(order[i]),
                                         xd_m=x[i], spacing_left_m=x[i]-x[i-1],
                                         V=snap['V'][i], theta_incoming=snap['theta_in'][i],
                                         theta_committed=snap['theta_out'][i], slip=snap['slip'][i],
                                         at_floor=snap['V'][i]<=VMIN*(1+1e-8), inferred_V=snap['inferred']))
    # Compare with the original exported mesh, not just nominal PRM refinement.
    meshes = [np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding=None)
              for path in args.run.glob('initial_mesh_*.csv')]
    mesh = np.concatenate(meshes)
    nearby = []
    for target in ([onset_x] if first is not None else []) + [25000.,40000.,80000.]:
        i = int(np.argmin(np.abs(x-target)))
        cellmask = ((np.abs(mesh['x']-geometry[i,2]) <= mesh['h']/2+1e-8)
                    & (np.abs(mesh['y']-geometry[i,3]) <= mesh['h']/2+1e-8))
        for cell in mesh[cellmask]:
            nearby.append(dict(xd_m=x[i], node=int(order[i]), cell=str(cell['cell']),
                               h_m=float(cell['h']), level=int(cell['level']),
                               x=float(cell['x']), y=float(cell['y'])))
    records(out/'node_spacing_cells.csv', nearby)
    records(out/'onset_metrics.csv', profile_metrics)
    records(out/'onset_neighbours.csv', node_records)
    records(out/'state_reconstruction_check.csv', history_metrics)
    records(out/'deep_nodal_proxy.csv', deep)
    wavelengths = []
    for step in (0,11,19,27,109,187,223,263,1060,2864,5187):
        i = by_step[step]
        dips, _, _ = extrema(data['V_m_per_s'][i], x)
        separation = np.diff(x[dips])
        wavelengths.append(dict(step=step,time_yr=times[i],dips=len(dips),
                                troughs_km=';'.join(f'{v/1000:.6f}' for v in x[dips]),
                                median_separation_m=float(np.median(separation)) if len(separation) else '',
                                minimum_separation_m=float(np.min(separation)) if len(separation) else ''))
    records(out/'trough_spacing.csv', wavelengths)
    # Independent serialization check: native arrays remain in stored node
    # order, unlike the sorted CSV working arrays used above.
    pairs = []
    for entry in ET.parse(args.run/'reconstructed_faults.pvd').findall('.//DataSet'):
        path = args.run/entry.attrib['file']
        step = int(path.stem.split('-')[-1])
        if path.is_file() and step in by_step: pairs.append((step,path))
    checks = []
    for target in (0,11,27,223,5187):
        step,path = min(pairs,key=lambda pair:abs(pair[0]-target))
        arrays = {a.attrib['Name']:np.fromstring(a.text,sep=' ')
                  for a in ET.parse(path).findall('.//PointData/DataArray')}
        row = dict(step=step)
        for native,field in [('slip_rate','V_m_per_s'),('slip_state','Theta_s'),('cumulative_slip','slip_m')]:
            row[native+'_absolute_difference'] = float(np.max(np.abs(arrays[native][order]-data[field][by_step[step]])))
        checks.append(row)
    records(out/'native_csv_agreement.csv', checks)
    checkpoints = []
    for path in sorted((args.run/'restart').glob('*/bp3_accepted_state.txt')):
        s,t = path.read_text().split()
        checkpoints.append(dict(path=str(path.parent),step=int(s),time_yr=float(t)/YEAR))
    summary = dict(first_resolved_dip=first,
                   onset_definition='5–18 km; V_i < 0.9 min(V_i-1,V_i+1); both neighbours >1e-12 m/s',
                   last_complete_slip_step=profile_metrics[-1]['step'],
                   last_complete_slip_year=profile_metrics[-1]['time_yr'],
                   max_reconstructed_theta_relative_error=max_theta_error,
                   max_deep_reconstructed_theta_relative_error=max_control_theta_error,
                   checkpoints=checkpoints, missing_profiles=missing,
                   first_solver_bound_contact=next(dict(step=s,time_yr=float(r['time'])/YEAR,
                                                      lower_active=int(r['lower_active']))
                                                   for s,r in accepted.items() if int(r['lower_active'])),
                   first_bound_saved_profile=next((r for r in profile_metrics if r['shallow_at_floor'] and not r['floor_count_is_inferred']),None),
                   last_deep_proxies=deep[-3:],
                   limitation='q and sigma are projected fields. Nodal products are not the production '
                   'JxW*chi*N weak friction integral or an independent residual. No operator probe performed.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if first is not None:
        fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
        for step in [onset_step-2,onset_step-1,onset_step,onset_step+1,onset_step+3]:
            if step not in snapshots: continue
            s=snapshots[step]; mask=np.abs(x-onset_x)<=800
            axes[0].semilogy(x[mask]/1000,s['V'][mask],'.-',label=f'step {step}, {s["time"]/YEAR:.3f} yr')
            axes[1].plot(x[mask]/1000,s['theta_in'][mask]/YEAR,'.-')
        axes[0].legend(fontsize=8);axes[0].set_ylabel('Accepted V (m/s)')
        axes[1].set_ylabel('Incoming Θ (yr)');axes[1].set_xlabel('Down-dip distance (km)')
        for ax in axes:ax.grid(alpha=.2)
        fig.suptitle('First resolved shallow dip — raw adjacent nodes, no smoothing')
        fig.savefig(out/'first_dip_neighbours.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
