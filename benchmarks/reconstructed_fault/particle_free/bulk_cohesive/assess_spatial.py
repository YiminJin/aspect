#!/usr/bin/env python3
"""One nested-mesh comparison of fresh aligned bulk-cohesive trajectories.

Phase is evaluated as its physical Q1 field. Constitutive driving is evaluated
only at the original native history points, then physically averaged.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from assess_native import csv, gauss, norm3, G, ETA, KF, ELL, SEED_R, SEED_P
from assess_orientation import rule
from assess_energy import physical_fields, reactions, TIMES, RADII, ANGLES

CASES = {'coarse': Path('build-particle-free/bulk-energy/runs/half'),
         'refined': Path('build-particle-free/bulk-spatial/runs/refined-qualified')}


def layout(path):
    nodes = csv(path/'phase-0000.csv'); nodes.sort(order='dof')
    x, y = np.unique(nodes['x']), np.unique(nodes['y'])
    dx, dy = x[1]-x[0], y[1]-y[0]
    nc = (len(x)-1)*(len(y)-1)
    # The dense export and compact q use the same actual active-cell/QP order.
    with (path/'bulk-0000.csv').open() as stream:
        header = stream.readline().strip().split(',')
    assert tuple(header[i] for i in (1, 2, 3, 4)) == ('x', 'y', 'JxW', 'phi')
    dense = np.loadtxt(path/'bulk-0000.csv', delimiter=',', skiprows=1,
                       usecols=(1, 2, 3, 4)).reshape(nc, 64, 4)
    midpoint = dense[:, :, :2].mean(axis=1)
    ix = np.rint((midpoint[:, 0]-dx/2)/dx).astype(int)
    iy = np.rint((midpoint[:, 1]-dy/2)/dy).astype(int)
    grid_ids = np.lexsort((nodes['x'], nodes['y'])).reshape(len(y), len(x))
    ids = np.stack([grid_ids[iy, ix], grid_ids[iy, ix+1],
                    grid_ids[iy+1, ix], grid_ids[iy+1, ix+1]], axis=-1)
    xy = np.stack([nodes['x'], nodes['y']], axis=-1)
    vertices = xy[ids]
    points, weights, N, gradN = rule(vertices, 8)
    point_error = float(np.max(abs(points-dense[:, :, :2])))
    weight_error = float(np.max(abs(weights-dense[:, :, 2])))
    assert point_error < 1e-14 and weight_error < 1e-18
    assert abs(np.sum(weights)-1.5) < 1e-12 and np.all(weights > 0)
    phi_qp = np.einsum('qi,ci->cq', N, nodes['phi'][ids])
    assert np.max(abs(phi_qp-dense[:, :, 3])) < 1e-14
    return dict(nodes=nodes, node_xy=xy, x=x, y=y, dx=dx, dy=dy,
                grid_ids=grid_ids, ids=ids, vertices=vertices, points=points,
                weights=weights, N=N, gradN=gradN,
                geometry_checks=dict(point_error=point_error, weight_error=weight_error))


def kernel(L):
    centers = np.concatenate([np.stack([.5+r*np.cos(np.deg2rad(ANGLES)),
                                       .5+r*np.sin(np.deg2rad(ANGLES))], axis=-1)
                              for r in RADII])
    points = L['points'].reshape(-1, 2); weights = L['weights'].ravel()
    tree = cKDTree(points); rows, cols, values = [], [], []
    normalizations, counts = [], []
    for j, indices in enumerate(tree.query_ball_point(centers, .0625)):
        indices = np.asarray(indices); difference = points[indices]-centers[j]
        k = (1-np.sum(difference*difference, axis=-1)/.0625**2)**3*weights[indices]
        normalizations.append(float(sum(k))); counts.append(len(indices))
        k /= normalizations[-1]
        rows.extend([j]*len(indices)); cols.extend(indices); values.extend(k)
    exact = np.pi*.0625**2/4  # All support disks are inside the physical box.
    L['kernel_checks'] = dict(exact_constant_integral=exact,
        maximum_normalization_relative_error=float(np.max(abs(np.asarray(normalizations)-exact))/exact),
        point_count_range=[min(counts), max(counts)])
    return csr_matrix((values, (rows, cols)), shape=(len(centers), len(points)))


def q1(L, phi, points):
    p = phi[L['grid_ids']]
    i = np.clip(np.searchsorted(L['x'], points[..., 0], side='right')-1, 0, len(L['x'])-2)
    j = np.clip(np.searchsorted(L['y'], points[..., 1], side='right')-1, 0, len(L['y'])-2)
    u = (points[..., 0]-L['x'][i])/L['dx']; v = (points[..., 1]-L['y'][j])/L['dy']
    a, b, c, d = p[j, i], p[j, i+1], p[j+1, i], p[j+1, i+1]
    value = (1-u)*(1-v)*a+u*(1-v)*b+(1-u)*v*c+u*v*d
    gradient = np.stack([((1-v)*(b-a)+v*(d-c))/L['dx'],
                         ((1-u)*(c-a)+u*(d-b))/L['dy']], axis=-1)
    return value, gradient


def masks(points):
    cap = (points[..., 0] > .5+1e-12) & (np.linalg.norm(points-.5, axis=-1) < .35)
    return {'cap': cap, 'lower': cap & (points[..., 1] < .5),
            'upper': cap & (points[..., 1] > .5)}


def initial_reference(L, order):
    points, weights, N, _ = rule(L['vertices'], order)
    represented = np.einsum('qi,ci->cq', N, L['nodes']['phi'][L['ids']])
    exact = np.interp(np.hypot(np.maximum(points[..., 0]-.5, 0), points[..., 1]-.5),
                      SEED_R, SEED_P, right=0.)
    regions = dict(whole=np.ones_like(weights, dtype=bool), **masks(points))
    return {name: dict(L2=float(np.sqrt(np.sum(weights[m]*(represented-exact)[m]**2))),
                       relative_L2=float(np.sqrt(np.sum(weights[m]*(represented-exact)[m]**2)
                                                 / np.sum(weights[m]*exact[m]**2))))
            for name, m in regions.items()}


def audit_case(name, path, L, output):
    ledger = csv(path/'bulk-accepted.csv'); times = np.r_[0., ledger['time']]
    initial_record = np.fromfile(path/'state-0000.bin')
    ndg = int(csv(path/'fixture.csv')['DG_DoFs'][0])
    history_end = ndg+5*L['weights'].size
    assert np.all(initial_record[:history_end] == 0.)
    assert np.array_equal(initial_record[history_end:-3], L['nodes']['phi'])
    assert np.all(initial_record[-3:] == 0.)
    assert np.all(np.fromfile(path/'native-history-0000.bin') == 0.)
    assert abs(times[-1]-.16) < 1e-15
    assert np.array_equal(ledger['publications'], np.arange(1, len(ledger)+1))
    assert np.all(csv(path/'gates.csv')['passed'] == 1)
    assert max(ledger['mechanical_residual']) <= 1e-9
    assert max(ledger['phase_KKT']) <= 1e-7 and max(ledger['projection_error']) <= 1e-12
    boundary = reactions(path, ledger, L)
    K = kernel(L); phase = []; arcs = {}; onset = {}; previous = None
    errors = {k: 0. for k in ['fracture', 'cohesive', 'bulk_snapshot', 'Db_rate', 'Dc', 'Df',
                              'traction_loss', 'phase_remainder', 'fracture_loss', 'stored_change']}
    sampling = dict(q_absolute=0., phi_absolute=0., driving_relative=0.)
    node_masks = masks(L['node_xy']); dense = []
    nodal_exact = np.interp(np.hypot(np.maximum(L['node_xy'][:, 0]-.5, 0), L['node_xy'][:, 1]-.5),
                            SEED_R, SEED_P, right=0.)
    nodal_seed_error = float(np.max(abs(L['nodes']['phi']-nodal_exact)))
    assert nodal_seed_error < 1e-12
    for step, time in enumerate(times):
        pnodes = csv(path/f'phase-{step:04d}.csv'); pnodes.sort(order='dof')
        phi = pnodes['phi']; phase.append(phi.copy())
        assert np.array_equal(pnodes['x'], L['nodes']['x']) and np.array_equal(pnodes['y'], L['nodes']['y'])
        assert np.all(phi >= 0.) and np.all(phi < 1.)
        if step: assert np.all(phi >= phase[-2])
        q = np.fromfile(path/f'quadrature-q-{step:04d}.bin').reshape(L['weights'].shape)
        p, gradient, h, hp, a, Y = physical_fields(phi, q, L)
        w = L['weights']; Ef = KF*np.sum(w*(p+ELL**2*np.sum(gradient**2, axis=-1)))
        Ec = np.sum(w*h*q*q/(2*G))
        Eb = 0. if not step else ledger['bulk'][step-1]
        if step == 0: E0 = Ef; assert np.all(q == 0.)
        if step == 0 or abs(time-.15) < 1e-12 or abs(time-.16) < 1e-12:
            with (path/f'bulk-{step:04d}.csv').open() as stream:
                header = stream.readline().strip().split(',')
            columns = (3, 4, 8, 9, 10, 12, 15)
            assert tuple(header[i] for i in columns) == ('JxW', 'phi', 'tauxx', 'tauyy', 'tauxy', 'q', 'driving')
            d = np.loadtxt(path/f'bulk-{step:04d}.csv', delimiter=',', skiprows=1,
                           usecols=columns)
            full_bulk = np.sum(d[:, 0]*norm3(d[:, 2:5])/(4*G))
            sampling['q_absolute'] = max(sampling['q_absolute'], float(np.max(abs(d[:, 5]-q.ravel()))))
            sampling['phi_absolute'] = max(sampling['phi_absolute'], float(np.max(abs(d[:, 1]-p.ravel()))))
            sampling['driving_relative'] = max(sampling['driving_relative'], float(np.max(abs(d[:, 6]-Y.ravel()))/max(1., np.max(abs(d[:, 6])))))
            assert sampling['q_absolute'] == 0. and sampling['phi_absolute'] < 1e-14
            assert sampling['driving_relative'] < 1e-12
            errors['bulk_snapshot'] = max(errors['bulk_snapshot'], abs(float(full_bulk-Eb)))
            dense.append(step)
        delta = phi-L['nodes']['phi']
        for flank in ('lower', 'upper'):
            growing = np.flatnonzero(node_masks[flank] & (delta > 1e-12))
            if len(growing) and flank not in onset:
                onset[flank] = dict(previous_time=float(times[step-1]), time=float(time), step=step,
                    nodes=[dict(x=float(pnodes['x'][i]), y=float(pnodes['y'][i]),
                                radius=float(np.hypot(pnodes['x'][i]-.5, pnodes['y'][i]-.5)),
                                initial_phase=float(L['nodes']['phi'][i]),
                                angle=float(np.rad2deg(np.arctan2(pnodes['y'][i]-.5, pnodes['x'][i]-.5))))
                           for i in growing])
        if step:
            row = ledger[step-1]; op, og, oh, oq, oa = previous; dt = row['dt']
            values = dict(fracture=Ef, cohesive=Ec, Db_rate=dt*2*G/ETA*Eb,
                Dc=dt*2*G/ETA*Ec, Df=1000*np.sum(w*abs((1+dt*G/ETA)*a-oa)),
                traction_loss=np.sum(w*oh*(q-oq)**2/(2*G)),
                phase_remainder=np.sum(w*q*q/(2*G)*(h-oh-hp*(p-op))),
                fracture_loss=KF*ELL**2*np.sum(w*np.sum((gradient-og)**2, axis=-1)))
            for key, value in values.items():
                errors[key] = max(errors[key], abs(float(value-row['Db' if key == 'Db_rate' else key])))
            errors['stored_change'] = max(errors['stored_change'], abs(float(Eb+Ec+Ef-E0-np.sum(ledger['energy_change'][:step]))))
        previous = p, gradient, h, q, a
        for common in TIMES:
            if abs(time-common) < 1e-12:
                arcs[str(common)] = np.stack([K@p.ravel(), K@q.ravel(), K@Y.ravel()], axis=-1)
    assert max(errors.values()) < 1e-10, errors
    D = ledger['Db']+ledger['Dc']+ledger['Df']; E = ledger['energy_change']
    losses = sum(ledger[k] for k in ('stress_loss', 'traction_loss', 'phase_remainder', 'fracture_loss'))
    rBE = boundary['endpoint'][1:]-E-D
    assert max(abs(rBE-losses+ledger['phase_work']+ledger['mechanical_work'])) < 1e-9
    rtr = boundary['trap'][1:]-E-D; scale = sum(abs(boundary['trap']))
    record = dict(cells=len(w), phase_dofs=len(phi), dx=L['dx'], dy=L['dy'],
        dt=float(ledger['dt'][0]), accepted_steps=len(ledger), onset=onset,
        nodal_seed_error=nodal_seed_error, fresh_all_histories_zero=True,
        initial_reference_Gauss24=initial_reference(L, 24),
        geometry=L['geometry_checks'], kernel_checks=L['kernel_checks'],
        independent_errors=errors, dense_sampling_checks=sampling, dense_stress_steps=dense,
        boundary_checks=boundary['errors'], mechanical_max=float(max(ledger['mechanical_residual'])),
        KKT_max=float(max(ledger['phase_KKT'])), source_max=float(max(ledger['projection_error'])),
        ledger_error_max=float(max(abs(ledger['ledger_error']))),
        energy=dict(initial_fracture=E0, added_fracture=float(ledger['fracture'][-1]-E0),
            endpoint_work=float(sum(boundary['endpoint'])), trapezoidal_work=float(sum(boundary['trap'])),
            storage_change=float(sum(E)), physical_dissipation=float(sum(D)),
            historical_endpoint_relative=float(abs(sum(rBE))/max(1.,abs(sum(boundary['endpoint'])))),
            trapezoidal_signed_remainder=float(sum(rtr)), trapezoidal_total_variation=float(sum(abs(rtr))),
            trapezoidal_variation_relative=float(sum(abs(rtr))/scale)))
    phase = np.asarray(phase)
    assert np.array_equal(phase[times <= .05+1e-12], np.broadcast_to(phase[0], phase[times <= .05+1e-12].shape))
    stop = np.flatnonzero(abs(times-.15) < 1e-12)[-1]
    assert np.array_equal(phase[-1], phase[stop])
    record['initial_hold_drift'] = float(np.max(abs(phase[times <= .05+1e-12]-phase[0])))
    np.savez_compressed(output/f'{name}-fields.npz', time=times, phase=phase,
                        xy=L['node_xy'], grid_ids=L['grid_ids'], x=L['x'], y=L['y'],
                        **{'arc_'+k: v for k, v in arcs.items()})
    (output/f'{name}.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def comparison(layouts, output):
    data = {name: dict(np.load(output/f'{name}-fields.npz')) for name in CASES}
    fine = layouts['refined']; points = fine['points']; weights = fine['weights']; regions = masks(points)
    initial = {n: q1(layouts[n], d['phase'][0], points) for n, d in data.items()}
    init_difference = initial['coarse'][0]-initial['refined'][0]
    initial_metrics = {n: dict(L2=float(np.sqrt(np.sum(weights[m]*init_difference[m]**2))),
                             relative_to_fine_L2=float(np.sqrt(np.sum(weights[m]*init_difference[m]**2)
                                                              / np.sum(weights[m]*initial['refined'][0][m]**2))))
                       for n, m in dict(whole=np.ones_like(weights, dtype=bool), **regions).items()}
    rows = []
    for time in TIMES:
        physical = {}; increments = {}; nodal = {}; metrics = {}
        for n, d in data.items():
            index = np.flatnonzero(abs(d['time']-time) < 1e-12)[-1]
            physical[n] = q1(layouts[n], d['phase'][index], points)
            increments[n] = physical[n][0]-initial[n][0]
            delta_nodes = d['phase'][index]-d['phase'][0]; nm = masks(d['xy'])
            nodal[n] = delta_nodes
            added = KF*(increments[n]+ELL**2*(np.sum(physical[n][1]**2, axis=-1)-np.sum(initial[n][1]**2, axis=-1)))
            metrics[n] = {flank: dict(growth_L2=float(np.sqrt(np.sum(weights[m]*increments[n][m]**2))),
                peak=float(np.max(delta_nodes[nm[flank]])), integrated_growth=float(np.sum(weights[m]*increments[n][m])),
                added_fracture=float(np.sum(weights[m]*added[m]))) for flank, m in regions.items()}
        differences = {}
        for flank, m in regions.items():
            f = metrics['refined'][flank]; c = metrics['coarse'][flank]
            presence = ('both' if min(f['growth_L2'], c['growth_L2']) > 1e-14 else
                        'refined_only' if f['growth_L2'] > 1e-14 else
                        'coarse_only' if c['growth_L2'] > 1e-14 else 'neither')
            differences[flank] = dict(growth_presence=presence,
                growth_L2_absolute=float(np.sqrt(np.sum(weights[m]*(increments['coarse']-increments['refined'])[m]**2))))
            fine_nodes = fine['node_xy']; node_mask = masks(fine_nodes)[flank]
            coarse_delta = q1(layouts['coarse'], nodal['coarse'], fine_nodes)[0]
            nodal_difference = float(np.max(abs(coarse_delta-nodal['refined'])[node_mask]))
            differences[flank]['growth_Linf_absolute'] = nodal_difference
            differences[flank]['growth_Linf_relative'] = nodal_difference/f['peak'] if f['peak'] > 1e-12 else None
            for key in ('peak', 'integrated_growth', 'added_fracture'):
                differences[flank][key+'_relative'] = abs(c[key]-f[key])/abs(f[key]) if abs(f[key]) > 1e-14 else None
            differences[flank]['growth_L2_relative'] = differences[flank]['growth_L2_absolute']/f['growth_L2'] if f['growth_L2'] > 1e-14 else None
        driving = {}
        for j, radius in enumerate(RADII):
            a = data['coarse']['arc_'+str(time)].reshape(3, len(ANGLES), 3)[j, :, 2]
            b = data['refined']['arc_'+str(time)].reshape(3, len(ANGLES), 3)[j, :, 2]
            driving[str(radius)] = {}
            for flank, selected in [('lower', ANGLES <= 0), ('upper', ANGLES >= 0)]:
                aa, bb = a[selected], b[selected]; theta = ANGLES[selected]
                driving[str(radius)][flank] = dict(relative_L2=float(np.linalg.norm(aa-bb)/np.linalg.norm(bb)) if np.linalg.norm(bb) > 1e-12 else None,
                    coarse_peak_angle=float(theta[np.argmax(aa)]) if max(aa) > 1e-12 else None,
                    refined_peak_angle=float(theta[np.argmax(bb)]) if max(bb) > 1e-12 else None,
                    coarse_peak=float(max(aa)), refined_peak=float(max(bb)))
        rows.append(dict(time=time, metrics=metrics, differences=differences, angular_driving=driving))
    return dict(initial_difference_on_common_fine_rule=initial_metrics, times=rows,
                comparison_measure='Fine physical Gauss8 JxW; each trajectory increment relative to its own fresh initial Q1 field.',
                physical_kernel='(1-rho^2/.0625^2)^3_+, normalized by original native JxW on each mesh',
                intended_growth_relative_tolerance=.05, intended_driving_relative_tolerance=.05,
                no_spatial_order_claimed=True, strict_reference_match_passed=False)


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    layouts = {}; records = {}
    for name, path in CASES.items():
        print('Assessing', name, flush=True)
        layouts[name] = layout(path); records[name] = audit_case(name, path, layouts[name], output)
    assert np.array_equal(csv(CASES['coarse']/'bulk-accepted.csv')['dt'],
                          csv(CASES['refined']/'bulk-accepted.csv')['dt'])
    result = dict(cases=records, identical_timestep_sequence=True,
                  comparison=comparison(layouts, output))
    (output/'assessment.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Spatial assessment complete.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
