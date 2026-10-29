#!/usr/bin/env python3
"""One continued-loading result: original-QP accounting and Q1 connectivity.

This reads accepted exports only. It does not advance or interpolate local
constitutive histories. Raster connectivity is a geometric diagnostic, with
explicit resolution checks; it is not a new phase evolution rule.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from scipy.optimize import brentq
from assess_native import csv, G, ETA, KF, ELL, M, SEED_R, norm3
from assess_spatial import layout, q1, masks, initial_reference, kernel, RADII, ANGLES
from assess_energy import physical_fields

BASE = Path('build-particle-free/bulk-spatial/runs/refined-qualified')
RUN = Path('build-particle-free/bulk-continued/runs/continued')
LEVELS = [.001, .01, .05, .1, .3, .5]


def phase(path, step):
    p = csv(path/f'phase-{step:04d}.csv'); p.sort(order='dof')
    return p['phi']


def crossing(x, values, level):
    if values[0] < level: return None
    below = np.flatnonzero(values < level)
    if not len(below): return float(x[-1])
    i = below[0]
    return float(x[i-1]+(x[i]-x[i-1])*(values[i-1]-level)/(values[i-1]-values[i]))


def contour_geometry(L, phi, subdivisions):
    x = np.linspace(0, L['x'][-1], (len(L['x'])-1)*subdivisions+1)
    y = np.linspace(0, L['y'][-1], (len(L['y'])-1)*subdivisions+1)
    X, Y = np.meshgrid(x, y)
    p = q1(L, phi, np.stack([X, Y], axis=-1))[0]
    centerline = phi[L['grid_ids'][len(L['y'])//2]]
    ray_r = np.linspace(0., .5*np.sqrt(2.), 4097)
    rays = {name: q1(L, phi, np.stack([.5+ray_r/np.sqrt(2.),
             .5+sign*ray_r/np.sqrt(2.)], axis=-1))[0]
             for name, sign in [('upper', 1), ('lower', -1)]}
    result = {}
    for level in LEVELS:
        components, count = label(p >= level)
        seed_id = components[len(y)//2, 0]; assert seed_id != 0
        connected = components == seed_id
        # Include interpolated crossings of the Q1 sampling raster edges,
        # rather than reporting the coordinate of its last above-level pixel.
        rows, cols = np.where(connected)
        right_edges = connected[:, :-1] & ~connected[:, 1:]
        j, i = np.where(right_edges)
        edge_x = x[i]+(x[i+1]-x[i])*(p[j, i]-level)/(p[j, i]-p[j, i+1])
        right = float(max(np.max(x[cols]), np.max(edge_x, initial=0.)))
        ju, iu = np.where(connected[:-1, :] & ~connected[1:, :])
        upper_y = y[ju]+(y[ju+1]-y[ju])*(p[ju, iu]-level)/(p[ju, iu]-p[ju+1, iu])
        jl, il = np.where(connected[1:, :] & ~connected[:-1, :]); jl = jl+1
        lower_y = y[jl]-(y[jl]-y[jl-1])*(p[jl, il]-level)/(p[jl, il]-p[jl-1, il])
        flanks = {}
        for name, sign in [('upper', 1), ('lower', -1)]:
            below = np.flatnonzero(rays[name] < level)
            ray_front = float(ray_r[-1])
            if len(below):
                end = int(below[0]); assert end > 0
                ray_front = float(brentq(lambda r: float(q1(L, phi, np.array(
                    [.5+r/np.sqrt(2.), .5+sign*r/np.sqrt(2.)]))[0])-level,
                    ray_r[end-1], ray_r[end], xtol=1e-13, rtol=1e-14))
            chosen = sign*(y[j]-.5) > 0
            f = connected & (X >= .5) & ((Y > .5) if sign > 0 else (Y < .5))
            jj, ii = np.where(f)
            flanks[name] = dict(rightmost=float(max(np.max(x[ii]), np.max(edge_x[chosen], initial=0.))),
                outer_y=float(max(np.max(y[jj]), np.max(upper_y[x[iu] >= .5], initial=0.)) if sign > 0
                              else min(np.min(y[jj]), np.min(lower_y[x[il] >= .5], initial=1.))),
                ray45_radius=ray_front)
        sizes = np.bincount(components.ravel()); sizes[0] = 0; sizes[seed_id] = 0
        disconnected = np.flatnonzero(sizes)
        result[str(level)] = dict(centerline=crossing(L['x'], centerline, level), rightmost=right,
            **flanks, disconnected_components=len(disconnected),
            disconnected_raster_area=float(sizes.sum()*(x[1]-x[0])*(y[1]-y[0])),
            largest_disconnected_pixels=int(sizes.max(initial=0)), touches_right=bool(connected[:, -1].any()))
    return result


def audit(path, output):
    output.mkdir(parents=True, exist_ok=True)
    L = layout(BASE); K = kernel(L); w = L['weights']; regions = masks(L['points'])
    ledger = csv(path/'bulk-accepted.csv')
    # The fixture row is flushed after all accepted exports. This also permits
    # a read-only progress audit without consuming a half-written current step.
    complete = int(np.atleast_1d(csv(path/'fixture.csv'))['step'][-1])
    ledger = ledger[ledger['step'] <= complete]; times = np.r_[0., ledger['time']]
    steps = len(ledger); assert np.array_equal(ledger['publications'], np.arange(1, steps+1))
    assert np.all(csv(path/'gates.csv')['passed'] == 1)
    assert max(ledger['mechanical_residual']) <= 1e-9
    assert max(ledger['phase_KKT']) <= 1e-7 and max(ledger['projection_error']) <= 1e-12
    phi0 = phase(path, 0); assert np.array_equal(phi0, L['nodes']['phi'])
    # Exact initial owner and solution identities, not a newly relaxed seed.
    initial_identity = {}
    for filename in ['state-0000.bin', 'native-history-0000.bin', 'solution-0000.bin']:
        initial_identity[filename] = (path/filename).read_bytes() == (BASE/filename).read_bytes()
    assert all(initial_identity.values())
    # std::map<CellId> local histories are in numeric CellId order, while the
    # dense/compact quadrature exports follow the actual active-cell order.
    cellids = np.loadtxt(BASE/'bulk-0000.csv', delimiter=',', skiprows=1,
                         usecols=0, dtype=str)[::64]
    numeric = np.array([int(c.split('_')[0]) for c in cellids])
    inverse = np.argsort(np.argsort(numeric)); assert len(np.unique(numeric)) == len(w)
    ndg = 27*len(w); nlocal = 5*w.size
    intact = np.hypot(np.maximum(L['points'][..., 0]-.5, 0), L['points'][..., 1]-.5) >= SEED_R[-1]
    initially_zero = np.einsum('qi,ci->cq', L['N'], phi0[L['ids']]) == 0.
    errors = {k: 0. for k in ['fracture', 'cohesive', 'Db', 'Dc', 'Df', 'traction_loss',
        'phase_remainder', 'fracture_loss', 'stored_change', 'boundary_increment',
        'boundary_work', 'generalized_work', 'rate_recurrence', 'local_a', 'local_q',
        'local_rate', 'slip_increment', 'absolute_slip_increment', 'native_FE_recurrence',
        'dense_bulk_storage', 'dense_phase', 'dense_q', 'dense_driving_relative']}
    history = []; phases = []; arcs = {}; connection = {}; abrupt = []; dense_tensors = {}
    first_disconnected = None; final_disconnected = []
    intact_initial = dict(continuum_intact_area=float(np.sum(w[intact])),
        represented_zero_area=float(np.sum(w[initially_zero])),
        represented_phase_outside_continuum_support=float(np.sum(w[intact]*np.einsum('qi,ci->cq', L['N'], phi0[L['ids']])[intact])))
    previous = None; Fprevious = None; localprevious = None; previous_forces = None
    prefix = dict(phase_max=0., q_max=0., ledger_max=0., dense_state_bitwise=None)
    old_ledger = csv(BASE/'bulk-accepted.csv')
    common_times = [.05, .1, .12875, .15, .155, .16, .165, .17, .175, .1775, .18, .185]
    for step, time in enumerate(times):
        phi = phase(path, step); phases.append(phi)
        assert np.all(np.isfinite(phi)) and np.all(phi >= 0) and np.all(phi < 1)
        if step: assert np.all(phi >= phases[-2])
        q = np.fromfile(path/f'quadrature-q-{step:04d}.bin').reshape(w.shape)
        p, grad, h, hp, a, Y = physical_fields(phi, q, L)
        Ef = KF*np.sum(w*(p+ELL**2*np.sum(grad*grad, axis=-1)))
        Ec = np.sum(w*h*q*q/(2*G)); Eb = 0. if not step else ledger['bulk'][step-1]
        densefile = path/f'bulk-{step:04d}.csv'
        if densefile.exists():
            d = np.loadtxt(densefile, delimiter=',', skiprows=1, usecols=(3, 4, 8, 9, 10, 12, 15))
            full_bulk = np.sum(d[:, 0]*norm3(d[:, 2:5])/(4*G))
            errors['dense_bulk_storage'] = max(errors['dense_bulk_storage'], abs(float(full_bulk-Eb)))
            errors['dense_phase'] = max(errors['dense_phase'], float(np.max(abs(d[:, 1]-p.ravel()))))
            errors['dense_q'] = max(errors['dense_q'], float(np.max(abs(d[:, 5]-q.ravel()))))
            errors['dense_driving_relative'] = max(errors['dense_driving_relative'], float(np.max(abs(d[:, 6]-Y.ravel()))/max(1., Y.max())))
            if time >= .175-1e-12: dense_tensors[step] = d[:, 2:5].copy()
        if not step: E0 = Ef
        dt = 0. if not step else ledger['dt'][step-1]
        inferred_rate = np.zeros_like(a) if not step else ((1+dt*G/ETA)*a-previous[4])/dt
        rf = path/f'quadrature-rate-{step:04d}.bin'
        rate = np.fromfile(rf).reshape(w.shape) if rf.exists() else inferred_rate
        if rf.exists(): errors['rate_recurrence'] = max(errors['rate_recurrence'], float(np.max(abs(rate-inferred_rate))))
        # Measured native force-displacement work including start/stop/hold.
        reaction = csv(path/f'boundary-reactions-{step:04d}.csv'); reaction.sort(order='dof')
        b = np.where(reaction['component'] == 0, reaction['y']-.5, 0.)
        dgamma = .018*dt if .05+1e-12 < time <= .175+1e-12 else 0.
        endpoint = np.sum(reaction['force']*reaction['increment'])
        trapezoid = 0. if previous_forces is None else np.sum(.5*(previous_forces+reaction['force'])*reaction['increment'])
        Q = np.sum(reaction['force']*b)
        errors['boundary_increment'] = max(errors['boundary_increment'], float(np.max(abs(reaction['increment']-dgamma*b))))
        errors['generalized_work'] = max(errors['generalized_work'], abs(float(endpoint-Q*dgamma)))
        previous_forces = reaction['force'].copy()
        delta = phi-phi0; growth = np.einsum('qi,ci->cq', L['N'], delta[L['ids']])
        dstep = np.zeros_like(phi) if not step else phi-phases[-2]
        growth_integral = np.sum(w*growth)
        # A separate native-node activation diagnostic catches weak nucleation
        # below the plotted superlevel sets. Eight-neighbor connectivity admits
        # diagonal positive Q1 corners; this is labeled as a graph diagnostic,
        # not exact nonzero-level bilinear topology.
        support = ((phi0 > 0.) | (delta > 1e-12))[L['grid_ids']]
        components, count = label(support, structure=np.ones((3, 3), dtype=int))
        seed_id = components[len(L['y'])//2, 0]
        newborn = ((phi0 == 0.) & (delta > 1e-12))[L['grid_ids']]
        remote_labels = np.unique(components[newborn & (components != seed_id)])
        remote_labels = remote_labels[remote_labels != 0]
        final_disconnected = []
        for number in remote_labels:
            jj, ii = np.where(components == number); native_indices = L['grid_ids'][jj, ii]
            final_disconnected.append(dict(nodes=len(ii), xmin=float(L['x'][ii].min()),
                xmax=float(L['x'][ii].max()), ymin=float(L['y'][jj].min()), ymax=float(L['y'][jj].max()),
                peak_increment=float(delta[native_indices].max())))
        if first_disconnected is None and len(remote_labels):
            first_disconnected = dict(step=step, start=float(times[step-1]), end=float(time),
                definition='native initially-zero node activation above 1e-12; eight-neighbor graph',
                components=final_disconnected.copy())
        m = dict(step=step, time=float(time), dt=float(dt), strain_increment=float(dgamma),
            Q=float(Q), endpoint_work=float(endpoint), trapezoidal_work=float(trapezoid),
            bulk=float(Eb), cohesive=float(Ec), fracture=float(Ef),
            peak_damage=float(delta.max()), peak_step_damage=float(dstep.max()),
            global_growth=float(growth_integral),
            continuum_intact_growth=float(np.sum(w[intact]*growth[intact])),
            represented_zero_growth=float(np.sum(w[initially_zero]*growth[initially_zero])),
            continuum_intact_peak=float(np.max(growth[intact])),
            represented_zero_peak=float(np.max(growth[initially_zero])),
            right_boundary_peak=float(np.max(delta[L['node_xy'][:, 0] == L['x'][-1]])),
            remote_peak=float(np.max(delta[L['node_xy'][:, 0] >= 1.])),
            disconnected_native_activation_components=int(len(remote_labels)),
            newly_active_zero_phase_nodes=int(np.sum(newborn)),
            disconnected_zero_phase_nodes=int(np.sum(newborn & (components != seed_id))),
            max_rate=float(np.max(abs(rate))), integrated_absolute_rate=float(np.sum(w*abs(rate))),
            max_driving=float(Y.max()))
        for name, mask in regions.items():
            m[name+'_growth'] = float(np.sum(w[mask]*growth[mask]))
            m[name+'_growth_L2'] = float(np.sqrt(np.sum(w[mask]*growth[mask]**2)))
            nodal_mask = masks(L['node_xy'])[name]
            m[name+'_peak'] = float(np.max(delta[nodal_mask]))
        if step:
            row = ledger[step-1]; op, og, oh, oq, oa = previous
            values = dict(fracture=Ef, cohesive=Ec, Db=dt*2*G/ETA*Eb,
                Dc=dt*2*G/ETA*Ec, Df=1000*np.sum(w*abs((1+dt*G/ETA)*a-oa)),
                traction_loss=np.sum(w*oh*(q-oq)**2/(2*G)),
                phase_remainder=np.sum(w*q*q/(2*G)*(h-oh-hp*(p-op))),
                fracture_loss=KF*ELL**2*np.sum(w*np.sum((grad-og)**2, axis=-1)))
            for key, value in values.items(): errors[key] = max(errors[key], abs(float(value-row[key])))
            errors['stored_change'] = max(errors['stored_change'], abs(float(Eb+Ec+Ef-E0-np.sum(ledger['energy_change'][:step]))))
            errors['boundary_work'] = max(errors['boundary_work'], abs(float(endpoint-row['Wext'])))
            increment = growth_integral-history[-1]['global_growth']
            old_increment = history[-1]['global_growth']-history[-2]['global_growth'] if step > 1 else 0.
            if m['peak_step_damage'] >= .001 or (old_increment > 1e-12 and increment > 2*old_increment):
                abrupt.append(dict(step=step, start=float(times[step-1]), end=float(time),
                    max_step_damage=m['peak_step_damage'], integrated_growth_increment=float(increment),
                    previous_integrated_increment=float(old_increment)))
        previous = p, grad, h, q, a
        rawfile = path/f'state-{step:04d}.bin'
        if rawfile.exists():
            raw = np.fromfile(rawfile); assert raw[-3] == step and raw[-1] == step and abs(raw[-2]-time) < 1e-14
            local = raw[ndg:ndg+nlocal].reshape(len(w), 64, 5)[inverse]
            for key, v in [('local_a', a-local[..., 0]), ('local_q', q-local[..., 3]), ('local_rate', rate-local[..., 4])]:
                errors[key] = max(errors[key], float(np.max(abs(v))))
            if localprevious is not None and localprevious[0] == step-1:
                oldlocal = localprevious[1]
                errors['slip_increment'] = max(errors['slip_increment'], float(np.max(abs(local[..., 1]-oldlocal[..., 1]-dt*rate))))
                errors['absolute_slip_increment'] = max(errors['absolute_slip_increment'], float(np.max(abs(local[..., 2]-oldlocal[..., 2]-dt*abs(rate)))))
            localprevious = step, local
            F = np.fromfile(path/f'native-history-{step:04d}.bin')
            solution = np.fromfile(path/f'solution-{step:04d}.bin')
            if Fprevious is not None and Fprevious[0] == step-1:
                target = (Fprevious[1]+dt*solution[:len(F)])/(1+dt*G/ETA)
                errors['native_FE_recurrence'] = max(errors['native_FE_recurrence'], float(np.max(abs(F-target))))
            Fprevious = step, F
        if time <= .15+1e-12:
            prefix['phase_max'] = max(prefix['phase_max'], float(np.max(abs(phi-phase(BASE, step)))))
            prefix['q_max'] = max(prefix['q_max'], float(np.max(abs(q-np.fromfile(BASE/f'quadrature-q-{step:04d}.bin').reshape(w.shape)))))
            if step:
                prefix['ledger_max'] = max(prefix['ledger_max'], max(abs(float(ledger[k][step-1]-old_ledger[k][step-1])) for k in ledger.dtype.names))
            if abs(time-.15) < 1e-12:
                prefix['dense_state_bitwise'] = all((path/f'{s}-0120.bin').read_bytes() == (BASE/f'{s}-0120.bin').read_bytes() for s in ['state', 'native-history', 'solution'])
        if step == 0 or step == steps or time >= .12-1e-12:
            connection[str(step)] = contour_geometry(L, phi, 8)
        if any(abs(time-t) < 1e-12 for t in common_times):
            arcs[str(step)] = np.stack([K@p.ravel(), K@q.ravel(), K@Y.ravel()], axis=-1)
        history.append(m)
    assert max(errors.values()) < 1e-10, errors
    assert prefix['phase_max'] == prefix['q_max'] == prefix['ledger_max'] == 0.
    if times[-1] >= .15: assert prefix['dense_state_bitwise']
    phase_array = np.asarray(phases)
    assert np.array_equal(phase_array[times <= .05+1e-12], np.broadcast_to(phi0, phase_array[times <= .05+1e-12].shape))
    # One diagnostic raster refinement at the endpoint; no new native trajectory.
    finer = contour_geometry(L, phases[-1], 16)
    shifts = {level: float(abs(finer[level]['rightmost']-connection[str(steps)][level]['rightmost'])) for level in finer}
    topology_changes = {level: int(finer[level]['disconnected_components']-connection[str(steps)][level]['disconnected_components']) for level in finer}
    # Check the saved fields at topology changes and large apparent front
    # jumps too. This refines only the read-only diagnostic raster, never the
    # physical mesh, quadrature, or local constitutive histories.
    diagnostic_steps = {steps}; previous_step = 0
    for current_step in sorted(map(int, connection)):
        if current_step == 0: continue
        before = connection[str(previous_step)]; after = connection[str(current_step)]
        if any(before[v]['disconnected_components'] != after[v]['disconnected_components']
               or abs(after[v]['rightmost']-before[v]['rightmost']) > L['dx']/2 for v in after):
            diagnostic_steps.update([previous_step, current_step])
        previous_step = current_step
    raster_checks = {}
    for current_step in sorted(diagnostic_steps):
        sampled = finer if current_step == steps else contour_geometry(L, phases[current_step], 16)
        original = connection[str(current_step)]
        raster_checks[str(current_step)] = dict(time=float(times[current_step]),
            max_rightmost_difference=float(max(abs(sampled[v]['rightmost']-original[v]['rightmost']) for v in original)),
            component_count_changes={v: int(sampled[v]['disconnected_components']-original[v]['disconnected_components']) for v in original})
    onset = {}
    nodal_regions = dict(initially_zero=phi0 == 0.,
        remote=L['node_xy'][:, 0] >= 1., right_boundary=L['node_xy'][:, 0] == L['x'][-1],
        **masks(L['node_xy']))
    for name, mask in nodal_regions.items():
        growing = np.any((phase_array-phi0)[:, mask] > 1e-12, axis=1)
        if growing.any():
            first = int(np.flatnonzero(growing)[0]); active = np.flatnonzero(mask & (phase_array[first]-phi0 > 1e-12))
            onset[name] = dict(step=first, start=float(times[first-1]), end=float(times[first]),
                locations=[dict(x=float(L['node_xy'][i, 0]), y=float(L['node_xy'][i, 1]),
                    initial_phase=float(phi0[i]), increment=float(phase_array[first, i]-phi0[i])) for i in active])
        else: onset[name] = None
    D = ledger['Db']+ledger['Dc']+ledger['Df']; E = ledger['energy_change']
    endpoint = np.array([r['endpoint_work'] for r in history[1:]])
    trap = np.array([r['trapezoidal_work'] for r in history[1:]])
    rbe = endpoint-E-D; rtr = trap-E-D
    numerical = sum(ledger[k] for k in ['stress_loss', 'traction_loss', 'phase_remainder', 'fracture_loss'])
    rw = ledger['phase_work']+ledger['mechanical_work']
    assert np.max(abs(rbe-numerical+rw)) < 1e-9
    stages = {}
    for name, mask in [('initial_hold', ledger['time'] <= .05+1e-12),
        ('original_loading', (ledger['time'] > .05+1e-12)&(ledger['time'] <= .15+1e-12)),
        ('added_loading', (ledger['time'] > .15+1e-12)&(ledger['time'] <= .175+1e-12)),
        ('final_hold', ledger['time'] > .175+1e-12)]:
        stages[name] = {key: float(np.sum(value[mask])) for key, value in
            [('endpoint_work', endpoint), ('trapezoidal_work', trap), ('storage_change', E),
             ('physical_dissipation', D), ('trapezoidal_signed_remainder', rtr), ('trapezoidal_absolute_remainder', abs(rtr))]}
    energy = dict(stages=stages, initial_fracture=float(E0), endpoint_work=float(sum(endpoint)),
        trapezoidal_work=float(sum(trap)), storage_change=float(sum(E)), physical_dissipation=float(sum(D)),
        endpoint_remainder=float(sum(rbe)), endpoint_screen_relative=float(abs(sum(rbe))/max(1, abs(sum(endpoint)))),
        trapezoidal_signed_remainder=float(sum(rtr)), trapezoidal_absolute_remainder=float(sum(abs(rtr))),
        trapezoidal_variation_relative=float(sum(abs(rtr))/sum(abs(trap))) if sum(abs(trap)) > 0 else None,
        maximum_cumulative_trapezoidal_remainder=float(max(abs(np.cumsum(rtr)))),
        terms={k: float(sum(ledger[k])) for k in ['Db', 'Dc', 'Df', 'stress_loss', 'traction_loss',
            'phase_remainder', 'fracture_loss', 'phase_work', 'mechanical_work', 'pressure_work']})
    energy['terms']['velocity_residual_work'] = float(sum(ledger['mechanical_work']-ledger['pressure_work']))
    hold_control = None
    stops = np.flatnonzero(abs(times-.175) < 1e-12)
    if len(stops) and times[-1] > .175+1e-12:
        start = int(stops[-1]); factor = float(np.prod(1/(1+ledger['dt'][start:]*G/ETA)))
        expected_Q = factor*history[start]['Q']; measured_Q = history[-1]['Q']
        hold_control = dict(steps=steps-start, Maxwell_factor=factor,
            max_phase_change=float(np.max(abs(phase_array[start:]-phase_array[start]))),
            max_local_slip_rate=max(r['max_rate'] for r in history[start+1:]),
            expected_Q=expected_Q, measured_Q=measured_Q,
            relative_reaction_relaxation_error=abs(measured_Q-expected_Q)/abs(expected_Q))
        initial_local = np.fromfile(path/f'state-{start:04d}.bin')[ndg:ndg+nlocal].reshape(-1, 5)
        final_local = np.fromfile(path/f'state-{steps:04d}.bin')[ndg:ndg+nlocal].reshape(-1, 5)
        hold_control['max_signed_slip_change'] = float(np.max(abs(final_local[:, 1]-initial_local[:, 1])))
        hold_control['max_absolute_slip_change'] = float(np.max(abs(final_local[:, 2]-initial_local[:, 2])))
        if start in dense_tensors and steps in dense_tensors:
            expected = factor*dense_tensors[start]; difference = dense_tensors[steps]-expected
            hold_control['physical_stress_L2_relative'] = float(np.sqrt(np.sum(w.ravel()*norm3(difference))/np.sum(w.ravel()*norm3(expected))))
            hold_control['physical_stress_max_absolute_Pa'] = float(np.sqrt(norm3(difference)).max())
    record = dict(accepted_steps=steps, final_time=float(times[-1]), declared_endpoint=.185,
        reached_endpoint=bool(abs(times[-1]-.185) < 1e-12), dt=float(ledger['dt'][0]),
        cells=len(w), initial_identity=initial_identity, identical_prefix=prefix,
        initialization_reference=initial_reference(L, 24), initially_intact_definition=intact_initial, independent_errors=errors,
        mechanical_max=float(max(ledger['mechanical_residual'])), KKT_max=float(max(ledger['phase_KKT'])),
        source_max=float(max(ledger['projection_error'])), ledger_max=float(max(abs(ledger['ledger_error']))),
        connectivity_levels=LEVELS, raster_subdivisions=8, refined_raster_subdivisions=16,
        final_raster_rightmost_difference=shifts, final_raster_component_count_change=topology_changes,
        refined_final_contours=finer, topology_transition_raster_checks=raster_checks,
        onset=onset, first_disconnected_native_activation=first_disconnected,
        final_disconnected_native_components=final_disconnected,
        energy=energy, hold_control=hold_control, abrupt_intervals=abrupt, final=history[-1])
    (output/'assessment.json').write_text(json.dumps(record, indent=2)+'\n')
    (output/'contours.json').write_text(json.dumps(connection, indent=2)+'\n')
    (output/'history.json').write_text(json.dumps(history, indent=2)+'\n')
    np.savez_compressed(output/'fields.npz', time=times, phase=phase_array, xy=L['node_xy'],
        grid_ids=L['grid_ids'], x=L['x'], y=L['y'], **{'arc_'+k: v for k, v in arcs.items()})
    cols = ['time', 'endpoint_work', 'trapezoidal_work', 'storage_change', 'physical_dissipation',
        'endpoint_remainder', 'trapezoidal_remainder']
    values = [times[1:], endpoint, trap, E, D, rbe, rtr]
    for name in ['bulk', 'cohesive', 'fracture', 'Db', 'Dc', 'Df', 'stress_loss', 'traction_loss',
                 'phase_remainder', 'fracture_loss', 'phase_work', 'mechanical_work', 'pressure_work']:
        cols.append(name); values.append(ledger[name])
    cols.append('velocity_residual_work'); values.append(ledger['mechanical_work']-ledger['pressure_work'])
    np.savetxt(output/'energy.csv', np.stack(values, axis=-1), delimiter=',', header=','.join(cols), comments='')
    brief = {k: record[k] for k in ['accepted_steps', 'final_time', 'reached_endpoint', 'identical_prefix', 'final']}
    brief['flagged_intervals'] = len(abrupt)
    print(json.dumps(brief, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/reconstructed_fault/particle_free/evidence/bulk_continued'))
    args = parser.parse_args(); audit(args.run, args.output)
