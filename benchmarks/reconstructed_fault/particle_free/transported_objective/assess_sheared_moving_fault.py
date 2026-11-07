#!/usr/bin/env python3
"""Four-run shear sensitivity study; reads native exports without changing them.

Full plane-strain stress norm; material-label traction quadrature is split at
both compared Cartesian grids. Bulk tubes use the reference curve in Eulerian
coordinates. No exact affine solution is assumed at the natural top boundary.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'DEAL_II_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import csv
import json
import re
import numpy as np
from compare_bounded_profile import ROOT, RUNS, load, reduce, tensor, quadrature, dot3, work_traction
from diagnose_bounded_energy import powers

OUT = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/sheared_moving_fault'
CASES = ['shear32-dt100', 'shear32-dt050', 'shear32-dt025', 'shear64-dt025']


def masks(a):
    return dict(all=np.ones(len(a), bool), central=(a >= .2) & (a <= .8),
                ends=(a < .2) | (a > .8), left=a < .2, right=a > .8)


def metric(error2, base2, w):
    num, den = float(w@error2), float(w@base2)
    return dict(absolute_L2=float(np.sqrt(max(0., num))),
                reference_L2=float(np.sqrt(max(0., den))),
                relative_L2=float(np.sqrt(max(0., num)/den)) if den > 0 else None)


def surface_rule(states, order=6):
    """Common label grid split at every label, region edge and DG cell crossing."""
    breaks = [0., .2, .8, 1.]
    for s in states:
        X = s.surface[:, 1:3]; labels = np.linspace(0, 1, len(X))
        breaks.extend(labels)
        for j, d in enumerate(np.diff(X, axis=0)):
            for axis in (0, 1):
                if abs(d[axis]) < 1e-15:
                    continue
                q = (np.linspace(-2, 2, s.n+1)-X[j, axis])/d[axis]
                q = q[(q > 0) & (q < 1)]
                breaks.extend(labels[j]+q*(labels[j+1]-labels[j]))
    b = np.unique(breaks); g, w = np.polynomial.legendre.leggauss(order)
    return (b[:-1, None]+.5*(g+1)*np.diff(b)[:, None]).ravel(), (.5*np.diff(b)[:, None]*w).ravel()


def surface_values(s, labels, cumulative=None):
    X = s.surface[:, 1:3]; count = len(X)-1
    j = np.minimum((labels*count).astype(int), count-1); q = labels*count-j
    d = X[j+1]-X[j]; length = np.linalg.norm(d, axis=1)
    t = d/length[:, None]; n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    points = X[j]+q[:, None]*d; tau = tensor(s, points); v = s.at(points)
    shear = t[:, 0]*(tau[:, 0]*n[:, 0]+tau[:, 2]*n[:, 1])+t[:, 1]*(tau[:, 2]*n[:, 0]+tau[:, 1]*n[:, 1])
    normal = tau[:, 0]*n[:, 0]**2+tau[:, 1]*n[:, 1]**2+2*tau[:, 2]*n[:, 0]*n[:, 1]
    interp = lambda a: (1-q)*a[j]+q*a[j+1]
    work, _ = work_traction(s)
    result = dict(shear_traction=shear, deviatoric_normal_traction=normal,
                  total_normal_traction=normal-v[:, 17], effective_compression=5+v[:, 17]-normal,
                  slip_rate=interp(s.surface[:, 6]), age=interp(s.surface[:, 3]),
                  normalization=interp(s.surface[:, 5]), work_shear_traction=interp(work),
                  geometry=points, jacobian=count*length)
    # Fixture background is a per-material-label property, not a function of
    # the moved x coordinate. Original vertices are uniform from x=-.8 to .8.
    background = 2.7+.15*np.cos(3*np.linspace(-.8, .8, len(X)))
    result['driving_shear_traction'] = shear+interp(background)
    if cumulative is not None:
        result['cumulative_slip'] = interp(np.asarray(cumulative))
    return result


def nearest_labels(s, x):
    X = s.surface[:, 1:3]; best = np.full(len(x), np.inf); labels = np.zeros(len(x))
    for j, d in enumerate(np.diff(X, axis=0)):
        q = np.clip((x-X[j])@d/(d@d), 0, 1)
        dist = np.linalg.norm(x-X[j]-q[:, None]*d, axis=1)
        take = dist < best; best[take] = dist[take]; labels[take] = (j+q[take])/(len(X)-1)
    return labels, best


def compare(a, b, ca=None, cb=None, bulk_order=8, region_reference=None):
    region_reference = b if region_reference is None else region_reference
    x, w = quadrature(max(a.n, b.n), bulk_order)
    ta, tb = tensor(a, x), tensor(b, x)
    error2, base2 = dot3(ta-tb, ta-tb), dot3(tb, tb)
    labels, distance = nearest_labels(region_reference, x)
    bulk = {'whole_domain': metric(error2, base2, w)}
    for name, selected in masks(labels).items():
        selected &= distance <= .2
        bulk[name] = metric(error2[selected], base2[selected], w[selected])
    labels, weights = surface_rule([a, b, region_reference])
    aa, bb = surface_values(a, labels, ca), surface_values(b, labels, cb)
    measure = surface_values(region_reference, labels)['jacobian']
    surface = {}
    for region, selected in masks(labels).items():
        ww = weights[selected]*measure[selected]; vals = {}
        for key in aa:
            if key in ('jacobian', 'geometry'):
                continue
            e, z = aa[key][selected]-bb[key][selected], bb[key][selected]
            vals[key] = metric(e*e, z*z, ww)
        delta = np.linalg.norm(aa['geometry'][selected]-bb['geometry'][selected], axis=1)
        vals['geometry'] = dict(absolute_L2=float(np.sqrt(ww@(delta*delta))), Linf=float(delta.max()))
        surface[region] = vals
    node_delta = np.linalg.norm(a.surface[:, 1:3]-b.surface[:, 1:3], axis=1)
    # Region masks must form a partition of the reported tube/curve norms.
    for regions in [bulk]+[{r:surface[r][k] for r in surface} for k in surface['all']]:
        for parent, parts in [('all', ('central', 'ends')), ('ends', ('left', 'right'))]:
            total = regions[parent]['absolute_L2']**2
            summed = sum(regions[p]['absolute_L2']**2 for p in parts)
            assert abs(total-summed) <= 2e-13*max(total, summed, 1e-30)
    return dict(stress=bulk, surface=surface, geometry_vertex_Linf=float(node_delta.max()))


def diagnostics(name):
    reduced, final = reduce(name)
    cmd, rows, meta, states = load(name)
    assert meta['G'] == 10 and meta['reference_eta'] == 10 and meta['sense'] == 1 and meta['direct'] == 1
    assert cmd['arguments']['eta'] == '1'
    initial = states[0]; spin = np.zeros(len(initial.surface)); geometry_replay = 0.
    terms = []; signed = absolute = active = 0.; previous = None
    min_clearance = 2.; cumulative = np.zeros(len(initial.surface)); trajectories = []
    for row in rows:
        step = int(row['step']); dt = float(row['dt']); s = states[step]
        assert int(row['history_publications']) == step and int(row['geometry_publications']) == step
        assert int(row['retries']) == 0
        old = states.get(step-1, s)
        if step:
            X = old.surface[:, 1:3]; v = old.remote_at(X)
            spin += .5*dt*(v[:, 14]-v[:, 13])
            geometry_replay = max(geometry_replay, float(np.linalg.norm(s.surface[:, 1:3]-X-dt*v[:, 1:3], axis=1).max()))
            cumulative += dt*s.surface[:, 6]
        clearance = float(2-np.abs(s.surface[:, 1:3]).max()); min_clearance = min(min_clearance, clearance)
        e = powers(s, old, initial, meta['G'], cmd['arguments']['eta'])
        e.update(step=step, time=float(row['time']), dt=dt)
        if previous is not None:
            rhs = e['mechanical_power']+e['background_power']-e['friction_power']-e['radiation_power']-e['maxwell_loss']+e['dilatation_power']-e['flux']
            e['coupled_endpoint_remainder'] = e['storage']-previous['storage']-dt*rhs
            signed += e['coupled_endpoint_remainder']; absolute += abs(e['coupled_endpoint_remainder'])
            active += .5*dt*sum(abs(z[k]) for z in (e, previous) for k in ('mechanical_power', 'slip_power', 'maxwell_loss', 'dilatation_power', 'flux'))
        terms.append(e); previous = e
        d = s.surface[-1, 1:3]-s.surface[0, 1:3]
        trajectories.append(dict(time=float(row['time']), clearance=clearance,
                                 spin_mean=float(spin.mean()), spin_min=float(spin.min()), spin_max=float(spin.max()),
                                 chord_angle=float(np.arctan2(d[1], d[0])), geometry=s.surface[:, 1:3].tolist()))
    assert min_clearance > 0 and geometry_replay < 2e-11
    assert len(rows)-1 == round(.4/cmd['arguments']['dt']) and abs(float(rows[-1]['time'])-.4) < 1e-14
    assert np.array_equal(cumulative, np.asarray(reduced['endpoint_cumulative_slip']))
    q12 = powers(final, states[max(states)-1], initial, meta['G'], cmd['arguments']['eta'], 12)
    transport = {}; reaction = {}
    for line in (RUNS/name/'run.log').read_text().splitlines():
        if 'Direct transport energy:' not in line and 'Direct source energy:' not in line:
            continue
        r = {k: float(v) for k, v in re.findall(r'(\w+)=([^\s]+)', line)}
        if 'Direct transport energy:' in line:
            transport[int(r['step'])] = r
        elif r['publication'] == 1:
            reaction[int(r['step'])] = r
    ledger = []
    for step, t in transport.items():
        r = reaction[step]; observed = terms[step]['storage']
        assert abs(observed-r['projected']) < 1e-11*max(1, abs(observed))
        ledger.append(dict(step=step, transport_change=t['advected']-t['accepted'],
                           rotation_projection_change=t['rotated_projected']-t['advected'],
                           reaction_change=r['local']-t['rotated_projected'], source_projection_loss=r['loss']))
    sums = {key:dict(signed=sum(t[key] for t in ledger), absolute=sum(abs(t[key]) for t in ledger)) for key in ledger[0] if key != 'step'}
    profile_labels = (np.arange(800)+.5)/800
    profiles = surface_values(final, profile_labels, cumulative)
    tau_profile = tensor(final, profiles['geometry'])
    profiles['stress_norm'] = np.sqrt(dot3(tau_profile, tau_profile))
    profiles['labels'] = profile_labels
    profiles = {k:v.tolist() for k,v in profiles.items() if k != 'jacobian'}
    substeps = [int(v) for v in re.findall(r'advection substeps=(\d+)', (RUNS/name/'run.log').read_text())]
    reduced.update(coupled_work=dict(steps=terms, signed=signed, absolute=absolute, active_scale=active,
                                    relative_active=absolute/active, final_q12_minus_q8={k:q12[k]-terms[-1][k] for k in q12}),
                   stage_energy=dict(steps=ledger, sums=sums), publication_counters=rows,
                   spin_at_labels=spin.tolist(), trajectories=trajectories,
                   min_curve_clearance=min_clearance, geometry_replay_Linf=geometry_replay,
                   final_profiles=profiles, advection_substeps=substeps)
    return reduced, initial, final


def boundary_diagnostics(final):
    """Locate failed global refinement using saved fields only, with grid-aligned windows."""
    cases = {}; pairs = {}; maps = {}
    for name, s in final.items():
        x, w = quadrature(s.n, 4); v = s.at(x); tau = tensor(s, x); sq = dot3(tau, tau)
        j = int(sq.argmax())
        cases[name] = dict(sampled_max_stress_norm=float(np.sqrt(sq[j])), sampled_max_position=x[j].tolist(),
                           sampled_max_speed=float(np.linalg.norm(v[:, 1:3], axis=1).max()), boundary_flux={})
        g, gw = np.polynomial.legendre.leggauss(8)
        line = (-2+s.h*(np.arange(s.n)[:, None]+.5*(g+1))).ravel(); weights = np.tile(.5*s.h*gw, s.n)
        for axis, side, label in [(0, -1, 'left'), (0, 1, 'right'), (1, -1, 'bottom'), (1, 1, 'top')]:
            points = np.zeros((len(line), 2)); points[:, axis] = 2*side; points[:, 1-axis] = line
            a = s.at(points); tt = tensor(s, points); flux = side*a[:, axis+1]*dot3(tt, tt)/40
            cases[name]['boundary_flux'][label] = float(weights@flux)
            if label == 'left':
                selected = line > 1.5
                cases[name]['boundary_flux']['left_upper_half_metre'] = float(weights[selected]@flux[selected])
    for name, i, j in [('dt_to_half', 0, 1), ('half_to_quarter', 1, 2), ('space', 2, 3)]:
        a, b = final[CASES[i]], final[CASES[j]]; n = max(a.n, b.n); h = 4/n
        x, w = quadrature(n, 4); delta = tensor(a, x)-tensor(b, x); errors = w*dot3(delta, delta)
        region = dict(rim=np.max(abs(x), axis=1) > 1.75, interior=np.max(abs(x), axis=1) <= 1.75,
                      left=x[:, 0] < -1.75, right=x[:, 0] > 1.75, top=x[:, 1] > 1.75, bottom=x[:, 1] < -1.75,
                      upper_left=(x[:, 0] < -1.5) & (x[:, 1] > 1.5), upper_right=(x[:, 0] > 1.5) & (x[:, 1] > 1.5))
        pairs[name] = dict(total_stress_L2=float(np.sqrt(errors.sum())),
                           squared_error_fractions={k:float(errors[z].sum()/errors.sum()) for k, z in region.items()})
        ij = np.floor((x+2)/h).astype(int)
        cell = np.bincount(ij[:, 0]*n+ij[:, 1], weights=errors, minlength=n*n).reshape(n, n)
        maps[name] = np.sqrt(cell/(h*h))
    result = dict(cases=cases, comparisons=pairs,
                  interpretation='Saved-state localization, not a new native test. Stress peaks are samples at tensor Gauss4 points, not proven Linf. Rim width .25 m; corner boxes .5 by .5 m. Windows overlap except rim/interior. Fractions refer to squared full-plane-strain stress L2 differences. Boundary flux is outward positive, using endpoint physical stress and velocity. Attribution to a particular boundary/operator defect is not established.')
    (OUT/'boundary-diagnosis.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(OUT/'boundary-difference-maps.npz', **maps)


def main():
    results = {}; initial = {}; final = {}
    for name in CASES:
        results[name], initial[name], final[name] = diagnostics(name)
        print(name, 'reduced', flush=True)
    for name in CASES[1:3]:
        assert np.array_equal(initial[CASES[0]].coeff, initial[name].coeff), 'Temporal initialization differs'
        assert np.array_equal(initial[CASES[0]].surface, initial[name].surface)
    assert all(np.array_equal(initial[CASES[0]].surface[:, :3], s.surface[:, :3]) for s in initial.values())
    pairs = {}
    for key, i, j in [('dt_to_half', 0, 1), ('half_to_quarter', 1, 2), ('dt_to_quarter', 0, 2), ('space', 2, 3)]:
        a, b = CASES[i], CASES[j]
        ca, cb = results[a]['endpoint_cumulative_slip'], results[b]['endpoint_cumulative_slip']
        ref = final[CASES[3]] if key == 'space' else final[CASES[2]]
        pairs[key] = dict(cases=[a, b], final=compare(final[a], final[b], ca, cb, region_reference=ref))
        if key == 'space':
            pairs[key]['initial'] = compare(initial[a], initial[b])
    # Norm ratio of successive differences; avoid interpreting denominator changes as order.
    orders = dict(stress={}, surface={})
    a, b = pairs['dt_to_half']['final'], pairs['half_to_quarter']['final']
    order = lambda x,y: float(np.log2(x/y)) if x > 0 and y > 0 else None
    for region in a['stress']:
        orders['stress'][region] = order(a['stress'][region]['absolute_L2'], b['stress'][region]['absolute_L2'])
    for region in a['surface']:
        orders['surface'][region] = {key:order(a['surface'][region][key]['absolute_L2'], b['surface'][region][key]['absolute_L2']) for key in a['surface'][region]}
    # Saved-state diagnostic refinement only; no additional native run.
    refined = compare(final[CASES[2]], final[CASES[3]], bulk_order=12)
    result = dict(cases=results, comparisons=pairs, temporal_orders=orders,
                  spatial_stress_q12=refined['stress'], temporal_initialization_bitwise_equal=True,
                  surface_initial_geometry_bitwise_equal=True,
                  conventions='Protocol: Eulerian bulk tube radius .2 m, material coordinate central [.2,.8], outer fifths endpoints; common split label Gauss6 traction; q8 bulk. Normal traction tensile-positive tau_nn-p; effective compression 5+p-tau_nn. Cumulative slip is endpoint sum(dt*V), excluding initialization. All differences are numerical sensitivities, not continuum errors.')
    (OUT/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    boundary_diagnostics(final)
    summary = []
    for name, r in results.items():
        last = r['coupled_work']['steps'][-1]
        summary.append(dict(case=name, cells=final[name].n**2, dt=r['arguments']['dt'],
                            steps=len(r['steps'])-1, labels=len(final[name].surface),
                            wall_s=r['wall_s'], linear_solves=r['linear_solves'], linear_iterations=r['linear_iterations'],
                            min_curve_clearance=r['min_curve_clearance'], spin_mean=np.mean(r['spin_at_labels']),
                            spin_min=min(r['spin_at_labels']), spin_max=max(r['spin_at_labels']),
                            endpoint_signed=r['endpoint_signed'], endpoint_absolute=r['endpoint_absolute'],
                            endpoint_relative=r['endpoint_relative'], trapezoid_signed=r['trapezoid_signed'],
                            trapezoid_absolute=r['trapezoid_absolute'], trapezoid_relative=r['trapezoid_relative'],
                            coupled_signed=r['coupled_work']['signed'], coupled_absolute=r['coupled_work']['absolute'],
                            coupled_relative=r['coupled_work']['relative_active'],
                            final_face_power=last['face_mechanical_power'], final_weak_power=last['weak_mechanical_power'],
                            final_flux=last['flux'], profile_mass_change=r['profile_mass']/r['initial_mass']-1,
                            normalization_relative_error=r['normalization_integral_relative_error']))
    with (OUT/'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    with (OUT/'differences.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['comparison', 'region', 'quantity', 'absolute_L2', 'reference_L2', 'relative_L2', 'Linf'])
        writer.writeheader()
        for name, pair in pairs.items():
            for region, m in pair['final']['stress'].items():
                writer.writerow(dict(comparison=name, region=region, quantity='stress', **m))
            for region, fields in pair['final']['surface'].items():
                for key, m in fields.items():
                    writer.writerow(dict(comparison=name, region=region, quantity=key, **m))
    entry = json.loads((OUT/'entry.json').read_text())
    verified = {}
    for key in ('source_hashes', 'binaries', 'protected_files'):
        actual = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in entry[key]}
        assert actual == entry[key], key
        verified[key] = len(actual)
    (OUT/'verification.json').write_text(json.dumps(verified, indent=2)+'\n')
    print('Four-case comparison, geometry/source/bounds/publication checks and provenance verified.', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--boundary-only', action='store_true', help='Localize global differences using only saved final fields')
    if parser.parse_args().boundary_only:
        final = {}
        for name in CASES:
            _, _, _, states = load(name, final_only=True); final[name] = states[max(states)]
        boundary_diagnostics(final)
    else:
        main()
