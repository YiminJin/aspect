#!/usr/bin/env python3
"""One axial sensitivity comparison; actual native fields, no history remap.

All interpolation below is of exported physical Q1 fields or of a diagnostic
test function. Constitutive quantities are never recomputed from averages.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import os
import struct
import zlib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from numpy.polynomial.legendre import leggauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from p5c_analysis import read, save
from p5c_specimen_analysis import grid, history

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'build-particle-free/p5c-axial'
BASE = ROOT / 'build-particle-free/p5c-native'
OUT = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/p5c_axial'
LABEL = {'fixed': 'Fixed-direction cohesive flow', 'radial': 'Radial tensor flow (direction free)'}
COLORS = {'fixed': '#0072b2', 'radial': '#d55e00'}
LEVELS = [.01, .05, .1, .2, .4]


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)


def verify_resolved_inputs():
    def values(tree, prefix=''):
        flat = {}
        for key, value in tree.items():
            name = prefix+key
            if 'alias' in value:
                continue  # The resolved target parameter is compared below.
            if 'value' in value:
                flat[name] = value['value']
            else:
                flat.update(values(value, name+'/'))
        return flat
    result = {}
    allowed = {'Output directory', 'Geometry model/Box/X repetitions', 'Termination criteria/End step'}
    for mode in ['fixed', 'radial']:
        for stage in ['prefix', 'active']:
            suffix = '-half' if stage == 'active' else ''
            original = BASE/'runs'/f'seed-{mode}{suffix}'
            old = values(json.loads((original/'parameters.json').read_text()))
            new = values(json.loads((WORK/'runs'/f'{mode}-{stage}'/'parameters.json').read_text()))
            assert old.keys() == new.keys()
            differences = {key: {'96x128': old[key], '192x128': new[key]} for key in old if old[key] != new[key]}
            assert set(differences) <= allowed, differences
            result[f'{mode}-{stage}'] = differences
    save(OUT/'resolved-input-comparison.json', result)
    rows = []
    with (ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/p5c_native/parameters.csv').open() as stream:
        for row in csv.DictReader(stream):
            value = row['fixed_tensor']
            fine = '192' if row['parameter'] == 'Nx' else '.0078125' if row['parameter'] == 'dx' else value
            name = {'P4_dt': 'dt_prefix', 'focused_half_dt': 'dt_active', 'focused_halving_start': 'dt_switch_time'}.get(row['parameter'], row['parameter'])
            rows.append({'parameter': name, '96_fixed': value, '96_radial': value,
                         '192_fixed': fine, '192_radial': fine, 'units': row['units']})
    write_csv(OUT/'parameters.csv', rows)


def combine(mesh, mode):
    sources = ([BASE/'runs'/f'seed-{mode}', BASE/'runs'/f'seed-{mode}-half'] if mesh == 96
               else [WORK/'runs'/f'{mode}-prefix', WORK/'runs'/f'{mode}-active'])
    target = WORK/'analysis'/f'{mesh}-{mode}'
    target.mkdir(parents=True, exist_ok=True)
    for j, source in enumerate(sources):
        for pattern in ['phase-*.csv', 'means-*.csv', 'boundary-reactions-*.csv', 'phase-forces-*.csv']:
            for path in source.glob(pattern):
                if len(path.stem[-4:]) != 4 or not path.stem[-4:].isdigit():
                    continue
                step = int(path.stem[-4:])
                if (j == 0 and step > 95) or (j == 1 and step <= 95):
                    continue
                link = target/path.name
                if not link.exists():
                    os.link(path, link)
    for name in ['seed-prescription.csv', 'fracture-parameters.csv']:
        if not (target/name).exists():
            os.link(sources[0]/name, target/name)
    for name in ['bulk-accepted.csv', 'bulk-stability.csv', 'bulk-profile.csv', 'resources.csv']:
        first = (sources[0]/name).read_text().splitlines()
        second = (sources[1]/name).read_text().splitlines()
        (target/name).write_text('\n'.join([first[0], *[r for r in first[1:] if int(r.split(',')[0]) <= 95],
                                          *[r for r in second[1:] if int(r.split(',')[0]) > 95]])+'\n')
    return target


def q1_norm(x, y, values):
    """Exact consistent Q1 L2 norm on this Cartesian grid, Gauss2."""
    z, w = leggauss(2)
    total = 0.
    area = np.diff(y)[:, None]*np.diff(x)[None, :]
    for u, wu in zip((z+1)/2, w/2):
        for v, wv in zip((z+1)/2, w/2):
            p = ((1-u)*(1-v)*values[:-1, :-1]+u*(1-v)*values[:-1, 1:]
                 +(1-u)*v*values[1:, :-1]+u*v*values[1:, 1:])
            total += wu*wv*np.sum(area*p*p)
    return float(np.sqrt(total))


def sample(x, y, p, xx, yy):
    X, Y = np.meshgrid(xx, yy)
    return RegularGridInterpolator((y, x), p)(np.column_stack([Y.ravel(), X.ravel()])).reshape(X.shape)


def analysis_controls():
    x, y = np.array([0., 1.]), np.array([0., 2.])
    tests = [('constant', np.ones((2, 2)), 2.),
             ('affine_x_plus_2y', np.array([[0., 1.], [4., 5.]]), 46/3),
             ('mixed_xy', np.array([[0., 0.], [0., 2.]]), 8/9)]
    controls = {}
    for name, values, exact in tests:
        computed = q1_norm(x, y, values)**2
        defect = abs(computed-exact)/max(1, exact)
        assert defect <= 1e-12
        controls[name] = {'squared_L2': computed, 'analytic_squared_L2': exact, 'scaled_defect': defect}
    save(OUT/'analysis-controls.json', controls)


def verify(run, mesh):
    a = read(run/'bulk-accepted.csv')
    s = read(run/'bulk-stability.csv')
    profile = read(run/'bulk-profile.csv')
    r = read(run/'resources.csv')
    assert np.array_equal(a['step'], np.arange(1, 138))
    assert np.array_equal(a['publications'], a['step'])
    dt = np.where(a['step'] <= 95, .00125, .000625)
    times = .00125*np.minimum(a['step'], 95)+.000625*np.maximum(a['step']-95, 0)
    assert np.max(abs(a['dt']-dt)) < 1e-15 and np.max(abs(a['time']-times)) < 2e-14
    assert max(a['mechanical_residual']) <= 1e-9
    assert max(a['phase_KKT']) <= 1e-7 and max(a['projection_error']) <= 1e-12
    bound = 1e-10*np.maximum.reduce([np.ones(len(a)), abs(a['Wext']), a['bulk']+a['cohesive']+a['fracture']])
    assert max(abs(a['ledger_error'])/bound) <= 1
    assert all(s['stable'] == 1) and max(s['critical_masters']) <= 1024
    assert all(profile['ranks'] == 4) and all(r['freeze'] == 1)
    with (run/'resources.csv').open() as stream:
        fingerprints = {row['particle_fingerprint'] for row in csv.DictReader(stream)}
    assert len(fingerprints) == 1 and all(r['particles'] == mesh*128*9)
    nonempty = s[s['critical_masters'] > 0]
    return {'steps': 137, 'schedule_time_defect_s': float(max(abs(a['time']-times))),
            'max_mechanical_residual': float(max(a['mechanical_residual'])),
            'max_phase_KKT': float(max(a['phase_KKT'])), 'max_source_error': float(max(a['projection_error'])),
            'max_abs_BE_ledger_J_m': float(max(abs(a['ledger_error']))),
            'max_BE_bound_ratio': float(max(abs(a['ledger_error'])/bound)),
            'max_critical_masters': int(max(s['critical_masters'])),
            'minimum_nonempty_relaxed_curvature': float(min(nonempty['minimum_curvature'])) if len(nonempty) else None,
            'stability_retries': int(sum(profile['stability_retries'])),
            'particles': int(r['particles'][0]), 'particle_fingerprint': next(iter(fingerprints)),
            'max_rank_peak_KiB': int(max(r['max_rank_KiB'])),
            **{name+'_s': float(sum(profile[name])) for name in ['assembly', 'communication', 'linear', 'history', 'total', 'stability']}}


def common_kernel(run, step, centers):
    """Same coarse-Q1 hat on both grids: sum w_i g_i / sum w_i mass_i.

    Native phase gradients include the complete fracture weak term, including
    gradient contributions. The fine grid represents this test hat exactly.
    """
    positions = read(run/f'phase-{step:04}.csv')
    forces = read(run/f'phase-forces-{step:04}.csv')
    by_id = {int(row['dof']): row for row in positions}
    x = np.array([by_id[int(i)]['x'] for i in forces['dof']])
    y = np.array([by_id[int(i)]['y'] for i in forces['dof']])
    result = []
    for xc, yc in centers:
        w = np.maximum(0, 1-abs(x-xc)/.015625)*np.maximum(0, 1-abs(y-yc)/.0078125)
        mass = float(w@forces['mass'])
        result.append({'x_m': xc, 'y_m': yc, 'kernel_integral_m2': mass,
                       'mechanical_driving_Pa': float(-w@forces['mechanical_gradient']/mass),
                       'fracture_resistance_Pa': float(w@forces['fracture_gradient']/mass),
                       'total_weak_residual_Pa': float(w@forces['residual']/mass)})
    return result


def independent_fracture_kernel(run, step, center):
    """Independent AT1 weak force, Gauss2 exact for these Q1 Cartesian fields."""
    x, y, phi = grid(read(run/f'phase-{step:04}.csv'), 'phi')
    X, Y = np.meshgrid(x, y)
    xc, yc = center
    w = np.maximum(0, 1-abs(X-xc)/.015625)*np.maximum(0, 1-abs(Y-yc)/.0078125)
    hx, hy = np.diff(x)[None, :], np.diff(y)[:, None]
    z, weights = leggauss(2)
    numerator = denominator = 0.
    for u, wu in zip((z+1)/2, weights/2):
        for v, wv in zip((z+1)/2, weights/2):
            value = ((1-u)*(1-v)*w[:-1, :-1]+u*(1-v)*w[:-1, 1:]
                     +(1-u)*v*w[1:, :-1]+u*v*w[1:, 1:])
            def gradient(p):
                return (((1-v)*(p[:-1, 1:]-p[:-1, :-1])+v*(p[1:, 1:]-p[1:, :-1]))/hx,
                        ((1-u)*(p[1:, :-1]-p[:-1, :-1])+u*(p[1:, 1:]-p[:-1, 1:]))/hy)
            px, py = gradient(phi)
            wx, wy = gradient(w)
            numerator += wu*wv*np.sum(hx*hy*8*(value+2*.046875**2*(px*wx+py*wy)))
            denominator += wu*wv*np.sum(hx*hy*value)
    return float(numerator/denominator)


def collect():
    OUT.mkdir(exist_ok=True)
    verify_resolved_inputs()
    analysis_controls()
    runs = {(mesh, mode): combine(mesh, mode) for mesh in [96, 192] for mode in ['fixed', 'radial']}
    for mesh in [96, 192]:
        assert np.array_equal(grid(read(runs[mesh, 'fixed']/'phase-0000.csv'), 'phi')[2],
                              grid(read(runs[mesh, 'radial']/'phase-0000.csv'), 'phi')[2])
        for mode in ['fixed', 'radial']:
            initial_means = read(runs[mesh, mode]/'means-0000.csv')
            for key in ['tauxx', 'tauyy', 'tauxy', 'Y', 'Wxx', 'Wxy', 'Axx', 'Axy', 'Kxx', 'Kxy', 'L', 'bulk', 'cohesive']:
                assert np.count_nonzero(initial_means[key]) == 0, (mesh, mode, key)
    summaries, histories, checks, tables, spatial = {}, {}, {}, [], []
    for (mesh, mode), run in runs.items():
        name = f'{mesh}-{mode}'
        summary, rows = history(run, OUT, name)
        checks[name] = verify(run, mesh)
        x, y, initial = grid(read(run/'phase-0000.csv'), 'phi')
        for row in rows:
            phi = grid(read(run/f'phase-{row["step"]:04}.csv'), 'phi')[2]
            row['consistent_growth_L2_m'] = q1_norm(x, y, phi-initial)
            row['growth_L2_m'] = row['consistent_growth_L2_m']
        write_csv(OUT/f'{name}-history.csv', rows)
        phi = grid(read(run/'phase-0137.csv'), 'phi')[2]
        X, Y = np.meshgrid(x, y)
        active = phi-initial > 1e-8
        summary['growth_nodes'] = int(active.sum())
        summary['growth_x_interval_m'] = [float(X[active].min()), float(X[active].max())] if active.any() else None
        summary['growth_y_interval_m'] = [float(Y[active].min()), float(Y[active].max())] if active.any() else None
        summary['initial_fracture_J_m'] = float(read(run/'bulk-accepted.csv')['fracture'][0])
        summary['added_fracture_J_m'] = rows[-1]['fracture']-summary['initial_fracture_J_m']
        summary['consistent_growth_L2_m'] = rows[-1]['consistent_growth_L2_m']
        mass = np.ones_like(initial)*(x[1]-x[0])*(y[1]-y[0])
        mass[[0, -1], :] *= .5
        mass[:, [0, -1]] *= .5
        summary['outer_flank_nodal_growth_m2'] = float(np.sum(mass*(phi-initial)*(abs(Y-.5) > .05)))
        summaries[name], histories[mesh, mode] = summary, rows
        tables.append({'Nx': mesh, 'Ny': 128, 'constitutive_mode': LABEL[mode],
                       'dx_m': 1.5/mesh, 'dy_m': 1/128,
                       'initial_representation_L2_relative': summary['initialization']['L2_relative_representation'],
                       'initial_fracture_J_m': summary['initial_fracture_J_m'],
                       'onset_lower_s': summary['onset_bracket_s'][0], 'onset_upper_s': summary['onset_bracket_s'][1],
                       'integrated_growth_m2': rows[-1]['integral_growth_m2'],
                       'peak_increment': rows[-1]['peak_increment'],
                       'growth_L2_m': rows[-1]['consistent_growth_L2_m'],
                       'added_fracture_J_m': summary['added_fracture_J_m'],
                       'growth_nodes': summary['growth_nodes'],
                       'growth_xmin_m': summary['growth_x_interval_m'][0], 'growth_xmax_m': summary['growth_x_interval_m'][1],
                       'growth_ymin_m': summary['growth_y_interval_m'][0], 'growth_ymax_m': summary['growth_y_interval_m'][1],
                       'outer_flank_nodal_growth_m2': summary['outer_flank_nodal_growth_m2'],
                       'initially_intact_peak_increment': rows[-1]['peak_initially_intact_growth']})
    for mode in ['fixed', 'radial']:
        coarse, fine = runs[96, mode], runs[192, mode]
        cx, cy, ci = grid(read(coarse/'phase-0000.csv'), 'phi')
        fx, fy, fi = grid(read(fine/'phase-0000.csv'), 'phi')
        pi = sample(cx, cy, ci, fx, fy)
        for step in range(138):
            cp = grid(read(coarse/f'phase-{step:04}.csv'), 'phi')[2]
            fp = grid(read(fine/f'phase-{step:04}.csv'), 'phi')[2]
            difference = sample(cx, cy, cp-ci, fx, fy)-(fp-fi)
            norm = q1_norm(fx, fy, fp-fi)
            spatial.append({'mode': LABEL[mode], 'step': step,
                            'time_s': histories[192, mode][step]['time_s'],
                            'initial_fields_L2_difference_m': q1_norm(fx, fy, pi-fi),
                            'initial_fields_peak_difference': float(np.max(abs(pi-fi))),
                            'growth_fields_L2_difference_m': q1_norm(fx, fy, difference),
                            'growth_relative_L2_difference': q1_norm(fx, fy, difference)/norm if np.max(fp-fi) > 1e-8 else None,
                            'peak_growth_difference': histories[96, mode][step]['peak_increment']-histories[192, mode][step]['peak_increment']})
    # A single common saved time before any of the four native onsets.
    pregrowth = min(s['onset_bracket_s'][0] for s in summaries.values())
    pre_step = min(range(138), key=lambda i: abs(histories[96, 'fixed'][i]['time_s']-pregrowth))
    centers = [(float(x), float(y)) for y in [.5-.0859375, .5-.078125, .5-.046875, .5-.0234375, .5-.0078125, .5,
                                             .5+.0078125, .5+.0234375, .5+.046875, .5+.078125, .5+.0859375]
               for x in np.arange(.96875, 1.125+.001, .015625)]
    kernels, native, independent = [], [], []
    for (mesh, mode), run in runs.items():
        for step in sorted(set([pre_step, 119, 129, 137])):
            for row in common_kernel(run, step, centers):
                kernels.append({'Nx': mesh, 'mode': LABEL[mode], 'step': step,
                                'time_s': histories[mesh, mode][step]['time_s'], **row})
        for center in [(1.078125, .5), (1.078125, .4921875), (1.09375, .5)]:
            exported = common_kernel(run, pre_step, [center])[0]['fracture_resistance_Pa']
            reference = independent_fracture_kernel(run, pre_step, center)
            assert abs(exported-reference)/8 <= 1e-12
            independent.append({'Nx': mesh, 'mode': LABEL[mode], 'x_m': center[0], 'y_m': center[1],
                                'native_resistance_Pa': exported, 'independent_resistance_Pa': reference,
                                'defect_over_Kf': abs(exported-reference)/8})
        # Native nodal growth condition, independently of common-kernel averages.
        onset_time = summaries[f'{mesh}-{mode}']['onset_bracket_s'][1]
        onset_step = min(range(138), key=lambda i: abs(histories[mesh, mode][i]['time_s']-onset_time))
        for step in sorted(set([pre_step, onset_step, 137])):
            a = read(run/f'phase-{step:04}.csv'); f = read(run/f'phase-forces-{step:04}.csv')
            lookup = {int(row['dof']): row for row in a}
            initial = {int(row['dof']): row['phi'] for row in read(run/'phase-0000.csv')}
            for row in f:
                point = lookup[int(row['dof'])]
                if .96 <= point['x'] <= 1.15 and abs(point['y']-.5) <= .11:
                    native.append({'Nx': mesh, 'mode': LABEL[mode], 'step': step,
                                   'time_s': histories[mesh, mode][step]['time_s'], 'x_m': point['x'], 'y_m': point['y'],
                                   'phi_initial': initial[int(row['dof'])], 'increment': point['phi']-initial[int(row['dof'])],
                                   'mechanical_driving_Pa': -row['mechanical_gradient']/row['mass'],
                                   'fracture_resistance_Pa': row['fracture_gradient']/row['mass'],
                                   'total_weak_residual_Pa': row['residual']/row['mass'],
                                   'obstacle_gap': row['phi']-row['obstacle']})
    reductions = {}
    for mesh in [96, 192]:
        fixed, radial = summaries[f'{mesh}-fixed'], summaries[f'{mesh}-radial']
        reductions[str(mesh)] = {key: 1-radial['final'][key]/fixed['final'][key]
                                for key in ['integral_growth_m2', 'peak_increment']}
    entry = json.loads((WORK/'entry.json').read_text())
    for section in ['protected', 'artifacts']:
        for name, digest in entry[section].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    costs = {}
    for mode in ['fixed', 'radial']:
        for stage in ['prefix', 'active']:
            label = f'p5c-axial-{mode}-{stage}'+('-frozen' if mode == 'fixed' and stage == 'prefix' else '')
            command = json.loads((ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence'/f'{label}.json').read_text())
            assert command['exit_code'] == 0
            costs[f'{mode}-{stage}'] = command
            gates = read(WORK/'runs'/f'{mode}-{stage}'/'gates.csv')
            assert all(gates['passed'] == 1)
            # Endpoint restart actually contains the published accepted tensor state.
            final = 95 if stage == 'prefix' else 137
            found = []
            for checkpoint in (WORK/'runs'/f'{mode}-{stage}'/'restart').glob('[0-9][0-9]'):
                raw = zlib.decompress((checkpoint/'resume.z').read_bytes()[16:])
                tag = f'bulk cohesive {mode} tensor cell history v1'.encode()
                q, step, publications, time = struct.unpack_from('<IIQd', raw, raw.index(tag)+len(tag))
                if step == final:
                    assert q == 8 and publications == final
                    found.append({'directory': str(checkpoint.relative_to(ROOT)), 'step': step, 'publications': publications, 'time_s': time})
            assert len(found) == 1
            checks[f'192-{mode}'][f'{stage}_checkpoint'] = found[0]
    write_csv(OUT/'comparison.csv', tables)
    write_csv(OUT/'spatial-growth.csv', spatial)
    write_csv(OUT/'common-kernel-forces.csv', kernels)
    write_csv(OUT/'native-cap-balance.csv', native)
    write_csv(OUT/'independent-fracture-kernel.csv', independent)
    result = {'summaries': summaries, 'checks': checks, 'radial_reductions': reductions,
              'common_pregrowth_time_s': pregrowth, 'common_pregrowth_step': pre_step,
              'strict_reference_match_passed': False, 'protected_hashes_verified': len(entry['protected'])}
    save(OUT/'summary.json', result)
    save(OUT/'run-commands.json', costs)
    return runs, histories, result


def figures(runs, histories, result):
    # Same physical cells for constitutive-field means: aggregate each pair of
    # refined x cells into one original cell; no pointwise state reconstruction.
    cache = {}
    steps = [0, 95, 109, 115, 119, 123, 127, 129, 133, 137]
    for summary in result['summaries'].values():
        onset_time = summary['onset_bracket_s'][1]
        onset_step = min(range(138), key=lambda i: abs(histories[96, 'fixed'][i]['time_s']-onset_time))
        steps.extend([onset_step-1, onset_step])
    steps = sorted(set(steps))
    for (mesh, mode), run in runs.items():
        for step in steps:
            x, y, phi = grid(read(run/f'phase-{step:04}.csv'), 'phi')
            initial = grid(read(run/'phase-0000.csv'), 'phi')[2]
            a = read(run/f'means-{step:04}.csv')
            cx, cy, tau = grid(a, 'tauxy')
            drive, wx, wy = [grid(a, key)[2] for key in ['Y', 'Wxx', 'Wxy']]
            cache[mesh, mode, step] = (x, y, phi, initial, cx, cy, tau, drive, wx, wy)
    dmax = max(float((v[2]-v[3]).max()) for v in cache.values())
    taumax = max(float(abs(v[6]).max()) for v in cache.values())
    ymax = max(float(v[7].max()) for v in cache.values())
    amax = max(float(abs(np.degrees(np.arctan2(v[8], v[9])))[np.hypot(v[8], v[9]) > 1e-12].max(initial=0)) for v in cache.values()) or 1.
    extent = [.96, 1.15, .39, .61]
    node_rows, cell_rows = [], []
    for (mesh, mode, step), values in cache.items():
        x, y, phi, initial, cx, cy, tau, drive, wx, wy = values
        X, Y = np.meshgrid(x, y)
        mask = (X >= extent[0]) & (X <= extent[1]) & (Y >= extent[2]) & (Y <= extent[3])
        for j, i in zip(*np.where(mask)):
            node_rows.append({'Nx': mesh, 'mode': LABEL[mode], 'time_s': histories[mesh, mode][step]['time_s'],
                              'x_m': x[i], 'y_m': y[j], 'phi_initial': initial[j, i], 'phi': phi[j, i],
                              'increment': phi[j, i]-initial[j, i]})
        X, Y = np.meshgrid(cx, cy)
        mask = (X >= extent[0]) & (X <= extent[1]) & (Y >= extent[2]) & (Y <= extent[3])
        for j, i in zip(*np.where(mask)):
            defined = np.hypot(wx[j, i], wy[j, i]) > 1e-12
            cell_rows.append({'Nx': mesh, 'mode': LABEL[mode], 'time_s': histories[mesh, mode][step]['time_s'],
                              'x_m': cx[i], 'y_m': cy[j], 'physical_tauxy_Pa': tau[j, i], 'Y_Pa': drive[j, i],
                              'Wxx_per_s': wx[j, i], 'Wxy_per_s': wy[j, i],
                              'active_tensor_angle_deg': float(np.degrees(np.arctan2(wx[j, i], wy[j, i]))) if defined else None})
    write_csv(OUT/'cap-phase-nodes.csv', node_rows)
    write_csv(OUT/'cap-physical-cell-means.csv', cell_rows)

    def draw(step, whole=False):
        fig, axes = plt.subplots(4, 4, figsize=(12.4, 11), constrained_layout=True)
        for row, (mesh, mode) in enumerate([(96, 'fixed'), (192, 'fixed'), (96, 'radial'), (192, 'radial')]):
            x, y, phi, initial, cx, cy, tau, drive, wx, wy = cache[mesh, mode, step]
            angle = np.ma.masked_where(np.hypot(wx, wy) <= 1e-12, np.degrees(np.arctan2(wx, wy)))
            fields = [(x, y, phi-initial, 0, dmax, 'magma', 'Phase increment Δφ'),
                      (cx, cy, tau, -taumax, taumax, 'RdBu_r', 'Physical τxy [Pa]'),
                      (cx, cy, drive, 0, ymax, 'inferno', 'Mechanical driving Y [Pa]'),
                      (cx, cy, angle, -amax, amax, 'twilight_shifted', 'Active tensor flow angle [°]')]
            for col, (xx, yy, z, lo, hi, cmap, title) in enumerate(fields):
                ax = axes[row, col]
                im = ax.pcolormesh(xx, yy, z, shading='nearest', vmin=lo, vmax=hi, cmap=cmap)
                ax.contour(x, y, initial, levels=LEVELS, colors='#31c8e3', linewidths=.65, linestyles='dashed')
                ax.contour(x, y, phi, levels=LEVELS, colors='white', linewidths=.55)
                if not whole and col == 0:
                    X, Y = np.meshgrid(x, y)
                    near = (X >= extent[0]) & (X <= extent[1]) & (Y >= extent[2]) & (Y <= extent[3])
                    ax.scatter(X[near], Y[near], s=5, c='#00d8ff', marker='.', alpha=.6)
                    active = near & (phi-initial > 1e-8)
                    ax.scatter(X[active], Y[active], s=12, facecolors='none', edgecolors='lime', linewidths=.6)
                ax.set_aspect('equal')
                ax.set_xlim(0 if whole else extent[0], 1.5 if whole else extent[1])
                ax.set_ylim(0 if whole else extent[2], 1 if whole else extent[3])
                if whole:
                    ax.plot(1, .5, 'c+', ms=5)
                ax.set_xlabel('x [m]', fontsize=8)
                ax.set_ylabel('y [m]', fontsize=8)
                ax.tick_params(labelsize=7)
                ax.set_title(f'{LABEL[mode]}\n{mesh}×128: {title}', fontsize=8)
                fig.colorbar(im, ax=ax, shrink=.75, pad=.02).ax.tick_params(labelsize=7)
        time = histories[96, 'fixed'][step]['time_s']
        fig.suptitle(f't = {time:.5f} s; Δt = '+('0.00125 s' if step <= 95 else '0.000625 s')+
                     '\nDashed cyan = initial contours; white = current (.01/.05/.1/.2/.4). Dots = bulk vertices; green rings = growing phase nodes.'+
                     '\nFlow angle describes the constitutive tensor increment, not the crack angle.', fontsize=10)
        return fig
    for step in [119, 129, 137]:
        fig = draw(step)
        fig.savefig(OUT/f'cap-{step:04}.png', dpi=170)
        plt.close(fig)
    fig = draw(137, whole=True)
    fig.savefig(OUT/'whole-domain.png', dpi=150)
    plt.close(fig)
    images = []
    for step in steps:
        fig = draw(step)
        stream = io.BytesIO()
        fig.savefig(stream, format='png', dpi=80)
        plt.close(fig)
        stream.seek(0)
        images.append(Image.open(stream).convert('P', palette=Image.Palette.ADAPTIVE))
    images[0].save(OUT/'axial-comparison.gif', save_all=True, append_images=images[1:], duration=650, loop=0)
    # Initialization maps, common scales, separate from physical growth.
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), constrained_layout=True)
    cx, cy, cp, _, _, _, _, _, _, _ = cache[96, 'fixed', 0]
    fx, fy, fp, _, _, _, _, _, _, _ = cache[192, 'fixed', 0]
    delta = fp-sample(cx, cy, cp, fx, fy)
    for ax, x, y, p, title in zip(axes, [cx, fx, fx], [cy, fy, fy], [cp, fp, delta],
                                ['96×128 initial phase', '192×128 initial phase', 'Initial representation difference']):
        limit = float(abs(delta).max())
        difference = ax is axes[-1]
        im = ax.pcolormesh(x, y, p, shading='nearest', cmap='RdBu_r' if difference else 'viridis',
                           vmin=-limit if difference else 0, vmax=limit if difference else .6)
        reference = fp if difference else p
        ax.contour(x, y, reference, levels=LEVELS, colors='white', linewidths=.6)
        X, Y = np.meshgrid(x, y)
        keep = (X >= .96) & (X <= 1.15) & (abs(Y-.5) <= .11)
        ax.scatter(X[keep], Y[keep], s=4, color='black', alpha=.4)
        ax.set_aspect('equal'); ax.set_xlim(.96, 1.15); ax.set_ylim(.39, .61)
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=.7)
    fig.savefig(OUT/'initial-representation.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for (mesh, mode), rows in histories.items():
        t = [r['time_s'] for r in rows]
        label = f'{LABEL[mode]}, {mesh}×128'
        style = '--' if mesh == 96 else '-'
        for ax, key, scale in zip(axes, ['integral_growth_m2', 'peak_increment', 'top_horizontal_reaction_N_per_m'], [1e6, 1, 1]):
            ax.plot(t, [r[key]*scale for r in rows], style, color=COLORS[mode], label=label)
            ax.axvline(.14, color='.5', linewidth=.7)
            ax.set_xlabel('time [s]'); ax.set_xlim(.125, .145)
    for ax, title in zip(axes, ['Integrated Δφ [10⁻⁶ m²]', 'Peak Δφ', 'Top horizontal reaction [N/m]']):
        ax.set_ylabel(title); ax.grid(alpha=.2)
    axes[0].legend(fontsize=6)
    fig.savefig(OUT/'growth-reaction.png', dpi=180); plt.close(fig)
    kernels = read(OUT/'common-kernel-forces.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for mesh in [96, 192]:
        for mode in ['fixed', 'radial']:
            keep = (kernels['Nx'] == mesh) & (kernels['mode'] == LABEL[mode]) & (abs(kernels['y_m']-.5) < 1e-12) & (kernels['step'] == result['common_pregrowth_step'])
            a = kernels[keep]
            for ax, key in zip(axes, ['mechanical_driving_Pa', 'fracture_resistance_Pa']):
                ax.plot(a['x_m'], a[key], '--o' if mesh == 96 else '-s', markersize=3, color=COLORS[mode], label=f'{LABEL[mode]}, {mesh}×128')
                ax.set_xlabel('kernel center x [m]'); ax.set_xlim(1.045, 1.11); ax.grid(alpha=.2)
    axes[0].set_ylabel('Mechanical driving [Pa]'); axes[1].set_ylabel('Fracture resistance [Pa]')
    axes[1].set_ylim(0, 11)
    axes[0].legend(fontsize=6)
    fig.suptitle(f'Before all four onsets: t={result["common_pregrowth_time_s"]:.6f} s; common physical Q1 hat at y=.5 m')
    fig.savefig(OUT/'pregrowth-balance.png', dpi=180); plt.close(fig)
    save(OUT/'plot-definitions.json', {'constitutive_labels': LABEL, 'flow_angle': 'atan2(mean Wxx,mean Wxy), degrees; only hypot(mean Wxx,mean Wxy)>1e-12/s. Tensor component angle, NOT crack angle.',
         'phase': 'native Q1 phase; increment relative to each mesh own initial field; contours '+str(LEVELS),
         'stress_driving': 'actual Gauss8 cell averages of physical deviatoric stress, mechanical Y and W; no constitutive recomputation after averaging',
         'common_force_kernel': 'coarse Q1 tensor-product hat centered at specified physical x,y; radii .015625 m in x,.0078125 m in y; exactly representable on both grids. Sum w_i g_i / sum w_i mass_i. Includes complete native fracture gradient term.',
         'native_balance': 'each native nodal test functional divided by its own lumped mass; distinct basis supports across meshes, retained separately from common kernels',
         'growth_L2': 'consistent Q1 L2 integral (Gauss2 exact); coarse growth evaluated on refined mesh for physical-field differences',
         'steps': steps, 'scales': {'delta_phi_max': dmax, 'tau_limit_Pa': taumax, 'Y_max_Pa': ymax, 'alpha_limit_deg': amax}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    runs, histories, result = collect()
    if not args.no_figures:
        figures(runs, histories, result)
    print(json.dumps({'radial_reductions': result['radial_reductions'], 'pre_growth_time_s': result['common_pregrowth_time_s']}))
