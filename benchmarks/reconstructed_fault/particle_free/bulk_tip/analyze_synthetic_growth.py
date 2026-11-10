#!/usr/bin/env python3
"""Reduce the two selected freely evolving synthetic BE histories (no solver changes)."""
import argparse
import json
import os
from pathlib import Path

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/aspect-synthetic-growth-mpl')
import numpy as np
from analyze_time_update import read, frame, balance, compare

ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / 'build-particle-free/bulk-tip/runs'
OUT = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/synthetic_growth'
H = .125
LEVELS = (.001, .01, .1, .5)


def regions(x, y):
    """Disjoint geometric regions relative to the equilibrated seed support."""
    axial = abs(x - 4) >= 2
    transverse = ~axial & (abs(y - 2) >= .25)
    inside = ~axial & ~transverse
    return dict(axial_intact=axial, transverse_intact=transverse,
                old_tip_band=inside & (abs(x - 4) >= 1.5),
                old_central_band=inside & (abs(x - 4) < 1.5))


def sampled_extents(points):
    result = {}
    for level in LEVELS:
        mask = points.phi >= level
        result[str(level)] = [float(f(points[k][mask])) for k in ('x', 'y')
                              for f in (np.min, np.max)] if np.any(mask) else None
    return result


def polynomial_sections(points):
    """Threshold extents of the Q2 traces on y=2 and x=4, including both DG sides.

    Recover each polynomial from all 36 Gauss samples, check the reconstruction,
    then solve its 1-D quadratic on the requested section. No smoothing or
    contour interpolation across DG faces. These are threshold locations, not
    a mesh-independent crack-front definition.
    """
    extents = {axis: {str(l): [] for l in LEVELS} for axis in ('axial', 'normal')}
    max_defect = 0.
    ordered = points[np.lexsort((points.q, points.cell))]
    # Group once: a fine-mesh trace must not scan every point for every cell.
    groups = np.split(ordered, np.flatnonzero(ordered.cell[1:] != ordered.cell[:-1]) + 1)
    for p in groups:
        cx, cy = float(np.mean(p.x)), float(np.mean(p.y))
        axes = []
        if abs(cy - 2) <= H / 2 + 1e-10:
            axes.append(('axial', cx, (2 - cy) * 2 / H))
        if abs(cx - 4) <= H / 2 + 1e-10:
            axes.append(('normal', cy, (4 - cx) * 2 / H))
        if not axes:
            continue
        X, Y = (p.x - cx) * 2 / H, (p.y - cy) * 2 / H
        v = np.polynomial.polynomial.polyvander2d(X, Y, [2, 2]).reshape(-1, 9)
        c = np.linalg.lstsq(v, p.phi, rcond=None)[0].reshape(3, 3)
        max_defect = max(max_defect, float(max(abs(v @ c.ravel() - p.phi))))
        for axis, center, fixed in axes:
            coefficients = (c if axis == 'axial' else c.T) @ np.array([1, fixed, fixed**2])
            for level in LEVELS:
                polynomial = coefficients.copy()
                polynomial[0] -= level
                roots = np.polynomial.polynomial.polyroots(polynomial)
                breaks = sorted([-1., 1.] + [float(r.real) for r in roots
                                             if abs(r.imag) < 1e-10 and -1 < r.real < 1])
                for left, right in zip(breaks[:-1], breaks[1:]):
                    if np.polynomial.polynomial.polyval((left + right) / 2, polynomial) >= 0:
                        extents[axis][str(level)].extend([center + H * left / 2, center + H * right / 2])
    for levels in extents.values():
        for key, values in levels.items():
            levels[key] = [min(values), max(values)] if values else None
    assert max_defect < 1e-11, max_defect
    return dict(extents=extents, reconstruction_max_error=max_defect)


def energy(rows, initial_fracture, start):
    mask = rows.time > start + 1e-10
    r = rows[mask]
    parent = initial_fracture if start == 0 else float(rows.fracture[np.argmin(abs(rows.time - start))])
    if start:
        assert min(abs(rows.time - start)) < 1e-10
    result = {}
    for key in ('Wext', 'energy_change', 'Db', 'Dc', 'Df', 'stress_loss',
                'traction_loss', 'phase_remainder', 'fracture_loss', 'phase_work',
                'mechanical_work', 'ledger_error'):
        result[key] = dict(signed=float(sum(r[key])), absolute=float(sum(abs(r[key]))))
    error = r.Wext - r.energy_change - r.Db - r.Dc - r.Df
    added = float(r.fracture[-1] - parent)
    result.update(added_fracture=added,
                  physical_work_error=dict(signed=float(sum(error)), absolute=float(sum(abs(error)))),
                  signed_error_over_added_fracture=float(sum(error) / added) if added > 1e-12 else None,
                  absolute_error_over_added_fracture=float(sum(abs(error)) / added) if added > 1e-12 else None)
    # Existing BE ledger, independently reduced before subtracting numerical terms.
    predicted = r.stress_loss + r.traction_loss + r.phase_remainder + r.fracture_loss - r.phase_work - r.mechanical_work + r.ledger_error
    result['ledger_reconstruction_defect'] = float(max(abs(error - predicted)))
    assert result['ledger_reconstruction_defect'] < 1e-10
    return result


def analyze(label, accepted_end=.06, allow_stopped=False):
    global H
    directory = RUNS / label / 'output'
    command = json.loads((directory.parent / 'command.json').read_text())
    refinement = int(command['arguments']['refinement'])
    H = .125 / 2**refinement
    assert 'exit_code' in command, 'Only reduce stopped/completed histories.'
    assert command['exit_code'] == 0 or allow_stopped
    assert not command['arguments']['fixed_phase'] and not command['arguments']['resume_from']
    rows = read(directory / 'bulk-accepted.csv')
    rows = rows[rows.time <= accepted_end + 1e-10]
    assert abs(rows.time[-1] - accepted_end) < 1e-10
    assert np.allclose(rows.time, np.arange(1, len(rows) + 1) * rows.dt[0], rtol=0, atol=1e-12)
    initial = frame(directory, 0)
    for name, mask in regions(initial.x, initial.y).items():
        if name.endswith('intact'):
            assert np.all(initial.phi[mask] == 0), 'Region definition does not match the initial seed.'
    seed = balance(directory, 'tip-seed-fracture-r*.csv')
    initial_energy = float(read(directory / 'tip-initial.csv').fracture[0])
    result = dict(label=label, command=command, initial_fracture=initial_energy,
                  accepted_end=float(rows.time[-1]),
                  trajectory_complete=command['exit_code']==0 and abs(rows.time[-1]-command['arguments']['end'])<1e-10,
                  initial_extents=sampled_extents(initial), initial_sections=polynomial_sections(initial),
                  energy_full=energy(rows, initial_energy, 0), energy_window=energy(rows, initial_energy, .04),
                  gates={k: float(max(abs(rows[k]))) for k in ('mechanical_residual', 'phase_KKT', 'ledger_error')},
                  onset={}, steps=[])
    result['cost'] = dict(wall_seconds=command['seconds'], intervals=len(rows), ranks=command['arguments']['ranks'])
    for name in ('tip-profile.csv', 'tip-solver.csv'):
        data = read(directory / name)
        result['cost'][name] = {key: float(sum(data[key])) for key in data.dtype.names
                                if key not in ('step', 'sparse_entries', 'owned_phase_rows')}
    saved = {0: initial}
    for row in rows:
        step = int(row.step)
        p = frame(directory, step)
        b = balance(directory, f'tip-balance-{step:04d}-r*.csv')
        assert len(np.unique(b.x)) == 64 * 2**refinement and len(np.unique(b.y)) == 32 * 2**refinement
        assert np.allclose(np.diff(np.unique(b.x)), H, rtol=0, atol=1e-12)
        s = read(directory / f'tip-sections-{step:04d}.csv')
        assert np.array_equal(p.cell, initial.cell) and np.array_equal(p.q, initial.q)
        assert np.array_equal(b.cell, seed.cell) and np.array_equal(b.i, seed.i)
        delta = p.phi - initial.phi
        # Exact quadratic SIPG chain rule: includes face energy, assigned to
        # rows by their cell. Regional values can be signed; never clip them.
        fracture = .5 * (b.fracture_gradient + seed.fracture_gradient) * (b.B - seed.B)
        partition_error = float(sum(fracture) - (row.fracture - initial_energy))
        assert abs(partition_error) < 1e-9, partition_error
        assert min(b.B - b.lower) > -1e-10 and max(b.B) < 1 + 1e-10
        regional = {}
        row_regions = regions(b.x, b.y)
        for name, mask in regions(p.x, p.y).items():
            regional[name] = dict(max_delta_phi=float(max(delta[mask])),
                                  integrated_delta_phi=float(sum(p.weight[mask] * delta[mask])),
                                  added_fracture=float(sum(fracture[row_regions[name]])),
                                  volume_fracture_change=float(sum(p.weight[mask] * (p.fracture_density[mask] - initial.fracture_density[mask]))))
        local_work = row.dt * p.weight * p.w * s.force[p.section.astype(int)]
        power_defect = float(sum(local_work) - row.Df)
        assert abs(power_defect) < 1e-9
        abs_rate = float(sum(s.ds * s.integral_abs_w))
        opposite = multiple = 0
        ordered = p[np.lexsort((p.y, p.section))]
        for section in np.split(ordered, np.flatnonzero(np.diff(ordered.section)) + 1):
            threshold = max(1e-10, 1e-6 * max(abs(section.w)))
            opposite += int(np.any(section.w > threshold) and np.any(section.w < -threshold))
            active = abs(section.w) > threshold
            multiple += int(sum(active & ~np.r_[False, active[:-1]]) > 1)
        item = dict(step=step, time=float(row.time), max_delta_phi=float(max(delta)),
                    min_delta_phi=float(min(delta)), max_coefficient_increment=float(max(b.B - seed.B)),
                    regions=regional, sampled_extents=sampled_extents(p),
                    added_fracture=float(row.fracture - initial_energy), fracture_partition_error=partition_error,
                    max_slip=float(max(abs(s.slip))), max_V=float(max(abs(s.V))),
                    age_range=[float(min(s.age)), float(max(s.age))],
                    cancellation_fraction=1 - float(sum(s.ds * abs(s.V))) / abs_rate if abs_rate else 0.,
                    opposite_sign_sections=opposite, multiple_band_sections=multiple,
                    positive_local_work=float(sum(np.maximum(local_work, 0))),
                    negative_local_work=float(sum(np.minimum(local_work, 0))), power_identity_error=power_defect)
        boundary = np.minimum.reduce([p.x, 8 - p.x, p.y, 4 - p.y]) <= .25
        remote = (abs(p.x - 4) > 3) | (abs(p.y - 2) > 1)
        item['max_boundary_phase_increment'] = float(max(delta[boundary]))
        item['max_remote_phase_increment'] = float(max(delta[remote]))
        for threshold in (1e-6, 1e-4, .001):
            for name, increment in [('any', max(delta))] + [(k, v['max_delta_phi']) for k, v in regional.items()]:
                if increment > threshold:
                    result['onset'].setdefault(f'{name}>{threshold:g}', float(row.time))
        if abs(row.time - .04) < 1e-10 or step == int(rows.step[-1]):
            item['polynomial_sections'] = polynomial_sections(p)
            saved[float(row.time)] = p
            if step == int(rows.step[-1]):
                np.savez_compressed(OUT / f'{label}-final.npz', x=p.x, y=p.y, phi=p.phi,
                                    initial_phi=initial.phi, weight=p.weight,
                                    row_x=b.x, row_y=b.y, row_fracture_change=fracture)
        result['steps'].append(item)
    return result, rows, saved


def plots(cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for label, (r, rows, saved) in cases.items():
        dt = rows.dt[0]
        tag = f'dt={dt:g} s'
        steps = r['steps']
        axes[0, 0].plot(rows.time, [s['max_delta_phi'] for s in steps], label=tag)
        axes[0, 1].plot(rows.time, [s['added_fracture'] for s in steps], label=tag)
        error = rows.Wext - rows.energy_change - rows.Db - rows.Dc - rows.Df
        line, = axes[0, 2].plot(rows.time, np.cumsum(error), label=tag + ' signed')
        axes[0, 2].plot(rows.time, np.cumsum(abs(error)), '--', color=line.get_color(), label=tag + ' abs')
        axes[1, 0].plot(rows.time, [s['max_slip'] for s in steps], label=tag)
        axes[1, 1].plot(rows.time, [s['regions']['axial_intact']['max_delta_phi'] for s in steps], label=tag + ' axial')
        axes[1, 1].plot(rows.time, [s['regions']['transverse_intact']['max_delta_phi'] for s in steps], '--', label=tag + ' transverse')
    final_case = list(cases.values())[-1][0]
    names = list(final_case['steps'][-1]['regions'])
    x = np.arange(len(names))
    for i, (r, rows, _) in enumerate(cases.values()):
        axes[1, 2].bar(x + i * .35, [r['steps'][-1]['regions'][n]['added_fracture'] for n in names],
                       width=.35, label=f'dt={rows.dt[0]:g}')
    axes[1, 2].set_xticks(x + .175, ['axial\nintact', 'transverse\nintact', 'old tip\nband', 'old central\nband'])
    for ax, title, unit in zip(axes.flat,
                              ['Maximum phase increment', 'Added fracture energy', 'Physical work error',
                               'Maximum cumulative section slip', 'Initially intact regions', 'SIPG energy partition'],
                              ['phase', 'J/m', 'J/m', 'm', 'phase increment', 'J/m']):
        ax.set(title=title, ylabel=unit)
        if ax is not axes[1, 2]:
            ax.set_xlabel('time (s)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle('Freely evolving synthetic specimen — BE, h=0.125 m, phase active from t=0')
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT / f'history.{suffix}', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(len(cases), 3, figsize=(12, 3.5 * len(cases)), squeeze=False, layout='constrained')
    delta_max = max(r['steps'][-1]['max_delta_phi'] for r, _, _ in cases.values())
    for i, (r, rows, saved) in enumerate(cases.values()):
        final = saved[max(saved)]
        initial = saved[0]
        for j, (values, title) in enumerate([(initial.phi, 'equilibrated seed'), (final.phi, 'phase at 0.06 s'),
                                            (final.phi - initial.phi, 'phase increment')]):
            order = np.lexsort((final.x, final.y))
            x, y = np.unique(final.x), np.unique(final.y)
            z = values[order].reshape(len(y), len(x))
            im = axes[i, j].pcolormesh(x, y, z, shading='nearest', cmap='magma' if j == 2 else 'viridis',
                                       vmin=0, vmax=delta_max if j == 2 else 1, rasterized=True)
            axes[i, j].set(title=f'dt={rows.dt[0]:g}: {title}', xlabel='x (m)', ylabel='y (m)', aspect='equal')
            axes[i, j].axvline(2, color='white', ls=':', lw=.7)
            axes[i, j].axvline(6, color='white', ls=':', lw=.7)
            fig.colorbar(im, ax=axes[i, j], shrink=.7)
    fig.suptitle('Gauss samples of native DGQ2 — no smoothing; thresholds are not sharp cracks')
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT / f'phase.{suffix}', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('labels', nargs='+')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {}
    for label in args.labels:
        cases[label] = analyze(label)
        (OUT / f'{label}.json').write_text(json.dumps(cases[label][0], indent=2, allow_nan=False) + '\n')
        print(label, json.dumps({k: cases[label][0][k] for k in ('onset', 'energy_full', 'energy_window', 'gates')}), flush=True)
    comparison = {}
    if len(cases) == 2:
        (label_a, (a, ra, fa)), (label_b, (b, rb, fb)) = cases.items()
        for name in fa[0].dtype.names:
            if fa[0][name].dtype.kind == 'f':
                identical = np.array_equal(fa[0][name], fb[0][name], equal_nan=True)
            else:
                identical = np.array_equal(fa[0][name], fb[0][name])
            assert identical, f'Initial quadrature field {name} differs.'
        sa = balance(RUNS / label_a / 'output', 'tip-seed-fracture-r*.csv')
        sb = balance(RUNS / label_b / 'output', 'tip-seed-fracture-r*.csv')
        assert np.array_equal(sa, sb), 'Initial Bernstein fields and fracture gradients must be identical.'
        comparison['identical_initial_state'] = True
        for t in (.04, .06):
            ta = min(fa, key=lambda k: abs(k-t))
            tb = min(fb, key=lambda k: abs(k-t))
            comparison[str(t)] = compare(fa[ta], fb[tb])
            coarse, fine = fa[ta], fb[tb]
            diff = coarse.phi - fine.phi
            increment_norm = np.sqrt(sum(fine.weight * (fine.phi - fb[0].phi)**2))
            comparison[str(t)]['phase_increment_relative_L2'] = float(
                np.sqrt(sum(fine.weight * diff**2)) / increment_norm) if increment_norm else None
            comparison[str(t)]['phase_regions'] = {
                name: dict(max_abs=float(max(abs(diff[mask]))),
                           L2=float(np.sqrt(sum(fine.weight[mask] * diff[mask]**2))))
                for name, mask in regions(fine.x, fine.y).items()}
            for label, rows in ((label_a, ra), (label_b, rb)):
                step = int(rows.step[np.argmin(abs(rows.time - t))])
                s = read(RUNS / label / 'output' / f'tip-sections-{step:04d}.csv')
                if label == label_a:
                    first = s
                else:
                    for key in ('slip', 'age', 'V'):
                        diff = first[key] - s[key]
                        norm = np.sqrt(sum(s.ds * s[key]**2))
                        error = np.sqrt(sum(s.ds * diff**2))
                        comparison[str(t)][key] = dict(max_abs=float(max(abs(diff))), L2=float(error),
                                                       relative_L2=float(error / norm) if norm else None)
        comparison['note'] = 'Two levels measure sensitivity; they do not establish a convergence order. BE V is an endpoint-rate observation.'
    (OUT / 'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False) + '\n')
    plots(cases)


if __name__ == '__main__':
    main()
