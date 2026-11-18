"""Compare same-peak screens, including work on identical nested phase variations."""
import json
import os
from analyze_resolution_matched import OUT, RUNS, read, gather, RectBernstein, masks
import numpy as np


def cell_grid(rows, field, hx, hy):
    nx, ny = round(8/hx), round(4/hy)
    grid = np.empty((ny, nx, 3, 3))
    ix = np.rint(rows.x[::9]/hx-.5).astype(int)
    iy = np.rint(rows.y[::9]/hy-.5).astype(int)
    grid[iy, ix] = rows[field].reshape(-1, 3, 3)
    return grid


def adjoint_normal_bisection(fine):
    """T^T on forces, where T exactly restricts one Q2 Bernstein polynomial."""
    low, high = fine[::2], fine[1::2]
    result = np.empty_like(low)
    result[:, :, 0] = low[:, :, 0]+.5*low[:, :, 1]+.25*low[:, :, 2]+.25*high[:, :, 0]
    result[:, :, 1] = .5*low[:, :, 1]+.5*low[:, :, 2]+.5*high[:, :, 0]+.5*high[:, :, 1]
    result[:, :, 2] = .25*low[:, :, 2]+.25*high[:, :, 0]+.5*high[:, :, 1]+high[:, :, 2]
    return result


def restrict_normal_bisection(coarse):
    fine = np.empty((2*coarse.shape[0], *coarse.shape[1:]))
    middle = .25*coarse[:, :, 0]+.5*coarse[:, :, 1]+.25*coarse[:, :, 2]
    fine[::2, :, 0] = coarse[:, :, 0]
    fine[::2, :, 1] = .5*(coarse[:, :, 0]+coarse[:, :, 1])
    fine[::2, :, 2] = middle
    fine[1::2, :, 0] = middle
    fine[1::2, :, 1] = .5*(coarse[:, :, 1]+coarse[:, :, 2])
    fine[1::2, :, 2] = coarse[:, :, 2]
    return fine


def main():
    labels = ['resolution60-coarse-screen', 'resolution60-fine-screen']
    summaries = [json.loads((OUT/(label+'-analysis.json')).read_text()) for label in labels]
    directories = [RUNS/label/'output' for label in labels]
    c, f = summaries
    assert c['hx'] == f['hx'] and c['hy'] == 2*f['hy']
    # Independent adjoint check on mixed-sign forces and positive trial increments.
    rng = np.random.default_rng(619)
    trial = rng.random((4, 3, 3, 3)); force = rng.normal(size=(8, 3, 3, 3))
    defect = abs(np.sum(restrict_normal_bisection(trial)*force)-np.sum(trial*adjoint_normal_bisection(force)))
    assert defect < 1e-12
    native = [gather(d, 'tip-balance-0001-r', ['cell', 'i']) for d in directories]
    seeds = [gather(d, 'tip-seed-fracture-r', ['cell', 'i']) for d in directories]
    seed_grids = []
    for n, seed, s in zip(native, seeds, summaries):
        assert np.array_equal(n.cell, seed.cell)
        n = n.copy(); n.B = seed.B
        seed_grids.append(cell_grid(n, 'B', s['hx'], s['hy']))
    difference = restrict_normal_bisection(seed_grids[0])-seed_grids[1]
    mass = np.array([[1/5, 1/10, 1/30], [1/10, 2/15, 1/10], [1/30, 1/10, 1/5]])
    def norm2(x):
        return float(np.einsum('abij,ik,jl,abkl->', x, mass, mass, x)*f['hx']*f['hy'])
    result = dict(seed_coefficient_max_difference=float(np.max(abs(difference))),
                  seed_L2_difference=np.sqrt(norm2(difference)),
                  seed_relative_L2_difference=np.sqrt(norm2(difference)/norm2(seed_grids[1])),
                  normal_bisection_adjoint_defect=float(defect), comparisons=[])
    fine_steps = {s['step']:s for s in f['steps']}
    for cs in c['steps']:
        if cs['step'] not in fine_steps:
            continue
        fs = fine_steps[cs['step']]
        assert cs['time'] == fs['time'] and cs['stage_time'] == fs['stage_time']
        step = cs['step']
        a, b = [gather(d, f'tip-balance-{step:04d}-r', ['cell', 'i']) for d in directories]
        fine_force = cell_grid(b, 'driving_load', f['hx'], f['hy'])-cell_grid(b, 'stage_fracture_gradient', f['hx'], f['hy'])
        coarse_force = cell_grid(a, 'driving_load', c['hx'], c['hy'])-cell_grid(a, 'stage_fracture_gradient', c['hx'], c['hy'])
        projected = adjoint_normal_bisection(fine_force)
        masses = cell_grid(a, 'mass', c['hx'], c['hy'])
        ix = np.rint(a.x/c['hx']-.5).astype(int); iy = np.rint(a.y/c['hy']-.5).astype(int)
        j = a.i.astype(int)
        common_c = (coarse_force/(c['K']*masses))[iy, ix, j//3, j%3]
        common_f = (projected/(c['K']*masses))[iy, ix, j//3, j%3]
        regional = {}
        for name, mask in masks(a.x, a.y).items():
            regional[name] = dict(coarse_max=float(max(common_c[mask])), fine_on_same_variation_max=float(max(common_f[mask])),
                                  maximum_row_difference=float(max(abs(common_c[mask]-common_f[mask]))))
        sa, sb = [read(d/f'tip-sections-{step:04d}.csv') for d in directories]
        assert np.array_equal(sa.s, sb.s) and np.array_equal(sa.ds, sb.ds)
        sections = {}
        for field in ['V', 'slip', 'age', 'force']:
            diff = sa[field]-sb[field]
            denominator = sum(sb.ds*sb[field]**2)
            sections[field] = dict(maximum=float(max(abs(diff))), relative_L2=float(np.sqrt(sum(sb.ds*diff**2)/denominator)) if denominator else None)
        result['comparisons'].append(dict(time=cs['time'], stage_time=cs['stage_time'],
                                          common_variations=regional, sections=sections,
                                          relative_integrated_V=(cs['integrated_V']-fs['integrated_V'])/fs['integrated_V'] if fs['integrated_V'] else None))
    (OUT/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/aspect-resolution-matched-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    names = ['central_band', 'old_tip_taper', 'outer_tip_taper', 'axial_ahead', 'remote']
    for index, name in enumerate(names):
        for s, style in zip(summaries, ['--', '-']):
            axes[0, 0].plot([x['stage_time'] for x in s['steps']], [x['regions'][name]['max_surplus'] for x in s['steps']],
                            style, color=f'C{index}', label=name.replace('_', ' ') if s is f else None)
        axes[0, 1].plot([x['stage_time'] for x in result['comparisons']],
                        [x['common_variations'][name]['fine_on_same_variation_max'] for x in result['comparisons']], color=f'C{index}', label=name.replace('_', ' '))
        axes[0, 1].plot([x['stage_time'] for x in result['comparisons']],
                        [x['common_variations'][name]['coarse_max'] for x in result['comparisons']], '--', color=f'C{index}')
    for s, label, style in zip(summaries, ['hy=.03125 m', 'hy=.015625 m'], ['.--', '.-']):
        axes[1, 0].plot([x['stage_time'] for x in s['steps']], [x['integrated_V'] for x in s['steps']], style, label=label)
        axes[1, 1].plot([x['time'] for x in s['steps']], [x['width'][0]['W_over_hn'] for x in s['steps']], style, label=label)
    for ax in axes[0]:
        ax.axhline(0, color='k', linewidth=.7)
        ax.legend(fontsize=7)
    for ax in axes[1]:
        ax.legend(fontsize=8)
    for ax, title, ylabel in zip(axes.flat,
                                ['Native row driving surplus', 'Driving on identical coarse phase variations', 'Integrated signed fault slip rate', 'Measured mechanical pulse resolution'],
                                ['max (D−F)/(K mass)', 'max (D−F)/(K mass)', 'integral V ds (m²/s)', 'W / hy']):
        ax.set(title=title, ylabel=ylabel, xlabel='Midpoint stage time (s)' if ax != axes[1, 1] else 'Endpoint time (s)')
        ax.grid(alpha=.2)
    fig.suptitle('Peak-0.60 fixed-phase screens: dashed coarse, solid fine; same physical input')
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/('screen-comparison.'+ext), dpi=170)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
