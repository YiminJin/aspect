"""Signed production-work split and cell-exact Q2 velocity/strip diagnostics.

Limited to the saved straight Cartesian 2D frozen BP3 probes. No simulation,
point averaging, smoothing, or new physical profile is introduced.
"""
import csv
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.spatial import cKDTree
from analyze_mechanical_modes import table

HERE = Path(__file__).resolve().parent
OUT = HERE/'first_long_run/mechanical-velocity-decomposition'
OLD = HERE/'first_long_run/mechanical-bulk-refinement'
WIDTH = 1200.
SURFACE = table(OLD/'capture/surface.csv')
TOP = np.array([SURFACE['x'][-1], SURFACE['y'][-1]])
T = TOP-np.array([SURFACE['x'][0], SURFACE['y'][0]])
T /= np.linalg.norm(T)
N = np.array([-T[1], T[0]])


class Field:
    def __init__(self, rows):
        self.ids = list(rows['cell'])
        assert len(set(self.ids)) == len(self.ids), 'Duplicate MPI cell ownership'
        self.index = {name: i for i, name in enumerate(self.ids)}
        self.xy = np.column_stack([rows['x'], rows['y']])
        self.h = rows['h']
        self.values = np.stack([np.column_stack([rows[f'ux{i}'], rows[f'uy{i}']]) for i in range(9)], axis=1).reshape(-1, 3, 3, 2)
        self.tree = cKDTree(self.xy+self.h[:, None]/2)

    def evaluate(self, points, indices=None):
        points = np.atleast_2d(points)
        if indices is None:
            _, candidates = self.tree.query(points, k=16)
            delta = points[:, None, :]-self.xy[candidates]
            inside = np.all((delta >= -1e-8) & (delta <= self.h[candidates, None]+1e-8), axis=2)
            assert inside.any(axis=1).all(), 'Velocity export does not cover requested points'
            indices = candidates[np.arange(len(points)), inside.argmax(axis=1)]
        ref = (points-self.xy[indices])/self.h[indices, None]
        def basis(z):
            return np.column_stack([2*(z-.5)*(z-1), 4*z*(1-z), 2*z*(z-.5)])
        def derivative(z):
            return np.column_stack([4*z-3, 4-8*z, 4*z-1])
        lx, ly = basis(ref[:, 0]), basis(ref[:, 1])
        u = np.einsum('qj,qi,qjid->qd', ly, lx, self.values[indices])
        ux = np.einsum('qj,qi,qjid->qd', ly, derivative(ref[:, 0]), self.values[indices])/self.h[indices, None]
        uy = np.einsum('qj,qi,qjid->qd', derivative(ref[:, 1]), lx, self.values[indices])/self.h[indices, None]
        dn = ux*N[0]+uy*N[1]; ds = ux*T[0]+uy*T[1]
        return np.column_stack([u@T, u@N, dn@T, ds@N])

    def column(self, xd):
        center = TOP-T*xd
        intervals = (self.xy-center)/N
        other = (self.xy+self.h[:, None]-center)/N
        lo = np.maximum(-WIDTH, np.minimum(intervals, other).max(axis=1))
        hi = np.minimum(WIDTH, np.maximum(intervals, other).min(axis=1))
        ids = np.flatnonzero(hi-lo > 1e-8)
        ids = ids[np.argsort(lo[ids])]
        assert abs(lo[ids[0]]+WIDTH) < 1e-8 and abs(hi[ids[-1]]-WIDTH) < 1e-8
        assert np.max(abs(lo[ids[1:]]-hi[ids[:-1]])) < 2e-8, 'Ray intervals overlap or leave gaps'
        # Q2 restricted to an inclined line is degree four. Three-point Gauss
        # integrates the velocity and both first gradients exactly on each cell.
        z, w = leggauss(3)
        half = (hi[ids]-lo[ids])/2
        normal = (lo[ids]+hi[ids])[:, None]/2+half[:, None]*z
        p = center+normal.reshape(-1, 1)*N
        u = self.evaluate(p, np.repeat(ids, 3)).reshape(-1, 3, 4)
        integrals = np.einsum('i,j,ijk->k', half, w, u)
        ends = self.evaluate(center+np.array([-WIDTH, WIDTH])[:, None]*N)
        difference = ends[1, 0]-ends[0, 0]
        return np.array([difference, integrals[1], integrals[2], integrals[3]])


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scale = json.loads((OLD/'comparison.json').read_text())['target_scale_factor']
    summary, profiles, field_map = {}, [], {}
    for mesh in ('coarse', 'fine'):
        run = OUT/mesh
        assert 'MECHANICAL MODES VERIFIED' in (run/'run.log').read_text()
        parts, modes = table(run/'mechanical_shear_parts.csv'), table(run/'mechanical_modes.csv')
        old = table((HERE/'first_long_run/mechanical-modes-preflight-fixed' if mesh == 'coarse' else OLD/'fine')/'mechanical_modes.csv')
        rows = np.concatenate([np.atleast_1d(table(p)) for p in sorted(run.glob('mechanical_velocity_cells_rank*.csv'))])
        raw = np.concatenate([np.atleast_1d(table(p)) for p in sorted(run.glob('mechanical_mode_qp_rank*.csv'))])
        results = []
        for i, part in enumerate(parts):
            mode = part['mode']; assert mode == old['mode'][i]
            agreement = float(modes['mechanical_shear'][i]/old['mechanical_shear'][i]-1)
            assert abs(agreement) < 1e-10
            field = Field(rows[rows['mode'] == mode]); field_map[mesh, mode] = field
            selected = raw[raw['mode'] == mode]
            covered = np.array([name in field.index for name in selected['cell']])
            selected = selected[covered]
            points = np.column_stack([selected['x'], selected['y']])
            sampled = field.evaluate(points, np.array([field.index[name] for name in selected['cell']]))
            errors = {}
            for j, key in enumerate(('u_s', 'u_n', 'd_us_dn', 'd_un_ds')):
                error = np.linalg.norm(sampled[:, j]-selected[key])/np.linalg.norm(selected[key])
                errors[key] = float(error); assert error < 1e-10
            xd = np.arange(15000., 18000.1, 10.)
            column = np.array([field.column(s) for s in xd])
            derivatives = []
            for ds in (.5, .25):
                # s increases up dip, whereas the plotted x_d increases down dip.
                derivative = np.array([(field.column(s-ds)[1]-field.column(s+ds)[1])/(2*ds) for s in xd])
                derivatives.append(derivative)
            closure = np.linalg.norm(column[:, 0]-column[:, 2])/max(np.linalg.norm(column[:, 2]), 1e-30)
            # For the alternating mode the boundary difference is almost zero.
            # Judge cancellation error against the input velocity, not that
            # near-zero difference; retain both diagnostics in the report.
            closure_input = np.max(abs(column[:, 0]-column[:, 2]))/1e-12
            assert closure_input < 1e-10
            fd_errors = [float(np.linalg.norm(d-column[:, 3])/np.linalg.norm(column[:, 3])) for d in derivatives]
            assert fd_errors[1] < 1e-3
            nodal_xd = (TOP-np.column_stack([SURFACE['x'], SURFACE['y']]))@T
            taper = np.where(abs(nodal_xd-16500) < 1500,
                             np.cos(np.pi*(nodal_xd-16500)/3000)**2, 0.)
            if mode != 'broad_3000m': taper *= np.cos(2*np.pi*(nodal_xd-16500)/(600 if mode == 'six_nodes_600m' else 200))
            order = np.argsort(nodal_xd)
            amplitude = np.interp(xd, nodal_xd[order], 1e-12*taper[order])
            for j, s in enumerate(xd):
                profiles.append(dict(mesh=mesh, mode=mode, xd_m=float(s),
                                     input_Q1_dV=float(amplitude[j]),
                                     cross_band_us_difference=float(column[j, 0]),
                                     integral_un=float(column[j, 1]), integral_d_us_dn=float(column[j, 2]),
                                     integral_d_un_ds=float(column[j, 3]),
                                     d_ds_integral_un_FD_half_m=float(derivatives[0][j]),
                                     d_ds_integral_un_FD_quarter_m=float(derivatives[1][j]),
                                     integrated_bulk_shear=float(column[j, 0]+column[j, 3])))
            def projection(value): return float(np.trapezoid(amplitude*value, xd)/np.trapezoid(amplitude*amplitude, xd))
            results.append(dict(mode=str(mode), k_d_us_dn=float(part['signed_d_us_dn']*scale),
                                k_d_un_ds=float(part['signed_d_un_ds']*scale),
                                k_relax=float(part['bulk_relaxation']*scale), sum_relative_error=float(part['closure_relative']),
                                repeat_shear_relative_change=agreement, Q2_native_QP_relative_errors=errors,
                                column_fundamental_theorem_relative_error=float(closure),
                                column_fundamental_theorem_max_error_over_input=float(closure_input),
                                column_derivative_FD_relative_errors=fd_errors,
                                unweighted_column_modal_projections=dict(cross_band=projection(column[:, 0]),
                                                                         along_normal=projection(column[:, 3]),
                                                                         sum=projection(column[:, 0]+column[:, 3])),
                                center_column=dict(xd_m=16500, cross_band=float(column[150, 0]),
                                                   along_normal=float(column[150, 3]), sum=float(column[150, 0]+column[150, 3]))))
        summary[mesh] = results
    with (OUT/'normal_columns.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(profiles[0]));writer.writeheader();writer.writerows(profiles)
    report = dict(coordinate_convention=dict(tangent=T.tolist(), normal=N.tolist(), s='up dip; d/ds = -d/dxd', normal_window_m=[-WIDTH, WIDTH]),
                  coefficient_units='Pa/(m/s), common step-11 kappa; velocities are independent of uniform kappa scaling',
                  signed_production_work=summary,
                  execution={mesh: json.loads((OUT/mesh/'execution.json').read_text()) for mesh in ('coarse', 'fine')})
    (OUT/'decomposition.json').write_text(json.dumps(report, indent=2)+'\n')
    # Three-mode columns show the change in mechanism without equating a
    # diffuse cross-band velocity difference to a sharp-interface slip rate.
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for row, mesh in enumerate(('coarse', 'fine')):
        for col, mode in enumerate(('broad_3000m', 'six_nodes_600m', 'alternating_200m')):
            p = [r for r in profiles if r['mesh'] == mesh and r['mode'] == mode]
            x = np.array([r['xd_m'] for r in p])/1000
            for key, label in (('cross_band_us_difference', 'cross-band velocity difference'),
                               ('integral_d_un_ds', 'd/ds integral u_n dn'), ('integrated_bulk_shear', 'sum: integrated bulk shear')):
                axes[row, col].plot(x, [r[key]/1e-12 for r in p], label=label)
            axes[row, col].plot(x, [r['input_Q1_dV']/1e-12 for r in p], 'k:', linewidth=.8, label='input Q1 dV')
            axes[row, col].set_title(mesh+' / '+mode);axes[row, col].grid(alpha=.25)
            axes[row, col].set_xlabel('Down dip x_d (km)')
        axes[row, 0].set_ylabel('Velocity / 1e-12 m/s');axes[row, 0].legend(fontsize=7)
    fig.suptitle('Unweighted kinematic integrals over the same [-1200,1200] m normal window')
    fig.tight_layout();fig.savefig(OUT/'integrated_shear_components.png', dpi=180);plt.close(fig)

    normal = np.linspace(-WIDTH, WIDTH, 241); xd = np.linspace(16000, 17000, 201)
    X, Z = np.meshgrid(xd, normal)
    panels = {}
    for mesh in ('coarse', 'fine'):
        field = field_map[mesh, 'alternating_200m']
        panels[mesh] = field.evaluate(TOP-X.ravel()[:, None]*T+Z.ravel()[:, None]*N)[:, :2].reshape(len(normal), len(xd), 2)/1e-12
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)
    for col, label in enumerate(('u_s', 'u_n')):
        limit = max(np.max(abs(value[:, :, col])) for value in panels.values())
        for row, mesh in enumerate(('coarse', 'fine')):
            im = axes[row, col].pcolormesh(X/1000, Z, panels[mesh][:, :, col], cmap='RdBu_r', vmin=-limit, vmax=limit, shading='auto')
            axes[row, col].set_title(mesh+' '+label+' / 1e-12 m/s');axes[row, col].set_xlabel('Down dip x_d (km)')
            axes[row, col].set_ylabel('Normal coordinate n (m)')
        fig.colorbar(im, ax=axes[:, col], label='Velocity / 1e-12 m/s')
    fig.suptitle('200-m input: actual Q2 perturbation velocity (same scales across meshes)')
    fig.savefig(OUT/'alternating_velocity_band.png', dpi=180);plt.close(fig)

    samples = [];fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for mesh, style in (('coarse', '-'), ('fine', '--')):
        field = field_map[mesh, 'alternating_200m']
        for s in (16500., 16550.):
            u = field.evaluate(TOP-s*T+normal[:, None]*N)
            for component in (0, 1): axes[0, component].plot(normal, u[:, component]/1e-12, style, label=f'{mesh}, xd={s/1000:g} km')
            samples.extend(dict(mesh=mesh, kind='normal', xd_m=s, n_m=float(n), us=float(v[0]), un=float(v[1])) for n, v in zip(normal, u))
        for n in (0., 400., WIDTH):
            u = field.evaluate(TOP-xd[:, None]*T+n*N)
            for component in (0, 1): axes[1, component].plot(xd/1000, u[:, component]/1e-12, style, label=f'{mesh}, n={n:g} m')
            samples.extend(dict(mesh=mesh, kind='tangential', xd_m=float(s), n_m=n, us=float(v[0]), un=float(v[1])) for s, v in zip(xd, u))
    for col, label in enumerate(('u_s', 'u_n')):
        for row in (0, 1):
            axes[row, col].set_ylabel(label+' / 1e-12 m/s');axes[row, col].grid(alpha=.25);axes[row, col].legend(fontsize=7)
        axes[0, col].set_xlabel('Normal coordinate (m)');axes[1, col].set_xlabel('Down dip x_d (km)')
    fig.tight_layout();fig.savefig(OUT/'alternating_velocity_cuts.png', dpi=180);plt.close(fig)
    with (OUT/'alternating_velocity_cuts.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]));writer.writeheader();writer.writerows(samples)
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
