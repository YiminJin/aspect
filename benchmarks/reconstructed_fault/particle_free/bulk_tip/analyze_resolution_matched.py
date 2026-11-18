"""Native rectangular-cell observations for the peak-0.60 resolution screen."""
import argparse
import gzip
import json
import os
from pathlib import Path
import re

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / 'build-particle-free/bulk-tip/runs'
OUT = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/resolution_matched_tip'


def read(path):
    if not path.exists():
        path = path.with_suffix(path.suffix + '.gz')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        names = stream.readline().strip().split(',')
        dtype = [(n, 'U32' if n == 'cell' else 'f8') for n in names]
        return np.atleast_1d(np.loadtxt(stream, delimiter=',', dtype=dtype)).view(np.recarray)


def gather(directory, stem, keys):
    files = sorted(set(p.with_suffix('') if p.suffix == '.gz' else p for p in directory.glob(stem + '*')))
    arrays = [read(p) for p in files if p.name.endswith('.csv')]
    values = np.concatenate(arrays).view(np.recarray)
    return values[np.lexsort(tuple(values[k] for k in reversed(keys)))]


def masks(x, y):
    axial = abs(x - 4)
    normal = abs(y - 2)
    return dict(central_band=(axial < 1.5) & (normal < .25),
                central_flanks=(axial < 1.5) & (normal >= .25) & (normal < .5),
                old_tip_taper=(axial >= 1.5) & (axial < 2) & (normal < .25),
                outer_tip_taper=(axial >= 1.875) & (axial < 2) & (normal < .25),
                axial_ahead=(axial >= 2) & (axial < 2.5) & (normal < .25),
                remote=(axial > 3) | (normal > 1),
                boundary=np.minimum.reduce([x, 8-x, y, 4-y]) <= .25)


class RectBernstein:
    """Native B_(ix+3*iy) coefficients, with separate physical hx and hy."""
    def __init__(self, rows, hx, hy, coefficients=None):
        self.hx, self.hy = hx, hy
        self.nx, self.ny = round(8/hx), round(4/hy)
        assert np.array_equal(rows.i, np.tile(np.arange(9), self.nx*self.ny))
        self.cx, self.cy = rows.x[::9], rows.y[::9]
        ix = np.rint(self.cx/hx-.5).astype(int)
        iy = np.rint(self.cy/hy-.5).astype(int)
        assert max(abs(self.cx-(ix+.5)*hx)) < 1e-12
        assert max(abs(self.cy-(iy+.5)*hy)) < 1e-12
        assert len(set(zip(ix, iy))) == self.nx*self.ny
        self.coefficients = np.empty((self.ny, self.nx, 3, 3))
        self.coefficients[iy, ix] = (rows.B if coefficients is None else coefficients).reshape(-1, 3, 3)

    def evaluate(self, x, y):
        x, y = np.broadcast_arrays(x, y)
        ix = np.clip(np.floor(x/self.hx).astype(int), 0, self.nx-1)
        iy = np.clip(np.floor(y/self.hy).astype(int), 0, self.ny-1)
        u, v = x/self.hx-ix, y/self.hy-iy
        bx = np.stack([(1-u)**2, 2*u*(1-u), u*u], axis=-1)
        by = np.stack([(1-v)**2, 2*v*(1-v), v*v], axis=-1)
        return np.einsum('...i,...ij,...j->...', by, self.coefficients[iy, ix], bx)

    def width(self):
        def compliance(phi):
            return 4*phi*(1+phi)/(1-phi)**2
        results = []
        for side in (-1, 1):
            x = 4 + side*1e-10
            peak = max(float(self.evaluate(x, 2-1e-12)), float(self.evaluate(x, 2+1e-12)))
            threshold = compliance(peak)/2
            crossings = []
            for direction in (-1, 1):
                # Search outward through native cell interiors; jumps are not smoothed.
                distance = np.arange(round(2/self.hy)*100+1)*self.hy/100
                y = 2 + direction*np.maximum(distance, 1e-12)
                values = compliance(self.evaluate(x, y)) - threshold
                j = np.flatnonzero(values < 0)[0]
                lo, hi = distance[j-1], distance[j]
                for _ in range(48):
                    mid = (lo+hi)/2
                    if compliance(float(self.evaluate(x, 2+direction*mid))) >= threshold:
                        lo = mid
                    else:
                        hi = mid
                crossings.append((lo+hi)/2)
            integrals = []
            for order in (24, 48):
                z, w = np.polynomial.legendre.leggauss(order)
                y = (np.arange(self.ny)[:, None]+.5+.5*z)*self.hy
                integrals.append(float(np.sum(compliance(self.evaluate(x, y))*w)*self.hy/2))
            results.append(dict(x_trace=side, peak=peak, W=sum(crossings), W_over_hn=sum(crossings)/self.hy,
                                compliance_integral=integrals[1], integration_difference=abs(integrals[1]-integrals[0])))
        return results


def reduce(label):
    run = RUNS / label
    directory = run / 'output'
    command = json.loads((run / 'command.json').read_text())
    text = (run / 'input.prm').read_text()
    def param(name):
        return float(re.findall(r'^\s*set '+re.escape(name)+r'\s*=\s*([^\n#]+)', text, re.M)[-1])
    hx, hy = 8/param('X repetitions'), 4/param('Y repetitions')
    assert param('Initial global refinement') == 0 and hx == .0625
    assert param('X extent') == 8 and param('Y extent') == 4
    K = param('Critical energy release rates')/((8/3)*param('Length scale'))
    accepted = read(directory / 'bulk-accepted.csv')
    native1 = gather(directory, 'tip-balance-0001-r', ['cell', 'i'])
    seed = gather(directory, 'tip-seed-fracture-r', ['cell', 'i'])
    assert np.array_equal(seed.cell, native1.cell) and np.array_equal(seed.i, native1.i)
    initial = RectBernstein(native1, hx, hy, seed.B)
    p0 = gather(directory, 'tip-points-0000-r', ['cell', 'q'])
    reconstruction = float(max(abs(initial.evaluate(p0.x, p0.y)-p0.phi)))
    assert reconstruction < 2e-12
    # Validate geometry and measures independently of old square-cell reducers.
    z, w = np.polynomial.legendre.leggauss(6)
    q = p0.q.astype(int)
    cx = np.repeat(native1.x[::9], 36); cy = np.repeat(native1.y[::9], 36)
    assert np.array_equal(p0.cell, np.repeat(native1.cell[::9], 36))
    quadrature_error = max(max(abs(p0.x-(cx+.5*hx*z[q % 6]))), max(abs(p0.y-(cy+.5*hy*z[q // 6]))))
    weight_error = float(max(abs(p0.weight-hx*hy*.25*w[q % 6]*w[q // 6])))
    assert quadrature_error < 2e-12 and weight_error < 1e-14
    assert abs(sum(p0.weight)-32) < 1e-9
    assert max(abs(native1.mass-hx*hy/9)) < 1e-14
    result = dict(label=label, command=command, hx=hx, hy=hy, cells=round(32/(hx*hy)),
                  seed_width=initial.width(), native_polynomial_defect=reconstruction,
                  quadrature_coordinate_defect=float(quadrature_error), quadrature_weight_defect=weight_error,
                  volume=float(sum(p0.weight)), K=K, steps=[])
    clocks = read(directory / 'tip-time-location.csv')
    for r in accepted:
        step = int(r.step)
        native = gather(directory, f'tip-balance-{step:04d}-r', ['cell', 'i'])
        points = gather(directory, f'tip-points-{step:04d}-r', ['cell', 'q'])
        sections = read(directory / f'tip-sections-{step:04d}.csv')
        assert len(sections) == 128*6 and abs(sum(sections.ds)-8) < 1e-12
        assert np.array_equal(points.cell, p0.cell) and np.array_equal(points.q, p0.q)
        normal_measure = np.bincount(points.section.astype(int), weights=points.weight/sections.ds[points.section.astype(int)])
        assert max(abs(normal_measure-4)) < 2e-11
        poly = RectBernstein(native, hx, hy)
        phase_error = float(max(abs(native.B-seed.B)))
        fixed = command['arguments']['fixed_phase']
        if fixed:
            assert phase_error == 0
            assert max(abs(native.stage_fracture_gradient-native.fracture_gradient)) < 1e-10
        surplus = (native.driving_load-native.stage_fracture_gradient)/(K*native.mass)
        assert max(abs(native.residual+native.driving_load-native.stage_fracture_gradient)) < 1e-10
        nm, pm = masks(native.x, native.y), masks(points.x, points.y)
        regions = {}
        for name, mask in nm.items():
            admissible = mask & (native.B < 1-1e-12)
            ids = np.flatnonzero(admissible)
            i = ids[np.argmax(surplus[ids])]
            pv = points[pm[name]]
            delta = points.phi[pm[name]]-p0.phi[pm[name]]
            regions[name] = dict(max_surplus=float(surplus[i]), location=[float(native.x[i]), float(native.y[i])],
                                 cell=str(native.cell[i]), local_index=int(native.i[i]), B=float(native.B[i]),
                                 positive_rows=int(sum(surplus[admissible]>1e-7)),
                                 integrated_positive_surplus=float(sum(np.maximum(surplus[admissible], 0)*K*native.mass[admissible])),
                                 max_Y_over_K=float(max(pv.Y)/K), max_abs_shear=float(max(abs(pv.tauxy))),
                                 max_delta_phi=float(max(delta)), phase_increment_integral=float(sum(pv.weight*delta)),
                                 positive_w=float(sum(pv.weight*np.maximum(pv.w, 0))),
                                 negative_w=float(sum(pv.weight*np.minimum(pv.w, 0))))
            # Exact quadratic SIPG energy increment, partitioned by native rows.
            regions[name]['added_fracture'] = float(sum(.5*(native.B[mask]-seed.B[mask])*(native.fracture_gradient[mask]+seed.fracture_gradient[mask])))
        index = points.section.astype(int)
        local_power = float(sum(points.weight*points.w*sections.force[index]))
        section_power = float(sum(sections.ds*sections.V*sections.force))
        assert abs(local_power-section_power) < 1e-9
        integrated_w = np.bincount(index, weights=points.weight*points.w)/sections.ds
        assert max(abs(integrated_w-sections.V)) < 1e-12
        absolute = float(sum(sections.ds*sections.integral_abs_w))
        order = np.lexsort((points.y, points.section))
        rates = points.w[order].reshape(len(sections), -1)
        threshold = np.maximum(1e-10, 1e-6*np.max(abs(rates), axis=1))[:, None]
        active = abs(rates) > threshold
        starts = active & ~np.column_stack([np.zeros(len(sections), dtype=bool), active[:, :-1]])
        opposite = np.any(rates > threshold, axis=1) & np.any(rates < -threshold, axis=1)
        row = {n:float(r[n]) for n in r.dtype.names}
        result['steps'].append(dict(step=step, time=float(r.time), stage_time=float(clocks.stage_time[clocks.step==step][0]),
                                   nominal_stage_shear=float(1e6*-np.expm1(-clocks.stage_time[clocks.step==step][0]/100)),
                                   regions=regions, width=poly.width(), phase_coefficient_change=phase_error,
                                   integrated_V=float(sum(sections.ds*sections.V)), integrated_slip=float(sum(sections.ds*sections.slip)),
                                   max_V=float(max(abs(sections.V))), cancellation=1-float(sum(sections.ds*abs(sections.V)))/absolute if absolute else 0.,
                                   positive_w_integral=float(sum(sections.ds*sections.positive_w)),
                                   negative_w_integral=float(sum(sections.ds*sections.negative_w)),
                                   opposite_sign_sections=int(sum(opposite)), multiple_band_sections=int(sum(np.sum(starts, axis=1)>1)),
                                   power_adjoint_error=local_power-section_power, normal_measure_error=float(max(abs(normal_measure-4))),
                                   accepted=row, physical_work_error=float(r.Wext-r.energy_change-r.Db-r.Dc-r.Df)))
    result['cost'] = dict(process_seconds=command.get('seconds'), profiles=[{n:float(r[n]) for n in r.dtype.names} for r in read(directory/'tip-profile.csv')])
    (OUT/(label+'-analysis.json')).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(label=label, seed=result['seed_width'], last=result['steps'][-1], cost=result['cost']), indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    args = parser.parse_args()
    reduce(args.label)
