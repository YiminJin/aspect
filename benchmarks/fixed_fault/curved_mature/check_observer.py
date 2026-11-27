"""Exercise the installed mature observer with changed Maxwell data and a law without age."""
import argparse
from pathlib import Path

import numpy as np

from check_curve import association, compliance
from check_restart import OBSERVE, check_clocks, fault, read_rows, run


def check_observations(output, viscosity, modulus, initial_interval, has_age):
    check_clocks(output, [0, 1])
    # The initial parent is stress-free, and step zero publishes no stress.
    # Both offsets are therefore entirely due to the signed slip contribution.
    for step, dt in enumerate((initial_interval, .0125)):
        labels = read_rows(output/f'material-{step}-surface-0.csv')
        fields = read_rows(output/f'material-{step}-rank0.csv')
        public = output/f'fault-labels-{step:05}.txt'
        age_column = ['Theta_s'] if has_age else []
        expected = ['i', 'x', 'y', *age_column, 'Z_Pa_m', 'I_m', 'V_m_s', 'width_m', 'eta', 'slip']
        if not labels or list(labels[0]) != expected or len(fields) != 16**2*9:
            raise ValueError(f'{output}, step {step}: incomplete mature observer output')
        if ('age(s)' in public.read_text().splitlines()[0]) != has_age:
            raise ValueError(f'{public}: age columns differ between observers')
        # The public file and observer must describe the same IDs, rates and cumulative slip.
        public_values = np.loadtxt(public)
        np.testing.assert_array_equal(public_values[:, 0], np.arange(5))
        np.testing.assert_array_equal([int(r['i']) for r in labels], np.arange(5))
        np.testing.assert_allclose(public_values[:, -2:], [(float(r['slip']), float(r['V_m_s']))
                                                           for r in labels], rtol=1e-12, atol=0., equal_nan=False)
        vertices = np.array([(float(r['x']), float(r['y'])) for r in labels])
        xy = np.array([(float(r['x']), float(r['y'])) for r in fields])
        segment, xi, within = association(xy, vertices)
        edges = np.diff(vertices, axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        tangent = (edges/lengths[:, None])[segment]
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        stations = np.r_[0., np.cumsum(lengths)]
        foot = stations[segment]+xi*lengths[segment]
        speed = np.interp(foot, stations, [float(r['V_m_s']) for r in labels])
        integral = np.interp(foot, stations, [float(r['I_m']) for r in labels])
        phase = np.array([float(r['phi']) for r in fields])
        chi = np.where(within > 0, compliance(phase)/integral, 0.)
        # For opposite-tangent slip the offset is +kappa chi V (t⊗n + n⊗t).
        xx, yy = (2*tangent*normal).T
        shear = tangent[:, 0]*normal[:, 1]+tangent[:, 1]*normal[:, 0]
        kappa = -viscosity*np.expm1(-modulus*dt/viscosity)
        expected = np.column_stack([xx, yy, shear])*(kappa*chi*speed)[:, None]
        measured = np.array([[float(r['offset_'+c]) for c in ('xx', 'yy', 'xy')] for r in fields])
        observed_eta = np.array([float(r['eta']) for r in labels+fields])
        print(f'{output.parent.name}, step {step}: eta {observed_eta.min():g}..{observed_eta.max():g} Pa s, '
              f'kappa {kappa:.12g} Pa s, peak |offset| {np.max(abs(expected)):.9g} Pa, '
              f'error {np.max(abs(measured-expected)):.4g} Pa', flush=True)
        np.testing.assert_array_equal(observed_eta, np.full(len(observed_eta), viscosity))
        np.testing.assert_allclose(measured, expected, rtol=0., atol=2e-13, equal_nan=False)


def run_controls(prefix, observer, dashpot, destination, case):
    example = prefix/'benchmarks/fixed_fault/curved_mature'
    destination.mkdir(parents=True, exist_ok=False)
    settings = OBSERVE + 'set End time = .0125\n'
    if case in ('both', 'changed'):
        changed = settings + fault('set Shear modulus = 17\nset Initialization interval = .025')
        changed += 'subsection Material model\n subsection Simple model\n set Viscosity = 24\n end\nend\n'
        out = run('changed', example/'curved_mature.prm', [observer], root=destination, prefix=prefix, settings=changed)
        check_observations(out, 24., 17., .025, True)
    if case in ('both', 'state-free'):
        # A state-free resistance law accepts a vertex table without an initial age column.
        table = destination/'stateless-vertices.txt'
        rows = [line.split() for line in (example/'vertices.txt').read_text().splitlines()
                if line.strip() and not line.startswith('#')]
        columns = [rows[0].index(c) for c in ('id', 'x_m', 'y_m', 'background_shear_Pa')]
        table.write_text(''.join(' '.join(row[c] for c in columns)+'\n' for row in rows))
        stateless = settings + fault(f'set Fault vertex file = {table}\nset Resistance law = linear dashpot')
        out = run('state-free', example/'curved_mature.prm', [observer, dashpot], root=destination,
                  prefix=prefix, settings=stateless)
        check_observations(out, 10., 10., .1, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prefix', 'observer', 'dashpot', 'destination'):
        parser.add_argument(name, type=lambda path: Path(path).resolve())
    parser.add_argument('--case', choices=['changed', 'state-free', 'both'], default='both')
    run_controls(**vars(parser.parse_args()))
