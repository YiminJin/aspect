"""Check the prescribed curve against FE phase and velocity from moving fault samples."""
import argparse
import csv
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss


def table(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def basis(nodes, x):
    """Values of the three cardinal polynomials at each reference coordinate."""
    return np.column_stack([np.prod((x[:, None] - np.delete(nodes, i)) /
                                    (nodes[i] - np.delete(nodes, i)), axis=1)
                            for i in range(3)])


class Fields:
    """Q2 on the example's unit square, with a separate polynomial per cell."""

    def __init__(self, output, step):
        shards = sorted(output.glob(f'material-{step}-rank*.csv'))
        rows = [r for path in shards for r in table(path)]
        xy = np.array([(float(r['x']), float(r['y'])) for r in rows])
        self.n = int(round(np.sqrt(len(rows) / 9)))
        self.nodes = (leggauss(3)[0] + 1.) / 2.
        cells = np.floor(xy * self.n).astype(int)
        local = np.argmin(abs((xy * self.n - cells)[:, :, None] - self.nodes), axis=2)
        keys = [(r['cell'], int(r['q'])) for r in rows]
        if (self.n**2 * 9 != len(keys) or len(set(keys)) != len(keys) or
                len(np.unique(np.hstack([cells, local]), axis=0)) != len(keys)):
            raise ValueError('Expected all nine distinct Q3 samples of every cell')
        np.testing.assert_allclose(xy, (cells + self.nodes[local])/self.n, rtol=0., atol=2e-14)
        self.values = {}
        for name in ('phi', 'vx', 'vy', 'Lxy', 'Lyx', 'tauxx', 'tauyy', 'tauxy', 'offset_xy'):
            values = np.empty((self.n, self.n, 3, 3))
            values[cells[:, 0], cells[:, 1], local[:, 0], local[:, 1]] = [float(r[name]) for r in rows]
            self.values[name] = values
        self.xy = xy
        self.measured = {k: np.array([float(r[k]) for r in rows]) for k in ('station', 'chi')}

    def at(self, name, xy):
        unit = np.atleast_2d(xy) * self.n
        cells = np.clip(np.floor(unit).astype(int), 0, self.n-1)
        x, y = (unit - cells).T
        return np.einsum('pi,pij,pj->p', basis(self.nodes, x),
                         self.values[name][cells[:, 0], cells[:, 1]], basis(self.nodes, y))


def compliance(phi):
    return phi*(1+phi)/(1-phi)**2


def normal_integrals(v, tangents, lengths, fields, order, ray_samples=None):
    """Integrate the two-sided physical ray, then solve the consistent trace mass system."""
    n = len(v)
    rhs, mass = np.zeros(n), np.zeros((n, n))
    # The public section uses three along-segment integration stations. Split
    # the normal integral at Cartesian faces and independently refine its rule.
    stations, ds = leggauss(3)
    points, weights = leggauss(order)
    grid_faces = np.linspace(0., 1., fields.n + 1)
    for j, (t, length) in enumerate(zip(tangents, lengths)):
        normal = np.array([-t[1], t[0]])
        for xi, w in zip((1+stations)/2, ds/2):
            center = v[j] + xi*length*t
            face_distances = [(grid_faces-center[d])/normal[d] for d in range(2) if normal[d] != 0.]
            lo = max(min(z[0], z[-1]) for z in face_distances)
            hi = min(max(z[0], z[-1]) for z in face_distances)
            cuts = np.unique(np.clip(np.concatenate(face_distances), lo, hi))
            a, b = cuts[:-1], cuts[1:]
            d = ((a+b)[:, None] + (b-a)[:, None]*points)/2
            phi = fields.at('phi', center + d.flatten()[:, None]*normal).reshape(d.shape)
            integral = ((b-a)/2) @ (compliance(phi) @ weights)
            if ray_samples is not None:
                ray_samples.append((j, xi, lo, hi, integral))
            shape = np.array([1-xi, xi])
            rhs[j:j+2] += length*w*integral*shape
            mass[j:j+2, j:j+2] += length*w*np.outer(shape, shape)
    return np.linalg.solve(mass, rhs)


def association(xy, v):
    """Independent Euclidean feet. At overlaps prefer the closest eligible normal."""
    edges = np.diff(v, axis=0)
    squared_lengths = np.sum(edges*edges, axis=1)
    offsets = xy[:, None, :] - v[:-1]
    xi = np.einsum('psi,si->ps', offsets, edges)/squared_lengths
    within = (xi >= 0.) & (xi <= 1.)
    xi = np.clip(xi, 0., 1.)
    points = v[:-1] + xi[:, :, None]*edges
    distances = np.linalg.norm(xy[:, None, :]-points, axis=2)
    count = within.sum(axis=1)
    distances[~within & (count[:, None] > 0)] = np.inf
    segment = distances.argmin(axis=1)
    return segment, xi[np.arange(len(xy)), segment], count


def cell_rule(order, n):
    """Physical points and area-average weights for identical square cells."""
    z, w = leggauss(order)
    i, j, a, b = np.indices((n, n, order, order))
    xy = np.column_stack([(i+(1+z[a])/2).ravel(), (j+(1+z[b])/2).ravel()])/n
    return xy, np.outer(w, w)/4


def check_rotated_parent(output, fields, v, s, integrals, dt):
    # At the second physical interval, the old accepted stress is nonzero.
    # Recover its rotated cell mean two ways. The native observer gives the
    # constitutive offset, before cut-cell projection into accepted DG stress.
    previous = Fields(output, 1)
    epsilon = -1.  # This example prescribes opposite-tangent motion.
    shear_modulus, viscosity = 10., 10.
    relax = np.exp(-dt*shear_modulus/viscosity)
    kappa = -viscosity*np.expm1(-dt*shear_modulus/viscosity)
    tangents = np.diff(v, axis=0)/np.diff(s)[:, None]
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    def weighted_speed(xy, step):
        segment, xi, count = association(xy, v)
        stations = s[segment]+xi*np.diff(s)[segment]
        rows = table(output/f'material-{step}-surface-0.csv')
        rate = np.interp(stations, s, [float(r['V_m_s']) for r in rows])
        chi = compliance(fields.at('phi', xy))/np.interp(stations, s, integrals)
        return segment, np.where(count > 0, chi*rate, 0.)

    n = fields.n
    q3, w3 = cell_rule(3, n)
    k, slip = weighted_speed(q3, 2)
    dyad_xy = tangents[k, 0]*normals[k, 1] + tangents[k, 1]*normals[k, 0]
    slip_stress = -kappa*epsilon*slip*dyad_xy
    relaxed_parent = fields.at('offset_xy', q3) - slip_stress
    observed = np.einsum('ijab,ab->ij', relaxed_parent.reshape(n,n,3,3), w3)

    q8, w8 = cell_rule(8, n)
    _, parent_slip = weighted_speed(q8, 1)
    vorticity = previous.at('Lyx', q8)-previous.at('Lxy', q8)
    half_difference = .5*(previous.at('tauxx', q8)-previous.at('tauyy', q8))
    shear = previous.at('tauxy', q8)
    # (t ⊗ n)_yx − (t ⊗ n)_xy = −1 for a CCW normal.
    def rotated_means(sense):
        twice_angle = dt*(vorticity + sense*parent_slip)
        rotated = half_difference*np.sin(twice_angle)+shear*np.cos(twice_angle)
        return relax*np.einsum('ijab,ab->ij', rotated.reshape(n,n,8,8), w8)
    actual = np.max(abs(observed-rotated_means(epsilon)))
    wrong = np.max(abs(observed-rotated_means(-epsilon)))
    if not (np.isfinite(actual) and np.isfinite(wrong)):
        raise ValueError(f'Nonfinite rotated-parent shear error: correct {actual:g}, wrong sign {wrong:g} Pa')
    if actual > 2e-13 or wrong <= 1e-7:
        raise ValueError(f'Rotated parent shear: correct {actual:g}, wrong sign {wrong:g} Pa')
    print(f'Cell-mean rotated parent shear: max error {actual:.4g} Pa, wrong slip spin {wrong:.6g} Pa')


def check(output, vertex_file):
    with vertex_file.open() as stream:
        records = [s.split('#', 1)[0].split() for s in stream]
    records = [r for r in records if r]
    vertices = [dict(zip(records[0], r)) for r in records[1:]]
    v = np.array([(float(r['x_m']), float(r['y_m'])) for r in vertices])
    np.testing.assert_array_equal([int(r['id']) for r in vertices], np.arange(len(v)))
    clock = np.loadtxt(output/'statistics', usecols=(0, 1, 2), ndmin=2)
    np.testing.assert_allclose(clock, [[0, 0, 0], [1, .0125, .0125], [2, .025, .0125]], atol=1e-15, rtol=0)
    labels = table(output/'material-2-surface-0.csv')
    np.testing.assert_array_equal([int(r['i']) for r in labels], np.arange(len(v)))
    np.testing.assert_array_equal([(float(r['x']), float(r['y'])) for r in labels], v)
    np.testing.assert_array_equal(v[[0, -1], 0], [0, 1])
    initial = table(output/'material-0-surface-0.csv')
    np.testing.assert_array_equal([float(r['Theta_s']) for r in initial],
                                  [float(r['initial_age_s']) for r in vertices])
    edges = np.diff(v, axis=0)
    length = np.linalg.norm(edges, axis=1)
    t = edges / length[:, None]
    normals = np.column_stack([-t[:, 1], t[:, 0]])
    s = np.r_[0, np.cumsum(length)]
    fields = Fields(output, 2)
    rays = []
    I = normal_integrals(v, t, length, fields, 24, rays)
    I16 = normal_integrals(v, t, length, fields, 16)
    np.testing.assert_allclose(I, I16, atol=5e-14, rtol=0)
    observed_I = np.array([float(r['I_m']) for r in labels])
    np.testing.assert_allclose(observed_I, I, atol=2e-13, rtol=0)
    k, xi, n_admitted = association(fields.xy, v)
    station = s[k]+xi*length[k]
    phi = fields.at('phi', fields.xy)
    expected_chi = np.where(n_admitted > 0, compliance(phi)/np.interp(station, s, I), 0.)
    np.testing.assert_allclose(fields.measured['station'], station, rtol=0., atol=2e-14)
    np.testing.assert_allclose(fields.measured['chi'], expected_chi, rtol=0., atol=3e-12)
    if not ((n_admitted > 1).any() and (n_admitted == 0).any()):
        raise ValueError('Fixture must sample both overlapping and uncovered normal supports')
    # A chord can give a different closest physical point even within the band.
    chord = v[-1]-v[0]
    chord_foot = v[0]+np.clip((fields.xy-v[0])@chord/(chord@chord), 0, 1)[:, None]*chord
    local_foot = v[k]+xi[:, None]*edges[k]
    delta = np.linalg.norm(chord_foot-local_foot, axis=1)
    i = np.argmax(np.where((phi > .29) & (n_admitted > 0), delta, -1))
    if delta[i] < .1:
        raise ValueError('Chord control failed to distinguish the prescribed bend')
    plus, minus = (v[:-1]+v[1:])/2 + .065*normals, (v[:-1]+v[1:])/2 - .065*normals
    def velocity(xy):
        return np.column_stack([fields.at(name, xy) for name in ('vx', 'vy')])
    updip = np.einsum('ij,ij->i', -t, velocity(plus)-velocity(minus))
    if not np.all(updip > 0):
        raise ValueError('Expected relative up-dip motion on each segment')
    check_rotated_parent(output, fields, v, s, I, clock[-1, 2])
    for j, xi, lo, hi, h in (rays[0], rays[5], rays[6]):
        print(f'clipped ray at segment {j}, xi={xi:.8f}: [{lo:.9f}, {hi:.9f}] m, integral {h:.12g} m')
    print('cumulative stations (m)', s)
    print('projected normal integrals (m)', I, 'max absolute defect', max(abs(observed_I-I)))
    print('max station (m) and chi (1/m) errors', max(abs(fields.measured['station']-station)),
          max(abs(fields.measured['chi']-expected_chi)))
    print('Q3 samples in overlap / outside all segment normals', sum(n_admitted > 1), sum(n_admitted == 0))
    print('active point / local foot / chord foot (m)', fields.xy[i], local_foot[i], chord_foot[i])
    for j in range(len(t)):
        print(f'segment {j}, n={normals[j]}, +/- (m)={plus[j]}, {minus[j]}, relative up-dip (m/s)={updip[j]:.12g}')
    return fields, v, updip


def plot(path, fields, v, updip):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, (ax, speed) = plt.subplots(1, 2, figsize=(9.2, 4.1), layout='constrained',
                                    gridspec_kw={'width_ratios': [1.25, 1]})
    pixels = (np.arange(256)+.5)/256
    xx, yy = np.meshgrid(pixels, pixels)
    phi = fields.at('phi', np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    img = ax.imshow(phi, origin='lower', extent=(0, 1, 0, 1), cmap='Blues', vmin=0, vmax=.35)
    level = .03+.32/np.e
    ax.contour(xx, yy, phi, levels=[level], colors='#13606b', linestyles='--', linewidths=1.3)
    trace, = ax.plot(*v.T, '-o', color='#853a25', lw=2, ms=4, label='Prescribed trace')
    contacts = ax.scatter(*v[[0,-1]].T, marker='s', s=42, color='#602514', clip_on=False,
                          label='Boundary contacts', zorder=4)
    length = np.linalg.norm(np.diff(v, axis=0), axis=1)
    tangents = np.diff(v, axis=0)/length[:, None]
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    centers = .5*(v[1:]+v[:-1])
    for center, normal in zip(centers, normals):
        cut = center+.065*np.array([-normal, normal])
        ax.plot(*cut.T, 'o-', color='#282a37', lw=.9, ms=2.6)
    ax.legend(handles=[trace, contacts,
                       Line2D([], [], color='#13606b', ls='--', label='Excess phase at 1/e'),
                       Line2D([], [], color='#282a37', marker='o', ms=3, label='Velocity sample pairs')],
              loc='lower left', framealpha=.95, fontsize=8)
    ax.set(xlabel='x (m)', ylabel='y (m)', title='Accepted phase and fault at 25 ms',
           aspect='equal', xlim=(0, 1), ylim=(0, 1))
    fig.colorbar(img, ax=ax, shrink=.84, label='Phase')
    mid = np.cumsum(length)-.5*length
    speed.axhline(0, color='.4', lw=.8)
    speed.plot(mid, 1000*updip, 'o-', color='#853a25')
    speed.set(xlabel='Arclength from left contact (m)', ylabel='Relative up-dip bulk velocity (mm/s)',
              title='Velocity across 0.13 m at 25 ms', xlim=(0, length.sum()))
    fig.savefig(path, dpi=210)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='Accepted moving fault samples from curved_mature.prm')
    parser.add_argument('--vertices', type=Path, default=Path(__file__).with_name('vertices.txt'))
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    fields, vertices, updip = check(args.output, args.vertices)
    if args.figure:
        plot(args.figure, fields, vertices, updip)
