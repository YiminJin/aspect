"""Compare two-step native section controls using published physical observations."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import erf


def read(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf-8'))


def fields(folder, step):
    pattern = f'tip-points-{step:04}-r*.csv'
    paths = sorted((*folder.glob(pattern), *folder.glob(pattern+'.gz')))
    assert len({p.name.removesuffix('.gz') for p in paths}) == len(paths)
    return np.concatenate([read(p) for p in paths])


def covariance(reference, other, degrees, step):
    a, b = fields(reference, step), fields(other, step)
    radians = np.deg2rad(degrees)
    t = np.array([np.cos(radians), np.sin(radians)])
    n = np.array([-t[1], t[0]])
    R = np.stack([t, n], axis=1)
    xy = np.stack((a['x'], a['y']), axis=1)
    transformed = np.stack((b['x'], b['y']), axis=1) @ R
    distance, ix = cKDTree(transformed).query(xy)
    assert len(a) == len(b) == len(np.unique(ix)) and distance.max() < 2e-12
    b = b[ix]
    errors = {key: float(np.max(abs(b[key]-a[key]))) for key in
              ('phi', 'w', 'weight', 'signed_shear', 'absolute_shear', 'Y', 'qcoh')}
    for prefix in ('tau', 'A'):
        def tensor(data):
            xx, yy, xy = (data[prefix+suffix] for suffix in ('xx', 'yy', 'xy'))
            return np.array([[xx, xy], [xy, yy]]).transpose(2, 0, 1)
        errors[prefix] = float(abs(R.T @ tensor(b) @ R - tensor(a)).max())
        errors[prefix+'zz'] = float(abs(b[prefix+'zz']-a[prefix+'zz']).max())
    errors['point_distance'] = float(distance.max())
    for file, keys in [('tip-sections', ('ds', 'age', 'slip', 'V', 'force')),
                       ('tip-planes', ('formed', 'parent_formed'))]:
        u, v = [read(p/f'{file}-{step:04}.csv') for p in (reference, other)]
        for key in keys:
            errors[key] = float(abs(u[key]-v[key]).max())
        if file == 'tip-planes':
            first = 'beta' if 'beta' in u.dtype.names else 'angle'
            second = 'beta' if 'beta' in v.dtype.names else 'angle'
            errors['beta'] = float(abs(u[first]-v[second]).max())
    return errors


def line_check(folder, step, degrees):
    """Intersect a line with the four specimen half spaces, independently of mesh edges."""
    p = fields(folder, step)
    s = read(folder/f'tip-planes-{step:04}.csv')
    d = read(folder/f'tip-sections-{step:04}.csv')
    a = np.deg2rad(degrees)
    t, n = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    xy = np.stack((p['x'], p['y']), axis=1)
    xi, zeta = xy @ t, xy @ n
    ids = p['section'].astype(int)
    def reduce(values):
        return np.bincount(ids, weights=p['weight']*values, minlength=len(s))
    stations = s['xi']
    dxi = s['reference_dxi']
    low, high = np.array([0., 0.]), np.array([2., 1.])
    extent = np.sort(np.stack([(low-stations[:, None]*t)/n,
                               (high-stations[:, None]*t)/n]), axis=0)
    lo, hi = extent[0].max(axis=1), extent[1].min(axis=1)
    assert (hi>lo).all()
    analytic = (1+stations/5)*(hi-lo)+(hi**3-lo**3)/3
    result = dict(section_count=len(s), point_count=len(p),
                  xi_error=float(abs(xi-stations[ids]).max()),
                  length_error=float(abs(reduce(1.)/dxi-(hi-lo)).max()),
                  polynomial_integral_error=float(abs(reduce(1+xi/5+zeta**2)/dxi-analytic).max()),
                  area_error=float(abs(p['weight'].sum()-2.)),
                  rate_error=float(abs(reduce(p['w'])/s['ds']-d['V']).max()),
                  power_error=float(abs(p['weight'] @ (d['force'][ids]*p['w'])-s['ds'] @ (d['force']*d['V']))))
    assert max(result[k] for k in result if k.endswith('error')) < 2e-11
    # Localized smooth transverse rate with a closed-form chord integral.
    width = .09
    center = np.array([1., .5]) @ n
    exact = .5*np.sqrt(np.pi)*width*(erf((hi-center)/width)-erf((lo-center)/width))
    pulse = reduce(np.exp(-((zeta-center)/width)**2))/dxi
    result['gaussian_max_abs_m'] = float(abs(pulse-exact).max())
    result['gaussian_relative_l2'] = float(np.sqrt(dxi @ (pulse-exact)**2 / (dxi @ exact**2)))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('other', type=Path)
    parser.add_argument('--degrees', type=float, default=0.)
    parser.add_argument('--step', type=int, default=2)
    parser.add_argument('--cut', action='store_true')
    args = parser.parse_args()
    result = (line_check(args.other, args.step, args.degrees) if args.cut else
              covariance(args.reference, args.other, args.degrees, args.step))
    print(json.dumps(result, indent=2))
