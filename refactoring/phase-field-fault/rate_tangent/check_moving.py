"""Independent centered difference of one later layered Picard section assembly."""
import sys
from pathlib import Path
import numpy as np


def check(directory):
    out = Path(directory)
    dtype = [('cell', 'U32'), ('q', int), ('x', float), ('y', float), ('phase', float),
             ('left', int)] + [(key, np.longdouble) for key in ('xi','chi','weight','shear','compression','defect')]
    points = np.loadtxt(out/'review-points-0', dtype=dtype)
    bands = np.loadtxt(out/'review-bands-0-0', dtype=np.longdouble)
    accepted = np.genfromtxt(out/'material-1-surface-0.csv', names=True, delimiter=',')
    candidate = np.genfromtxt(out/'material-2-surface-0.csv', names=True, delimiter=',')
    np.testing.assert_array_equal(bands[:, 0], accepted['i'])
    np.testing.assert_allclose(bands[:, 4], accepted['Theta_s'], rtol=2e-16)
    np.testing.assert_allclose(bands[:, 1:3], np.stack([candidate['x'], candidate['y']], axis=1), rtol=2e-16)
    if points['defect'].max() < 1e-6:
        raise ValueError('Missing projected section stress defect')

    n = len(bands)
    left, xi = points['left'], points['xi']
    def interpolate(v):
        return (1-xi)*v[left] + xi*v[left+1]
    eta = 5.5 + 4.5*np.tanh(points['y']/.35)
    stiffness = -eta*np.expm1(-.125/eta)
    age = interpolate(bands[:, 4])
    rates = bands[:, 5]
    direction = rates*(.25+.45*np.cos(.6*(np.arange(n)-9)))
    chi = points['chi']
    psi = np.exp((.6+.015*np.log(age*.001/.008))/.025)/.002
    def residual(v, k):
        speed = interpolate(v)
        resistance = points['compression']*.025*np.arcsinh(psi*speed) + speed
        traction = points['shear']-resistance-k*chi*speed
        weighted = points['weight']*chi*traction
        rows = np.zeros(n,dtype=np.longdouble)
        np.add.at(rows,left,(1-xi)*weighted)
        np.add.at(rows,left+1,xi*weighted)
        return rows

    action = bands[:, 6]*direction
    action[1:] += bands[:-1, 7]*direction[:-1]
    action[:-1] += bands[:-1, 7]*direction[1:]
    norm = np.max(np.abs(action))
    def difference(h, k):
        return (residual(rates-h*direction,k)-residual(rates+h*direction,k))/(2*h)
    np.testing.assert_allclose(residual(rates,stiffness), bands[:,8], rtol=0., atol=2e-14*max(bands[:,9]))
    coarse = difference(1e-4,stiffness)
    fine = difference(5e-5,stiffness)
    error = np.max(abs(action-fine))/norm
    mutation = np.max(abs(action-difference(5e-5,interpolate(bands[:,10]))))/norm
    if error>=2e-8 or mutation<=2e-6:
        raise ValueError(f'Pointwise tangent error {error}, label stiffness mutation {mutation}')
    print(f'{len(points)} Q8 points, frozen projection defect {max(points["defect"]):.8g} Pa, '
          f'action norm {norm:.9g}, relative error {error:.4g}, '
          f'Richardson estimate {max(abs(fine-coarse))/(3*norm):.4g}, label stiffness error {mutation:.4g}')


if __name__ == '__main__':
    check(sys.argv[1])
