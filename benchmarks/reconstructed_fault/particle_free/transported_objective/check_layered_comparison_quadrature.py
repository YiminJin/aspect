#!/usr/bin/env python3
"""Analytic squared-norm check for shifted, non-nested discontinuous Q2 fields."""
import json
from types import SimpleNamespace
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from assess_layered_campaign import OUT, common_bulk_quadrature


def main():
    def grid(h, cells):
        return SimpleNamespace(lo=np.array([-1., -1.]), hi=np.array([1., 1.]),
                               h=h, shape=np.array([cells, cells]))
    a, b = grid(1., 2), grid(2./3., 3)
    # A(x+.2,y)-B(x,y) = .4*x+c+.3*y^2. Its jumps are at
    # x=-.2 (an A face) and x=1/3 (a B face). Integrate analytically.
    exact = 0.
    for left, right, c in [(-1., -.2, .04), (-.2, 1./3., 1.04),
                           (1./3., .8, -.96)]:
        d1, d2, d3 = right-left, right**2-left**2, right**3-left**3
        exact += 2*(.16*d3/3+.4*c*d2+c*c*d1)
        exact += .4*(.2*d2+c*d1)+.036*d1
    checks = []
    for order in (3, 8, 12):
        x, w = common_bulk_quadrature(a, b, .2, order)
        difference = .4*x[:,0]+.04+(x[:,0]>-.2)-2*(x[:,0]>1./3.)+.3*x[:,1]**2
        integral = float(w @ difference**2)
        error = abs(integral-exact)
        assert abs(float(w.sum())-3.6)<1e-12
        assert error<1e-12, (order, integral, exact)
        checks.append(dict(order=order, integral=integral, error=error))
    result = dict(analytic_squared_norm=exact, checks=checks)
    (OUT/'comparison-quadrature-check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
