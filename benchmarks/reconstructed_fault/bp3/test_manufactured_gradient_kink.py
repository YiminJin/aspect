"""Cheap algebra/continuum controls for the independent diagnostic only."""
import numpy as np
from scipy import sparse

from manufactured_gradient_kink import (LX, LY, D, assemble, basis, chi,
                                       elastic_symbol, quadrature, reference)
from manufactured_trace_comparison import element_transforms


def check_pointwise_work():
    nx, ny, nf = 4, 8, 8
    a, b, normal, direct, _, _, _, mass, x, wx, XF = assemble(nx, ny, nf, 'split')
    y, wy = quadrature(np.linspace(0, LY, ny+1), 3)
    X, Xd = [basis(x, nx, 2, LX, d) for d in (False, True)]
    Y, Yd = [basis(y, ny, 2, LY, d) for d in (False, True)]
    P = sparse.kron(basis(x, nx, 1, LX), basis(y, ny, 1, LY))
    dx, dy = sparse.kron(Xd, Y), sparse.kron(X, Yd)
    S = sparse.kron(XF, np.ones((len(y), 1)))
    weight = np.kron(wx, wy)
    loc = np.tile(chi(y), len(x))
    rng = np.random.default_rng(384791)
    u, w = rng.normal(size=(2, a.shape[0]))
    v = rng.normal(size=nf)
    nvel = 4*nx*ny

    def fields(state):
        ux, uy, p = state[:nvel], state[nvel:2*nvel], state[2*nvel:]
        return dx@ux, dy@uy, (dy@ux+dx@uy)/2, P@p

    ex, ey, es, p = fields(u)
    wxp, wyp, wsp, wp = fields(w)
    pointwise = weight@(2*(ex*wxp+ey*wyp+2*es*wsp)-p*(wxp+wyp)-wp*(ex+ey))
    np.testing.assert_allclose(w@(a@u), pointwise, rtol=3e-14, atol=2e-12)
    np.testing.assert_allclose(w@(b@v), weight@(2*loc*(S@v)*wsp), rtol=3e-14, atol=2e-12)
    np.testing.assert_allclose(v@(normal@u), weight@(loc*(S@v)*(p-2*ey)), rtol=3e-14, atol=2e-12)
    np.testing.assert_allclose(v@direct@v, weight@((loc**2+D*loc)*(S@v)**2), rtol=3e-14, atol=2e-12)
    np.testing.assert_allclose(mass@np.ones(nf), LX/nf, atol=2e-14)


def check_reference():
    # Eliminate velocity with the transverse incompressibility projector,
    # independently of the closed scalar symbol in the reference routine.
    rng = np.random.default_rng(7681)
    for k, l in rng.uniform(.1, 10., (100, 2)):
        wave = np.array([k, l])
        traction = np.array([l, k])
        projection = np.eye(2)-np.outer(wave, wave)/(wave@wave)
        eliminated = 1-traction@projection@traction/(wave@wave)
        np.testing.assert_allclose(eliminated, 4*k*k*l*l/(k*k+l*l)**2, atol=1e-14)
    k = np.linspace(-20, 20, 51)
    left, right = element_transforms(.25, .125, k)
    x, w = quadrature(np.array([.25, .375]), 24)
    numerical_left = np.exp(-1j*np.outer(k, x))@(w*(1-(x-.25)/.125))/LX
    numerical_right = np.exp(-1j*np.outer(k, x))@(w*((x-.25)/.125))/LX
    np.testing.assert_allclose(left, numerical_left, atol=1e-15)
    np.testing.assert_allclose(right, numerical_right, atol=1e-15)
    matrix, load, target = reference(64, 1024)
    assert np.linalg.eigvalsh(matrix)[0] > 0
    np.testing.assert_allclose(matrix@target, load, atol=2e-14)
    # Uniform frozen friction alone gives the known consistent-mass inverse
    # response; positivity of its entries is not inverse monotonicity.
    mass = np.diag(np.full(50, 4.))+np.diag(np.ones(49), 1)+np.diag(np.ones(49), -1)
    rhs = np.zeros(50)
    rhs[0] = -1
    v = np.linalg.solve(mass, rhs)
    np.testing.assert_allclose(v[:15], (-2+np.sqrt(3.))**np.arange(1, 16), rtol=2e-13)


if __name__ == '__main__':
    check_pointwise_work()
    check_reference()
    print('PASS: pointwise Stokes/source/normal work, basis Fourier transforms, '
          'independent elimination symbol, manufactured load, reaction stencil.')
