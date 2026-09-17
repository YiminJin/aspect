"""Independent frozen-history Q2/Q1--Q1 work-measure kink experiment.

The load is manufactured from an analytic Fourier Stokes elimination, not
from the tested FE operator. No ASPECT trajectory or production file is read
or modified. See stage_K5_gradient_kink_addendum.md for scope and equations.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/aspect-kink-mpl')
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.linalg import circulant, solve

LX = LY = 8.
D = .036
HERE = Path(__file__).resolve().parent


def basis(points, cells, degree, length, derivative=False):
    h = length/cells
    element = np.floor(points/h).astype(int) % cells
    z = (points/h) % 1
    if degree == 1:
        val = np.array([-np.ones_like(z), np.ones_like(z)]) / h if derivative else np.array([1-z, z])
    elif derivative:
        val = np.array([4*z-3, 4-8*z, 4*z-1])/h
    else:
        val = np.array([(1-z)*(1-2*z), 4*z*(1-z), z*(2*z-1)])
    indices = (degree*element[:, None] + np.arange(degree+1)) % (degree*cells)
    return sparse.coo_matrix((val.T.ravel(),
                              (np.repeat(np.arange(len(points)), degree+1), indices.ravel())),
                             shape=(len(points), degree*cells)).tocsr()


def quadrature(breaks, order):
    z, w = leggauss(order)
    h = np.diff(breaks)
    return ((breaks[:-1, None]+h[:, None]*(z+1)/2).ravel(),
            (h[:, None]*w/2).ravel())


def product(left, weight, right):
    return (left.T @ sparse.diags(weight) @ right).tocsr()


def chi(y):
    n = y-LY/2
    return np.where(abs(n) < 1, (1+np.cos(np.pi*n))/2, 0.)


def chi_fourier(k):
    # Transform of the compact cosine, including its removable singularities.
    z = np.asarray(k)
    result = np.zeros_like(z)
    generic = (abs(z) > 1e-12) & (abs(abs(z)-np.pi) > 1e-12)
    result[generic] = np.pi**2*np.sin(z[generic])/(z[generic]*(np.pi**2-z[generic]**2))
    result[abs(z) <= 1e-12] = 1
    result[abs(abs(z)-np.pi) <= 1e-12] = .5
    return result


def elastic_symbol(k, cutoff):
    transverse = 2*np.pi*np.arange(-2*cutoff, 2*cutoff+1)/LY
    c2 = chi_fourier(transverse)**2
    elastic = np.zeros_like(k)
    for j, wave in enumerate(k):
        if wave == 0:
            elastic[j] = 1/LY
        else:
            elastic[j] = np.sum(c2*4*wave**2*transverse**2/(wave**2+transverse**2)**2)/LY
    return elastic


def reference(nf, cutoff, profile='kink'):
    """Independent continuum symbol and analytic triangle weak load."""
    modes = np.arange(-cutoff, cutoff+1)
    k = 2*np.pi*modes/LX
    elastic = elastic_symbol(k, cutoff)
    h = LX/nf
    hat_basis = h/LX*np.sinc(k*h/(2*np.pi))**2
    # Continuous triangle centered at 3 with half-width 1, height -1.
    target_hat = -np.sinc(k/(2*np.pi))**2*np.exp(-3j*k)/LX
    x = np.arange(nf)*h
    weak = np.real(np.exp(1j*np.outer(x, k)) @ (LX*hat_basis*elastic*target_hat))
    folded = np.bincount(modes % nf, weights=LX*hat_basis**2*elastic, minlength=nf)
    matrix = circulant(np.fft.ifft(folded).real*nf)
    target = -np.maximum(1-abs(x-3), 0.)
    # The local reaction is integrated analytically, avoiding a needless
    # slowly converging Fourier representation of the Q1 mass matrix.
    mass_column = np.zeros(nf)
    mass_column[0] = 2*h/3
    mass_column[[1, -1]] = h/6
    exact_mass = circulant(mass_column)
    matrix += D*exact_mass
    weak += D*exact_mass@target
    np.testing.assert_allclose(matrix@target, weak, rtol=2e-12, atol=2e-14)
    if profile == 'smooth':
        target = np.cos(2*np.pi*x/LX)
        e1 = elastic[np.flatnonzero(modes == 1)[0]]
        weak = h*np.sinc(h/LX)**2*(e1+D)*target
    return matrix, weak, target


def assemble(nx, ny, nf, rule):
    """Build the full saddle system and the exactly paired source work."""
    xbreaks = np.linspace(0, LX, nx+1)
    if rule == 'split':
        xbreaks = np.unique(np.r_[xbreaks, np.linspace(0, LX, nf+1)])
    x, wx = quadrature(xbreaks, 3)
    y, wy = quadrature(np.linspace(0, LY, ny+1), 3)
    X, Xd, XP, XF = (basis(x, n, p, LX, deriv) for n, p, deriv in
                    [(nx, 2, False), (nx, 2, True), (nx, 1, False), (nf, 1, False)])
    Y, Yd, YP = (basis(y, ny, p, LY, deriv) for p, deriv in
                [(2, False), (2, True), (1, False)])
    mx, kx, dx = product(X, wx, X), product(Xd, wx, Xd), product(X, wx, Xd)
    my, ky, dy = product(Y, wy, Y), product(Yd, wy, Yd), product(Y, wy, Yd)
    px, pdx = product(X, wx, XP), product(Xd, wx, XP)
    py, pdy = product(Y, wy, YP), product(Yd, wy, YP)
    kron = sparse.kron
    axx = 2*kron(kx, my)+kron(mx, ky)
    ayy = kron(kx, my)+2*kron(mx, ky)
    axy = kron(dx, dy.T)
    divx, divy = -kron(pdx, py), -kron(px, pdy)
    a = sparse.bmat([[axx, axy, divx], [axy.T, ayy, divy], [divx.T, divy.T, None]], format='csc')

    xs, xd = product(X, wx, XF), product(Xd, wx, XF)
    localization = chi(y)
    cy = np.asarray(Y.T @ (wy*localization))[:, None]
    cdy = np.asarray(Yd.T @ (wy*localization))[:, None]
    cpy = np.asarray(YP.T @ (wy*localization))[:, None]
    b = sparse.vstack([kron(xs, cdy), kron(xd, cy), sparse.csr_matrix((nx*ny, nf))], format='csc')
    # Actual normal-stress action: p - 2*d_y u_y. It must cancel in this
    # reflection-symmetric setup, not in an arbitrary BP3 configuration.
    normal = sparse.hstack([sparse.csr_matrix((nf, 4*nx*ny)),
                            -2*kron(xs.T, cdy.T),
                            kron(product(XP, wx, XF).T, cpy.T)], format='csr')
    mass = product(XF, wx, XF).toarray()
    integral = float(wy@localization)
    square = float(wy@localization**2)
    direct = mass*(square+D*integral)
    # Only translation and pressure gauges are fixed; the operator/load are
    # checked again with those rows restored after each solve.
    keep = np.ones(a.shape[0], dtype=bool)
    keep[[0, 4*nx*ny, 8*nx*ny]] = False
    return a, b, normal, direct, keep, integral, square, mass, x, wx, XF


def run(nx, ny, nf, rule, load_rule, profile, out):
    started = time.monotonic()
    ref, load, target = reference(nf, 512, profile)
    ref2, load2, _ = reference(nf, 1024, profile)
    uncertainty = float(np.max(abs(load2-load)))
    np.testing.assert_allclose(load2, load, rtol=1e-9, atol=1e-12)
    a, b, normal, direct, keep, integral, square, mass, xq, wx, XF = assemble(nx, ny, nf, rule)
    exact_load = load2.copy()
    if load_rule == 'quadrature':
        k = 2*np.pi*np.arange(-1024, 1025)/LX
        target_hat = -np.sinc(k/(2*np.pi))**2*np.exp(-3j*k)/LX
        traction = np.real(np.exp(1j*np.outer(xq, k)) @ (elastic_symbol(k, 1024)*target_hat))
        traction -= D*np.maximum(1-abs(xq-3), 0.)
        if profile == 'smooth':
            e1 = elastic_symbol(np.array([2*np.pi/LX]), 1024)[0]
            traction = (e1+D)*np.cos(2*np.pi*xq/LX)
        load2 = np.asarray(XF.T@(wx*traction))
    factor = splu(a[keep][:, keep])
    responses = np.zeros((a.shape[0], nf))
    # Limit dense-RHS workspace without changing factorization or equations.
    for j in range(0, nf, 8):
        responses[keep, j:j+8] = factor.solve(b[keep, j:j+8].toarray())
    relative_bulk = np.linalg.norm(a@responses-b.toarray())/np.linalg.norm(b.toarray())
    assert relative_bulk < 2e-9, relative_bulk
    mechanical = direct-b.T@responses
    symmetry = float(np.max(abs(mechanical-mechanical.T)))
    assert symmetry < 1e-10, symmetry
    normal_max = float(np.max(abs(normal@responses)))
    assert normal_max < 2e-9, normal_max

    x = np.arange(nf)*LX/nf
    free = x < 4
    value = target.copy()
    value[free] = solve(mechanical[np.ix_(free, free)],
                        load2[free]-mechanical[np.ix_(free, ~free)]@value[~free], assume_a='sym')
    residual = mechanical@value-load2
    surface_residual = float(np.linalg.norm(residual[free])/np.linalg.norm(load2[free]))
    assert surface_residual < 1e-10, surface_residual
    reference_value = target.copy()
    reference_value[free] = solve(ref2[np.ix_(free, free)],
                                  exact_load[free]-ref2[np.ix_(free, ~free)]@value[~free], assume_a='sym')
    if profile == 'kink':
        np.testing.assert_allclose(reference_value, target, atol=3e-12, rtol=3e-12)

    error = value-target
    junction = nf//2
    impulse = np.zeros(nf)
    impulse[junction] = 1
    impulse[free] = solve(mechanical[np.ix_(free, free)], -mechanical[free, junction], assume_a='sym')
    ref_impulse = np.zeros(nf)
    ref_impulse[junction] = 1
    ref_impulse[free] = solve(ref2[np.ix_(free, free)], -ref2[free, junction], assume_a='sym')
    # Affine continuum wave and constant controls verify signs and normalization.
    ones = np.ones(nf)
    constant_error = float(np.max(abs(mechanical@ones-(integral**2/LY+D*integral)*mass@ones)))
    assert abs(integral-1) < 2e-10
    # A Q2 bulk velocity cannot exactly integrate the cosine source. Its
    # constant-rate consistency error is measured, not asserted to vanish.
    assert np.ptp((mechanical@ones)/(mass@ones)) < 1e-10
    result = dict(nx=nx, ny=ny, nf=nf, rule=rule, load_rule=load_rule, profile=profile, h_bulk=LX/nx, h_fault=LX/nf,
                  max_error=float(max(abs(error))), rms_error=float(np.sqrt(error@mass@error/LX)),
                  last_free_error=float(error[junction-1]),
                  penultimate_error=float(error[junction-2]),
                  impulse_neighbor=float(impulse[junction-1]),
                  reference_impulse_neighbor=float(ref_impulse[junction-1]),
                  schur_neighbor=float(mechanical[junction-1, junction]),
                  reference_schur_neighbor=float(ref2[junction-1, junction]),
                  discrete_reference_max_error=float(max(abs(reference_value-target))),
                  FE_minus_reference_max_error=float(max(abs(value-reference_value))),
                  fresh_bulk_residual=relative_bulk, fresh_surface_residual=surface_residual,
                  normal_action_max=normal_max, matrix_symmetry_error=symmetry,
                  localization_integral=integral, localization_square=square,
                  constant_error=constant_error, reference_load_change=uncertainty,
                  load_quadrature_change=float(max(abs(load2-exact_load))),
                  seconds=time.monotonic()-started,
                  peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with (out/'summary.json').open('w') as stream:
        json.dump(result, stream, indent=2)
    with (out/'profile.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['s', 'target', 'FE', 'reference', 'error', 'impulse', 'reference_impulse', 'load', 'residual'])
        writer.writerows(zip(x, target, value, reference_value, error, impulse, ref_impulse, load2, residual))
    np.savez_compressed(out/'matrices.npz', surface=mechanical, reference=ref2, direct=direct, mass=mass)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nx', type=int, required=True)
    parser.add_argument('--ny', type=int, default=32)
    parser.add_argument('--nf', type=int, default=64)
    parser.add_argument('--rule', choices=['split', 'native'], default='split')
    parser.add_argument('--load-rule', choices=['exact', 'quadrature'], default='exact')
    parser.add_argument('--profile', choices=['kink', 'smooth'], default='kink')
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    output = HERE/'gradient-kink'/args.label
    output.mkdir(parents=True, exist_ok=False)
    run(args.nx, args.ny, args.nf, args.rule, args.load_rule, args.profile, output)
