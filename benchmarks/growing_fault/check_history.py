"""Independent midpoint balance from the onecut mesh, stage velocity and native endpoints."""
import argparse
from pathlib import Path
import math

import numpy as np

from compare import rows


def fracture_energy(patches, dx):
    # Independent scalar energy, integrated once per interior face. Gauss(4)
    # is exact for every polynomial product here. Outward normal points from
    # the current cell to its positive-axis neighbor.
    ell, Kf, penalty = .25, 26.6666666667/((8/3)*.25), 10
    q, w = np.polynomial.legendre.leggauss(4)
    q, w = (q+1)/2, w/2
    def basis(x):
        return np.stack(((1-x)**2, 2*x*(1-x), x*x), axis=-1)
    def derivative(x):
        return np.stack((2*x-2, 2-4*x, 2*x), axis=-1)/dx
    def evaluate(p, x, y):
        return [np.einsum('...i,ij,...j->...', a, p, b) for a,b in
                ((basis(y),basis(x)), (basis(y),derivative(x)), (derivative(y),basis(x)))]
    energy = face_energy = 0.
    for (i,j), p in patches.items():
        phi, gx, gy = evaluate(p, q[None,:], q[:,None])
        energy += dx**2 * (w @ (phi+ell**2*(gx*gx+gy*gy)) @ w)
        for axis, neighbor in enumerate(((i+1,j),(i,j+1))):
            if neighbor not in patches:
                continue
            side = (np.ones_like(q),q) if axis==0 else (q,np.ones_like(q))
            other = (np.zeros_like(q),q) if axis==0 else (q,np.zeros_like(q))
            inside = evaluate(p,*side)
            outside = evaluate(patches[neighbor],*other)
            jump = inside[0]-outside[0]
            normal_sum = inside[1+axis]+outside[1+axis]
            face_energy += dx*ell**2*(w @ (penalty*9/dx*jump*jump-normal_sum*jump))
    return Kf*(energy+face_energy), Kf*face_energy


def check(output):
    # Velocity components are continuous Q2, ordered vertices, faces, center.
    # Phase is DG tensor Bernstein, with x the fastest coefficient index.
    nodal_order = [(0, 0), (2, 0), (0, 2), (2, 2), (0, 1),
                   (2, 1), (1, 0), (1, 2), (1, 1)]
    dx, dt, modulus, viscosity = .25, .005, 1e6, 1e8
    points = rows(output.glob('points-0000-r*.csv'), ('cell', 'q'))
    coordinates = np.array([[float(points[k][axis]) for axis in ('x', 'y')] for k in sorted(points)])
    cells = np.array([k[0] for k in sorted(points)])
    elements = np.floor(coordinates/dx).astype(int)
    xy = coordinates/dx - elements
    basis = [np.array([2*t*t-3*t+1, 4*t*(1-t), 2*t*t-t]) for t in xy.T]
    derivative = [np.array([4*t-3, 4-8*t, 4*t-1])/dx for t in xy.T]

    def read_frame(step):
        tag = f'{step:04}'
        pts = rows(output.glob(f'points-{tag}-r*.csv'), ('cell', 'q'))
        if pts.keys() != points.keys():
            raise ValueError('Native identities changed')
        keys = sorted(pts)
        def array(names):
            return np.array([[float(pts[k][name]) for name in names] for k in keys])
        fields = rows(output.glob(f'fields-{tag}-r*.csv'), ('cell', 'component', 'local'))
        phase = np.array([float(fields[k]['value']) for k in sorted(fields) if k[1] == 7])
        sections = rows([output/f'sections-{tag}.csv'], ('section',))
        values = {name: np.array([float(sections[k][name]) for k in sorted(sections)])
                  for name in ('beta', 'age', 'V', 'slip', 'ds')}
        return (fields, phase, values, array(['tau'+ij for ij in ('xx','yy','zz','xy')]),
                array(['A'+ij for ij in ('xx','yy','zz','xy')]), array(['signed_shear','absolute_shear']))

    area = np.array([float(points[k]['weight']) for k in sorted(points)])
    strip = np.array([int(points[k]['section']) for k in sorted(points)])
    cell_position = {c: tuple(elements[np.flatnonzero(cells == c)[0]]) for c in set(cells)}
    def discrete_fracture(fields):
        patches = {pos: np.array([float(fields[c,7,i]['value']) for i in range(9)]).reshape(3,3)
                   for c,pos in cell_position.items()}
        return fracture_energy(patches, dx)
    parent = read_frame(0)
    previous_fracture, _ = discrete_fracture(parent[0])
    if np.min(parent[1]) >= 0:
        raise ValueError('The seed no longer exercises a negative incoming Bernstein coefficient')
    for step in (1, 2, 3):
        current = read_frame(step)
        fields, phase, sections, tau, strain, shear = current
        _, old_phase, old_sections, old_tau, old_strain, old_shear = parent
        if np.min(phase-old_phase) < -2e-15:
            raise ValueError('An endpoint decreased a Bernstein coefficient')
        grad = np.zeros((len(strip), 2, 2))
        for cell in sorted(set(cells)):
            selected = cells == cell
            for component in range(2):
                for i, (a, b) in enumerate(nodal_order):
                    v = float(fields[cell,component,i]['value'])
                    grad[selected,component,0] += v*derivative[0][a,selected]*basis[1][b,selected]
                    grad[selected,component,1] += v*basis[0][a,selected]*derivative[1][b,selected]
        divergence = np.trace(grad, axis1=1, axis2=2)
        D = np.column_stack([grad[:,0,0]-divergence/3, grad[:,1,1]-divergence/3,
                             -divergence/3, (grad[:,0,1]+grad[:,1,0])/2])
        beta = math.pi/6 + sections['beta'][strip]
        C, S = np.cos(beta), np.sin(beta)
        plane = np.column_stack([-C*S, C*S, np.zeros_like(S), (C*C-S*S)/2])
        increment = shear[:,0]-old_shear[:,0]
        slip_rate = increment/dt
        transfer = np.bincount(strip, weights=area*slip_rate)/sections['ds']
        np.testing.assert_allclose(transfer, sections['V'], atol=2e-14, rtol=1e-12)
        np.testing.assert_allclose(shear[:,1]-old_shear[:,1], abs(increment), atol=1e-15, rtol=1e-12)
        np.testing.assert_allclose(sections['slip']-old_sections['slip'], dt*transfer, atol=1e-15)
        rate = abs(transfer)/.01
        source = np.full_like(rate, dt)
        np.divide(-np.expm1(-dt*rate), rate, out=source, where=rate!=0)
        np.testing.assert_allclose(sections['age'], old_sections['age']*np.exp(-dt*rate)+source, atol=1e-13, rtol=0)
        bulk = (tau-old_tau)/(2*modulus*dt)+(tau+old_tau)/(4*viscosity)-D
        cohesive = (strain-old_strain)/dt+modulus/viscosity*(strain+old_strain)/2
        fault = plane*slip_rate[:,None]
        bulk_error, cohesive_error = np.max(abs(bulk+fault)), np.max(abs(cohesive-fault))
        np.testing.assert_allclose(bulk, -fault, atol=2e-13, rtol=0)
        np.testing.assert_allclose(cohesive, fault, atol=2e-13, rtol=0)
        print(f'Step {step}: native Maxwell and cohesive rate defects {bulk_error:.4g}, '
              f'{cohesive_error:.4g} /s; reversed bulk slip sign {np.max(abs(bulk-fault)):.4g} /s')
        book = rows([output/f'balance-{step:04}.csv'], ('step',))[step,]
        fracture, face_energy = discrete_fracture(fields)
        error = fracture-float(book['fracture'])
        np.testing.assert_allclose(error, 0., atol=1e-12)
        if abs(face_energy)<1e-5:
            raise ValueError('Face work is too small to discriminate omission of SIPG')
        print(f'  fracture {fracture:.12g}, added {fracture-previous_fracture:.12g}, '
              f'face contribution {face_energy:.12g}, energy error {error:.3g} J/m')
        previous_fracture = fracture
        parent = current


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    check(parser.parse_args().output)
