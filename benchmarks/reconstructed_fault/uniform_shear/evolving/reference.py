#!/usr/bin/env python3
"""Independent finite-step K3 preflight. No ASPECT executable or constitutive calls.

P1 normal weak solve with Gauss integration, a banded Newton Jacobian, natural
boundaries, and a bracketed homogeneous mechanical solve. This is not CPDI.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.integrate import cumulative_simpson, cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.linalg import solve_banded
from scipy.optimize import brentq


def read_parameters(path):
    sections, values = [], {}
    for raw in path.read_text().splitlines():
        line = raw.split('#', 1)[0].strip()
        if line.startswith('subsection '):
            sections.append(line[11:])
        elif line == 'end':
            sections.pop()
        elif line.startswith('set '):
            key, value = line[4:].split('=', 1)
            values['/'.join(sections+[key.strip()])] = value.strip()
    return values


class Reference:
    def __init__(self, parameters, cells, order, ramp_peak=.009, initial_data=None):
        self.parameters = parameters
        self.ramp_peak = ramp_peak
        material = 'Material model/Phase field fault/'
        get = lambda name: float(parameters[material+name])
        self.activation = get('Phase field activation threshold')
        self.eta = get('Reference viscosities')
        self.G = get('Elastic shear moduli')
        self.Hc = get('Cohesions')**2/(2*self.G)
        self.ell = float(parameters['Phase field model/Length scale'])
        self.Ec = get('Critical energy release rates')/((8/3)*self.ell)
        self.m = self.Ec/self.Hc
        assert parameters['Phase field model/Geometric function type'] == 'AT1'
        assert float(parameters['Phase field model/Degradation curvature parameter']) == 1
        assert get('Minimum viscosity') <= self.eta <= get('Maximum viscosity')
        self.mu0, self.af, self.bf = (get(n) for n in ('Reference friction coefficients',
                                    'Direct effect parameters', 'Evolution effect parameters'))
        self.Dc, self.Vref, self.Vmin = (get(n) for n in ('Characteristic slip distance',
                                       'Reference slip rate', 'Minimum slip rate'))
        self.damping = get('Radiation damping coefficients')
        self.sigma = float(parameters['Surface pressure'])
        self.core = .6  # prescribed fault.txt, not a phase boundary condition
        # Endpoint-regularized stationary first integral, independently inverted.
        t = np.linspace(0, np.pi/2, 32769)
        phi = self.core*np.sin(t)**2
        jac = 2*self.ell*np.sqrt(1+self.core)*(1-phi)/np.sqrt(3-self.core-(1+self.core)*phi)
        distance = cumulative_simpson(jac, x=t, initial=0)
        self.support = float(distance[-1])
        self.profile = PchipInterpolator((self.support-distance)[::-1], phi[::-1])
        cutoff = brentq(lambda y: float(self.profile(y))-self.activation, 0, self.support)
        self.initial_data = initial_data
        extra_nodes = []
        if initial_data is not None:
            # Conditional 1-D reduction of INITIAL exports only. Keep the
            # particle-row P0 H and published Q1 phi distinct; do not force
            # them to solve the independent continuum discretization exactly.
            self.saved_phase = np.genfromtxt(initial_data/'phase_0.csv', delimiter=',', names=True)
            particles = np.genfromtxt(initial_data/'particles_0.csv', delimiter=',', names=True)
            self.saved_surface = np.genfromtxt(initial_data/'surface_0.csv', delimiter=',', names=True)
            self.phase_y = np.unique(self.saved_phase['y'])
            self.phase_values = np.array([np.mean(self.saved_phase['phi'][self.saved_phase['y'] == y])
                                          for y in self.phase_y])
            rows = np.unique(particles['y'])
            self.history_edges = np.r_[-.5, .5*(rows[1:]+rows[:-1]), .5]
            self.history_values = np.array([np.average(particles['H'][particles['y'] == y],
                                             weights=particles['volume'][particles['y'] == y]) for y in rows])
            self.saved_q = float(np.average(particles['tau_xy'], weights=particles['volume']))
            extra_nodes = np.r_[self.history_edges, self.phase_y]
            self.initial_provenance = {name: hashlib.sha256((initial_data/name).read_bytes()).hexdigest()
                                      for name in ('phase_0.csv', 'particles_0.csv', 'surface_0.csv')}
        # Split at actual H initialization discontinuities and support boundaries.
        self.y = np.unique(np.r_[np.linspace(-.5, .5, cells+1),
                                 -cutoff, cutoff, -self.support, self.support, extra_nodes])
        if initial_data is not None:
            # A saved symmetric row midpoint can be -3e-19 instead of zero.
            # Merge only roundoff-coincident independent integration splits.
            self.y = self.y[np.r_[True, np.diff(self.y) > 32*np.finfo(float).eps]]
        self.dy = np.diff(self.y)
        points, weights = np.polynomial.legendre.leggauss(order)
        self.shape = np.array([(1-points)/2, (1+points)/2])
        self.qy = self.y[:-1, None]+self.dy[:, None]*(points+1)/2
        self.w = self.dy[:, None]*weights/2
        self.admitted = abs(self.qy) < self.support
        prescribed = self.stationary(self.qy)
        g, _ = self.degradation(prescribed)
        _, hc = self.degradation(self.core)
        self.H0 = np.where(prescribed > self.activation,
                           self.Ec*self.core/(hc*g*g), self.Hc)
        if initial_data is not None:
            self.H0 = self.history_values[np.searchsorted(self.history_edges[1:], self.qy)]
        self.cutoff = cutoff

    def stationary(self, y):
        return np.where(abs(y) < self.support,
                        self.profile(np.minimum(abs(y), self.support)), 0.)

    def degradation(self, phi):
        A = (1-phi)**2+self.m*phi*(1+phi)
        g = (1-phi)**2/A
        return g, 1/g-1

    def samples(self, phi):
        return phi[:-1, None]*self.shape[0]+phi[1:, None]*self.shape[1]

    def residual(self, phi, H, jacobian=False):
        p = self.samples(phi)
        A = (1-p)**2+self.m*p*(1+p)
        dA = 2*(p-1)+self.m*(1+2*p)
        # Connected rational branch, not a lower-bound obstacle.
        if not np.all(np.isfinite(p) & (p <= 1) & (A > 0)
                      & ((p >= 0) | (dA >= 0) | (self.m*(self.m-8) < 0))):
            return None
        B = (1-p)*(1+3*p)
        dg = -self.m*B/A**2
        reaction = H*dg+self.Ec
        K = 2*self.Ec*self.ell**2/self.dy
        gradient = K*np.diff(phi)
        left = np.sum(self.w*self.shape[0]*reaction, axis=1)-gradient
        right = np.sum(self.w*self.shape[1]*reaction, axis=1)+gradient
        r = np.r_[left, 0.]+np.r_[0., right]
        if not jacobian:
            return r
        ddg = self.m*(2*B*dA-(2-6*p)*A)/A**3
        reaction_jac = self.w*H*ddg
        ll = np.sum(reaction_jac*self.shape[0]**2, axis=1)+K
        rr = np.sum(reaction_jac*self.shape[1]**2, axis=1)+K
        lr = np.sum(reaction_jac*self.shape[0]*self.shape[1], axis=1)-K
        band = np.zeros((3, len(phi)))
        band[1] = np.r_[ll, 0.]+np.r_[0., rr]
        band[0, 1:] = lr
        band[2, :-1] = lr
        return r, band

    def phase(self, H, guess):
        # Small elements amplify subtraction of nearby nodal values. Retain
        # extra precision in the independent iterate/residual, not a relaxed
        # stopping tolerance; the banded correction itself remains double.
        phi = np.asarray(guess, dtype=np.longdouble).copy()
        r0 = np.linalg.norm(self.residual(phi, H))
        history = []
        for iteration in range(50):
            r, matrix = self.residual(phi, H, True)
            norm = np.linalg.norm(r)
            history.append(float(norm))
            if norm <= max(1e-11*r0, 2e-11):
                return np.asarray(phi, dtype=float), history
            direction = solve_banded((1, 1), matrix, np.asarray(-r, dtype=float))
            for backtrack in range(25):
                alpha = 2.**(-backtrack)
                trial = phi+alpha*direction
                residual = self.residual(trial, H)
                if residual is not None and np.linalg.norm(residual) < (1-1e-4*alpha)*norm:
                    phi = trial
                    break
            else:
                raise RuntimeError(f'Reference phase line search exhausted: {norm:g}')
        raise RuntimeError('Reference phase iteration exhaustion')

    def localization(self, phi):
        assert min(phi) >= -1e-4 and max(phi) < 1, 'Reference profile inadmissible'
        g, h = self.degradation(np.maximum(self.samples(phi), 0))
        return g, h, float(np.sum(self.w*h))

    def mechanics(self, old, dt, U, I):
        beta = math.exp(-dt*self.G/self.eta)
        kappa = -self.eta*math.expm1(-dt*self.G/self.eta)
        def evaluate(V):
            q = beta*old['q']+kappa*(U-V)
            C = (kappa*V+beta*old['Ih']*old['C'])/I
            mu = self.af*math.asinh(V/(2*self.Vref)*math.exp(
                (self.mu0+self.bf*math.log(old['Theta']*self.Vref/self.Dc))/self.af))
            return q-C-self.sigma*mu-self.damping*V, q, C
        lower = evaluate(self.Vmin)[0]
        assert lower > 0, 'No admissible interior mechanical root'
        upper = max(U, self.Vref)
        while evaluate(upper)[0] > 0:
            upper *= 2
        V = brentq(lambda v: evaluate(v)[0], self.Vmin, upper, xtol=1e-16, rtol=1e-14)
        F, q, C = evaluate(V)
        assert abs(F) < 1e-7
        return dict(V=V, q=q, C=C, Ih=I, residual=F, F_at_Vmin=lower), beta, kappa

    def velocity(self, response, beta, kappa, old, phi, old_phi, U):
        # Fine independent integration for velocity and timestep/crossing estimates.
        y = np.unique(np.r_[np.linspace(-.5, .5, 16385), self.y])
        _, h = self.degradation(np.maximum(np.interp(y, self.y, phi), 0))
        _, hp = self.degradation(np.maximum(np.interp(y, self.y, old_phi), 0))
        v = h/response['Ih']*response['V']+beta*old['C']/kappa*(h*old['Ih']/response['Ih']-hp)
        u = -U/2+(response['q']-beta*old['q'])/kappa*(y+.5)+cumulative_trapezoid(v, y, initial=0)
        return y, u

    def run(self, accepted_times=None):
        H = self.H0.copy()
        if self.initial_data is None:
            phi, initial_iterations = self.phase(H, self.stationary(self.y))
        else:
            phi = np.interp(self.y, self.phase_y, self.phase_values)
            initial_iterations = []  # Supplied production solve, not a reference convergence claim.
        phi0 = phi.copy()
        g, hp, I = self.localization(phi)
        C0 = float(np.sum(self.w*self.admitted*g*np.sqrt(2*self.G*H))/np.sum(self.w*self.admitted))
        old = dict(q=1500., C=C0, Theta=200., Ih=I)
        if self.initial_data is not None:
            surface = self.saved_surface
            order = np.argsort(surface['x'])
            x = surface['x'][order]
            for target, field in (('C', 'C'), ('Ih', 'Ih'), ('Theta', 'Theta')):
                old[target] = float(np.trapezoid(surface[field][order], x)/(x[-1]-x[0]))
            old['q'] = self.saved_q
            C0 = old['C']
        initial, beta, kappa = self.mechanics(old, 2, 1e-4, I)
        vy, vu = self.velocity(initial, beta, kappa, old, phi, phi, 1e-4)
        cfl = float(self.parameters['CFL number'])
        degree = float(self.parameters['Discretization/Temperature polynomial degree'])
        level=int(self.parameters['Mesh refinement/Initial global refinement'])
        nx=int(self.parameters['Geometry model/Box/X repetitions'])*2**level
        ny=int(self.parameters['Geometry model/Box/Y repetitions'])*2**level
        mesh_h = min(.25/nx,1/ny)
        # The strengthening law imposes no RSF splitting cap. Max first step
        # comes from the resolved fixture; the K3 maximum dt is explicitly 2 s.
        assert self.af > self.bf
        convective = cfl*mesh_h/(degree*max(abs(vu)))
        dt = min(2., float(self.parameters['Maximum first time step']), convective)
        if accepted_times:
            dt = accepted_times[0]['dt']
        report = dict(reference_kind='independent continuum initialization', ramp_peak_m_per_s=self.ramp_peak,
                      activation_threshold=self.activation, activation_distance_m=self.cutoff,
                      support_half_width_m=self.support, cells=len(self.dy), quadrature_order=self.w.shape[1],
                      initial=dict(Ih=I, C=C0, phi_max=float(max(phi)), phi_min=float(min(phi)),
                                   H_max=float(max(H.flat)), evaluated=initial,
                                   retained_histories=old.copy(),
                                   reference_phase_residual=float(np.linalg.norm(self.residual(phi, H))),
                                   omitted_fraction=float(np.sum(self.w*hp*(~self.admitted))/I),
                                   phase_residual_history=initial_iterations),
                      first_timestep=dict(selected_s=dt, convection_estimate_s=convective,
                                          RSF_splitting_cap='infinity: a>b', maximum_s=2.), steps=[])
        report['time_sequence_source'] = ('exact accepted ASPECT time/dt/U exports'
                                          if accepted_times is not None else 'reference timestep prediction')
        if self.initial_data is not None:
            report['reference_kind'] = 'conditional production-initialized 1-D reduction, not CPDI'
            report['initial_data'] = str(self.initial_data)
            report['initial_data_sha256'] = self.initial_provenance
            report['initial']['stored_Ih_minus_profile_integral'] = old['Ih']-I
        t = 0.
        snapshots = [(phi.copy(), H.copy())]
        total_crossings = 0
        density=int(self.parameters['Particles/Generator/Reference cell/Number of particles per cell per direction'])
        py = -.5+(np.arange(density*ny)+.5)/(density*ny)
        px = (np.arange(density*nx)+.5)*.25/(density*nx)
        particle_x = np.broadcast_to(px[:, None], (len(px), len(py))).copy()
        preceding_velocity = np.interp(py, vy, vu)
        previous_dt = None
        for k in range(1, 3 if accepted_times is None else len(accepted_times)+1):
            if accepted_times is not None:
                dt = accepted_times[k-1]['dt']
                assert abs(t+dt-accepted_times[k-1]['time']) < 1e-12
            t += dt
            H_input = H.copy()
            previous_phi = phi.copy()
            previous_I = old['Ih']
            phase_entry = float(np.linalg.norm(self.residual(phi, H)))
            phi, iterations = self.phase(H, phi)
            g, h, I = self.localization(phi)
            U = 1e-4+(self.ramp_peak-1e-4)*min(t/2, 1.)
            if accepted_times is not None:
                U = accepted_times[k-1]['U']
            response, beta, kappa = self.mechanics(old, dt, U, I)
            history = beta*old['C']/kappa*(h*previous_I/I-hp)
            crack = h/I*response['V']+history
            intact = g == 1
            assert not np.any(intact & (hp > 0)), 'Inadmissible healing'
            A = response['C']/g
            B = np.divide(beta*hp*old['C'], 1-g, out=np.zeros_like(g), where=~intact)
            candidate = dt/(2*kappa)*(A-B)*(A+B)
            equivalent = np.divide(dt*(.5*kappa*crack**2+beta*hp*old['C']*crack),
                                   (1-g)**2, out=candidate.copy(), where=~intact)
            assert np.allclose(candidate, equivalent, rtol=2e-10, atol=1e-8)
            # Production-associated support update; unassociated histories stay put.
            H = np.where(self.admitted, np.maximum(H, candidate), H)
            decay = math.exp(-response['V']*dt/self.Dc)
            theta = old['Theta']*decay+self.Dc/response['V']*(-math.expm1(-response['V']*dt/self.Dc))
            omitted = float(np.sum(self.w*h*(~self.admitted))/I)
            normalization_scale = max(abs(response['V']), self.Vref)
            full_error = abs(float(np.sum(self.w*crack))-response['V'])/normalization_scale
            supported_error = abs(float(np.sum(self.w*crack*self.admitted))-response['V'])/normalization_scale
            # RK2 runs BEFORE mechanics: its second velocity is the extrapolated
            # linearization, not this step's as-yet-unknown accepted solution.
            new_vy, new_vu = self.velocity(response, beta, kappa, old, phi, previous_phi, U)
            last_velocity = np.interp(py, vy, vu)
            predicted_velocity = (last_velocity if previous_dt is None else
                                  last_velocity+dt/previous_dt*(last_velocity-preceding_velocity))
            displacement = dt*.5*(last_velocity+predicted_velocity)
            transported = particle_x+displacement
            crossings = int(np.count_nonzero((transported < 0) | (transported >= .25)))
            particle_x = transported % .25
            preceding_velocity = last_velocity
            previous_dt = dt
            total_crossings += crossings
            row = dict(step=k, time_s=t, dt_s=dt, U=U, **response, Theta=theta,
                       phi_max=float(max(phi)), phi_change_max=float(max(abs(phi-previous_phi))),
                       Ih_relative_change=I/previous_I-1, H_max_ratio=float(max(H.flat)/max(H_input.flat)),
                       H_change_max=float(np.max(H-H_input)), H_max=float(np.max(H)), phase_entry_residual=phase_entry,
                       phase_residual_history=iterations, omitted_fraction=omitted,
                       full_slip_normalization_error=full_error,
                       supported_slip_normalization_error=supported_error,
                       supported_instantaneous_integral=float(np.sum(self.w*self.admitted*h/I*response['V'])),
                       supported_history_integral=float(np.sum(self.w*self.admitted*history)),
                       full_history_integral=float(np.sum(self.w*history)),
                       omitted_history_integral=float(np.sum(self.w*history*(~self.admitted))),
                       phi_envelope_pass=bool(max(phi)<.8), containment_pass=bool(omitted<=1e-4),
                       normalization_pass=bool(supported_error<=1e-4),
                       estimated_periodic_crossing_events=crossings,
                       maximum_particle_displacement_m=float(max(abs(displacement))))
            report['steps'].append(row)
            old = dict(q=response['q'], C=response['C'], Theta=theta, Ih=I)
            hp = h
            vy, vu = new_vy, new_vu
            dt = min(2., cfl*mesh_h/(degree*max(abs(vu))))
            snapshots.append((phi.copy(), H.copy()))
        # H0 counterfactual is the same independent initial problem, not a second
        # production run. Its residual distinguishes feedback from an iterate lag.
        report['feedback'] = dict(phi2_minus_phi0_max=float(max(abs(phi-phi0))),
            phi2_residual_with_H0=float(np.linalg.norm(self.residual(phi, self.H0))),
            total_estimated_crossing_events=total_crossings)
        report['completed_real_steps'] = len(report['steps'])
        report['smoke_budget_pass'] = len(report['steps']) == 2 and all(
            s['phi_envelope_pass'] and s['containment_pass'] and s['normalization_pass']
            for s in report['steps'])
        return report, snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameters', type=Path, required=True)
    parser.add_argument('--cells', type=int, default=2048)
    parser.add_argument('--quadrature', type=int, default=6)
    parser.add_argument('--ramp-peak', type=float, default=.009)
    parser.add_argument('--initial-data', type=Path,
                        help='Conditional reference: use only saved timestep-zero phase/particle/surface CSVs')
    parser.add_argument('--accepted-times', type=Path,
                        help='Use the contiguous time_k.csv dt/time/U prefix from ASPECT, never its evolved histories')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    model = Reference(read_parameters(args.parameters), args.cells, args.quadrature,
                      args.ramp_peak, args.initial_data)
    accepted_times = None
    if args.accepted_times is not None:
        accepted_times = []
        for k in range(1, len(list(args.accepted_times.glob('time_*.csv')))):
            path = args.accepted_times/f'time_{k}.csv'
            if not path.exists(): break
            row = np.genfromtxt(path, delimiter=',', names=True)
            assert int(row['step']) == k
            accepted_times.append({name: float(row[name]) for name in ('time', 'dt', 'U')})
    report, snapshots = model.run(accepted_times)
    if args.accepted_times is not None:
        report['accepted_time_directory'] = str(args.accepted_times)
    report['resolved_parameters'] = str(args.parameters)
    report['parameter_sha256'] = hashlib.sha256(args.parameters.read_bytes()).hexdigest()
    report['elapsed_seconds'] = time.monotonic()-start
    args.output.mkdir(parents=True, exist_ok=True)
    for k, (phi, H) in enumerate(snapshots):
        np.savetxt(args.output/f'phase-{k}.csv', np.c_[model.y, phi], delimiter=',', header='y,phi', comments='')
        np.savetxt(args.output/f'history-{k}.csv', np.c_[model.qy.ravel(), H.ravel()], delimiter=',', header='y,H', comments='')
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
