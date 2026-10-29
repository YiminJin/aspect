#!/usr/bin/env python3
"""Equilibrated cross-band reduction of the qualified tensor normal-map law.

This benchmark eliminates the 1-D mixed mechanical equations exactly. Both K
components, uniform physical shear traction and a pressure-reference scalar
remain joint unknowns. No independent prescribed-X material inverse is used.
"""
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from gradient_joint import Strip, norm_tau, old_tau
from gradient_point import project

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'build-particle-free/mature-gradient'
OUT = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/mature_gradient'
G, ETA, P0, ELL, KF = 1e6, 1e8, 10000., .046875, 8.


class CrossFault(Strip):
    def __init__(self, n, length):
        super().__init__(n, length)
        self.y = self.x - .125
        self.H = float(sum(self.mass))
        z = np.clip((np.abs(self.y) - ELL/2)/(ELL/2), 0., 1.)
        shoulder = 1 - 10*z**3 + 15*z**4 - 6*z**5
        self.phi = .02 + (.999 - .02)*shoulder
        self.h = 4*self.phi*(1+self.phi)/(1-self.phi)**2
        self.I = float(self.mass @ self.h)

    def evaluate_joint(self, old, displacement, dt, values, tangent=True):
        K = values[:2*self.n].reshape(self.n, 2)
        shear = G*values[-2]
        reference = values[-1]
        beta = 1/(1+dt*G/ETA)
        b = 2*G*beta*(1+1/self.h)
        delta = K - old['K']
        A = beta*(old['A']+delta)
        Q = 2*G*A/self.h[:, None]
        tau = np.column_stack([-2*G*A[:, 0], 2*G*A[:, 0], np.full(self.n, shear)])
        pressure = P0 + tau[:, 1] - G*reference
        T = np.zeros((self.n, 3))
        T[:, 2] = shear + 2*G*A[:, 1]
        gamma = (T[:, 2]/beta - old['T'][:, 2])/G
        X = self.Xop @ K
        F = np.column_stack([.5*(tau[:, 0]-tau[:, 1]), tau[:, 2]]) - Q - X
        Z = F + b[:, None]*delta
        trial = T.copy()
        trial[:, 0] -= 2*G*beta*old['A'][:, 0]
        trial[:, 1] += 2*G*beta*old['A'][:, 0]
        trial[:, 2] -= 2*G*beta*old['A'][:, 1]
        trial_contact = pressure - .5*(trial[:, 0]+trial[:, 1]) - np.hypot(.5*(trial[:, 0]-trial[:, 1]), trial[:, 2])
        if min(trial_contact) <= 0:
            raise RuntimeError('Elastic trial leaves physical closed contact; no pressure clipping.')
        force = np.empty_like(K)
        masks = np.empty(self.n, dtype=int)
        rows, columns, entries = [], [], []
        for i in range(self.n):
            projected, J, mask = project(Z[i], tau[i], pressure[i], jac=tangent)
            masks[i] = mask
            force[i] = F[i]-projected if mask else -b[i]*delta[i]
            if not tangent:
                continue
            P, Pt, Pp = J[:, :2], J[:, 2:5], J[:, 5]
            complement = np.eye(2)-P
            dF = np.diag([-2*beta*(1+1/self.h[i]), -2*beta/self.h[i]])
            stress_action = np.array([[-2*beta, 0.], [2*beta, 0.], [0., 0.]])
            pressure_action = np.array([2*beta, 0.])
            block = complement @ dF - Pt @ stress_action - np.outer(Pp, pressure_action) - b[i]/G*P
            blocks = {i: block}
            for j, v in zip(self.pattern[i].indices, self.pattern[i].data):
                blocks[j] = blocks.get(j, np.zeros((2, 2))) - complement*v/G
            for j, block in blocks.items():
                for a in range(2):
                    for c in range(2):
                        rows.append(2*i+a); columns.append(2*j+c); entries.append(block[a, c])
            shear_action = complement[:, 1] - Pt[:, 2]
            for a in range(2):
                rows.extend([2*i+a, 2*i+a])
                columns.extend([2*self.n, 2*self.n+1])
                entries.extend([shear_action[a], Pp[a]])
        compatibility = (self.mass @ gamma - displacement)/self.H
        gauge = reference - self.mass @ tau[:, 1]/(G*self.H)
        residual = np.r_[force.ravel()/G, compatibility, gauge]
        force_scale = np.maximum.reduce([np.ones(self.n), np.sqrt(norm_tau(T)), np.sqrt(2*np.sum(Q*Q, axis=1)),
                                         np.sqrt(2*np.sum(X*X, axis=1)), .1*pressure])
        force_norm = float(np.max(np.sqrt(2*np.sum(force*force, axis=1))/force_scale))
        normal_contact = pressure-.5*(tau[:, 0]+tau[:, 1])-np.hypot(.5*(tau[:, 0]-tau[:, 1]), tau[:, 2])
        if min(normal_contact) <= 0:
            raise RuntimeError('Returned stress leaves closed contact.')
        result = dict(T=T, A=A, Q=Q, K=K.copy(), W=delta/dt, tau=tau, X=X, F=F, pressure=pressure,
                      L=old['L']+2*np.linalg.norm(delta, axis=1)[:, None], gamma=gamma,
                      masks=masks, force=force, residual=residual, force_norm=force_norm,
                      norm=max(force_norm, abs(compatibility), abs(gauge)), shear=shear,
                      normal_traction=-P0+G*reference, compatibility=compatibility, pressure_gauge=gauge,
                      contact_min=float(min(normal_contact)), trial_contact_min=float(min(trial_contact)),
                      values=values.copy())
        if tangent:
            # These two equations are the exact displacement compatibility and
            # mean-pressure constraint, not a frozen-pressure approximation.
            for j in range(self.n):
                rows.extend([2*self.n, 2*self.n+1])
                columns.extend([2*j+1, 2*j])
                entries.extend([2*self.mass[j]/self.H, -2*beta*self.mass[j]/self.H])
            rows.extend([2*self.n, 2*self.n+1]); columns.extend([2*self.n, 2*self.n+1]); entries.extend([1/beta, 1.])
            result['J'] = sp.coo_matrix((entries, (rows, columns)), shape=(2*self.n+2, 2*self.n+2)).tocsr()
        return result

    def trial(self, old, displacement, dt, label='trial'):
        beta = 1/(1+dt*G/ETA)
        previous_tau = old_tau(old)
        previous_shear = float(self.mass @ previous_tau[:, 2]/self.H)
        values = np.r_[old['K'].ravel(), beta*(previous_shear/G+displacement/self.H),
                       2*beta*float(self.mass @ old['A'][:, 0]/self.H)]
        evaluations = 0
        for iteration in range(80):
            current = self.evaluate_joint(old, displacement, dt, values)
            evaluations += 1
            if current['norm'] <= 1e-12:
                K = values[:2*self.n].reshape(self.n, 2)
                K[current['masks'] == 0] = old['K'][current['masks'] == 0]
                check = self.evaluate_joint(old, displacement, dt, values)
                evaluations += 1
                if check['norm'] <= 1e-12:
                    check.update(iterations=iteration, evaluations=evaluations)
                    return check
                current = check
            direction = spsolve(current['J'], -current['residual'])
            for i in np.flatnonzero(current['masks'] == 0):
                direction[2*i:2*i+2] = old['K'][i]-values[2*i:2*i+2]
            for line in range(30):
                alpha = 2.**(-line)
                candidate = values + alpha*direction
                try:
                    other = self.evaluate_joint(old, displacement, dt, candidate, False)
                    evaluations += 1
                except RuntimeError:
                    continue
                if other['norm'] < current['norm']*(1-1e-4*alpha):
                    values = candidate
                    break
            else:
                np.savez(WORK/(label+'-failed.npz'), values=values, direction=direction,
                         **{'old_'+key: v for key, v in old.items()})
                raise RuntimeError(f'{label}: joint line search failed at iteration {iteration}; norm={current["norm"]}')
        np.savez(WORK/(label+'-failed.npz'), values=values, **{'old_'+key: v for key, v in old.items()})
        raise RuntimeError(f'{label}: 80-iteration bound; no publication.')

    def account(self, old, new, displacement, dt):
        tau, q = new['tau'], new['Q']
        initial_tau, initial_q = old_tau(old), old['Q']
        bulk = float(self.mass @ norm_tau(tau)/(4*G))
        cohesive = float(self.mass @ (self.h*2*np.sum(q*q, axis=1))/(4*G))
        gradient = self.gradient_energy(new['K'])
        storage = bulk + cohesive + gradient - float(self.mass @ (norm_tau(initial_tau)+self.h*2*np.sum(initial_q*initial_q, axis=1))/(4*G)) - self.gradient_energy(old['K'])
        Db = float(self.mass @ norm_tau(tau)/(2*ETA))
        Dc = float(self.mass @ (self.h*2*np.sum(q*q, axis=1))/(2*ETA))
        dissipation = 2*np.sum(new['F']*new['W'], axis=1)
        Df = float(self.mass @ dissipation)
        work = new['shear']*displacement
        loss_stress = float(self.mass @ norm_tau(tau-initial_tau)/(4*G))
        loss_cohesive = float(self.mass @ (self.h*2*np.sum((q-initial_q)**2, axis=1))/(4*G))
        loss_gradient = self.gradient_energy(new['K']-old['K'])
        error = work-storage-dt*(Db+Dc+Df)-loss_stress-loss_cohesive-loss_gradient
        gradient_power = float(self.mass @ (2*np.sum(new['X']*new['W'], axis=1)))
        gradient_error = dt*gradient_power-gradient+self.gradient_energy(old['K'])-loss_gradient
        assert abs(error) <= 1e-10 and abs(gradient_error) <= 1e-10, (error, gradient_error)
        assert Db >= 0 and Dc >= 0 and Df >= 0 and min(dissipation) >= 0, min(dissipation)
        assert abs(self.mass @ new['gamma']-displacement) <= 1e-12
        assert abs(self.mass @ (new['pressure']-P0)/self.H) <= 1e-8
        source = new['T'][:, 2]-1/(1+dt*G/ETA)*(old['T'][:, 2]+G*new['gamma'])
        source_error = float(max(abs(source)/np.maximum(1., abs(new['T'][:, 2]))))
        assert source_error <= 1e-12
        V = float(self.mass @ (2*new['W'][:, 1]))
        delta = float(self.mass @ (2*new['K'][:, 1]))
        cohesive_power = float(self.mass @ (2*np.sum(q*new['W'], axis=1)))
        return dict(bulk_J_m2=bulk, cohesive_J_m2=cohesive, gradient_J_m2=gradient,
                    storage_change_J_m2=storage, Db_W_m2=Db, Dc_W_m2=Dc, Df_W_m2=Df,
                    work_J_m2=work, BE_stress_J_m2=loss_stress, BE_cohesive_J_m2=loss_cohesive,
                    BE_gradient_J_m2=loss_gradient, ledger_error_J_m2=error, gradient_identity_error_J_m2=gradient_error,
                    delta_m=delta, V_m_s=V, Rgradient_Pa=gradient_power/V if V > 1e-12 else 0.,
                    Rcohesive_Pa=cohesive_power/V if V > 1e-12 else 0., Rfriction_Pa=Df/V if V > 1e-12 else 0.,
                    source_error=source_error)
