#!/usr/bin/env python3
"""Bounded mature-band sliding: independent mechanics/energy checks and data."""
import csv
import hashlib
import json
import time
import numpy as np
import mpmath as mp
from mature_gradient_strip import CrossFault, WORK, OUT, G, ETA, P0, ELL, KF


def write(name, rows):
    with (OUT/name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def local_reference(H, I, old_shear, old_q, displacement, dt):
    """50-digit homogeneous-traction scalar elimination, no normal-map call."""
    with mp.workdps(50):
        H, I, s0, q0, dU, dt = [mp.mpf(float(x)) for x in [H, I, old_shear, old_q, displacement, dt]]
        beta = 1/(1+dt*mp.mpf(G)/ETA)
        mu, R = mp.mpf('.1'), mp.mpf(1000)
        trial = beta*(s0+G*dU/H)
        if trial-beta*q0 <= mp.sqrt(R*R-mu*mu*trial*trial):
            return float(trial), float(beta*q0)
        D = beta*(H*s0+I*q0+G*dU)
        a = H+I; A = a*a+I*I*mu*mu
        shear = (a*D+I*mp.sqrt(A*R*R-mu*mu*D*D))/A
        q = shear-mp.sqrt(R*R-mu*mu*shear*shear)
        return float(shear), float(q)


def action_check(mesh, old, new, displacement, dt):
    direction = np.zeros(2*mesh.n+2)
    direction[0:-2:2] = 5e-6*np.sin(4*np.pi*mesh.y/.25)
    direction[1:-2:2] = 4e-6*np.cos(2*np.pi*mesh.y/.25)
    direction[-2] = 10/G; direction[-1] = 3/G
    base = mesh.evaluate_joint(old, displacement, dt, new['values'])
    action = base['J'] @ direction
    epsilon = 1e-5
    for reductions in range(13):
        a = mesh.evaluate_joint(old, displacement, dt, new['values']+epsilon*direction, False)
        b = mesh.evaluate_joint(old, displacement, dt, new['values']-epsilon*direction, False)
        if np.array_equal(base['masks'], a['masks']) and np.array_equal(base['masks'], b['masks']):
            break
        epsilon *= .5
    else:
        raise RuntimeError('No smooth same-branch derivative probe after 12 halvings.')
    error = float(np.linalg.norm((a['residual']-b['residual'])/(2*epsilon)-action)/np.linalg.norm(action))
    assert error <= 1e-6, error
    return dict(relative_action_error=error, epsilon=epsilon, branch_reductions=reductions,
                includes='Both K components, mechanical shear and solved pressure reference, physical pressure and changing conjugate directions.')


def trajectory(n, length, dt, case):
    mesh = CrossFault(n, length)
    old = mesh.initial()
    ref_shear = ref_q = 0.
    start = time.monotonic()
    history, profiles = [], []
    max_reference_error = 0.
    checks = {}
    velocity = 1e-3; displacement = velocity*dt
    for step in range(1, round(2/dt)+1):
        digest = hashlib.sha256(b''.join(v.tobytes() for v in old.values())).hexdigest()
        new = mesh.trial(old, displacement, dt, label=case)
        assert digest == hashlib.sha256(b''.join(v.tobytes() for v in old.values())).hexdigest()
        energy = mesh.account(old, new, displacement, dt)
        if length == 0:
            ref_shear, ref_q = local_reference(mesh.H, mesh.I, ref_shear, ref_q, displacement, dt)
            max_reference_error = max(max_reference_error, abs(new['shear']-ref_shear)/1000.,
                                      float(max(abs(new['Q'][:, 1]-ref_q)))/1000.)
            assert max_reference_error <= 1e-6
        rate = 2*new['W'][:, 1]
        equiv = 2*np.linalg.norm(new['W'], axis=1)
        k = 2*new['K'][:, 1]
        signed = float(mesh.mass @ k)
        width = float((mesh.mass @ equiv)**2/(mesh.mass @ (equiv*equiv))) if mesh.mass @ (equiv*equiv) > 0 else 0.
        accumulated_width = float(signed*signed/(mesh.mass @ (k*k))) if signed > 0 else 0.
        support = np.abs(mesh.y) <= ELL
        spill = float((mesh.mass[~support] @ equiv[~support])/(mesh.mass @ equiv)) if mesh.mass @ equiv > 1e-12 else 0.
        power_error = new['shear']-energy['Rcohesive_Pa']-energy['Rgradient_Pa']-energy['Rfriction_Pa'] if energy['V_m_s'] > 1e-12 else 0.
        assert abs(power_error) <= 1e-7, power_error
        row = dict(case=case, N=n, ell_s_m=length, dt_s=dt, step=step, time_s=step*dt,
                   imposed_displacement_m=step*displacement, imposed_velocity_m_s=velocity,
                   traction_Pa=new['shear'], normal_traction_Pa=new['normal_traction'],
                   pressure_perturbation_max_Pa=float(max(abs(new['pressure']-P0))),
                   force_error=new['force_norm'], compatibility_error_m=abs(mesh.H*new['compatibility']),
                   physical_contact_min_Pa=new['contact_min'], iterations=new['iterations'],
                   rate_participation_width_m=width, K_participation_width_m=accumulated_width,
                   outside_initial_support_rate_fraction=spill, max_K_engineering=float(max(abs(k))),
                   cumulative_absolute_slip_m=float(mesh.mass @ new['L'][:, 0]),
                   traction_partition_error_Pa=power_error, **energy)
        history.append(row)
        if any(abs(step*dt-t) < 1e-12 for t in [.25, .5, 1., 1.5, 2.]):
            for i in range(n):
                profiles.append(dict(case=case, time_s=step*dt, y_m=mesh.y[i], weight_m=mesh.mass[i],
                                     phi=mesh.phi[i], h=mesh.h[i], k_engineering=k[i], w_engineering_s1=rate[i],
                                     Kxx=new['K'][i, 0], Kxy=new['K'][i, 1], Axx=new['A'][i, 0], Axy=new['A'][i, 1],
                                     Qxx_Pa=new['Q'][i, 0], Qxy_Pa=new['Q'][i, 1], Xxx_Pa=new['X'][i, 0], Xxy_Pa=new['X'][i, 1],
                                     tauxx_Pa=new['tau'][i, 0], tauyy_Pa=new['tau'][i, 1], tauxy_Pa=new['tau'][i, 2],
                                     pprime_Pa=new['pressure'][i]-P0, mask=int(new['masks'][i])))
        if abs(step*dt-.5) < 1e-12 and n == 257 and length == .006 and dt == .01:
            checks['complete_joint_action'] = action_check(mesh, old, new, displacement, dt)
            retry = mesh.trial(old, displacement, dt, label=case+'-retry')
            assert all(np.array_equal(new[k], retry[k]) for k in old)
            checks['immutable_trial_and_retry'] = 'Bitwise on T,A,K,Q,L from the same accepted parent; benchmark only.'
        old = {key: new[key].copy() for key in old}
        (WORK/'progress.json').write_text(json.dumps(dict(case=case, step=step, time_s=step*dt, **energy))+'\n')
    # Initial representation is independent of the subsequent mechanical path.
    points = np.linspace(-.125, .125, 8193)
    z = np.clip((abs(points)-ELL/2)/(ELL/2), 0, 1)
    exact_phase = .02+(.999-.02)*(1-10*z**3+15*z**4-6*z**5)
    error = float(np.sqrt(np.trapezoid((np.interp(points, mesh.y, mesh.phi)-exact_phase)**2, points)/np.trapezoid(exact_phase**2, points)))
    fracture = KF*(mesh.mass @ mesh.phi + ELL*ELL*mesh.phi @ (mesh.L @ mesh.phi))
    return history, profiles, dict(case=case, N=n, length_m=length, dt_s=dt, seconds=time.monotonic()-start,
                                  steps=len(history), I_m=mesh.I, initial_phase_relative_L2=error,
                                  fixed_fracture_storage_J_m2=float(fracture), local_analytic_max_scaled_error=max_reference_error,
                                  **checks)


if __name__ == '__main__':
    histories, profiles, results = [], [], []
    cases = [(257, 0., .01, 'local'), (257, .001, .01, 'ls1'), (257, .003, .01, 'ls3'),
             (257, .006, .01, 'ls6'), (129, .006, .01, 'ls6-n129'),
             (513, .006, .01, 'ls6-n513'), (257, .006, .005, 'ls6-half')]
    for n, length, dt, case in cases:
        h, p, r = trajectory(n, length, dt, case)
        histories += h; profiles += p; results.append(r)
        write('history.csv', histories); write('profiles.csv', profiles)
        (OUT/'checks.json').write_text(json.dumps(results, indent=2)+'\n')
        print(json.dumps(r), flush=True)
