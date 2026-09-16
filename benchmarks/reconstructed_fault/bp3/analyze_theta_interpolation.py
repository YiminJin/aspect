"""Saved-data aging/interpolation commutator, with the split clock preserved.

No solve, history replacement, or production change. Complete alternate weak
loads cannot be recovered exactly from saved first/second moments. The full
load estimates below are explicitly reconstructed; selected raw samples are
evaluated exactly and never used as a complete quadrature rule.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded
from numpy.polynomial.legendre import leggauss

HERE = Path(__file__).resolve().parent
OUT = HERE / 'theta-interpolation-audit'
CASES = {'100m': HERE / 'junction-matched-qualified-local4/refined',
         '50m': HERE / 'fault-grid-50-local4'}
DC, V0, A, B, MU0 = .008, 1e-6, .025, .015, .6


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def column(rows, key):
    return np.array([float(r[key]) for r in rows])


def interpolate(values, j, xi):
    return (1-xi)*values[j] + xi*values[j+1]


def velocity(values, j, xi):
    # Match the surface assembler's ordinary affine evaluation and its exact
    # endpoint branches. A nodal lower bound is not imposed at interior QPs.
    return np.where(xi == 0, values[j], np.where(xi == 1, values[j+1],
                    values[j] + xi*(values[j+1]-values[j])))


def update(theta, v, dt):
    x = v*dt/DC
    return theta*np.exp(-x) - (DC/v)*np.expm1(-x)


def mu(v, theta):
    return A*np.arcsinh(v/(2*V0)*np.exp((MU0+B*np.log(theta*V0/DC))/A))


def band(diagonal, upper):
    result = np.zeros((3, len(diagonal)))
    result[1] = diagonal
    result[0, 1:] = result[2, :-1] = upper
    return result


def multiply(matrix, x):
    result = matrix[1]*x
    result[:-1] += matrix[0, 1:]*x[1:]
    result[1:] += matrix[2, :-1]*x[:-1]
    return result


def weak_moments(root, k):
    parts = [read(root/f'stress_weak_moments_{k}_rank{r}.csv') for r in range(4)]
    return {key:sum(column(p, key) for p in parts)
            for key in ['weight', 'sigma_load', 'friction_load']}


def write(name, rows):
    with (OUT/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    summary = {'timing': 'commit k versus friction in accepted mechanics k+1; no same-step implicit update',
               'full_alternative_weak_load_is_estimated': True, 'cases': []}
    profiles, samples, loads, element_loads = [], [], [], []
    source_files = [Path(__file__)]
    for label, root in CASES.items():
        parameters = (root/'parameters.prm').read_text()
        for key, value in [('Direct effect parameters', '0.010, 0.025'),
                           ('Evolution effect parameters', '0.015'),
                           ('Characteristic slip distance', '0.008'),
                           ('Reference friction coefficients', '0.6'),
                           ('Reference slip rate', '1e-6'),
                           ('Use regularized formulation', 'true')]:
            assert any(line.split('=')[1].split('#')[0].strip() == value
                       for line in parameters.splitlines() if line.strip().startswith('set '+key+' '))
        for k in [4, 12]:
            old, current, following = [read(root/f'fault_{step}.csv') for step in [k-1, k, k+1]]
            x = column(current, 'xd')
            junction = int(np.argmin(abs(x-40000)))
            assert abs(x[junction]-40000) < 1e-7
            assert current[junction]['prescribed'] == '1'
            assert all(current[i]['prescribed'] == '0' for i in [junction+1, junction+2])
            segments = [junction, junction+1]  # 40 -> last free -> preceding free, stored up-dip.
            dt = float(current[0]['dt'])
            old_theta, new_theta = column(old, 'Theta'), column(current, 'Theta')
            vk, vn = column(current, 'V'), column(following, 'V')
            np.testing.assert_allclose(new_theta, update(old_theta, vk, dt), rtol=1e-12, atol=0)
            for key in ['x', 'y']:
                np.testing.assert_array_equal(column(old, key), column(following, key))

            def states(j, xi):
                theta_a = interpolate(new_theta, j, xi)
                theta_b = update(interpolate(old_theta, j, xi), velocity(vk, j, xi), dt)
                return theta_a, theta_b

            item = dict(case=label, update_step=k, friction_step=k+1,
                        update_time=float(current[0]['time']), dt=dt,
                        friction_time=float(following[0]['time']), segments=segments,
                        preceding_nodes=[dict(xd=float(x[i]), V_k=float(vk[i]), V_next=float(vn[i]),
                          Theta_old=float(old_theta[i]), Theta_committed=float(new_theta[i]))
                          for i in range(junction, junction+3)], segment_results=[])
            for j in segments:
                xi = np.linspace(0, 1, 1001)
                ta, tb = states(j, xi)
                np.testing.assert_allclose(ta[[0, -1]], tb[[0, -1]], rtol=1e-12, atol=0)
                imax = int(np.argmax(ta/tb))
                mid_a, mid_b = states(j, .5)
                item['segment_results'].append(dict(segment=j, xd_down=float(x[j]), xd_up=float(x[j+1]),
                    midpoint_A=float(mid_a), midpoint_B=float(mid_b), midpoint_ratio=float(mid_a/mid_b),
                    maximum_ratio=float((ta/tb)[imax]), maximum_ratio_xd=float(interpolate(x, j, xi[imax])),
                    minimum_ratio=float(min(ta/tb))))
                for z, a, b in zip(xi[::10], ta[::10], tb[::10]):
                    v = float(velocity(vn, j, z))
                    profiles.append(dict(case=label, update_step=k, friction_step=k+1, segment=j, xi=z,
                        xd=float(interpolate(x, j, z)), V_k=float(velocity(vk, j, z)), V_next=v,
                        Theta_A=a, Theta_B=b, ratio=a/b,
                        mu_A=float(mu(v, a)), mu_B=float(mu(v, b))))

            # Exact pointwise check at the saved production sample locations.
            # These extrema-selected samples have tiny/nonrepresentative weights.
            seen = set()
            errors = []
            premature_errors, lagged_errors = [], []
            for rank in range(4):
                path = root/f'stress_samples_{k+1}_rank{rank}.csv'
                source_files.append(path)
                for r in read(path):
                    j = int(r['segment'])
                    identity = (rank, r['particle'], r['domain_q'])
                    if j not in segments or identity in seen:
                        continue
                    seen.add(identity)
                    xi = float(r['xi']); ta, tb = states(j, xi)
                    v = float(velocity(vn, j, xi))
                    np.testing.assert_allclose(v, float(r['V']), rtol=1e-12, atol=1e-25)
                    ma, mb = float(mu(v, ta)), float(mu(v, tb))
                    errors.append(abs(ma-float(r['mu'])))
                    premature_errors.append(abs(float(mu(v, interpolate(column(following, 'Theta'), j, xi)))-float(r['mu'])))
                    lagged_errors.append(abs(float(mu(v, interpolate(old_theta, j, xi)))-float(r['mu'])))
                    sigma, weight = float(r['sigma_n']), float(r['weight'])
                    delta = sigma*(mb-ma)
                    samples.append(dict(case=label, update_step=k, friction_step=k+1,
                        rank=rank, particle=r['particle'], domain_q=r['domain_q'], segment=j, xi=xi,
                        xd=float(interpolate(x, j, xi)), weight=weight, sigma_n=sigma, V_next=v,
                        Theta_A=float(ta), Theta_B=float(tb), mu_saved=float(r['mu']), mu_A=ma, mu_B=mb,
                        friction_A=sigma*ma, friction_B=sigma*mb,
                        delta_friction=delta, delta_weak_left=weight*(1-xi)*delta,
                        delta_weak_right=weight*xi*delta))
            assert errors and max(errors) < 2e-12
            item['selected_sample_count'] = len(errors)
            item['maximum_saved_mu_error'] = max(errors)
            item['maximum_mu_error_using_newly_committed_next_state'] = max(premature_errors)
            item['maximum_mu_error_using_extra_lagged_state'] = max(lagged_errors)

            # Reconstruct a signed line density of normal load from ALL saved
            # weak moments. This reproduces the Q1 normal weak loads exactly,
            # but not their unsaved within-element distribution.
            moments = weak_moments(root, k+1)
            length = x[:-1]-x[1:]
            diagonal = np.zeros(len(x))
            diagonal[:-1] += length/3; diagonal[1:] += length/3
            line_mass = band(diagonal, length/6)
            surface_mass = band(column(following, 'mass_diagonal'), column(following, 'mass_upper')[:-1])
            np.testing.assert_allclose(multiply(surface_mass, np.ones(len(x))), moments['weight'], rtol=1e-12)
            rho = solve_banded((1, 1), line_mass, moments['weight'])
            normal_density = solve_banded((1, 1), line_mass, moments['sigma_load'])
            sigma_q1 = solve_banded((1, 1), surface_mass, moments['sigma_load'])
            np.testing.assert_allclose(multiply(line_mass, normal_density), moments['sigma_load'], rtol=1e-12)
            methods = ['normal_load_Q1', 'density_times_projected_sigma', 'constant_50MPa']
            delta_by_order = {}
            for order in [64, 128, 256]:
                z, w = leggauss(order); z=(z+1)/2; w=w/2
                delta_by_order[order] = {}
                for method in methods:
                    baseline = np.zeros(len(x)); delta = np.zeros(len(x))
                    mass_diag = np.zeros(len(x)); mass_upper = np.zeros(len(x)-1)
                    for j in range(junction-1, junction+3):
                        density = interpolate(rho, j, z)
                        if method == 'normal_load_Q1':
                            normal = interpolate(normal_density, j, z)
                        elif method == 'density_times_projected_sigma':
                            normal = density*interpolate(sigma_q1, j, z)
                        else:
                            normal = density*50e6
                        ta = interpolate(new_theta, j, z)
                        v = velocity(vn, j, z)
                        ma = mu(v, ta)
                        change = np.zeros(len(z))
                        if j in segments:
                            _, tb = states(j, z)
                            change = mu(v, tb)-ma
                        for side, shape in enumerate([1-z, z]):
                            baseline[j+side] += length[j]*sum(w*shape*normal*ma)
                            element_delta = length[j]*sum(w*shape*normal*change)
                            delta[j+side] += element_delta
                            mass_diag[j+side] += length[j]*sum(w*shape*shape*density)
                            if order == 256 and j in segments:
                                element_loads.append(dict(case=label, update_step=k, friction_step=k+1,
                                    method=method, segment=j, xd_down=x[j], xd_up=x[j+1], node=j+side,
                                    last_unprescribed=int(j+side==junction+1),
                                    estimated_delta_weak_friction=element_delta,
                                    estimated_delta_friction_per_full_node_weight=element_delta/moments['weight'][j+side]))
                        mass_upper[j] += length[j]*sum(w*z*(1-z)*density)
                    delta_by_order[order][method] = delta
                    if order == 256:
                        item.setdefault('reconstruction_checks', {})[method] = dict(
                            mass_diagonal_max_relative_error=float(max(abs(mass_diag[i]/surface_mass[1,i]-1)
                              for i in range(junction, junction+3))),
                            mass_offdiagonal_max_relative_error=float(max(abs(mass_upper[j]/surface_mass[0,j+1]-1)
                              for j in segments)))
                        for i in range(junction, junction+3):
                            weight = moments['weight'][i]
                            actual = moments['friction_load'][i]/weight
                            predicted = baseline[i]/weight
                            change = delta[i]/weight
                            loads.append(dict(case=label, update_step=k, friction_step=k+1, method=method,
                                node=i, xd=x[i], last_unprescribed=int(i==junction+1),
                                weight=weight, actual_friction_density=actual,
                                reconstructed_baseline_density=predicted, baseline_difference=predicted-actual,
                                estimated_delta_friction_density=change,
                                estimated_alternative_density=actual+change,
                                estimated_delta_R_density=-change,
                                estimated_delta_weak_friction=delta[i]))
            item['delta_quadrature_128_to_256_max_Pa'] = {
                method: float(max(abs((delta_by_order[256][method]-delta_by_order[128][method]) /
                                     moments['weight']))) for method in methods}
            summary['cases'].append(item)
            source_files += [root/f'fault_{s}.csv' for s in [k-1, k, k+1]]
            source_files += [root/f'stress_weak_moments_{k+1}_rank{rank}.csv' for rank in range(4)]
            source_files.append(root/'parameters.prm')

    # Uniform fields commute, and exact endpoint contact must remain valid.
    dt=4e8; old=np.array([8e6, 8e6]); v=np.array([1e-20, 1e-20]); xi=np.linspace(0,1,101)
    np.testing.assert_allclose(interpolate(update(old,v,dt),0,xi),
                              update(interpolate(old,0,xi),velocity(v,0,xi),dt),rtol=1e-14)
    summary['uniform_and_contact_check'] = 'passed'
    summary['sha256'] = {str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(set(source_files))}
    write('theta_profiles.csv', profiles)
    write('selected_production_samples.csv', samples)
    write('weak_load_estimates.csv', loads)
    write('element_weak_load_estimates.csv', element_loads)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='sha256'}, indent=2))


if __name__ == '__main__':
    main()
