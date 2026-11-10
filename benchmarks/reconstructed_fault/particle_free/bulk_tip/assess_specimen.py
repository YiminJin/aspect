#!/usr/bin/env python3
"""Assess the retained specimen scales and signed fields; launch no simulations.

The dimensional lengths are screening indicators, not sharp-crack criteria.
Retained cases differ in loading and whether phase was fixed; they are not a
new controlled baseline/perturbation pair.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from analyze_time_update import frame, read, regions

ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / 'build-particle-free/bulk-tip/runs'
OUT = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/propagation_specimen'
FIXTURE = Path(__file__).with_name('localized.prm')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def parameter(name):
    matches = re.findall(r'^\s*set ' + re.escape(name) + r'\s*=\s*([^\n#]+)',
                         FIXTURE.read_text(), re.M)
    assert len(matches) == 1, (name, matches)
    return float(matches[0])


def retained_case(label, step, G, K, m):
    directory = RUNS / label / 'output'
    points = frame(directory, step)
    initial = frame(directory, 0)
    sections = read(directory / f'tip-sections-{step:04d}.csv')
    rows = read(directory / 'bulk-accepted.csv')
    rows = rows[rows.step <= step]
    command = json.loads((directory.parent / 'command.json').read_text())
    masks = regions(points.x, points.y)
    hp = m * (1 + 3 * points.phi) / (1 - points.phi)**3
    calculated_Y = hp * points.qcoh**2 / (2 * G)
    result = {
        'label': label, 'time': float(rows.time[-1]),
        'original_command': command['command'],
        'original_core_sha256': command['binary_sha256'],
        'original_checker_sha256': command.get('checker_sha256'),
        'input_sha256': sha(directory.parent / 'input.prm'),
        'fixed_phase': bool(command['arguments'].get('fixed_phase', False)),
        'phase_basis': 'Bernstein' if 'set Phase basis = Bernstein' in
            (directory.parent / 'input.prm').read_text() else 'see original input',
        'initial_fracture': float(read(directory / 'tip-initial.csv').fracture[0]),
        'max_phase_change': float(max(abs(points.phi - initial.phi))),
        'driving_identity_max_abs': float(max(abs(calculated_Y - points.Y))),
        'regions': {},
    }
    for name, mask in masks.items():
        p = points[mask]
        result['regions'][name] = {
            'max_Y_over_K': float(max(p.Y) / K),
            'max_abs_shear': float(max(abs(p.tauxy))),
            'max_abs_deviatoric_normal': float(max(max(abs(p.tauxx)), max(abs(p.tauyy)))),
            'max_abs_fault_rate': float(max(abs(p.w))),
            'fault_rate_positive_integral': float(sum(p.weight * np.maximum(p.w, 0))),
            'fault_rate_negative_integral': float(sum(p.weight * np.minimum(p.w, 0))),
        }
    absolute = float(sum(sections.ds * sections.integral_abs_w))
    net_absolute = float(sum(sections.ds * abs(sections.V)))
    result['section'] = {
        'positive_fault_rate_integral': float(sum(sections.ds * sections.positive_w)),
        'negative_fault_rate_integral': float(sum(sections.ds * sections.negative_w)),
        'absolute_fault_rate_integral': absolute,
        'absolute_section_rate_integral': net_absolute,
        'cancellation_fraction': 1 - net_absolute / absolute if absolute else 0,
        'max_abs_V': float(max(abs(sections.V))),
        'max_abs_force': float(max(abs(sections.force))),
        'max_abs_slip': float(max(abs(sections.slip))),
    }
    band_counts = []
    opposite_sign_sections = 0
    for section in np.unique(points.section):
        p = points[points.section == section]
        p = p[np.argsort(p.y)]
        threshold = max(1e-10, 1e-6 * max(abs(p.w)))
        active = abs(p.w) > threshold
        opposite_sign_sections += bool(np.any(p.w > threshold) and np.any(p.w < -threshold))
        band_counts.append(int(sum(active & ~np.r_[False, active[:-1]])))
    result['section']['opposite_sign_sections'] = opposite_sign_sections
    result['section']['multiple_band_sections'] = sum(n > 1 for n in band_counts)
    result['section']['max_bands'] = max(band_counts)
    work_positive = work_negative = 0.
    adjoint_error = 0.
    used = [directory.parent / 'input.prm', directory.parent / 'command.json',
            directory / 'bulk-accepted.csv', directory / 'tip-initial.csv']
    used += sorted(directory.glob('tip-points-0000-r*.csv'))
    for row in rows:
        n = int(row.step)
        p = frame(directory, n)
        section_path = directory / f'tip-sections-{n:04d}.csv'
        s = read(section_path)
        work = row.dt * p.weight * p.w * s.force[p.section.astype(int)]
        work_positive += float(sum(np.maximum(work, 0)))
        work_negative += float(sum(np.minimum(work, 0)))
        adjoint_error = max(adjoint_error, abs(float(sum(work)) - row.dt * sum(s.ds * s.force * s.V)))
        used += sorted(directory.glob(f'tip-points-{n:04d}-r*.csv')) + [section_path]
    remainder = rows.Wext - rows.energy_change - rows.Db - rows.Dc - rows.Df
    result['work'] = {
        'positive_local_friction_radiation': work_positive,
        'negative_local_friction_radiation': work_negative,
        'section_friction_radiation': float(sum(rows.Df)),
        'max_adjoint_defect': float(adjoint_error),
        'physical_remainder_signed': float(sum(remainder)),
        'physical_remainder_absolute': float(sum(abs(remainder))),
        'stress_loss': float(sum(rows.stress_loss)),
        'added_fracture': float(rows.fracture[-1] - result['initial_fracture']),
        'max_ledger_error': float(max(abs(rows.ledger_error))),
    }
    result['data_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(used))}
    return result


def main():
    G = parameter('Elastic shear moduli')
    eta = parameter('Reference viscosities')
    Gc = parameter('Critical energy release rates')
    strength = parameter('Cohesions')
    ell = parameter('Length scale')
    width, height = parameter('X extent'), parameter('Y extent')
    K = Gc / ((8 / 3) * ell)
    m = 2 * G * K / strength**2
    # These geometry/loading values are the declared localized.prm seed and rate.
    assert '.8*max(0,1-((x-2)/.8)^2)*max(0,1-((y-1)/.2)^2)' in FIXTURE.read_text()
    assert '.01*(y-1);0' in FIXTURE.read_text()
    a, peak, rate, time = .8, .8, .01, .08
    pN, mu = parameter('Surface pressure'), parameter('Reference friction coefficients')
    ref_rate = parameter('Reference slip rate')
    age = parameter('Bulk tip initial age')
    Dc = parameter('Characteristic slip distance')
    rsf_a, rsf_b = parameter('Direct effect parameters'), parameter('Evolution effect parameters')
    radiation = parameter('Radiation damping coefficients')
    force = pN * rsf_a * np.arcsinh(ref_rate / (2 * ref_rate) *
        np.exp((mu + rsf_b * np.log(age * ref_rate / Dc)) / rsf_a)) + radiation * ref_rate
    scales = {
        'G': G, 'eta': eta, 'Gc': Gc, 'strength_calibration': strength,
        'ell': ell, 'K': K, 'm': m, 'seed_half_length': a, 'seed_peak': peak,
        'side_ligament': width / 2 - a, 'normal_center_to_wall': height / 2,
        'a_over_ell': a / ell, 'side_ligament_over_ell': (width / 2 - a) / ell,
        'normal_distance_over_ell': height / (2 * ell),
        'maxwell_time': eta / G,
        'homogeneous_tau_at_008': eta * rate * (-np.expm1(-time * G / eta)),
        'intact_density_at_calibration': strength**2 / (2 * G),
        'energy_strength_length': G * Gc / strength**2,
        'a_over_energy_strength_length': a * strength**2 / (G * Gc),
        'reference_force_at_initial_age_and_V0': float(force),
        'net_stress_proxy': strength - float(force),
        'friction_adjusted_length_proxy': G * Gc / (strength - force)**2,
        'homogeneous_elastic_energy_at_calibration': width * height * strength**2 / (2 * G),
        'fully_developed_one_cell_each_tip_charge': 2 * .25 * Gc,
        'transverse_equilibrium_halfwidth_at_peak': 2 * ell * np.sqrt(peak),
        'transverse_equilibrium_fracture_per_length_at_peak': Gc * peak**1.5,
        'Gc_for_a_equal_length_no_friction_proxy': a * strength**2 / G,
        'Gc_for_a_equal_length_reference_friction_proxy': float(a * (strength - force)**2 / G),
    }
    scales['driving_factor_Y_over_K_per_squared_force_over_strength'] = {
        str(phi): (1 + 3 * phi) * (1 - phi) /
            (m * phi * (1 + phi) + (1 - phi)**2)**2
        for phi in [0., .01, .1, .5, .8]
    }
    # Independent integration of the one-dimensional AT1 equilibrium profile.
    z, weights = np.polynomial.legendre.leggauss(3)
    support = 2 * ell * np.sqrt(peak)
    n = .5 * support * (1 + z)
    phi = (np.sqrt(peak) - n / (2 * ell))**2
    gradient = -(np.sqrt(peak) - n / (2 * ell)) / ell
    numerical_energy = support * sum(weights * K * (phi + ell**2 * gradient**2))
    scales['AT1_profile_energy_quadrature_error'] = float(abs(numerical_energy - Gc * peak**1.5))
    assert scales['AT1_profile_energy_quadrature_error'] < 1e-12
    cases = [retained_case(label, step, G, K, m) for label, step in [
        ('bernstein-localized01', 16), ('tip-time-screen-wide-seed01', 4),
        ('tip-time-screen-seed01', 4)]]
    assert all(c['driving_identity_max_abs'] < 1e-10 for c in cases)
    assert all(c['work']['max_adjoint_defect'] < 1e-12 for c in cases)
    assert all(c['max_phase_change'] == 0 for c in cases if c['fixed_phase'])
    result = {
        'scope': 'Algebra and retained evidence only. No new coupled runs or parameter changes.',
        'fixture': str(FIXTURE.relative_to(ROOT)), 'fixture_sha256': sha(FIXTURE),
        'scales': scales, 'retained_cases': cases,
        'decision': 'Reject the present 4x2 m, a=.8 m, Gc=26.6667 specimen for the next propagation run. '
                    'The energy/strength length must be reduced relative to seed and body dimensions '
                    '(or those dimensions increased). No new baseline/perturbation pair passes the scaling gate.',
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'assessment.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'scales': scales, 'cases': [
        {k: c[k] for k in ['label', 'time', 'fixed_phase', 'max_phase_change', 'regions', 'section', 'work']}
        for c in cases]}, indent=2))


if __name__ == '__main__':
    main()
