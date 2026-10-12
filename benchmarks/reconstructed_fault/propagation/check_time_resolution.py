#!/usr/bin/env python3
"""Read-only numerical/preservation checks; accuracy failure is a valid outcome."""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from time_resolution import E, BASE, directory, load, Reference, G

ROOT = Path(__file__).resolve().parents[3]
DOCS = {
    'AGENTS.md', 'doc/reconstructed_fault/current_design.md',
    'doc/reconstructed_fault/CURRENT_STATUS.md',
    'doc/reconstructed_fault/propagation/STATUS.md',
    'doc/reconstructed_fault/propagation/PLAN.md',
    'benchmarks/reconstructed_fault/propagation/README.md',
}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    baseline = load(E / 'baseline.json')
    preserved = 0
    for name, value in load(E / 'incoming-normal-manifest.json')['hashes'].items():
        if name not in DOCS:
            assert sha(ROOT / name) == value, 'incoming artifact changed: ' + name
            preserved += 1
    prototype = 'benchmarks/reconstructed_fault/propagation/localization_1d.py'
    for name, value in baseline['hashes'].items():
        if name != prototype:
            assert sha(ROOT / name) == value, 'production/build changed: ' + name
    assert sha(E / 'localization_1d.py.before') == baseline['hashes'][prototype]
    diagnostic = ('        # Diagnostic snapshots of the actual native-initialization samples.\n'
                  '        self.initial_particle_y=pc.copy();self.initial_particle_H=H.copy()\n')
    assert (ROOT / prototype).read_text().replace(diagnostic, '') == (E / 'localization_1d.py.before').read_text()
    for name, value in load(E / 'manifest.json')['hashes'].items():
        assert sha(ROOT / name) == value, 'current artifact changed: ' + name
    for command, expected in [(['git', 'rev-parse', 'HEAD'], baseline['head']),
                              (['git', 'branch', '--show-current'], baseline['branch'])]:
        assert subprocess.check_output(command, cwd=ROOT, text=True).strip() == expected

    initial = np.load(E / 'baseline-diagnostics/initial.npz')
    assert np.count_nonzero(initial['r']) == 0 and initial['Z'] == initial['slip'] == 0
    assert len(initial['H']) == len(initial['particle_y']) == 768
    r = Reference(256, 1., 6)
    assert np.array_equal(r.initial, initial['phi'])
    assert np.array_equal(r.initial_particle_H, initial['H'])
    assert np.array_equal(r.initial_particle_y, initial['particle_y'])
    comparison = load(E / 'comparison.json')
    audit = load(E / 'audit.json')
    partitions = [64, 128, 256, 512, 1024]
    assert comparison['partitions'] == partitions and comparison['reference_denominator'] == 1024
    assert comparison['selected_denominator'] is None
    maximum_kkt = 0.
    maximum_healing = 0.
    for d in partitions:
        path = directory(d)
        f = np.load(path / 'fields.npz')
        hist = load(path / 'history.json')
        diagnostics = E / 'baseline-diagnostics' if d == 64 else path
        energy = load(diagnostics / 'energy.json')
        snapshot = np.load(diagnostics / 'initial.npz')
        assert all(np.array_equal(snapshot[k], initial[k]) for k in initial.files)
        assert len(hist) == len(energy) == d * 5 // 8 and hist[-1]['time'] == .625
        assert np.array_equal(f['initial'], initial['phi'])
        phi = np.vstack([f['initial'], f['phase']])
        R = np.vstack([f['incoming_r'], f['final_r']])
        Z = np.r_[f['incoming_Z'], f['final_Z']]
        assert np.count_nonzero(R[0]) == 0 and Z[0] == 0
        assert np.min(phi) >= 0 and np.max(phi) < 1
        maximum_healing = max(maximum_healing, float(np.max(-np.diff(phi, axis=0))))
        assert maximum_healing <= 4.922e-12
        assert abs(sum(.256 * (j - .5) / d**2 for j in range(1, len(hist) + 1)) - .05) < 1e-14
        old_fracture = r.fracture(phi[0])
        for j, (row, term) in enumerate(zip(hist, energy), 1):
            assert row['time'] == j / d and row['I'] > 0
            assert row['force'] < 1e-7 and abs(row['stick_slip']) < 1e-7
            assert abs(row['slip'] - (0. if j == 1 else hist[j-2]['slip']) - row['V'] / d) < 1e-14
            assert abs(Z[j] / row['I'] - row['q']) < 1e-9
            # Reevaluate the actual total-energy phase KKT signs at the incoming
            # accepted anchor, with the immutable previous-phase lower bound.
            r.r = R[j-1].copy()
            r.Z = Z[j-1]
            gradient = r.objective(phi[j])[1]
            defect = r.defect(gradient, phi[j], phi[j-1])
            maximum_kkt = max(maximum_kkt, float(defect))
            assert defect <= 1e-11 and abs(defect - row['KKT']) < 1e-13
            assert abs(term['fracture_change'] - term['fracture'] + old_fracture) < 1e-12
            old_fracture = term['fracture']
            expenditure = sum(term[k] for k in ['bulk_change', 'cohesive_change', 'fracture_change',
                                               'D_relax', 'bulk_viscous', 'cohesive_viscous', 'friction'])
            for measure in ['bulk', 'boundary']:
                assert abs(term[measure + '_work'] - expenditure - term[measure + '_work_error']) < 1e-12
            assert abs(term['bulk_work_error'] - row['mechanical_error']) < 1e-10
            assert abs(term['bulk_viscous'] + term['cohesive_viscous'] + term['friction'] - row['loss']) < 1e-10
        totals = load(diagnostics / 'totals.json') if d == 64 else load(path / 'summary.json')['energy_totals']
        for key, value in totals.items():
            assert abs(value['signed'] - sum(v[key] for v in energy)) < 1e-12
            assert abs(value['absolute'] - sum(abs(v[key]) for v in energy)) < 1e-12
            assert abs(value['maximum'] - max(abs(v[key]) for v in energy)) < 1e-12
        assert np.isclose(np.sum([v['fracture_change'] for v in energy]),
                          energy[-1]['fracture'] - r.fracture(phi[0]), atol=1e-12, rtol=0)
        proof = audit[str(d)]
        assert proof['maximum_prediction_error'] < 1e-10
        assert proof['coefficient_recurrence_error'] == proof['mechanical_replay_R_error'] == proof['mechanical_replay_Z_error'] == 0
        assert proof['direct_normalization_error'] < 1e-12
        for check in proof['selected_incoming_checks']:
            assert check['incoming_time'] == check['time'] - 1/d
            assert check['bulk_energy_6_24'] < 1e-11 and check['bulk_energy_24_48'] < 1e-11
            assert check['gradient_error_6_24'] < 1e-11 and check['Hessian_relative_error_6_24'] < 1e-11
            if check['free_nodes']:
                assert check['minimum_free_curvature'] > 0
        # Common rates are independent reconstructions from cumulative slip,
        # with exactly forty equal physical intervals in every path.
        slips = np.r_[0., [v['slip'] for v in hist]][np.arange(41) * (d//64)]
        rates = np.diff(slips)*64
        ref_hist = load(directory(1024) / 'history.json')
        ref_slips = np.r_[0., [v['slip'] for v in ref_hist]][np.arange(41)*16]
        ref_rates = np.diff(ref_slips)*64
        rate_error = max(abs(rates-ref_rates))/max(abs(ref_rates))
        assert abs(rate_error-comparison['errors'][str(d)]['errors']['V_common_relative']) < 1e-13

    last = comparison['errors']['512']['errors']
    assert last['phi'] > baseline['targets']['phi'] and last['q_Pa'] > baseline['targets']['q_Pa']
    assert not any(row['uncertainty_inclusive_pass'] for row in comparison['errors'].values())
    assert comparison['finest_reference_uncertainty']['phi']['estimated_finest_error'] > baseline['targets']['phi']
    assert comparison['finest_reference_uncertainty']['q_Pa']['estimated_finest_error'] > baseline['targets']['q_Pa']
    frozen_errors = []
    for d in [64, 256, 512]:
        summary = load(E / f'frozen-dt-1-{d}/summary.json')
        assert summary['maximum_phase_increment'] == 0 and summary['frozen_phase']
        energy = summary['energy_totals']
        assert energy['D_relax']['absolute'] < 1e-10
        assert abs(energy['bulk_work']['signed']-energy['boundary_work']['signed']) < 1e-9
        frozen_errors.append(energy['bulk_work_error']['absolute'])
    assert frozen_errors[0] > frozen_errors[1] > frozen_errors[2]
    spatial = audit['separate_spatial_comparison']
    assert spatial['phi'] < 1e-5 and spatial['q_Pa'] < .01 and spatial['slip'] < 1e-8
    assert spatial['bulk_storage'] < .001
    result = dict(preserved_incoming_artifacts=preserved, partitions=partitions,
                  identical_initial_fields=True, total_boundary_displacement_m=.05,
                  maximum_normalized_phase_KKT=maximum_kkt,
                  maximum_phase_healing=maximum_healing,
                  prototype_accepted_state_replay_and_Maxwell_recurrence_pass=True,
                  normalization_and_selected_incoming_derivatives_pass=True,
                  endpoint_force_and_contact_pass=True,
                  evolving_profile_work_identity_pass=True,
                  frozen_control_work_errors=frozen_errors,
                  provisional_temporal_accuracy_gate_pass=False,
                  finest_reference_sufficiently_resolved=False,
                  selected_native_timestep=None, new_native_run=False,
                  production_sources_or_model_changed=False, P3_ready=False)
    (E / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
