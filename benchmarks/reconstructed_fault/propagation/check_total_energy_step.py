#!/usr/bin/env python3
"""Independent saved-field/ledger and baseline gate for isolated candidate."""
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE/'total_energy_evidence'
BASE = ROOT/'build-propagation-integration/total-energy-prototype-baseline'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((OUT/'manifest.json').read_text())
for name, expected in manifest['hashes'].items():
    assert digest(ROOT/name) == expected, f'Changed artifact: {name}'
baseline = json.loads((BASE/'baseline.json').read_text())
allowed = set(manifest['authorized_document_changes'])
for name, expected in baseline['hashes'].items():
    assert digest(BASE/name) == expected, f'Changed preserved copy: {name}'
    if name not in allowed:
        assert digest(ROOT/name) == expected, f'Changed baseline: {name}'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip() == baseline['head']
assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT).decode().strip() == baseline['branch']
summary = json.loads((OUT/'coupled_blocks/summary.json').read_text())
assert summary['prototype_only'] and not summary['production_model_changed']
assert not summary['native_coupled_qualified']
source = ROOT/'build-propagation-integration/output-irreversibility_dt03125/material-phase-accepted-0.csv'
assert digest(source) == summary['input_sha256']
with source.open() as stream:
    lookup = {(float(r['x']), float(r['y'])): float(r['phi']) for r in csv.DictReader(stream)}
seed = [lookup[(i/16, j/16)] for i in range(17) for j in range(17)]
fields = json.loads((OUT/'coupled_blocks/phase-fields.json').read_text())
obstacle = json.loads((OUT/'coupled_blocks/unloaded-obstacle.json').read_text())
assert 0 < obstacle['unconstrained_gradient_min_J_per_m'] <= obstacle['unconstrained_gradient_max_J_per_m']
assert obstacle['constrained_KKT'] == 0
assert all(obstacle[k] for k in ['saturated_trial_rejected', 'immutable_bound_array',
                               'immutable_profile', 'state_preserved'])
assert len(fields) == 27 and set(summary['results']) == {'unloaded', 'uniform', 'nonuniform'}
for name, result in summary['results'].items():
    assert result['rejected_trial_preserves_state']
    derivative = result['derivatives']
    assert derivative['diagnostic_history_invariance'] and derivative['adjoint_defect'] < 1e-9
    tests = derivative['directional_checks']
    assert len(tests) == 9
    for i in [0, 3, 6]:
        assert min(t['error'] for t in tests[i:i+3]) < 1e-7
    assert len(result['steps']) == 9
    for ratio in [.00125, 1., 4.]:
        previous = seed
        for cycle in [1, 2, 3]:
            record = next(r for r in result['steps'] if r['ratio'] == ratio and r['cycle'] == cycle)
            field = next(r['phase'] for r in fields if r['control'] == name and r['ratio'] == ratio and r['cycle'] == cycle)
            phi = [x for row in field for x in row]
            assert len(phi) == 289 and all(old <= new < 1 for old, new in zip(previous, phi))
            assert max(phi) == record['phi_max'] and record['min_I'] > 0
            previous = phi
            assert record['beta'] == math.exp(-ratio)
            assert record['phase']['kkt'] <= 1e-11 and record['phase']['healing'] == 0
            assert record['mechanical_kkt'] < 1e-9 and record['diagnostic_history_decrease'] <= 0
            for key in ['bulk_viscous', 'fault_viscous', 'friction']:
                assert record[key] >= 0
            jump = record['bulk_profile_jump']+record['fault_profile_jump']+record['fracture_expenditure']
            assert jump <= 1e-10 and abs(jump+record['phase_release']) < 1e-12
            assert abs(record['phase_path_virtual_work']-jump) < 1e-9
            assert record['phase_path_error'] < 1e-9
            # Recompute total ledger from separately saved state/work/loss terms.
            gap = (record['external_work']-record['bulk_storage_change']-record['fault_storage_change']
                   -record['fracture_expenditure']-record['bulk_viscous']-record['fault_viscous']-record['friction'])
            assert abs(gap-record['phase_release']) < 1e-8
            assert abs(record['ordinary_bulk_work']-record['ordinary_fault_work']-record['friction']) < 1e-8
            for key, tolerance in [('total_defect', 1e-8), ('bulk_identity_defect', 1e-8),
                                   ('fault_identity_defect', 1e-8), ('temporal_quadrature_error', 1e-9),
                                   ('legacy_endpoint_error', 1e-10), ('residual_strain_reconstruction_error', 1e-12)]:
                assert abs(record[key]) < tolerance
            if name == 'unloaded':
                assert phi == seed
                assert record['max_slip_rate'] == record['diagnostic_history_max'] == 0
                assert record['phase_release'] == gap == 0
    elastic = result['elastic_limit']
    assert abs(elastic['total_defect']) < 1e-8
    assert elastic['bulk_viscous']+elastic['fault_viscous'] < 1e-20
    coupled = result['free_bulk_mode']
    assert abs(coupled['free_bulk_residual']) < 1e-9
    assert coupled['mechanical_kkt'] < 1e-9
    assert abs(coupled['total_defect']) < 1e-8
    assert abs(coupled['free_bulk_work']) < 1e-12
    assert abs(coupled['exchange_defect']) < 1e-8
    assert coupled['residual_strain_reconstruction_error'] < 1e-12
    if name != 'unloaded':
        assert abs(coupled['free_bulk_velocity_amplitude']) > 1e-6
    endpoint = result['endpoint_balance_comparator']
    if name != 'unloaded':
        assert abs(endpoint['exchange_defect']) > .1, 'Lost endpoint-work diagnostic failure'
        first = result['steps'][0]
        assert first['phase_release'] > 5., 'Do not hide finite relaxation loss'
print('PASS: isolated discrete derivative/adjoint, irreversibility, exact Maxwell ledgers and publication.')
print('Retained limitations: phase-release slack and endpoint-exchange failures; no production qualification/P3.')
