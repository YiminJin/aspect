#!/usr/bin/env python3
"""Verify bounded localization evidence and preservation; no stability/P3 claim."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / 'benchmarks/reconstructed_fault/propagation/localization_evidence'
BASELINE = ROOT / 'build-propagation-integration/localization-baseline'
RUNS = ROOT / 'build-propagation-integration/runs'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(path.read_text())


def csv(path):
    return np.genfromtxt(path, delimiter=',', names=True, ndmin=1)


def identical(left, right, names):
    for name in names:
        assert digest(left / name) == digest(right / name), name
    return len(names)


def main():
    baseline = load(BASELINE / 'baseline.json')
    for name, expected in baseline['hashes'].items():
        assert digest(BASELINE / name) == expected, 'snapshot: ' + name
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                   text=True).strip() == baseline['head']
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT,
                                   text=True).strip() == baseline['branch']
    # Preserve incoming manual notation edits exactly, including their whitespace.
    marker = 'diff --git a/doc/reconstructed_fault/pf_rsf.tex b/doc/reconstructed_fault/pf_rsf.tex\n'
    incoming = (BASELINE / 'incoming.diff').read_text()
    original_notation_diff = marker + incoming.split(marker)[1].split('diff --git ', 1)[0]
    assert subprocess.check_output(['git', 'diff', '--binary', '--',
                                    'doc/reconstructed_fault/pf_rsf.tex'], cwd=ROOT,
                                   text=True) == original_notation_diff
    changed = load(EVIDENCE / 'task-changes.json')
    for name, expected in baseline['hashes'].items():
        if name not in changed['modified_existing_files']:
            assert digest(ROOT / name) == expected, 'unchanged incoming: ' + name
    for name, expected in load(EVIDENCE / 'native-build.json').items():
        assert digest(ROOT / name) == expected, 'final build/source: ' + name

    # Historical manifests retain their historical source and binary identities.
    # Exempt only explicitly recorded edits/rebuilds; every old result is checked.
    historical = load(ROOT / 'benchmarks/reconstructed_fault/propagation/native_fe_evidence/manifest.json')
    preserved = 0
    for name, expected in historical['hashes'].items():
        if name not in changed['modified_existing_files']:
            assert digest(ROOT / name) == expected, 'historical artifact: ' + name
            preserved += 1
    for name, expected in load(EVIDENCE / 'manifest.json')['hashes'].items():
        assert digest(ROOT / name) == expected, 'new evidence: ' + name

    assessment = load(EVIDENCE / 'assessment.json')
    paths = {'affine': 'affine-final-executable',
             'compatible': 'compatible-operator-check',
             'wide': 'wide-operator-check'}
    for name, label in paths.items():
        records = assessment[name]
        assert records['completed'] and records['steps'] == 64 and records['time'] == 1
        assert records['healing'] == 0 and records['phi_max'] < 1
        assert records['mesh_dx'] == records['mesh_dy'] == .0625
        assert records['recurrence_error'] == 0
        assert max(records['mechanics_max']) <= 1e-8
        assert all(row['max_source_error'] < .005 for row in records['normalization'])
        run = RUNS / ('output-localization-' + label)
        log = (ROOT / 'build-propagation-integration' / ('localization-' + label + '.log')).read_text()
        targets = [float(v) for v in re.findall(r'Phase box target:.*? target=([^\s]+)', log)]
        defects = [float(v) for v in re.findall(r'Phase box converged:.*? KKT=([^\s]+)', log)]
        assert len(targets) == len(defects) == 65
        assert all(defect <= target for defect, target in zip(defects, targets))
        previous = csv(run / 'fe-phase-0.csv')['phi']
        for step in range(1, 65):
            current = csv(run / f'fe-phase-{step}.csv')['phi']
            assert min(current) >= 0 and max(current) < 1
            assert min(current - previous) >= -4.922e-12
            previous = current
            old = csv(run / f'fe-encoding-{step-1}.csv')
            new = csv(run / f'fe-encoding-{step}.csv')
            beta = np.exp(-.015625 / 100)
            kappa = -1e8 * np.expm1(-.015625 / 100)
            assert np.array_equal(new['d_R'], beta * old['d_R'] + kappa / 1e6 * new['u'])
        assert 'Exception' not in log

    stall = load(EVIDENCE / 'stall-assessment.json')
    assert stall['kkt'] > stall['target']
    assert stall['free_native_longdouble_difference'] < stall['target'] / 100
    assert stall['active_changes_one_ulp'] == [0, 0]
    assert stall['negative_modes'] == 2
    assert all(row['energy_change'] < 0 for row in stall['feasible_negative_curvature'])
    log = (ROOT / 'build-propagation-integration/localization-affine-final-executable.log').read_text()
    entry = log.split('*** Timestep 20:')[1].split('*** Timestep 21:')[0]
    defect, correction, energy, allowance = map(float, re.search(
        r'FE phase exact polish: KKT=([^\s]+) correction=([^\s]+) energy change=([^\s]+) arithmetic=([^\s]+)', entry).groups())
    assert defect < stall['target'] and correction < np.sqrt(np.finfo(float).eps)
    assert energy <= allowance

    affine_repeat = identical(RUNS / 'output-localization-affine-final-check',
                              RUNS / 'output-localization-affine-final-executable',
                              [f'fe-{field}-{step}.csv' for step in range(65)
                               for field in ['phase', 'surface', 'encoding']])
    bypass_repeat = identical(RUNS / 'output-localization-compatible-final',
                              RUNS / 'output-localization-compatible-operator-check',
                              [f'fe-{field}-{step}.csv' for step in range(8)
                               for field in ['phase', 'surface']])
    mature_repeat = identical(ROOT / 'build-propagation-integration/output-native-fe-mature-final',
                              RUNS / 'output-localization-mature-final',
                              ['boundary_state.csv', 'ih_automatic_completion_rank0.csv'])
    continuum = load(EVIDENCE / 'continuum-verification.json')
    assert continuum['common_time_refinement']['phi'] < .0004
    for row in continuum['stationary_controls']:
        assert row['KKT'] < 1e-11 and row['stress_error'] < 1e-8
        assert abs(row['mechanical_error']) < 1e-12
    for row in continuum['frozen_bulk_quadrature']:
        assert row['relative_energy'] < 1e-10 and row['relative_force_error'] < 1e-8
    for row in load(EVIDENCE / 'reference-verification.json'):
        assert min(v['gradient_error'] for v in row['energy_derivative']) < 1e-9
        assert min(v['hessian_error'] for v in row['energy_derivative']) < 1e-9
    for cells in [128, 256]:
        reference = EVIDENCE / f'one-d-h1-n{cells}-dt0.015625-exact-bulk'
        fields = np.load(reference / 'fields.npz')
        history = load(reference / 'history.json')
        assert len(history) == 64 and history[-1]['time'] == 1
        assert all(row['kkt'] <= 1e-11 for row in history)
        assert min(fields['phase'].ravel()) >= 0 and max(fields['phase'].ravel()) < 1
        assert min(np.diff(np.r_[fields['initial'][None, :], fields['phase']], axis=0).ravel()) >= -4.922e-12
    assert assessment['native_reference']['16']['full_phi'] < 1e-8
    result = dict(snapshot_files=len(baseline['hashes']),
                  preserved_historical_artifacts=preserved,
                  final_native_controls_completed=3,
                  unchanged_KKT_targets=True, bounds_and_irreversibility_pass=True,
                  affine_repeat_identical_files=affine_repeat,
                  transfer_bypass_identical_files=bypass_repeat,
                  mature_default_identical_files=mature_repeat,
                  reference_checks_pass=True,
                  negative_curvature_separately_reported=True,
                  native_spatial_time_convergence_verified=False,
                  energy_minimum_selection_verified=False, P3_ready=False)
    (EVIDENCE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
