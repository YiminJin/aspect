#!/usr/bin/env python3
"""Read-only numerical and preservation gate for the bounded P2 checkpoint."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / 'review_checkpoint_evidence'
BASELINE = ROOT / 'build-propagation-integration/review-checkpoint-baseline'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((OUT / 'manifest.json').read_text())
for name, expected in manifest['hashes'].items():
    assert digest(ROOT / name) == expected, f'Changed checkpoint artifact: {name}'
baseline = json.loads((BASELINE / 'baseline.json').read_text())
changed = set(manifest['authorized_existing_document_changes'])
for name, expected in baseline['hashes'].items():
    assert digest(BASELINE / name) == expected, f'Changed preserved copy: {name}'
    if name not in changed:
        assert digest(ROOT / name) == expected, f'Changed incoming artifact: {name}'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip() == baseline['base_commit']
assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT).decode().strip() == baseline['branch']
production = subprocess.check_output(['git', 'diff', 'HEAD', '--', 'include', 'source'], cwd=ROOT)
assert production == (OUT / 'production-since-frozen.diff').read_bytes(), 'Production diff changed'
assert manifest['production_changed_this_task'] is False
assert manifest['new_coupled_runs'] == 0

result = json.loads((OUT / 'normalization.json').read_text())
assert result['spacings_tested'] == [.0125, .00625]
assert result['source_target'] == .005 and result['selected_spacing'] == .00625
assert result['all_frozen_fields_pass'] and not result['coupled_model_qualified']
assert len(result['records']) == 12
for name, expected in result['input_hashes'].items():
    assert digest(ROOT / name) == expected, f'Changed frozen phase input: {name}'
for spacing in result['spacings_tested']:
    records = [r for r in result['records'] if r['spacing'] == spacing]
    assert len(records) == 6
    assert {(r['label'], r['state']) for r in records} == {
        (label, state) for label in ['irreversibility_dt125', 'irreversibility_dt0625', 'irreversibility_dt03125']
        for state in ['onset', 'strong']}
    for r in records:
        reference = r['reference_projection']
        assert 0 < r['phi_max'] < 1 and r['min_I'] > 0 and reference['min_I'] > 0
        assert r['constant_error'] < 2e-14 and r['reference_constant_error'] < 2e-14
        assert r['conservation_error'] < 2e-15 and r['reference_conservation_error'] < 2e-15
        assert r['projection_quadrature_uncertainty'] < 2e-12
        assert all(0 <= x <= 1 for x in [r['worst_x'], reference['worst_x']])
        error = max(r['max_source_error'], reference['max_source_error'])
        assert r['frozen_field_target_pass'] == (error < .005)
        path = OUT / f"normalization-{r['label']}-{r['step']}-{spacing:g}.csv"
        with path.open() as stream:
            samples = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        assert len(samples) == 4001 and samples[0]['x'] == 0 and samples[-1]['x'] == 1
        for row in samples:
            for suffix, column in [('native_Q3', 'consistent_native_Q3'), ('split_Q32', 'consistent_split_Q32')]:
                assert row[column] > 0 and row['direct'] > 0
                observed = row['direct'] / row[column] - 1
                assert abs(observed - row['source_error_' + suffix]) < 1e-15
        for suffix, maximum in [('native_Q3', r['max_source_error']), ('split_Q32', reference['max_source_error'])]:
            dense = max(abs(row['source_error_' + suffix]) for row in samples)
            assert dense <= maximum + 2e-12
            assert maximum - dense < 1e-8
        if spacing == .00625:
            assert error < .005 and r['native_Q3_vs_split_Q32'] < 2e-11
    if spacing == .0125:
        assert any(not r['frozen_field_target_pass'] for r in records), 'Unneeded second spacing'
    else:
        assert all(r['frozen_field_target_pass'] for r in records)
print('PASS: six frozen fields at .00625 m; .0125 fails; independent numerical and preservation checks.')
print('Production/model unchanged; no coupled qualification, new phase law, time refinement or P3.')
