#!/usr/bin/env python3
"""Replay the bounded consolidation checks; launch no simulator or MPI job."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from assess_pf6_domain_free import domain_absence, initialization, normalization_derivatives
from assess_pf6_local import quadrature_control
from assess_pf6_transfer import compare_runs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def identical_exports(first, second, prefixes=None):
    files = sorted(p.name for p in first.iterdir()
                   if p.suffix in ('.csv', '.txt') and p.name != 'transfer-cost.csv'
                   and (any(p.name.startswith(x) for x in prefixes) if prefixes else p.suffix == '.csv'))
    assert files
    assert all((first/name).read_bytes() == (second/name).read_bytes() for name in files)
    return dict(files=len(files), names=files, byte_identical=True)


def phase_trace(run):
    records = []
    current = None
    for line in (run/'run.log').read_text().splitlines():
        if 'Endpoint phase target:' in line:
            current = dict(target=float(re.search(r'target=(\S+)', line)[1]), iterations=0)
        elif 'Endpoint phase iteration ' in line:
            current['last_scaled_KKT'] = float(re.search(r'KKT=([^,]+)', line)[1])
            current['iterations'] += 1
        elif re.search(r'Endpoint (?:CPDI|fracture) reuse check:', line):
            current['reassembly_gradient_difference'] = float(re.search(r'gradient=(\S+)', line)[1])
        elif 'Endpoint cost:' in line:
            assert current is not None
            assert current['last_scaled_KKT'] + current['reassembly_gradient_difference'] <= current['target']
            records.append(current)
            current = None
    assert records
    return records


def physical_progress(run):
    folder = run/'run'
    phases = [{(r['x'], r['y']): float(r['phase']) for r in rows(folder/f'state-{s}-phase.csv')}
              for s in range(3)]
    assert phases[0].keys() == phases[1].keys() == phases[2].keys()
    healing = max(0., max(a[k]-b[k] for a, b in zip(phases, phases[1:]) for k in a))
    growth = max(phases[2][k]-phases[0][k] for k in phases[0])
    assert healing == 0. and growth > 0.
    energies = [rows(folder/f'state-{s}-energy.csv')[0] for s in range(3)]
    assert all(int(e['step']) == int(e['publications']) == s for s, e in enumerate(energies))
    return dict(total_phase_growth_Linf=growth, healing_Linf=healing,
                initial_phase_max=max(phases[0].values()), final_phase_max=max(phases[2].values()),
                states=energies,
                final_Z_max_Pa_m=max(float(r['Z']) for r in rows(folder/'state-2-surface.csv')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('continued', type=Path)
    p.add_argument('--protected', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    runs = [a.baseline, a.candidate, a.continued]
    records = [json.loads((run/'record.json').read_text()) for run in runs]
    assert all(r['exit_code'] == 0 for r in records)
    assert records[0]['source_hashes'].keys() == records[1]['source_hashes'].keys()
    assert records[0]['observer_sha256'] == records[1]['observer_sha256'] == records[2]['observer_sha256']
    assert records[0]['plugin_sha256'] == records[1]['plugin_sha256'] == records[2]['plugin_sha256']
    changed = {k: [v, records[1]['source_hashes'][k]] for k, v in records[0]['source_hashes'].items()
               if records[1]['source_hashes'][k] != v}
    assert set(changed) == {'source/simulator/solver/reconstructed_fault_endpoint_phase.cc',
                            'source/material_model/phase_field_fault/dg_history.cc'}
    assert records[1]['source_hashes'] == records[2]['source_hashes']
    normalized_inputs = [(run/'input.prm').read_text().replace(str(run.resolve()), '<RUN>') for run in runs[:2]]
    assert normalized_inputs[0] == normalized_inputs[1]
    gates = [rows(run/'run/transfer-gates.csv') for run in runs]
    assert all(g and all(row['passed'] == '1' for row in g) for g in gates)
    equivalence = compare_runs(a.baseline/'run', a.candidate/'run')
    restart = compare_runs(a.continued/'run', a.candidate/'run')
    trace = [phase_trace(run) for run in runs[:2]]
    assert trace[0] == trace[1]
    mechanical = [[line for line in (run/'run.log').read_text().splitlines()
                   if 'Relative nonlinear residuals (bulk, fault)' in line] for run in runs[:2]]
    assert mechanical[0] == mechanical[1] and mechanical[0]
    quadrature = quadrature_control(a.candidate, ['prescribed-initial', 'state-0', 'state-1', 'state-2'])
    assert all(r['physical_rule_passed'] and r['normalization_passed'] for r in quadrature.values())
    result = dict(
        scope='Serial fixed-reference cohesive domain-free DG endpoint consolidation; two dt=.125 s accepted steps',
        base_revision=records[0]['head'], source_changes=changed,
        normalized_inputs_identical=True,
        normalized_input_sha256=hashlib.sha256(normalized_inputs[0].encode()).hexdigest(),
        cases=[dict(path=str(run), record_sha256=sha(run/'record.json'),
                    executable_sha256=r['executable_sha256'], plugin_sha256=r['plugin_sha256'],
                    input_sha256=r['input_sha256'], seconds=r['seconds'], ranks=r['ranks'],
                    native_gates=len(g)) for run, r, g in zip(runs, records, gates)],
        entry_exports=identical_exports(a.baseline/'run', a.candidate/'run'),
        restart_exports=identical_exports(a.candidate/'run', a.continued/'run', ['state-2-', 'before-step-2-']),
        entry_comparisons=equivalence, restart_comparisons=restart,
        physical_progress=physical_progress(a.candidate), phase_solve_trace=trace[1],
        mechanical_residual_trace_identical=True, mechanical_residual_records=len(mechanical[0]),
        domain_absence=[domain_absence(run) for run in runs],
        initialization=initialization(a.candidate),
        normalization_derivatives=normalization_derivatives(a.candidate), quadrature=quadrature,
        strict_reference_match_passed=False,
        prior_accuracy_failures_reassessed=False, hanging_phase_evolution_guard_retained=True,
        limitations='Equivalence and bounded functional checks only; original mirror-branch, spatial/temporal and Gauss6 failures remain.')
    if a.protected:
        protected = json.loads(a.protected.read_text())
        root = Path(__file__).resolve().parents[5]
        for name, identity in protected['files'].items():
            assert sha(root/name) == identity['sha256']
            assert (root/name).stat().st_size == identity['bytes']
        result['protected_BP3'] = dict(files=len(protected['files']), unchanged=True,
                                       manifest=str(a.protected), manifest_sha256=sha(a.protected))
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, entry_exports=result['entry_exports']['files'],
                          restart_exports=result['restart_exports']['files'], native_gates=[len(g) for g in gates])))


if __name__ == '__main__':
    main()
