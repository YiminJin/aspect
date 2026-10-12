#!/usr/bin/env python3
"""Check restricted FE-state evidence without qualifying the failed coupled run."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / 'benchmarks/reconstructed_fault/propagation/native_fe_evidence'
BASELINE = ROOT / 'build-propagation-integration/native-fe-baseline'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((EVIDENCE / 'manifest.json').read_text())
    for name, expected in manifest['hashes'].items():
        assert digest(ROOT / name) == expected, name
    baseline = json.loads((BASELINE / 'baseline.json').read_text())
    allowed = json.loads((EVIDENCE / 'task-changes.json').read_text())
    unchanged = 0
    for name, expected in baseline['hashes'].items():
        assert digest(BASELINE / name) == expected, 'snapshot: ' + name
        if name not in allowed:
            assert digest(ROOT / name) == expected, 'incoming evidence: ' + name
            unchanged += 1
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    assert head == baseline['head'] and branch == baseline['branch']
    for entry in json.loads((EVIDENCE / 'restriction-checks.json').read_text()):
        assert entry['verified'] and entry['exit'] != 0
        assert entry['marker'] in (ROOT / entry['log']).read_text()
    mature = ROOT / 'build-propagation-integration/output-native-fe-mature-final'
    legacy = ROOT / 'build-propagation-integration/accuracy-energy-baseline/build-propagation-integration/output-irreversibility_mature_default'
    for name in ['boundary_state.csv', 'ih_automatic_completion_rank0.csv']:
        assert (mature / name).read_bytes() == (legacy / name).read_bytes()
    early = ROOT / 'build-propagation-integration/output-native-fe-endpoint64'
    final = ROOT / 'build-propagation-integration/output-native-fe-endpoint64-fixed'
    for step in range(16):
        for field in ['phase', 'surface']:
            name = f'fe-{field}-{step}.csv'
            assert (early / name).read_bytes() == (final / name).read_bytes()
    records = json.loads((EVIDENCE / 'assessment.json').read_text())
    assert records['qualification']['FE_representation_verified']
    assert not records['qualification']['coupled_completed']
    assert not records['qualification']['native_time_convergence_verified']
    assert not records['qualification']['P3_ready']
    assert records['completed_steps'] == 19 and records['final_time'] == .296875
    assert records['nodal_recurrence_error_m'] == 0
    for case in ['endpoint64-fixed', 'endpoint64-final', 'endpoint64-pdas', 'endpoint64-accurate-sum']:
        log = (ROOT / f'build-propagation-integration/native-fe-{case}.log').read_text()
        assert '*** Timestep 20:  t=0.3125' in log
        assert 'ExcNonlinearSolverNoConvergence' in log
    result = dict(snapshot_files=len(baseline['hashes']), unchanged_incoming_files=unchanged,
                  manifest_files=len(manifest['hashes']), representation_checks_pass=True,
                  coupled_qualification=False, P3_ready=False)
    (EVIDENCE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
