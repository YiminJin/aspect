#!/usr/bin/env python3
"""Preserve compact inputs/logs and hash the four-run shear evidence."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/sheared_moving_fault'
RUNS = ROOT/'build-particle-free/transported-objective/stage-c/material-runs'
CASES = ['shear32-dt100', 'shear32-dt050', 'shear32-dt025', 'shear64-dt025']


def main():
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    entry = json.loads((OUT/'entry.json').read_text())
    preserved = {}
    for key in ('source_hashes', 'binaries', 'protected_files'):
        actual = {p:sha(ROOT/p) for p in entry[key]}
        assert actual == entry[key], key
        preserved[key] = actual
    assert not subprocess.check_output(['git', 'diff', '--name-only', '--', 'source', 'include', 'CMakeLists.txt', 'cmake', 'tests'], text=True).strip()
    files = list(Path(__file__).parent.glob('*sheared_moving_fault.py'))
    files += [Path(__file__).with_name('run_material_native.py'), Path(__file__).with_name('compare_bounded_profile.py'),
              Path(__file__).with_name('diagnose_bounded_energy.py'), Path(__file__).with_name('assess_material_native.py'),
              Path(__file__).with_name('isolate_profile_projection.py'), Path(__file__).with_name('isolate_tensor_defect.py')]
    commands = {}; storage = {}
    for case, level, dt in zip(CASES, (5, 5, 5, 6), (.1, .05, .025, .025)):
        run = RUNS/case; command = json.loads((run/'command.json').read_text()); a = command['arguments']
        assert command['exit_code'] == 0 and command['head'] == entry['revision']
        assert a['level'] == level and a['dt'] == dt and a['ranks'] == 4
        assert a['speed'] == .4 and a['shear'] == .5 and a['initial_dt'] == .1 and a['spacing'] == .1
        assert a['eta'] == '1' and a['representation'] == 'direct physical stress'
        assert a['profile_method'] == 'bounded MacCormack' and a['inflow'] == 'accepted trace'
        assert a['support'] == 'full normal cross-section' and not a['true_checks'] and not a['reject']
        assert command['core_sha256'] == entry['binaries']['build-particle-free/native/aspect-release']
        assert command['plugin_sha256'] == entry['binaries']['build-particle-free/transported-objective/stage-c/checks/libtransported_material_energy.release.so']
        dest = OUT/'runs'/case; dest.mkdir(parents=True, exist_ok=True)
        selected = [run/p for p in ('command.json', 'input.prm', 'fault.txt', 'run.log')]
        selected += [run/'output'/p for p in ('material-summary.csv', 'material-model.csv')]
        step = round(.4/dt); stats = sorted((run/'output').glob(f'material-{step}-storage-rank*.csv'))
        selected += stats
        for path in selected:
            shutil.copy2(path, dest/path.name)
        files += selected+[run/'runner.py']+list((run/'output').glob('material-*.csv'))
        commands[case] = command
        storage[case] = [list(csv.DictReader(path.open()))[0] for path in stats]
    docs = ROOT/'doc/reconstructed_fault/particle_free'
    files += [docs/p for p in ('SHEARED_MOVING_FAULT_PROTOCOL.md', 'SHEARED_MOVING_FAULT_REPORT.md',
                               'STATUS.md', 'SOURCE_AUDIT.md', 'REVIEW_PLAN.md', 'RESEARCH_HANDOFF.md', 'MATURE_MOVING_FAULT.md')]
    files += [p for p in OUT.rglob('*') if p.is_file() and p.name != 'final-provenance.json']
    record = dict(utc=datetime.now(timezone.utc).isoformat(), root=str(ROOT), revision=entry['revision'],
                  branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
                  native_source_unchanged=True, no_rebuild=True, maximum_MPI_ranks=4, sequential_runs=True,
                  no_production_or_server_run=True, commands=commands, final_storage=storage,
                  preserved=preserved, files_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))},
                  data_losses=['Inherited loss of 39/45 native-pilot frames.', 'Earlier temporary build logs lost in inherited cleanup.'])
    (OUT/'final-provenance.json').write_text(json.dumps(record, indent=2)+'\n')
    print('Recorded four runs, unchanged source/binaries and all 63 protected artifacts.')


if __name__ == '__main__':
    main()
