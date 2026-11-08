#!/usr/bin/env python3
"""Read bounded recovery outputs and verify the preserved reference; run no solver."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'build-particle-free/propagation-recovery'
EVIDENCE = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/propagation_recovery'
sys.path.insert(0, str(ROOT / 'benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native'))
from verify import snapshot


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def index(label):
    with (WORK / 'runs' / label / 'output/cohesive-index.csv').open() as stream:
        return list(csv.DictReader(stream))


def compare(reference, candidate):
    a = {float(row['time_s']): Path(row['stem']) for row in index(reference)}
    b = {float(row['time_s']): Path(row['stem']) for row in index(candidate)}
    assert b.keys() <= a.keys()
    result = []
    for time, stem in b.items():
        values = snapshot(a[time], stem)
        assert all(values[name] for name in ('surface_bitwise', 'geometry_bitwise',
                                            'bulk_records_bitwise', 'bulk_center_files_bitwise'))
        assert values['publications'][0] == values['publications'][1]
        result.append(dict(time_s=time, **values))
    return result


def diagnostics(label):
    result = []
    for row in index(label):
        with Path(row['stem'] + '-diagnostics.csv').open() as stream:
            values = {entry['quantity']: float(entry['value']) for entry in csv.DictReader(stream)}
        assert values['publications'] == int(row['step'])
        values.update(step=int(row['step']), time_s=float(row['time_s']))
        result.append(values)
    return result


def main():
    protection = json.loads((EVIDENCE / 'entry-preservation.json').read_text())
    for group in ('sha256', 'binaries', 'historical_binaries'):
        for name, expected in protection[group].items():
            assert digest(ROOT / name) == expected, name
    source = json.loads((ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/dgq2_overnight/identity.json').read_text())
    for name, expected in source['source_sha256'].items():
        assert digest(ROOT / name) == expected, name

    executions = {}
    launch_dir = ROOT / 'benchmarks/reconstructed_fault/refactoring_boundary/evidence'
    for path in sorted(launch_dir.glob('propagation-recovery-*.json')):
        executions[path.stem] = json.loads(path.read_text())
    total = sum(row['seconds'] for row in executions.values())
    assert total < 3600, total
    for label, row in executions.items():
        assert row['exit_code'] == (124 if label.endswith('-growth') else 0), (label, row)

    episode = diagnostics('episode-short')
    accepted = episode[1:]
    assert episode[0]['step'] == 748 and len(accepted) == 4
    maxima = {key: max(abs(row[key]) for row in accepted) for key in
              ('weak_error_Pa', 'source_error_Pa', 'ledger_error_J_m', 'continuous_history_error')}
    sums = {key: sum(row[key] for row in accepted) for key in
            ('fracture_increment_J_m', 'friction_J_m', 'radiation_J_m',
             'physical_remainder_J_m', 'surface_remainder_J_m')}
    absolute_work = sum(abs(row['physical_remainder_J_m']) for row in accepted)
    summary = dict(
        unchanged_source_files=len(source['source_sha256']),
        preserved_files=len(protection['sha256']),
        preserved_binaries=len(protection['binaries']) + len(protection['historical_binaries']),
        current_plugin_sha256=digest(WORK / 'checks/libcohesive_rsf_native_checks.release.so'),
        executions=executions, aggregate_launch_seconds=total,
        rejection=compare('matched-control', 'rejection'),
        restart=compare('rejection', 'current-restart'),
        historical_zero=json.loads((EVIDENCE / 'historical-zero.json').read_text()),
        episode=dict(start_step=748, end_step=accepted[-1]['step'],
                     end_time_s=accepted[-1]['time_s'], accepted_steps=len(accepted),
                     maxima=maxima, signed_sums=sums, absolute_work_remainder_J_m=absolute_work,
                     absolute_work_over_new_fracture=absolute_work / sums['fracture_increment_J_m'],
                     absolute_work_over_sliding=absolute_work / (sums['friction_J_m'] + sums['radiation_J_m'])),
        growth_replay='Timed out at 330 seconds; only initial state exported; no completed physical step.',
        plugin_build_seconds_approx=(WORK / 'checks/libcohesive_rsf_native_checks.release.so').stat().st_mtime
                                  - (WORK / 'checks/CMakeCache.txt').stat().st_mtime,
        build_timing_method='Cache-generation to library modification times; approximate, excludes configure.',
        numerical_source_changed=False, next_long_run_launched=False)
    inputs = {}
    for path in WORK.glob('runs/*/parameters.prm'):
        inputs[str(path.relative_to(ROOT))] = digest(path)
    for path in (ROOT / 'benchmarks/reconstructed_fault/particle_free/propagation_recovery').glob('*'):
        if path.is_file():
            inputs[str(path.relative_to(ROOT))] = digest(path)
    summary['input_and_support_sha256'] = inputs
    summary['new_episode_checkpoint_sha256'] = {
        str(path.relative_to(ROOT)): digest(path)
        for path in sorted((WORK / 'runs/episode-short/output/restart').rglob('*'))
        if path.is_file()}
    summary['new_episode_checkpoint_marker'] = (
        WORK / 'runs/episode-short/output/restart/last_good_checkpoint.txt').read_text().strip()
    (EVIDENCE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: summary[key] for key in
                     ('aggregate_launch_seconds', 'unchanged_source_files', 'preserved_files', 'episode')}, indent=2))


if __name__ == '__main__':
    main()
