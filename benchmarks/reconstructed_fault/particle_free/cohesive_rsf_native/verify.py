#!/usr/bin/env python3
"""Compare existing native qualification outputs; never advance or convert state.

Run after the named bounded qualify.py controls. Results retain execution/binary
identities so reused component evidence is distinguished from final-source runs.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'build-particle-free/cohesive-rsf-native'
OUT = ROOT / 'benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_native'
HISTORY = ['jump_m', 'damage', 'theta_normalized', 'reference_m',
           'relaxed_jump_m', 'absolute_slip_m']


def indexed(label):
    rows = csv.DictReader((WORK / 'runs' / label / 'output/cohesive-index.csv').open())
    return {round(float(row['time_s']), 12): Path(row['stem']) for row in rows}


def file(stem, suffix):
    return Path(str(stem) + '-' + suffix)


def snapshot(a, b):
    aa = list(csv.DictReader(file(a, 'surface.csv').open()))
    bb = list(csv.DictReader(file(b, 'surface.csv').open()))
    assert len(aa) == len(bb)
    result = dict(
        surface_history_max_abs=max(abs(float(x[k])-float(y[k]))
                                    for x, y in zip(aa, bb) for k in HISTORY),
        identities_exact=[r['id'] for r in aa] == [r['id'] for r in bb],
        geometry_bitwise=file(a, 'geometry.csv').read_bytes() == file(b, 'geometry.csv').read_bytes(),
        surface_bitwise=file(a, 'surface.csv').read_bytes() == file(b, 'surface.csv').read_bytes())
    if file(a, 'physical-history.csv').exists() and file(b, 'physical-history.csv').exists():
        x = list(csv.reader(file(a, 'physical-history.csv').open()))
        y = list(csv.reader(file(b, 'physical-history.csv').open()))
        assert [r[0] for r in x] == [r[0] for r in y]
        delta = abs(np.array([r[1:] for r in x], float) - np.array([r[1:] for r in y], float))
        result.update(bulk_records_bitwise=file(a, 'physical-history.csv').read_bytes() ==
                      file(b, 'physical-history.csv').read_bytes(),
                      DG_coefficient_max_abs_Pa=float(delta[:, 8:35].max()),
                      native_coefficient_max_abs_m=float(delta[:, 35:].max()))
    pairs = [(p, file(b, p.name[len(a.name)+1:])) for p in
             sorted(a.parent.glob(a.name + '-bulk-r*.csv'))]
    result['bulk_center_files_bitwise'] = bool(pairs) and all(
        p.read_bytes() == q.read_bytes() for p, q in pairs)
    result['publications'] = [int(float(dict((r['quantity'], r['value']) for r in
        csv.DictReader(file(s, 'diagnostics.csv').open()))['publications'])) for s in (a, b)]
    return result


def sequence(a, b):
    x, y = indexed(a), indexed(b)
    common = sorted(x.keys() & y.keys())
    assert common
    rows = [dict(time_s=t, **snapshot(x[t], y[t])) for t in common]
    for row in rows:
        assert row['surface_history_max_abs'] == 0 and row['identities_exact'] and row['geometry_bitwise']
        assert row['publications'][0] == row['publications'][1]
        if 'bulk_records_bitwise' in row:
            assert row['bulk_records_bitwise']
        if (a, b) == ('native-main-initial', 'native-prefix-owned-check') and row['time_s'] == .44:
            # This control corrects the demonstrated nonfinite *diagnostic*
            # pressure at initialization. All persistent tensor/native records
            # above remain exact. Verify that pressure alone changed here.
            for p in x[row['time_s']].parent.glob(x[row['time_s']].name+'-bulk-r*.csv'):
                q = file(y[row['time_s']], p.name[len(x[row['time_s']].name)+1:])
                for old, new in zip(csv.DictReader(p.open()), csv.DictReader(q.open())):
                    assert all(old[k] == new[k] for k in old if k != 'pprime_Pa')
                    assert not math.isfinite(float(old['pprime_Pa']))
                    assert math.isfinite(float(new['pprime_Pa']))
            row['documented_initial_pressure_correction_only'] = True
        else:
            assert row['bulk_center_files_bitwise']
    return dict(reference=a, candidate=b, common_states=rows)


def main():
    cases = {
        'outer_cutback': ('general-cutback-control', 'general-rejection'),
        'outer_restart': ('general-cutback-control', 'general-restarted'),
        'admission_cutback': ('admission-cutback-control', 'admission-rejection'),
        'formed_restart': ('native-main-initial', 'formed-restarted'),
        'initialization_correction': ('native-main-initial', 'native-prefix-owned-check'),
        'final_source_cutback': ('general-cutback-control', 'final-rejection'),
        'final_source_completed_restart': ('final-rejection', 'final-completed-restart'),
        'late_current_restart': ('pilot-final-current', 'final-current-restarted'),
        'late_current_all_records': ('final-current-completed', 'final-current-restarted'),
        'packaged_anchor': ('native-prefix-owned-check', 'packaged-native-restart'),
    }
    result = {name: sequence(*pair) for name, pair in cases.items()}
    labels = sorted(set(label for pair in cases.values() for label in pair) |
                    {'general-r1', 'general-r2', 'general-r4', 'pilot-final-current',
                     'native-halfstep-mature', 'final-affine-history'})
    executions = {label: json.loads((WORK/'runs'/label/'execution.json').read_text()) for label in labels}
    result['executions'] = executions
    (OUT/'native-lifecycle.json').write_text(json.dumps(result, indent=2)+'\n')
    costs = [dict(case=k, ranks=v['ranks'], elapsed_s=v['elapsed_s'],
                  returncode=v['returncode'], binary_sha256=v['binary_sha256'])
             for k, v in executions.items()]
    with (OUT/'native-costs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(costs[0])); writer.writeheader(); writer.writerows(costs)
    original = json.loads((OUT/'entry-protection.json').read_text())
    changed, missing = [], []
    for name, digest in original.items():
        p = ROOT/name
        if not p.exists():
            missing.append(name)
        elif hashlib.file_digest(p.open('rb'), 'sha256').hexdigest() != digest:
            changed.append(name)
    protection = dict(audited_files=len(original), changed=changed, missing=missing,
                      bp3_changes=[p for p in changed if 'bp3' in p.lower()])
    assert not missing and not protection['bp3_changes']
    (OUT/'source-protection.json').write_text(json.dumps(protection, indent=2)+'\n')
    print(json.dumps(dict(lifecycle_cases=len(cases), changed_original_files=len(changed),
                          missing=missing, bp3_changes=protection['bp3_changes']), indent=2))


if __name__ == '__main__':
    main()
