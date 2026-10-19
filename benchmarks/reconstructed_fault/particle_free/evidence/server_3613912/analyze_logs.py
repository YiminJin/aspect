"""Reduce supplied replay logs only; does not launch ASPECT or modify raw files."""
from pathlib import Path
import csv
import hashlib
import json
import re

OUT = Path(__file__).resolve().parent
LOGS = OUT.parent.parent / 'bp3_dg_first_event'
original = (LOGS / 'BP3.o3610007').read_text()
replay = (LOGS / 'BP3.o3613912').read_text()
stderr = (LOGS / 'BP3.e3613912').read_text()
rows = []
for line in replay.splitlines():
    if not line.startswith('Work consistency trace: '):
        continue
    row = {}
    last_key = None
    for item in line.split(': ', 1)[1].split(', '):
        key, value = item.split('=', 1)
        if key == 'node':
            row[last_key + '_node'] = int(value)
        else:
            row[key] = int(value) if key == 'step' else float(value)
            last_key = key
    rows.append(row)
assert [r['step'] for r in rows] == list(range(810, 817))
assert 'Accepted work observer does not reproduce frozen mechanical traction.' in stderr
assert rows[-1]['original_raw_normal_Pa'] > rows[-1]['bound_Pa']
assert all(r['matching_raw_normal_Pa'] < r['bound_Pa'] and
           r['matching_shear_Pa'] < r['bound_Pa'] for r in rows)
assert rows[-1]['original_raw_normal_Pa_node'] == 4918


def step_lines(text):
    result = {}
    step = None
    for line in text.splitlines():
        match = re.match(r'\*\*\* Timestep (\d+):', line)
        if match:
            step = int(match[1])
            result[step] = []
        if step is not None:
            result[step].append(line)
    return result


old_steps, new_steps = step_lines(original), step_lines(replay)
comparison = {}
for row in rows:
    step = row['step']
    def selected(lines):
        return [line for line in lines if 'DG Maxwell publication:' in line or
                'Fault linear solve:' in line or
                'Relative nonlinear residuals (bulk, fault)' in line]
    before, after = selected(old_steps[step]), selected(new_steps[step])
    comparison[step] = {'mechanical_and_publication_log_lines_identical': before == after,
                        'compared_lines': len(before)}
    assert before == after and len(before) > 1

with (OUT.parent / 'server_3610007/accepted_steps.csv').open() as stream:
    accepted = {int(r['step']): r for r in csv.DictReader(stream)}
for row in rows[:-1]:
    assert row['time_s'] == float(accepted[row['step']]['time'])
    assert row['dt_s'] == float(accepted[row['step']]['dt'])

result = {
    'strict_reference_match_passed': False,
    'job': 3613912,
    'ranks': 160,
    'input_log_sha256': {name: hashlib.sha256((LOGS / name).read_bytes()).hexdigest()
                        for name in ['BP3.o3610007', 'BP3.o3613912', 'BP3.e3613912']},
    'trace_rows': rows,
    'frozen_frame_mechanism_reproduced': True,
    'existing_threshold_Pa': rows[-1]['bound_Pa'],
    'matching_frame_all_reported_steps_pass': True,
    'mechanical_and_publication_log_comparison': comparison,
    'steps_810_to_815_time_and_dt_match_original_accepted_csv': True,
    'latest_completed_checkpoint_in_stdout': {'step': 815, 'slot': '03',
                                              'metadata_verification_pending': True},
    'native_cells': 552081,
    'native_DoFs': 19498460,
    'nodal_failure_csv_available': (OUT / 'work_consistency_failure_816.csv').exists(),
    'runtime_source_changed_for_this_analysis': False,
    'limitation': 'Root maxima establish that matching the mechanical frames removes '
                  'the reported guard violation. Signed nodal load reconciliation '
                  'and verification of a repaired observer still require the failure CSV '
                  'and implementation/tests; no first-event accuracy claim.'
}
(OUT / 'log-analysis.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'step_816': rows[-1], 'all_logged_solver_and_publication_lines_identical': True}, indent=2))
