#!/usr/bin/env python3
"""Independent P1 numerical gate; expected-output matching is not qualification."""
import csv
import json
import math
import pathlib
import re
import sys

rows = list(csv.DictReader(pathlib.Path(sys.argv[1]).open()))
log = pathlib.Path(sys.argv[2]).read_text()
assert [r['stage'] for r in rows] == ['hold', 'hold_repeat', 'ramp_05', 'ramp_10', 'unload']
for row in rows:
    for key in row:
        if key != 'stage':
            row[key] = float(row[key])
            assert math.isfinite(row[key]), (row['stage'], key)
    assert row['target'] == max(1e-8 * row['initial_residual'], row['roundoff'])
    assert row['final_residual'] <= row['target'], row
    assert row['H_increment_min'] >= 0 and row['masked_particles'] >= 4
    assert row['uncertainty'] >= row['correction'] >= 0
for marker in ['P1 scratch residual/Jacobian/roundoff isolation verified',
               'P1 normal restoration verified', 'P1 exceptional restoration verified']:
    assert marker in log, marker
repeat = max(rows[1]['change'], rows[4]['change'])
uncertainty = max(r['uncertainty'] for r in rows)
tolerance = 10 * max(repeat, uncertainty)
repeat_resolved = repeat <= 10 * uncertainty
resolved_evolution = min(r['change'] for r in rows[2:4]) > 100 * tolerance
violations = []
for row in rows:
    for quantity, value in [('negative_node', -row['node_min']),
                            ('upper_node', row['node_max'] - 1),
                            ('negative_cpdi', -row['cpdi_min']),
                            ('upper_cpdi', row['cpdi_max'] - 1),
                            ('healing', row['healing'])]:
        if value > tolerance and not (row['stage'] == 'hold' and quantity == 'healing'):
            violations.append(dict(stage=row['stage'], quantity=quantity,
                                   value=value, tolerance=tolerance))
result = dict(equality_converged=True, resolved_evolution=resolved_evolution,
              repeats_resolved=repeat_resolved, uncertainty_estimate=uncertainty,
              repeat_drift=repeat, tolerance=tolerance, violations=violations,
              physically_admissible=not violations,
              qualified=repeat_resolved and resolved_evolution and not violations,
              production_mechanical_H_update_qualified=False,
              physical_restart_and_rollback_tested=False,
              nonlinear_iterations={})
# Separate production Newton work from auxiliary correction solves.
for stage, segment in re.findall(r'P1 solve (\w+)\n(.*?)P1 result', log, re.S):
    result['nonlinear_iterations'][stage] = len(re.findall(r'Iteration\s+\d+:', segment))
print(json.dumps(result, indent=2))
sys.exit(0 if result['qualified'] else 2)
