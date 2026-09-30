#!/usr/bin/env python3
"""Compare accepted state and the known step-two failure without relaxing checks."""
from pathlib import Path
import json
import re
import struct
import zlib

root = Path(__file__).resolve().parent
evidence = root/'evidence'
reference = root.with_name('restart_investigation')
logs = [(reference/'evidence/create-one.log').read_text(errors='replace'),
        (evidence/'cohesive-create-one.log').read_text(errors='replace'),
        (evidence/'cohesive-resume-one.log').read_text(errors='replace')]
markers = ('Iteration ', 'Fault linear solve:', 'Relative nonlinear residuals',
           'line search accepted after', 'Solving temperature system',
           'Phase-field residual target:', 'Stage-J history resolution:',
           'Stage-J feedback step', 'line search exhausted')

def decisions(log):
    # Fresh/resumed streams retain different scientific/field-width settings.
    # Compare the full-precision target as exact doubles, with no tolerance.
    # Failure messages on stdout/stderr are checked separately below because
    # their interleaving is nondeterministic; preserve solver trace ordering.
    result = []
    for line in log.splitlines():
        if not any(m in line for m in markers):
            continue
        line = re.sub(r'\s+', ' ', line).strip()
        line = re.sub(r'\s+([,:])', r'\1', line)
        if 'Phase-field residual target:' in line:
            line = re.sub(r'(?<==)[0-9.eE+\-]+',
                          lambda m: float(m[0]).hex(), line)
        result.append(line)
    return result

checks = {}
checks['uninterrupted-before-after-decisions'] = decisions(logs[0]) == decisions(logs[1])
step_two = [log[log.index('*** Timestep 2:'):] for log in logs]
checks['step-two-decisions-all-three'] = decisions(step_two[0]) == decisions(step_two[1]) == decisions(step_two[2])
checks['original-restored-history-assertions-pass'] = 'Stage-J checkpoint histories, V, geometry, and bulk: verified' in logs[2]
for label, log in zip(('reference', 'create', 'resume'), logs):
    checks[f'{label}/nonconvergence'] = 'Newton line search exhausted all admissible candidates' in log and 'Nonlinear solver failed to converge' in log
    checks[f'{label}/no-segfault'] = 'Segmentation fault' not in log
outputs = [reference/'output-create-one', root/'output-cohesive-create-one', root/'output-cohesive-resume-one']
checks['accepted-statistics-before-after'] = (outputs[0]/'statistics').read_bytes() == (outputs[1]/'statistics').read_bytes()
for name in ('mesh', 'mesh.info', 'mesh_fixed.data', 'mesh_variable.data'):
    data = [(out/'restart/01'/name).read_bytes() for out in outputs]
    checks[f'accepted-checkpoint/{name}'] = data[0] == data[1] == data[2]

def fingerprint(out):
    data = zlib.decompress((out/'restart/01/resume.z').read_bytes()[16:])
    key = b'StageJRestart'
    assert data.count(key) == 1
    start = data.index(key) + len(key)
    size = struct.unpack_from('<Q', data, start)[0]
    blob = data[start+8:start+8+size]
    assert len(blob) == size and b'serialization::archive' in blob[:40]
    return blob

checks['accepted-history-V-geometry-bulk-fingerprint'] = fingerprint(outputs[0]) == fingerprint(outputs[1]) == fingerprint(outputs[2])
(evidence/'cohesive-comparison.json').write_text(json.dumps(checks, indent=2)+'\n')
print(len(checks), 'cohesive checks; failures:', [k for k, v in checks.items() if not v])
assert all(checks.values())
