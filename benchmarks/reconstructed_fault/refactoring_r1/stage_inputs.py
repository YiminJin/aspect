#!/usr/bin/env python3
"""Stage the recorded R1 PRMs and immutable convex fixture in a fresh directory."""
import hashlib
import json
import pathlib
import shutil

root = pathlib.Path(__file__).resolve().parent
source = root.parents[2].parent / 'aspect/benchmarks/reconstructed_fault/bp3/output-cleanup-evidence/fixture-convex'
for directory in ('inputs', 'fixtures'):
    (root / directory).mkdir()  # Refuse to overwrite a reference run.
for name, text in json.loads((root / 'input_templates.json').read_text()).items():
    (root / 'inputs' / name).write_text(text)
provenance = []
for name in ('target_cells.txt', 'fault.txt', 'profile.txt', 'completion.txt'):
    origin, target = source / name, root / 'fixtures' / name
    shutil.copyfile(origin, target)
    provenance.append(dict(source=str(origin), copied_to=str(target),
                           sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
(root / 'evidence').mkdir(exist_ok=True)
(root / 'evidence/fixture-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
