#!/usr/bin/env python3
"""Record artifacts before runtime checks; qualification rechecks the hashes."""
from pathlib import Path
import hashlib, json

root = Path(__file__).resolve().parent
repo = root.parents[2]
target = root / 'evidence/executed-artifacts.json'
assert not target.exists(), 'Do not replace the pre-execution manifest.'
plugins = list((root / 'plugin-build').rglob('*.release.so'))
inputs = list((root / 'inputs').glob('candidate-*.prm'))
assert len(plugins) == 7 and len(inputs) == 13
paths = [repo / 'build-refactor-r4b-residual/aspect-release', *plugins, *inputs]
target.write_text(json.dumps({str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in paths}, indent=2) + '\n')
