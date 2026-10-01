#!/usr/bin/env python3
"""Run a four-rank repaired probe with an immutable pre/post-R4c executable."""
from pathlib import Path
import json, os, subprocess, sys
root = Path(__file__).resolve().parent
repo = root.parents[2]
version = sys.argv[1]
assert version in ('reference', 'candidate')
binary = repo/('build-refactor-r4b-residual/aspect-r4b-residual-qualified' if version == 'reference'
               else 'build-refactor-r4c/aspect-r4c-verified')
env = {k:v for k,v in os.environ.items() if not k.startswith('ASPECT_')}
env.update(json.loads((root/'evidence/environment.json').read_text()))
raise SystemExit(subprocess.run([sys.executable, str(root/'run_logged.py'), version+'-run', '720',
                                 'mpirun', '-np', '4', str(binary), str(root/f'inputs/{version}.prm')],
                                env=env, cwd=repo).returncode)
