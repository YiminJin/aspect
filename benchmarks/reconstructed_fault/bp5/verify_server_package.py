"""Verify copied scientific inputs/sources without a repository or environment search."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root/'manifest.json').read_text())
qualification = json.loads((root/'qualification.json').read_text())
assert qualification['launch_approved'], 'Bounded startup/restart qualification is missing'
for name, expected in manifest['files'].items():
    # These two site-facing documents may be customized; scientific inputs,
    # plugin sources, evidence and qualification remain hash-checked.
    if name in ['first_event.slurm', 'README.md']:
        continue
    path = root/name
    assert path.is_file(), f'Missing packaged input: {name}'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Changed packaged input: {name}'
print('Packaged inputs/source provenance verified; bounded four-rank startup/restart passed.')
print('Server executable/plugin ABI and longer-time physical accuracy remain separate requirements.')
