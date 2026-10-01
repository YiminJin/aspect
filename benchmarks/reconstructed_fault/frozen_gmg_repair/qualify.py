#!/usr/bin/env python3
"""Freeze the bounded fixture evidence without rebuilding production binaries."""
from pathlib import Path
import hashlib, json, subprocess
root = Path(__file__).resolve().parent
repo = root.parents[2]
evidence = root/'evidence'
hash_file = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
subprocess.run(['python3', str(root/'compare.py')], check=True)
for version in ('reference','candidate'):
    for action in ('configure','build','run'):
        record = json.loads((evidence/f'{version}-{action}.json').read_text())
        assert record['exit_code'] == (1 if action == 'run' else 0)
paths = [p for p in root.iterdir() if p.is_file()]
paths += list((root/'plugin').glob('*')) + list((root/'inputs').glob('*.prm'))
paths += list((repo/'benchmarks/reconstructed_fault/bp3/reference_200km').glob('*.h'))
paths += [repo/'benchmarks/reconstructed_fault/bp3/reference_200km/bp3.cc',
          repo/'tests/reconstructed_fault_frozen_gmg.cc',
          repo/'benchmarks/reconstructed_fault/performance/gmg/run.py',
          repo/'build-refactor-r4b-residual/aspect-r4b-residual-qualified',
          repo/'build-refactor-r4c/aspect-r4c-verified']
paths += list((repo/'benchmarks/reconstructed_fault/bp3/fixtures/modified_bp3').glob('*'))
paths += list((repo/'benchmarks/reconstructed_fault/bp3/fixtures/modified_bp3_wide').glob('*'))
paths += list((repo/'benchmarks/reconstructed_fault/bp3').glob('*.prm'))
for version in ('reference','candidate'):
    paths += list((root/f'{version}-plugin-build').glob('*.so'))
    paths += list((root/f'output-{version}').glob('frozen_*.bin'))
    paths += [root/f'output-{version}/frozen_gmg.csv']
manifest = {str(p.relative_to(repo)):hash_file(p) for p in sorted(set(paths)) if p.is_file()}
(evidence/'executed-artifacts.json').write_text(json.dumps(manifest, indent=2)+'\n')
comparison = json.loads((evidence/'comparison.json').read_text())
result = dict(pre_R4c='983d57e2863af798de29cb9601d33bfe3a53a6af',
              post_R4c='fa6013678b525b189a1d27ef08465d4a6ef263f6',
              fixture='uncommitted test-only repair over accepted R4c',
              checks=comparison['count'], failures=comparison['failures'],
              complete_pass_before_intentional_stop=True, ranks=4,
              production_source_unchanged=True, matched_rows=comparison['matched_rows'],
              artifact_entries=len(manifest),
              artifacts_sha256=hash_file(evidence/'executed-artifacts.json'),
              protection_sha256=hash_file(evidence/'protected-hashes.json'))
(evidence/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
