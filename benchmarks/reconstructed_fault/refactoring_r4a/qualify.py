#!/usr/bin/env python3
"""Check recorded R4a outcomes and freeze the executable and provenance."""
from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path(__file__).resolve().parent;repo=root.parents[2];e=root/'evidence'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
labels=['configure','build-includes','independent','plugin-configure','plugin-build']
labels += [f'{v}-{n}' for v in ('reference','candidate') for n in ('unit-1','unit-2','rollback-original-1','rollback-original-2','ordinary-amg')]
labels += [f'candidate-bp3-{mode}' for mode in ('legacy-one','legacy-two','automatic-one','automatic-two')]
for label in labels:
    assert json.loads((e/f'{label}.json').read_text())['exit_code']==0,label
for name in ('focused-comparison','lifecycle-comparison'):
    checks=json.loads((e/f'{name}.json').read_text());assert checks and all(checks.values()),name
assert all(json.loads((e/'source-verification.json').read_text())['checks'].values())
for name in ('bp3-legacy-one','bp3-legacy-two','bp3-automatic-one','bp3-automatic-two','rollback-original-1','rollback-original-2'):
    prm=(root/f'output-candidate-{name}/parameters.prm').read_text()
    libraries=next(line for line in prm.splitlines() if 'set Additional shared libraries' in line)
    assert 'refactoring_r4a/plugin-build/' in libraries,(name,libraries)
    assert all('refactoring_r4a/plugin-build/' in p for p in libraries.split('=',1)[1].split(',')),libraries
fields=json.loads((e/'state-comparison.json').read_text())
assert fields and all(f['max_abs']==0 for v in fields.values() for f in v['fields'].values())
executed=json.loads((e/'executed-artifacts.json').read_text())
assert all(h(repo/p)==v for p,v in executed.items()),'Executed binaries/plugins changed'
manifest=json.loads((e/'candidate-source-hashes.json').read_text())
assert all(h(repo/p)==v for p,v in manifest.items()),'Candidate source changed during qualification'
obj=repo/'build-refactor-r4a/CMakeFiles/aspect.exe.release.dir/source/simulator/solver/reconstructed_fault_stokes.cc.o'
assert obj.stat().st_mtime >= (repo/'source/simulator/solver/reconstructed_fault_stokes.cc').stat().st_mtime
symbols=subprocess.check_output(['nm','-C','--defined-only',str(obj)],text=True)
selected=[line for line in symbols.splitlines() if '::solve_reconstructed_fault_stokes()' in line and '{' not in line]
for dim in (2,3):
    assert sum(line.endswith(f' W aspect::Simulator<{dim}>::solve_reconstructed_fault_stokes()') for line in selected)==1,(dim,selected)
(e/'driver-symbols.txt').write_text('\n'.join(selected)+'\n')
ordinary=subprocess.check_output(['nm','-C','--defined-only',str(e/'solver-independent.o')],text=True)
for dim in (2,3):
    for name in ('solve_stokes','solve_advection'):
        assert any(f'aspect::Simulator<{dim}>::{name}(' in line and ' W ' in line for line in ordinary.splitlines()),(dim,name)
(e/'ordinary-symbols.txt').write_text('\n'.join(line for line in ordinary.splitlines() if ' W aspect::Simulator<' in line and any('::'+n+'(' in line for n in ('solve_stokes','solve_advection')))+'\n')
binary=repo/'build-refactor-r4a/aspect-release';qualified=binary.with_name('aspect-r4a-qualified')
if not qualified.exists():shutil.copy2(binary,qualified)
assert h(binary)==h(qualified)
artifacts=[qualified]+list((root/'plugin-build').rglob('*.so'))+list((root/'inputs').glob('*.prm'))
(e/'candidate-artifacts.json').write_text(json.dumps({str(p.relative_to(repo)):h(p) for p in artifacts},indent=2)+'\n')
record=dict(reference_revision=json.loads((e/'reference.json').read_text())['revision'],
 reference_sha256=h(repo/'build-refactor-r3b/aspect-maxwell-qualified'),candidate_sha256=h(qualified),
 candidate_revision='uncommitted R4a diff over d7b88b25e',passing_invocations=len(labels),
 exact_field_groups=len(fields),cache_work_checks=len(json.loads((e/'lifecycle-comparison.json').read_text())),
 focused_checks=len(json.loads((e/'focused-comparison.json').read_text())),instantiations=[2,3])
(e/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
