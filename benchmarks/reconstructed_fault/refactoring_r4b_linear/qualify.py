#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,shutil,subprocess
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
labels=['configure','independent-final','build','reference-plugin-configure','reference-plugin-build','plugin-configure','plugin-build']
for v in ('reference','candidate'):
 labels += [f'{v}-{name}-{rank}' for name in ('residual','exhaustion','pressure') for rank in (1,2)]+[f'{v}-gmg-1']
labels += [f'candidate-{name}-{rank}' for name in ('unit','rollback-original') for rank in (1,2)]
labels += ['candidate-bp3-'+mode for mode in ('legacy-one','legacy-two','automatic-one','automatic-two')]
for label in labels:
 assert json.loads((e/(label+'.json')).read_text())['exit_code']==(1 if '-exhaustion-' in label else 0),label
for name in ('source-verification','focused-comparison','lifecycle-comparison'):
 checks=json.loads((e/(name+'.json')).read_text());assert checks and all(checks.values()),name
fields=json.loads((e/'state-comparison.json').read_text());assert len(fields)==372
assert all(f['max_abs']==0 for row in fields.values() for f in row['fields'].values())
for name in ('candidate-source-hashes','executed-artifacts','reference-plugin-hashes'):
 m=json.loads((e/(name+'.json')).read_text());assert all(h(repo/p)==v for p,v in m.items()),name
for name in ('residual-1','residual-2','exhaustion-1','exhaustion-2','pressure-1','pressure-2','gmg-1','rollback-original-1','rollback-original-2','bp3-legacy-one','bp3-legacy-two','bp3-automatic-one','bp3-automatic-two'):
 prm=(r/f'output-candidate-{name}/parameters.prm').read_text()
 libraries=next(line for line in prm.splitlines() if 'set Additional shared libraries' in line)
 assert all('refactoring_r4b_linear/plugin-build/' in p for p in libraries.split('=',1)[1].split(',')),libraries
obj=repo/'build-refactor-r4b-linear/CMakeFiles/aspect.exe.release.dir/source/simulator/solver/reconstructed_fault_stokes.cc.o'
assert obj.stat().st_mtime>=max((repo/p).stat().st_mtime for p in ('include/aspect/simulator.h','source/simulator/solver/reconstructed_fault_stokes.cc'))
s=subprocess.check_output(['nm','-C','--defined-only',str(obj)],text=True)
lines=[line for line in s.splitlines() if '{' not in line and ' W aspect::Simulator<' in line and ('::solve_reconstructed_fault_stokes()' in line or '::solve_reconstructed_fault_condensed_system(' in line)]
for dim in (2,3):
 for name in ('solve_reconstructed_fault_stokes','solve_reconstructed_fault_condensed_system'):
  assert sum(f' W aspect::Simulator<{dim}>::{name}(' in line for line in lines)==1
(e/'linked-symbols.txt').write_text('\n'.join(lines)+'\n')
binary=repo/'build-refactor-r4b-linear/aspect-release';qualified=binary.with_name('aspect-r4b-linear-qualified')
if not qualified.exists():shutil.copy2(binary,qualified)
assert h(binary)==h(qualified)
record=dict(reference_revision=json.loads((e/'reference.json').read_text())['revision'],reference_sha256=h(repo/'build-refactor-r4a/aspect-r4a-qualified'),candidate_revision='uncommitted first R4b subpass',candidate_sha256=h(qualified),successful_builds_and_runtime_checks=len(labels),expected_exhaustion_failures=4,exact_field_groups=372,bp3_counter_and_decision_checks=28,focused_checks=len(json.loads((e/'focused-comparison.json').read_text())),instantiations=[2,3])
(e/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
