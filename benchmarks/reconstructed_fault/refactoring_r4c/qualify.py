#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,shutil
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
labels=['configure','build','independent','plugin-configure','plugin-build','reference-plugin-configure','reference-plugin-build','reference-frozen-plugin-configure','reference-frozen-plugin-build-final','frozen-plugin-configure','frozen-plugin-build']
labels += [f'candidate-{n}-{rank}' for n in ('residual','exhaustion','pressure','unit','rollback-original') for rank in (1,2)]
labels += ['candidate-gmg-1']+[f'candidate-bp3-{m}' for m in ('legacy-one','legacy-two','automatic-one','automatic-two')]
labels += [f'{v}-{n}' for v in ('reference','candidate') for n in ('ordinary-amg','bfbt','melt','fail-s','fail-budget','frozen-replay')]
for label in labels:
 j=json.loads((e/(label+'.json')).read_text());expected=1 if any(s in label for s in ('exhaustion','fail-','frozen-replay')) else 0
 assert j['exit_code']==expected,(label,j['exit_code'])
for name,count in [('source-verification',9),('focused-comparison',31),('extra-comparison',22),('lifecycle-comparison',28)]:
 checks=json.loads((e/(name+'.json')).read_text());assert len(checks)==count and all(checks.values()),name
fields=json.loads((e/'state-comparison.json').read_text());assert len(fields)==372 and all(f['max_abs']==0 for v in fields.values() for f in v['fields'].values())
frozen=json.loads((e/'frozen-comparison.json').read_text());assert all(frozen['checks'].values())
for name in ('candidate-source-hashes','candidate-executed-artifacts','reference-executed-artifacts','reference-frozen-artifact','protected-hashes'):
 m=json.loads((e/(name+'.json')).read_text());assert all(h(repo/p)==v for p,v in m.items()),name
for p in r.glob('output-candidate-*/parameters.prm'):
 libs=next(line for line in p.read_text().splitlines() if 'set Additional shared libraries' in line).split('=',1)[1].strip()
 assert not libs or all('refactoring_r4c/plugin-build/' in s for s in libs.split(',')),(p,libs)
assert len((e/'linked-symbols.txt').read_text().splitlines())==9
binary=repo/'build-refactor-r4c/aspect-release';qualified=binary.with_name('aspect-r4c-verified')
if not qualified.exists():shutil.copy2(binary,qualified)
assert h(binary)==h(qualified)
record=dict(reference=json.loads((e/'reference.json').read_text()),candidate='uncommitted R4c over 983d57e28',candidate_binary=str(qualified.relative_to(repo)),candidate_sha256=h(qualified),status='equivalence verified with pre-existing frozen-GMG fixture limitation',selected_build_runtime_outcomes=len(labels),expected_exhaustion_and_ordinary_failures=6,preexisting_frozen_probe_failures=2,source_checks=9,focused_checks=31,ordinary_checks=22,bp3_exact_field_groups=372,bp3_counter_decision_checks=28,frozen_matched_checks=len(frozen['checks']),frozen_GMG_completed=False,independent_compile=['ordinary','coupled','private header'],instantiations=[2,3])
(e/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
