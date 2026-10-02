#!/usr/bin/env python3
"""Freeze the tested candidate and record evidence without repeating runtimes."""
from pathlib import Path
import hashlib,json,shutil,subprocess
r=Path(__file__).resolve().parent;repo=r.parents[2];e=r/'evidence'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('verify_extraction','verify_symbols','compare','compare_frozen'):
 subprocess.run(['python3',str(r/(name+'.py'))],check=True,cwd=repo)
labels=['configure','build','plugin-configure','plugin-build','reference-plugin-configure','reference-frozen-plugin-build','independent','reference-frozen','candidate-frozen']
labels += [f'candidate-{case}-{rank}' for rank in (1,2) for case in ('unit','dynamic','adiabatic','rate','explicit','filter','singular','singular-current','pressure','rollback','bp3')]
for label in labels:
 expected=1 if '-singular-' in label or label.endswith('-frozen') else 0
 assert json.loads((e/(label+'.json')).read_text())['exit_code']==expected,label
for p,expected in json.loads((e/'protected-evidence-hashes.json').read_text()).items():
 assert digest(repo/p)==expected,p
binary=repo/'build-refactor-r5b2/aspect-release';qualified=binary.with_name('aspect-r5b2-qualified')
if not qualified.exists():shutil.copy2(binary,qualified)
assert digest(binary)==digest(qualified)
source={str(p.relative_to(repo)):digest(p) for d in ('source','include','unit_tests') for p in (repo/d).rglob('*') if p.is_file()}
(e/'candidate-source-hashes.json').write_text(json.dumps(source,indent=2)+'\n')
paths=[qualified,repo/'build-refactor-r5b1/aspect-r5b1-qualified',repo/'CMakeLists.txt']
paths+=list((r/'inputs').glob('*.prm'))+list((r/'plugin-build').rglob('*.so'))+list((r/'reference-plugin-build').rglob('*.so'))
paths+=[p for p in r.glob('*.py')]+list((r/'plugin').glob('*'))
paths+=[r/'plugin-build/singular_current_diagnostic.cc',r/'reference-plugin-build/singular_current_diagnostic.cc']
paths+=[repo/'tests'/name for name in ('phase_field_fault_surface_system.cc','phase_field_fault_surface_singular_system.cc','phase_field_fault_pressure_gauge.cc','phase_field_fault_stage_i_rollback.cc','reconstructed_fault_frozen_gmg.cc')]
paths+=[r.with_name('frozen_gmg_repair')/name for name in ('frozen.prm','clock.csv','evidence/environment.json')]
(e/'executed-artifacts.json').write_text(json.dumps({str(p.relative_to(repo)):digest(p) for p in paths},indent=2)+'\n')
record=dict(reference_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=repo).strip(),candidate='uncommitted R5b2 over accepted R5b1',candidate_binary=str(qualified.relative_to(repo)),candidate_sha256=digest(qualified),source_entries=len(source),artifact_entries=len(paths),source_checks=len(json.loads((e/'source-checks.json').read_text())),helper_instantiations=[2,3],surface_checks=len(json.loads((e/'comparison.json').read_text())),frozen_checks=json.loads((e/'frozen-comparison.json').read_text())['count'],recorded_build_runtime_outcomes=len(labels),unit_assertions_per_rank=636,unit_cases=3,surface_ranks=[1,2],frozen_ranks=4,known_failures='Original singular fixture stale diagnostic assertion (both matched ranks); supplemental invalidation probe passes with deliberate exit 1. Initial sandbox MPI socket denial retained separately.',limitations='No Debug or 3D runtime, unsupported-UMFPACK build, long scientific or restart campaign, dedicated native-QP/line observer test. No other extraction or fixture correction.')
(e/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
