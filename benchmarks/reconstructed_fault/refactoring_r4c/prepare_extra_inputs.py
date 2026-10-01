#!/usr/bin/env python3
from pathlib import Path
import json
r=Path(__file__).resolve().parent;repo=r.parents[2];rel=str(r.relative_to(repo));inp=r/'inputs';inp.mkdir(exist_ok=True)
archive=repo.parent/'aspect/benchmarks/reconstructed_fault/performance/gmg/frozen-wide-verified-local4'
oldroot=str(repo.parent/'aspect')
(inp/'frozen-clock.csv').write_bytes((archive/'clock.csv').read_bytes())
(inp/'frozen-run.prm').write_text((archive/'run.prm').read_text().replace(oldroot,str(repo)))
for v in ('reference','candidate'):
 plugin='reference-plugin-build' if v=='reference' else 'plugin-build'
 ordinary=(repo/'benchmarks/reconstructed_fault/refactoring_r4a/inputs/candidate-ordinary-amg.prm').read_text()
 (inp/f'{v}-ordinary-amg.prm').write_text(ordinary.replace('benchmarks/reconstructed_fault/refactoring_r4a/output-candidate-ordinary-amg',f'{rel}/output-{v}-ordinary-amg'))
 for name,test,lib in [('bfbt','nsinker_bfbt','nsinker_bfbt'),('melt','melt_transport_compressible_iterative','melt_transport_compressible_iterative'),('fail-s','stokes_solver_fail_S',None),('fail-budget','stokes_solver_fail',None)]:
  s=f'include $ASPECT_SOURCE_DIR/tests/{test}.prm\nset Output directory = $ASPECT_SOURCE_DIR/{rel}/output-{v}-{name}\n'
  if lib:s+=f'set Additional shared libraries = $ASPECT_SOURCE_DIR/{rel}/{plugin}/lib{lib}.release.so\n'
  # Select the assembled path under test where the fixture inherits automatic backend selection.
  s+='subsection Solver parameters\n  subsection Stokes solver parameters\n    set Stokes solver type = block AMG\n  end\nend\n'
  (inp/f'{v}-{name}.prm').write_text(s)
 s=f'include $ASPECT_SOURCE_DIR/{rel}/inputs/frozen-run.prm\nset Additional shared libraries = $ASPECT_SOURCE_DIR/{rel}/{plugin}/libbp3_legacy.release.so, $ASPECT_SOURCE_DIR/{rel}/{plugin}/libreconstructed_fault_frozen_gmg.release.so\nset Output directory = $ASPECT_SOURCE_DIR/{rel}/output-{v}-frozen\n'
 (inp/f'{v}-frozen.prm').write_text(s)
 (inp/f'{v}-frozen-replay.prm').write_text(s.replace('libbp3_legacy.release.so','libbp3_frozen_reference.release.so').replace(f'output-{v}-frozen',f'output-{v}-frozen-replay'))
env=json.loads((archive/'probe_provenance.json').read_text())['environment']
env={k:v.replace(oldroot,str(repo)) for k,v in env.items()}
env['ASPECT_BP3_TIMESTEP_SEQUENCE']=str(inp/'frozen-clock.csv')
(r/'evidence/frozen-environment.json').write_text(json.dumps(env,indent=2)+'\n')
