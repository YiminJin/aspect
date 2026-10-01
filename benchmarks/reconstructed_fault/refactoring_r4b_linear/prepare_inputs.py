#!/usr/bin/env python3
from pathlib import Path
r=Path(__file__).resolve().parent;repo=r.parents[2];rel=str(r.relative_to(repo));inputs=r/'inputs';inputs.mkdir(exist_ok=True)
for mode in ('legacy-one','legacy-two','automatic-one','automatic-two'):
 p=repo/f'benchmarks/reconstructed_fault/refactoring_r4a/inputs/candidate-bp3-{mode}.prm'
 (inputs/p.name).write_text(p.read_text().replace('benchmarks/reconstructed_fault/refactoring_r4a',rel))
for rank in (1,2):
 p=repo/f'benchmarks/reconstructed_fault/refactoring_r4a/inputs/candidate-rollback-original-{rank}.prm'
 (inputs/p.name).write_text(p.read_text().replace('benchmarks/reconstructed_fault/refactoring_r4a',rel))
for v in ('reference','candidate'):
 plugin='reference-plugin-build' if v=='reference' else 'plugin-build'
 for name,test in [('residual','phase_field_fault_residual_consistency'),('exhaustion','phase_field_fault_linear_exhaustion'),('pressure','phase_field_fault_pressure_gauge'),('gmg','phase_field_fault_stage_i')]:
  include=f'tests/{test}.prm' if name!='gmg' else 'benchmarks/reconstructed_fault/server_gmg/gmg_q1.prm'
  for rank in ((1,) if name=='gmg' else (1,2)):
   (inputs/f'{v}-{name}-{rank}.prm').write_text(f'''include $ASPECT_SOURCE_DIR/{include}
set Additional shared libraries = $ASPECT_SOURCE_DIR/{rel}/{plugin}/lib{test}.release.so
set Output directory = $ASPECT_SOURCE_DIR/{rel}/output-{v}-{name}-{rank}
''')
