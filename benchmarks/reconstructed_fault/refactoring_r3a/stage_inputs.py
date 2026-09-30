#!/usr/bin/env python3
"""Rebind existing qualified inputs; preserve numerical settings and assertions."""
from pathlib import Path
root=Path(__file__).resolve().parent; repo=root.parents[2]; inputs=root/'inputs'; inputs.mkdir()
base='$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/refactoring_r3a'
for variant in ('reference','candidate'):
 for name,prm,plugin in [('stage-j','phase_field_fault_stage_j_restart_create','phase_field_fault_stage_j_restart_create'),('temperature','phase_field_fault_stage_j_temperature','phase_field_fault_stage_j_temperature'),('frozen-stress','phase_field_fault_frozen_stress','phase_field_fault_frozen_stress'),('rollback-original','phase_field_fault_stage_i_rollback','phase_field_fault_stage_i_rollback'),('rollback-open-top','phase_field_fault_stage_i_rollback','phase_field_fault_stage_i_rollback')]:
  for rank in (1,2):
   text=f'include $ASPECT_SOURCE_DIR/tests/{prm}.prm\nset Additional shared libraries = {base}/plugin-build/lib{plugin}.release.so\nset Output directory = {base}/output-{variant}-{name}-{rank}\n'
   if name=='rollback-open-top':text+='subsection Boundary velocity model\n set Prescribed velocity boundary indicators = left:function, right:function, bottom:function\nend\n'
   (inputs/f'{variant}-{name}-{rank}.prm').write_text(text)
# Use the existing R2b proposal-3 BP3 PRMs and the exact same plugin libraries.
for name in ('legacy-one','legacy-two','automatic-one','automatic-two','automatic-split'):
 old=root.with_name('refactoring_r2b_cache')/'inputs'/f'candidate-bp3-{name}.prm'
 text=old.read_text()
 text=text.replace(f'refactoring_r2b_cache/output-candidate-bp3-{name}',f'refactoring_r3a/output-candidate-bp3-{name}')
 (inputs/f'candidate-bp3-{name}.prm').write_text(text)
print('Staged original history/rollback cases and matched BP3 inputs.')

# Distinct supplemental case; original Stage-J PRMs above are unchanged.
for variant in ('reference','candidate'):
 text=(inputs/f'{variant}-stage-j-2.prm').read_text().replace(f'output-{variant}-stage-j-2',f'output-{variant}-stage-j-open-top-2')
 text+='\n# Separate supplemental fixture: existing traction-free-top pattern.\nsubsection Boundary velocity model\n set Prescribed velocity boundary indicators = left:function, right:function, bottom:function\nend\n'
 (inputs/f'{variant}-stage-j-open-top-2.prm').write_text(text)
