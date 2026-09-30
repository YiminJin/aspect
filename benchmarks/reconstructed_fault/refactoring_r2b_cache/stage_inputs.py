#!/usr/bin/env python3
"""Rebind proposal-2 and existing cache fixtures; change only outputs and observers."""
from pathlib import Path
root=Path(__file__).resolve().parent
repo=root.parents[2]
inputs=root/'inputs';inputs.mkdir()
base='$ASPECT_SOURCE_DIR/'+root.relative_to(repo).as_posix()
audit=base+'/plugin-build/libcache_audit.release.so'
bp3='$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/refactoring_boundary/plugin-build/libbp3_restore_150x50.release.so'
for variant in ('reference','candidate'):
 for name in ('cache-remote-one','cache-remote-two','cache-cell-one','cache-cell-two','cache-independent','cache-no-composition','warm-cell-one','warm-cell-two'):
  prm='phase_field_fault_ih_no_composition.prm' if name=='cache-no-composition' else 'phase_field_fault_ih.prm'
  test='verify phase field fault I h' if name.startswith('warm') else 'verify fault I h cache'
  text=f'''include $ASPECT_SOURCE_DIR/tests/{prm}
set Additional shared libraries = {audit}
set Output directory = {base}/output-{variant}-{name}
subsection Solver parameters
 subsection Phase field solver parameters
  set Max nonlinear iterations = 50
 end
end
subsection Postprocess
 set List of postprocessors = particles, {test}, normalization cache audit
end
'''
  if 'cell' in name:
   text+='subsection Material model\n subsection Phase field fault\n  set I h integration backend = cell intervals\n end\nend\n'
  if name=='cache-independent':
   text+='subsection Material model\n subsection Phase field fault\n  set Elastic shear moduli = 1e10\n  set Cohesions = 1e5\n end\nend\n'
  (inputs/f'{variant}-{name}.prm').write_text(text)
 for mode in ('legacy','automatic'):
  source='qualified-legacy-one.prm' if mode=='legacy' else 'auto-bp3-base.prm'
  for ranks in ('one','two'):
   name=f'bp3-{mode}-{ranks}'
   text=f'''include $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/refactoring_boundary/inputs/{source}
set Additional shared libraries = {bp3}, {audit}
set Output directory = {base}/output-{variant}-{name}
subsection Postprocess
 set List of postprocessors = reconstructed fault BP3, visualization, particles, reconstructed faults, BP3 output complete, BP3 restored monitor, normalization cache audit
end
'''
   (inputs/f'{variant}-{name}.prm').write_text(text)
print('Staged matched inputs in',inputs)

# Resume the same accepted automatic checkpoint in separate output branches.
for variant in ('reference','candidate'):
 text=(inputs/f'{variant}-bp3-automatic-two.prm').read_text().replace(f'output-{variant}-bp3-automatic-two',f'output-{variant}-bp3-automatic-split')
 (inputs/f'{variant}-bp3-automatic-split.prm').write_text(text+'set Resume computation = true\n')
