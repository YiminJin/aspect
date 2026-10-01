#!/usr/bin/env python3
"""Reuse qualified inputs with separate output and candidate plugin paths."""
from pathlib import Path
root=Path(__file__).resolve().parent
repo=root.parents[2]
rel=str(root.relative_to(repo))
inputs=root/'inputs';inputs.mkdir(exist_ok=True)
for mode in ('legacy-one','legacy-two','automatic-one','automatic-two'):
    source=repo/f'benchmarks/reconstructed_fault/maxwell_cleanup/inputs/candidate-bp3-{mode}.prm'
    text=source.read_text().replace('benchmarks/reconstructed_fault/maxwell_cleanup',rel).replace('benchmarks/reconstructed_fault/refactoring_r2b_cache/plugin-build',rel+'/plugin-build')
    (inputs/source.name).write_text(text)
for variant in ('reference','candidate'):
    for ranks in (1,2):
        source=repo/f'benchmarks/reconstructed_fault/refactoring_r3b/inputs/candidate-rollback-original-{ranks}.prm'
        text=source.read_text().replace('benchmarks/reconstructed_fault/refactoring_r3b',rel).replace('output-candidate-',f'output-{variant}-')
        if variant=='candidate':text=text.replace('benchmarks/reconstructed_fault/refactoring_r3a/plugin-build',rel+'/plugin-build')
        (inputs/f'{variant}-rollback-original-{ranks}.prm').write_text(text)
    (inputs/f'{variant}-ordinary-amg.prm').write_text(f'''include $ASPECT_SOURCE_DIR/tests/convection_box_particles.prm
set Output directory = $ASPECT_SOURCE_DIR/{rel}/output-{variant}-ordinary-amg
subsection Solver parameters
  subsection Stokes solver parameters
    set Stokes solver type = block AMG
  end
end
''')
