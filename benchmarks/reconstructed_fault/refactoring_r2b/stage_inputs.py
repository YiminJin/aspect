#!/usr/bin/env python3
"""Rebind qualified R2a inputs without changing physical/numerical settings."""
from pathlib import Path

root = Path(__file__).resolve().parent
reference = root.with_name('refactoring_r2a')
inputs = root / 'inputs'
inputs.mkdir()  # Never overwrite an existing candidate run.
names = ('ih-converged', 'ih-converged-two', 'ih-no-composition-converged',
         'ih-cell', 'ih-cell-two',
         'bp3-base', 'bp3-one', 'bp3-two')
for name in names:
    text = (reference / 'inputs' / f'{name}.prm').read_text()
    text = text.replace('build-refactor-r2a/tests/', 'build-refactor-r2b/tests/')
    for part in ('inputs/', 'plugin-build/', 'output-'):
        text = text.replace('refactoring_r2a/' + part, 'refactoring_r2b/' + part)
    (inputs / f'{name}.prm').write_text(text)

# The existing lifecycle calls normalization repeatedly. Disabling only the
# completed-value cache exercises reuse of its unchanged cell traversals.
for name in ('ih-cell', 'ih-cell-two'):
    for variant, source in (('reference', reference / 'inputs'), ('candidate', inputs)):
        text = (source / f'{name}.prm').read_text()
        text += ('\nset Output directory = $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/'
                 f'refactoring_r2b/output-{variant}-{name}-warm\n')
        (inputs / f'{variant}-{name}-warm.prm').write_text(text)
