#!/usr/bin/env python3
"""Rebind qualified R1 inputs without changing physical/numerical settings."""
from pathlib import Path

root = Path(__file__).resolve().parent
reference = root.with_name('refactoring_r1')
inputs = root / 'inputs'
inputs.mkdir()  # Never overwrite an existing candidate run.
names = ('ih-converged', 'ih-converged-two', 'ih-no-composition-converged',
         'ih-cell', 'ih-cell-two', 'rollback-open-top', 'rollback-open-top-two',
         'bp3-base', 'bp3-one', 'bp3-two', 'bp3-split')
for name in names:
    text = (reference / 'inputs' / f'{name}.prm').read_text()
    text = text.replace('build-refactor-baseline/tests/', 'build-refactor-r2a/tests/')
    for part in ('inputs/', 'plugin-build/', 'output-'):
        text = text.replace('refactoring_r1/' + part, 'refactoring_r2a/' + part)
    (inputs / f'{name}.prm').write_text(text)
for ranks in (1, 2):
    (inputs / f'rollback-original-{ranks}.prm').write_text(
        'include $ASPECT_SOURCE_DIR/tests/phase_field_fault_stage_i_rollback.prm\n'
        'set Additional shared libraries = $ASPECT_SOURCE_DIR/build-refactor-r2a/tests/'
        'libphase_field_fault_stage_i_rollback.release.so\n'
        'set Output directory = $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/'
        f'refactoring_r2a/output-rollback-original-{ranks}\n')
