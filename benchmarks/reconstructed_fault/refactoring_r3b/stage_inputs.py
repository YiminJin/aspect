#!/usr/bin/env python3
"""Reuse qualified R3a/fixed-restart fixtures, changing output paths only."""
from pathlib import Path
import shutil
root = Path(__file__).resolve().parent
inputs = root/'inputs'
inputs.mkdir(exist_ok=True)
r3a = root.with_name('refactoring_r3a')
fix = root.with_name('restart_fix')
for variant in ('reference', 'candidate'):
    for name in ('stage-j', 'temperature', 'frozen-stress', 'rollback-original', 'rollback-open-top'):
        for ranks in (1, 2):
            text = (r3a/f'inputs/candidate-{name}-{ranks}.prm').read_text()
            text = text.replace(f'refactoring_r3a/output-candidate-{name}-{ranks}',
                                f'refactoring_r3b/output-{variant}-{name}-{ranks}')
            (inputs/f'{variant}-{name}-{ranks}.prm').write_text(text)
for mode in ('legacy-one', 'legacy-two', 'automatic-one', 'automatic-two', 'automatic-split'):
    name = f'candidate-bp3-{mode}.prm'
    text = (fix/'inputs'/name).read_text().replace('restart_fix/output-', 'refactoring_r3b/output-')
    (inputs/name).write_text(text)
for mode in ('create', 'resume'):
    name = f'cohesive-{mode}-one.prm'
    text = (fix/'inputs'/name).read_text().replace('restart_fix/output-', 'refactoring_r3b/output-')
    (inputs/name).write_text(text)
    if mode == 'resume':
        shutil.copytree(root.with_name('restart_investigation')/'output-create-one',
                        root/'output-cohesive-resume-one')
