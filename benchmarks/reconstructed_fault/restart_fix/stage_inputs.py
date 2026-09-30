#!/usr/bin/env python3
"""Prepare matched inputs; retain the original checkpoints and observers."""
from pathlib import Path
import shutil
root = Path(__file__).resolve().parent
inputs = root/'inputs'
inputs.mkdir(exist_ok=True)
r3a = root.with_name('refactoring_r3a')
investigation = root.with_name('restart_investigation')
for mode in ('legacy-one', 'legacy-two', 'automatic-one', 'automatic-two', 'automatic-split'):
    name = f'candidate-bp3-{mode}.prm'
    text = (r3a/'inputs'/name).read_text()
    text = text.replace('refactoring_r3a/output-', 'restart_fix/output-')
    (inputs/name).write_text(text)
for mode in ('create', 'resume'):
    text = (investigation/'inputs/create-one.prm').read_text()
    text = text.replace('restart_investigation/output-create-one', f'restart_fix/output-cohesive-{mode}-one')
    if mode == 'resume':
        text += '\nset Resume computation = true\n'
        shutil.copytree(investigation/'output-create-one', root/'output-cohesive-resume-one')
    (inputs/f'cohesive-{mode}-one.prm').write_text(text)
