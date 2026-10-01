#!/usr/bin/env python3
"""Reuse accepted first-subpass inputs, changing only artifact/output paths."""
from pathlib import Path

root = Path(__file__).resolve().parent
previous = root.with_name('refactoring_r4b_linear')
(root / 'inputs').mkdir(exist_ok=True)
inputs = list((previous / 'inputs').glob('candidate-*.prm'))
assert len(inputs) == 13
for source in inputs:
    (root / 'inputs' / source.name).write_text(
        source.read_text().replace(previous.name, root.name))
