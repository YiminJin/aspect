#!/usr/bin/env python3
"""Prepare retained growth and saddle controls for the bounded rank comparison."""
from pathlib import Path
import shutil

E = Path(__file__).resolve().parent / 'native_mpi_evidence'


def prepare():
    core = E / 'core'
    core.mkdir(exist_ok=False)
    old = E.parent / 'native_solver_evidence'
    shutil.copy2(old / 'core-all/core-6.txt', core / 'core-6.txt')
    (core / 'files.txt').write_text('core-6.txt\n')
    for label, directory, complete in [('core', core, True),
                                        ('saddle', old / 'saddle', False)]:
        source = old / ('core-all-coarse.prm' if complete else 'saddle-coarse.prm')
        text = source.read_text()
        text += ('\nsubsection Postprocess\n subsection P2 native endpoint\n'
                 f'  set Fixed entry input directory = {directory.resolve()}\n'
                 f'  set Verify complete fixed entry = {str(complete).lower()}\n end\nend\n')
        (E / (label + '.prm')).write_text(text)


if __name__ == '__main__':
    prepare()
