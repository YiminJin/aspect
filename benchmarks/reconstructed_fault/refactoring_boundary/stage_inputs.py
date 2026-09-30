#!/usr/bin/env python3
"""Stage isolated inputs, retaining the qualified R1/R2b fixture and plugins."""
from pathlib import Path
root = Path(__file__).resolve().parent
repo = root.parents[2]
inputs = root / 'inputs'
inputs.mkdir(exist_ok=True)
relative = root.relative_to(repo).as_posix()
ref = root.with_name('refactoring_r2b')
for name in ('base', 'one', 'two'):
    text = (ref / 'inputs' / f'bp3-{name}.prm').read_text()
    text = text.replace('refactoring_r2b/inputs/', 'refactoring_boundary/inputs/')
    text = text.replace('refactoring_r2b/output-', 'refactoring_boundary/output-')
    (inputs / f'bp3-{name}.prm').write_text(text)
base = (inputs / 'bp3-base.prm').read_text().replace('refactoring_r2b/plugin-build', 'refactoring_boundary/plugin-build')
base += '''
subsection Fault reconstruction
  set Boundary completion = automatic prescribed
end
subsection Postprocess
  subsection BP3
    set Bottom normalization completion file =
  end
end
'''
(inputs / 'auto-bp3-base.prm').write_text(base)
legacy = (inputs / 'bp3-base.prm').read_text().replace('refactoring_r2b/plugin-build', 'refactoring_boundary/plugin-build')
legacy += f'\nset Output directory = $ASPECT_SOURCE_DIR/{relative}/output-qualified-legacy-one\n'
(inputs / 'qualified-legacy-one.prm').write_text(legacy)
for name in ('one', 'two', 'split'):
    text = f'include $ASPECT_SOURCE_DIR/{relative}/inputs/auto-bp3-base.prm\n'
    text += f'set Output directory = $ASPECT_SOURCE_DIR/{relative}/output-qualified-bp3-{name}\n'
    if name == 'split':
        text += 'set Resume computation = true\n'
    (inputs / f'final-bp3-{name}.prm').write_text(text)
fixtures = {
    'interior': '.3 .5 .6\n.7 .5 .6\n',
    'interior_touch': '.3 .06 .6\n.7 .06 .6\n',
    'perpendicular': '.5 0 .6\n.5 1 .6\n',
    'oblique': '.25 0 .6\n.75 1 .6\n',
    'reversed': '.75 1 .6\n.25 0 .6\n',
    'left_top': '0 .25 .6\n.65 1 .6\n',
    'multiple': '.25 0 .6\n.3 .25 .6\n---\n1 .75 .6\n.75 .8 .6\n',
    'curved': '.25 0 .6\n.35 .35 .6\n.7 .5 .6\n.75 .85 .6\n',
    'corner': '0 0 .6\n.5 .5 .6\n',
    'tangential': '.2 0 .6\n.8 0 .6\n',
    'crossing': '.2 -.2 .6\n.4 .2 .6\n',
}
for name, text in fixtures.items():
    (inputs / f'{name}.txt').write_text(text)
for name in (*fixtures, 'oblique-two', 'multiple-two', 'h-driven', 'material'):
    geometry = name.removesuffix('-two')
    if name in ('h-driven', 'material'):
        geometry = 'oblique'
    text = f'''include $ASPECT_SOURCE_DIR/tests/phase_field_fault_boundary_completion.prm
set Additional shared libraries = $ASPECT_SOURCE_DIR/build-refactor-boundary/tests/libphase_field_fault_boundary_completion.release.so
set Output directory = $ASPECT_SOURCE_DIR/{relative}/output-{name}
subsection Fault reconstruction
  set Prescribed faults file = $ASPECT_SOURCE_DIR/{relative}/inputs/{geometry}.txt
end
'''
    if name == 'material':
        text += '''subsection Material model
  subsection Phase field fault
    set Elastic shear moduli = 1e10, 2e10
  end
end
'''
    (inputs / f'{name}.prm').write_text(text)
print('Staged boundary-completion inputs in', inputs)
