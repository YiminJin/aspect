#!/usr/bin/env python3
"""Prepare a short native-composition fixture from retained inputs; launch nothing."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[4]
BUILD = ROOT / 'build-particle-free/native-phase-integration'
RUNS = ROOT / 'build-particle-free/transported-objective/stage-c/material-runs'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('label')
p.add_argument('--kind', choices=['layered', 'growth'], default='layered')
p.add_argument('--end', type=float, default=.05)
p.add_argument('--level', type=int, default=4)
p.add_argument('--restart', type=Path)
p.add_argument('--slot', default='01')
p.add_argument('--reject', action='store_true')
p.add_argument('--q1', action='store_true', help='Native Q1 comparison; DGQ2 is the default.')
p.add_argument('--schedule')
p.add_argument('--inflow', default='0', help='Boundary composition function, in x,y,t.')
a = p.parse_args()
assert all(c.isalnum() or c == '-' for c in a.label)
out = RUNS / ('layered-phase-integration-' + a.label) if a.kind == 'layered' else BUILD / a.label
out.mkdir(parents=True, exist_ok=False)
if a.kind == 'layered':
    parent = RUNS / 'layered-dgphase-life-control01'
    text = (parent / 'input.prm').read_text().replace(str(parent), str(out))
    shutil.copyfile(parent / 'fault.txt', out / 'fault.txt')
    plugins = [BUILD/'checks/libtransported_material_energy.release.so', BUILD/'provider/liblayered_background.release.so']
    initial = '.35*max(0,1-(x/1.4)^2)^4*max(0,1-((y-.6*x)/.6)^2)^4'
    initial_section = 'set Model name = function\n subsection Function\n  set Function expression = ' + initial + '\n end'
    postprocess = 'material transaction checks, layered background'
else:
    parent = ROOT/'build-particle-free/dgq2-phase/growth03'
    text = (parent/'input.prm').read_text().replace(str(parent), str(out))
    plugins = [BUILD/'operators/libdg_native_phase_checks.release.so']
    initial_section = 'set Model name = native phase seed'
    postprocess = 'DG native phase checks'
# Replace path-bearing launch entries, leaving physical/material controls intact.
text = '\n'.join(line for line in text.splitlines()
                 if not line.strip().startswith(('set Additional shared libraries', 'set Output directory')))
text += '\nset Additional shared libraries = ' + ', '.join(map(str, plugins)) + '\n'
text += f'set Output directory = {out}/output\nset End time = {a.end:.17g}\n'
text += '''
subsection Compositional fields
 set Number of fields = 1
 set Names of fields = phase_field
 set Types of fields = phase field
 set Compositional field methods = phase field
 set Mapped particle properties =
 set List of normalized fields =
end
subsection Discretization
 set Phase polynomial degree = 0
 set Use discontinuous phase discretization = auto
 set Composition polynomial degree = 2
 set Use discontinuous composition discretization = true
end
'''
text += '\nsubsection Initial composition model\n ' + initial_section + '\nend\n'
text += '''
subsection Boundary composition model
 set List of model names = function
 set Fixed composition boundary indicators = left, right, bottom, top
 subsection Function
  set Function expression = 0
 end
end
subsection Material model
 subsection Phase field fault
  set FE initial phase profile file =
  subsection Material initial phase
   set Function expression = 0
  end
 end
end
'''
text += f'\nsubsection Postprocess\n set List of postprocessors = {postprocess}\nend\n'
if a.kind == 'layered':
    text += f'\nsubsection Mesh refinement\n set Initial global refinement = {a.level}\nend\n'
    text += '\nsubsection Checkpointing\n set Steps between checkpoint = 2\nend\n'
text += '\nsubsection Boundary composition model\n subsection Function\n  set Function expression = ' + a.inflow + '\n end\nend\n'
if a.reject:
    text += '\nsubsection Time stepping\n set List of model names = function, material transaction rejection\nend\n'
if a.schedule:
    text += '\nsubsection Time stepping\n subsection Function\n  set Function expression = ' + a.schedule + '\n end\nend\n'
if a.q1:
    text += '\nsubsection Discretization\n set Composition polynomial degree = 1\n set Use discontinuous composition discretization = false\nend\n'
if a.restart:
    (out/'output').mkdir()
    shutil.copytree(a.restart/'output/restart', out/'output/restart')
    (out/'output/restart/last_good_checkpoint.txt').write_text(str(int(a.slot)))
    text += '\nset Resume computation = true\n'
else:
    text += '\nset Resume computation = false\n'
(out/'input.prm').write_text(text)
print(out/'input.prm')
