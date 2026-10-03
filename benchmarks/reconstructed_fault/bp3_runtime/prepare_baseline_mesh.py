#!/usr/bin/env python3
"""Diagnostic only: replay generated leaves through the accepted legacy plugin."""
from prepare_inputs import parse,write,root
import csv
case='small-60-startup'
leaves=[]
for p in (root/('output-'+case)).glob('initial_mesh_*.csv'):
 leaves += [x['cell'] for x in csv.DictReader(p.open())]
assert len(leaves)==10634 and len(set(leaves))==len(leaves)
(root/'inputs/diagnostic-generated-leaves.txt').write_text('\n'.join(sorted(leaves))+'\n')
data={};parse(root/'inputs'/(case+'.prm'),data,[])
data[('Additional shared libraries',)]='$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_geometry/build/bp3/libbp3_restore_150x50.release.so'
data[('Output directory',)]='$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_runtime/output-baseline-graded'
data[('Postprocess','List of postprocessors')]='reconstructed fault BP3, particles, reconstructed faults, BP3 output complete, BP3 restored monitor'
data[('Mesh refinement','Strategy')]='BP3 saved mesh'
data[('Mesh refinement','BP3 saved mesh','Target cells file')]='$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_runtime/inputs/diagnostic-generated-leaves.txt'
data[('Postprocess','BP3 restored monitor','Stationary profile file')]='$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3/fixtures/bp3_150x50/profile.txt'
data[('Postprocess','BP3','Bottom normalization completion file')]=''
write(data,root/'inputs/baseline-graded.prm')
