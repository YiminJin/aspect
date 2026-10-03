#!/usr/bin/env python3
from prepare_inputs import parse,write,root
base={};parse(root/'inputs/dip-60.prm',base,[])
base={k:v for k,v in base.items() if 'Coupled replenishment' not in k and 'Crossing reference cell' not in k}
base[('Additional shared libraries',)]='$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_runtime/build/libbp3_restore_150x50.release.so'
base[('Postprocess','List of postprocessors')]='reconstructed fault BP3, particles, reconstructed faults, BP3 output complete, BP3 restored monitor'
base[('End time',)]='100';base[('Postprocess','BP3','Last accepted step')]='1'
for name in ['model-only','heterogeneous','legacy-rejected']:
 data=dict(base);data[('Output directory',)]='$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_runtime/output-'+name
 if name=='heterogeneous':data[('Material model','Phase field fault','Critical energy release rates')]='2e7, 4e7'
 if name=='legacy-rejected':data[('Fault reconstruction','Boundary completion')]='legacy'
 write(data,root/'inputs'/(name+'.prm'))
