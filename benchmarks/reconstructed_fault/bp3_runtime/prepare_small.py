#!/usr/bin/env python3
"""Standalone small resolved inputs, including the explicit truncated transition."""
from prepare_inputs import parse,write,root
import math
for dip in (60,45):
 for reverse in (False,True) if dip==45 else (False,):
  case='small-'+str(dip)+('-reverse' if reverse else '')
  data={};parse(root/'inputs/dip-60.prm',data,[])
  changes={
   ('Output directory',):'$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_runtime/output-'+case,
   ('Geometry model','Box','X extent'):'2000',('Geometry model','Box','Y extent'):'1000',
   ('Geometry model','Box','Box origin X coordinate'):'-1000',
   ('Geometry model','Box','X repetitions'):'4',('Geometry model','Box','Y repetitions'):'2',
   ('Mesh refinement','Initial global refinement'):'7',('Mesh refinement','Initial adaptive refinement'):'0',
   ('Mesh refinement','Minimum refinement level'):'1',
   ('Mesh refinement','Skip setup initial conditions on initial refinement'):'true',
   ('Phase field model','Length scale'):'20',
   ('Material model','Phase field fault','Critical energy release rates'):'1e5',
   ('Fault reconstruction','Structural point spacing'):'4',
   ('Postprocess','BP3','Allow truncated transition'):'true',
   ('Postprocess','BP3','Audit full state every step'):'false',
   ('Postprocess','BP3 restored monitor','Write detailed diagnostics'):'true',
   ('Postprocess','BP3','Last accepted step'):'1',
   ('Postprocess','List of postprocessors'):'reconstructed fault BP3, particles, reconstructed faults, BP3 output complete, BP3 restored monitor, BP3 profile check',
   ('End time',):'100',
  }
  data.update(changes)
  path=root/'inputs'/(case+'-fault.txt')
  dx=500/math.tan(math.radians(dip))
  # Native prescribed format retained, including peak=0.6.
  points=[(-dx,1000.),(dx,0.)]
  if reverse:points.reverse()
  path.write_text(''.join(f'{x:.17g} {y:.17g} 0.6\n' for x,y in points))
  data[('Fault reconstruction','Prescribed faults file')]='$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_runtime/inputs/'+path.name
  write(data,root/'inputs'/(case+'.prm'))
