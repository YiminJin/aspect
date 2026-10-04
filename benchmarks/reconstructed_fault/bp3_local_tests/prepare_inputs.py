#!/usr/bin/env python3
from pathlib import Path
import importlib.util
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prm',root.with_name('bp3_runtime')/'prepare_inputs.py')
prm=importlib.util.module_from_spec(spec);spec.loader.exec_module(prm)
base={};prm.parse(root.with_name('bp3_runtime')/'inputs/small-60.prm',base,[])
base.update({
 ('Additional shared libraries',):'$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_local_tests/build/libbp3_restore_150x50.release.so',
 ('Fault reconstruction','Prescribed faults file'):'$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_local_tests/inputs/fault.txt',
 ('Material model','Phase field fault','Initial time step'):'0.05',
 ('Maximum time step',):'0.05',('Maximum first time step',):'0.05',('End time',):'1',
 ('Time stepping','Reconstructed fault time step','Maximum logarithmic state change'):'0.1',
 ('Nonlinear solver failure strategy',):'cut timestep size',
 ('Time stepping','Repeat on nonlinear solver failure','Cut back factor'):'0.5',
 ('Postprocess','List of postprocessors'):'reconstructed fault BP3, reconstructed faults, BP3 output complete, BP3 restored monitor, BP3 local probes',
 ('Postprocess','BP3','Last accepted step'):'20',('Postprocess','BP3','Profile time interval'):'0.05',
 ('Postprocess','BP3','Write bulk and particle visualization'):'false',
 ('Postprocess','BP3 restored monitor','Write detailed diagnostics'):'false',
 ('Checkpointing','Steps between checkpoint'):'10',
 ('Particles','Point density kernel function'):'cutoff c1 dealii',('Particles','Bandwidth'):'0.3',
 ('Particles','Addition point density function granularity'):'6',
})
if __name__=='__main__':
 (root/'inputs/fault.txt').write_bytes((root.with_name('bp3_runtime')/'inputs/small-60-fault.txt').read_bytes())
 for policy in 'AB':
  data=base.copy();data[('Mesh refinement','BP3 local comparison','Policy')]=policy
  for pilot in [True,False]:
   name=('pilot-' if pilot else '')+policy
   data[('Output directory',)]='$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_local_tests/output-'+name
   data[('Postprocess','BP3','Last accepted step')]='1' if pilot else '20'
   prm.write(data,root/'inputs'/(name+'.prm'))
