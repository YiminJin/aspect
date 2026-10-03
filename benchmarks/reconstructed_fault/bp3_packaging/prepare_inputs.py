#!/usr/bin/env python3
"""Prepare standalone candidate and bounded tests; never required at runtime."""
from pathlib import Path
import sys,json,hashlib
import importlib.util
spec=importlib.util.spec_from_file_location('section3_prm',Path(__file__).resolve().parents[1]/'bp3_runtime/prepare_inputs.py')
prm=importlib.util.module_from_spec(spec);spec.loader.exec_module(prm)
parse,write=prm.parse,prm.write
root=Path(__file__).resolve().parent;repo=root.parents[2];bp3=root.with_name('bp3')
if __name__=='__main__':
 old={};parse(bp3/'bp3_150x50_first_event.prm',old,[])
 new={k:v for k,v in old.items() if not any(x in k for x in ['BP3 saved mesh','Stationary profile file','Bottom normalization completion file','Mature prestress file'])}
 new.update({
  ('Additional shared libraries',):'$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3/build-maintained/libbp3_restore_150x50.release.so',
  ('Output directory',):'$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3/output-fresh-candidate',
  ('Resume computation',):'false',
  ('Mesh refinement','Strategy'):'BP3 fault support',
  ('Mesh refinement','Initial global refinement'):'9',('Mesh refinement','Initial adaptive refinement'):'0',('Mesh refinement','Minimum refinement level'):'1',
  ('Fault reconstruction','Boundary completion'):'automatic prescribed',
  ('Fault reconstruction','Prescribed faults file'):'$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3/production/fault.txt',
  ('Particles','Particle generator name'):'reference cell',
  ('Particles','Generator','Reference cell','Number of particles per cell per direction'):'4',
  ('Particles','List of particle properties'):'BP3 frozen crack driving force, initial composition, maxwell stress',
  ('Particles','Interpolation scheme'):'BP3 history linear least squares',
  ('Particles','Minimum particles per cell'):'12',('Particles','Maximum particles per cell'):'24',
  ('Particles','Load balancing strategy'):'remove and add particles',
  ('Particles','Particle addition algorithm'):'point density function',('Particles','Particle removal algorithm'):'point density function',
  ('Particles','Point density kernel function'):'cutoff c1 dealii',('Particles','Bandwidth'):'0.3',
  ('Particles','Addition point density function granularity'):'6',
  ('Particles','Interpolator','Linear least squares','Use linear least squares limiter'):'true, false, false, true, true, true',
  ('Particles','Interpolator','Linear least squares','Use boundary extrapolation'):'false',
  ('Time stepping','Reconstructed fault time step','Maximum logarithmic state change'):'0.1',
  ('Nonlinear solver failure strategy',):'cut timestep size',
  ('Time stepping','Repeat on nonlinear solver failure','Cut back factor'):'0.5',
  ('Postprocess','BP3','Write bulk and particle visualization'):'false',
  ('Postprocess','BP3','Weakening region length'):'15000',
  ('Postprocess','BP3 restored monitor','Bottom velocity constraint'):'full',
 })
 production=bp3/'production/bp3_fresh.prm'
 write(new,production)
 production.write_text('''# Fresh 150 x 50 km BP3 candidate; NOT qualified for server execution.
 # Graded-boundary automatic-completion admission remains blocked (Section 3).
 # Physical/cap settings inherited from historical filter20 + first_event inputs.
 # Latest server input is unconfirmed. Provisional explicit choices: unweighted
 # log-state bound 0.1 (not 0.2); nonlinear-failure cutback 0.5 (old input aborted).
 # Native H/Maxwell limiter resolves this displayed mask by field names.
 '''+ '\n'.join(production.read_text().splitlines()[1:])+'\n')
 (bp3/'production/fault.txt').write_bytes((bp3/'fixtures/bp3_150x50/fault.txt').read_bytes())
 diffs=[{'parameter':' / '.join(k),'historical_explicit':old.get(k,'<not explicitly set; native default>'),'candidate':new.get(k,'<removed>')} for k in sorted(old.keys()|new.keys()) if old.get(k)!=new.get(k)]
 (root/'results/production_changes.json').write_text(json.dumps(diffs,indent=2)+'\n')
 # Reuse accepted cases but keep outputs and libraries isolated.
 for case in ['model-only','create','staggered','resume','direct','retry']:
  data={};parse(root.with_name('bp3_runtime')/'inputs'/(case+'.prm'),data,[])
  data[('Additional shared libraries',)]=new[('Additional shared libraries',)]
  if case!='model-only':data[('Additional shared libraries',)]+=', $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_packaging/build/observer/libfrozen_birth_observer.release.so'
  data[('Output directory',)]='$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_packaging/output-'+case
  write(data,root/'inputs'/(case+'.prm'))
 # Toggle only writer selection/cadence: checkpoint histories must stay identical.
 data={};parse(root/'inputs/staggered.prm',data,[])
 data[('Output directory',)]='$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_packaging/output-quiet'
 data[('Postprocess','BP3','Write bulk and particle visualization')]='false'
 data[('Postprocess','BP3','Profile time interval')]='1000000'
 data[('Postprocess','BP3','Heavy output time interval')]='1000000'
 data[('Postprocess','BP3','Audit full state every step')]='false'
 data[('Postprocess','BP3 restored monitor','Write detailed diagnostics')]='false'
 write(data,root/'inputs/quiet.prm')
 # Same checkpoint, new output schedules and writer selection are admissible.
 data={};parse(root/'inputs/resume.prm',data,[])
 data[('Output directory',)]='$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_packaging/output-reschedule'
 data[('Postprocess','BP3','Profile time interval')]='1000000'
 data[('Postprocess','BP3','Write bulk and particle visualization')]='false'
 write(data,root/'inputs/reschedule.prm')
 # No mesh allocation for production parsing.
 (root/'inputs/production-parse.prm').write_text('include $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3/production/bp3_fresh.prm\nset Output directory = $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_packaging/output-production-parse\n')
