"""Isolate focused inputs; retain the accepted data and outputs."""
from pathlib import Path
import importlib.util
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prm',root.parent/'bp3_runtime/prepare_inputs.py')
prm=importlib.util.module_from_spec(spec);spec.loader.exec_module(prm)
prefix='$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_birth_completion/'
for case in ['transport-A','transport-A-serial']:
 d={};prm.parse(root.parent/'bp3_local_tests/inputs'/f'{case}.prm',d,[])
 d[('Additional shared libraries',)]=prefix+'build/transport/libbp3_birth_transport.release.so'
 d[('Output directory',)]=prefix+'output-'+case+('-final' if case=='transport-A' else '')
 d[('Postprocess','List of postprocessors')]+=', birth identity audit'
 # The checkpoint after step 2 precedes the demonstrated reuse at step 3.
 d[('Checkpointing','Steps between checkpoint')]='3'
 prm.write(d,root/'inputs'/f'{case}.prm')
 d[('Resume computation',)]='true'
 d[('Output directory',)]=prefix+'output-'+case+'-resume'
 d[('Checkpointing','Steps between checkpoint')]='0'
 prm.write(d,root/'inputs'/f'{case}-resume.prm')
for case in ['direct','retry','direct2','retry2','create','staggered','resume','model-only']:
 d={};prm.parse(root.parent/'bp3_runtime/inputs'/f'{case}.prm',d,[])
 d[('Additional shared libraries',)]=prefix+'build/maintained/libbp3_restore_150x50.release.so'
 if case!='model-only': d[('Additional shared libraries',)]+=', '+prefix+'build/observer/libfrozen_birth_observer.release.so'
 d[('Output directory',)]=prefix+'output-'+case+('-final' if case in ('direct','direct2') else '')
 prm.write(d,root/'inputs'/f'{case}.prm')
# Full production configuration; only library/output selection and bounded stop differ.
d={};prm.parse(root.parent/'bp3/production/bp3_fresh.prm',d,[])
d[('Additional shared libraries',)]=prefix+'build/maintained/libbp3_restore_150x50.release.so, '+prefix+'build/qualification/libcompletion_qualification.release.so'
d[('Output directory',)]=prefix+'output-production-qualified'
d[('Postprocess','List of postprocessors')]+=', production completion qualification'
prm.write(d,root/'inputs/production.prm')
d[('Postprocess','Production qualification mode')]='mesh'
d[('Output directory',)]=prefix+'output-production-mesh'
prm.write(d,root/'inputs/production-mesh.prm')

for case in ['A','B']:
 d={};prm.parse(root.parent/'bp3_local_tests/inputs'/f'{case}.prm',d,[])
 d[('Additional shared libraries',)]=prefix+'build/local/libbp3_restore_150x50.release.so'
 d[('Output directory',)]=prefix+'output-local-'+case+'-final'
 d[('Postprocess','BP3','Last accepted step')]='2';d[('End time',)]='0.1'
 prm.write(d,root/'inputs'/f'local-{case}.prm')
for ranks,source in [(2,'transport-A'),(1,'transport-A-serial')]:
 for kind,param in [('retry','Repeat native birth step'),('control','Attach birth identity audit')]:
  d={};prm.parse(root/'inputs'/f'{source}.prm',d,[])
  d[('Output directory',)]=prefix+f'output-transport-{kind}-{ranks}'
  d[('Postprocess',param)]='true' if kind=='retry' else 'false'
  prm.write(d,root/'inputs'/f'transport-{kind}-{ranks}.prm')
(root/'inputs/production-parse.prm').write_text('include $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3/production/bp3_fresh.prm\nset Output directory = '+prefix+'output-production-parse\n')
