#!/usr/bin/env python3
"""Materialize local regression inputs; not required by the maintained runtime."""
from pathlib import Path
root=Path(__file__).resolve().parent
repo=root.parents[2]
old=root.with_name('bp3_geometry')
def parse(path, data, stack):
 for line in path.read_text().splitlines():
  s=line.split('#',1)[0].strip()
  if s.startswith('include '):
   parse(Path(s[8:].replace('$ASPECT_SOURCE_DIR',str(repo))),data,stack)
  elif s.startswith('subsection '): stack.append(s[11:].strip())
  elif s=='end': stack.pop()
  elif s.startswith('set '):
   k,v=s[4:].split('=',1); data[tuple(stack+[k.strip()])]=v.strip()
def write(data,path):
 tree={}
 for keys,v in data.items():
  node=tree
  for k in keys[:-1]:node=node.setdefault(k,{})
  node[keys[-1]]=v
 def emit(node,indent=''):
  lines=[]
  for k,v in node.items():
   if isinstance(v,dict):lines += [indent+'subsection '+k]+emit(v,indent+'  ')+[indent+'end']
   else:lines.append(indent+'set '+k+' = '+v)
  return lines
 path.write_text('# Local functional regression, not production BP3 qualification.\n'+'\n'.join(emit(tree))+'\n')
if __name__=='__main__':
 for case in ['dip-60','dip-45-serial','dip-45-reverse-serial','staggered','create','resume','serial-fixed','mpi','retry','direct','retry2','direct2','random-mpi-create','random-mpi-resume','random-mpi-direct']:
  source=old/'inputs'/('filter-'+case+'.prm')
  if not source.exists():continue
  data={};parse(source,data,[])
  data={k:v for k,v in data.items() if not any(x in k for x in ['BP3 saved mesh','Stationary profile file','Bottom normalization completion file'])}
  data[('Mesh refinement','Strategy')]='BP3 fault support'
  data[('Additional shared libraries',)]='$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_runtime/build/libbp3_restore_150x50.release.so, $ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_runtime/build/observer/libfrozen_birth_observer.release.so'
  data[('Postprocess','List of postprocessors')]+=', BP3 profile check'
  data[('Output directory',)]='$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_runtime/output-'+case
  # A self-contained fault file is the only runtime data fixture.
  source=Path(data[('Fault reconstruction','Prescribed faults file')].replace('$ASPECT_SOURCE_DIR',str(repo)))
  dest=root/'inputs'/source.name;dest.write_bytes(source.read_bytes())
  data[('Fault reconstruction','Prescribed faults file')]='$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_runtime/inputs/'+dest.name
  write(data,root/'inputs'/(case+'.prm'))
