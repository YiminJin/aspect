#!/usr/bin/env python3
"""One bounded production run, retaining input, executable identity and log."""
import argparse,hashlib,json,os,resource,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--lumped',action='store_true');p.add_argument('--spacing',type=float);p.add_argument('--padded',action='store_true');p.add_argument('--full-support',action='store_true');p.add_argument('--ranks',type=int,default=1);p.add_argument('--resume',help='Existing output label');a=p.parse_args()
repo=Path(__file__).resolve().parents[3];here=Path(__file__).resolve().parent
build=repo/'build-propagation-integration';evidence=here/'integration_evidence';evidence.mkdir(exist_ok=True)
output=build/('output-'+(a.resume or a.label));log=evidence/(a.label+'.log');assert not log.exists()
assert output.exists() if a.resume else not output.exists()
prm=build/(a.label+'.prm');plugin=build/'plugin/libproduction_integration.release.so';exe=build/'aspect'
text=f'include {here}/integration.prm\nset Additional shared libraries = {plugin}\nset Output directory = {output}\n'
if a.padded:
 text+=f'include {here}/padded.prm\ninclude {here}/integration.prm\nsubsection Geometry model\n subsection Box\n  set Box origin X coordinate = -0.5\n  set X extent = 2\n  set X repetitions = 2\n end\nend\nsubsection Fault reconstruction\n set Prescribed faults file = {repo}/tests/phase_field_fault_ih.txt\nend\n'
if a.full_support:
 assert not a.padded
 text+='subsection Fault reconstruction\n set Association support = full normal cross-section\nend\n'
if a.lumped:text+='subsection Material model\n subsection Phase field fault\n  set I h projection = row sum lumped\n end\nend\n'
if a.spacing:text+=f'subsection Fault reconstruction\n set Structural point spacing = {a.spacing}\nend\n'
if a.resume:text+='set Resume computation = true\n'
prm.write_text(text)
env=os.environ.copy();env.update(ASPECT_SOURCE_DIR=str(repo),OMP_NUM_THREADS='1',DEAL_II_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
env['PATH']='/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:'+env['PATH']
command=[str(exe),str(prm)]
if a.ranks>1:command=['/opt/openmpi/5.0.6/bin/mpiexec','-n',str(a.ranks)]+command
files=[exe.resolve(),plugin,prm,here/'production_integration.cc',repo/'source/simulator/phase_field.cc',repo/'source/simulator/solver_schemes.cc',repo/'source/material_model/phase_field_fault/history.cc']
input_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
start=time.monotonic()
with log.open('w') as f:
 try:code=subprocess.run(command,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
 except subprocess.TimeoutExpired:code=124
record=dict(command=command,code=code,seconds=time.monotonic()-start,ranks=a.ranks,rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,hashes=input_hashes)
record['output_bytes']=sum(f.stat().st_size for f in output.rglob('*') if f.is_file())
record['checkpoint_bytes']=sum(f.stat().st_size for f in (output/'restart').rglob('*') if f.is_file())
log.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='hashes'},indent=2));raise SystemExit(code)
