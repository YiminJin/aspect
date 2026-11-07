#!/usr/bin/env python3
"""Flatten the retained native growth fixture without its historical observer plugins."""
from pathlib import Path
import sys,os,json,hashlib,subprocess,time,signal,argparse
root=Path(__file__).resolve().parents[4]
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('label');parser.add_argument('--ranks',type=int,choices=[1,2,4],default=1);parser.add_argument('--end',type=float,default=.53125);args=parser.parse_args()
label=args.label;assert all(c.isalnum() or c in '-_' for c in label)
out=root/'build-particle-free/dgq2-phase'/label
out.mkdir(exist_ok=False)
sources={}
def expand(path):
 text=path.read_text();sources[str(path.relative_to(root))]=hashlib.sha256(text.encode()).hexdigest()
 lines=[]
 for line in text.splitlines():
  stripped=line.strip()
  if stripped.startswith('include '):
   name=stripped[8:].replace('$ASPECT_SOURCE_DIR',str(root));lines.extend(expand(Path(name)))
  elif stripped.startswith('set Additional shared libraries') and 'dgq2-phase/operators' not in line:continue
  elif stripped.startswith('set List of postprocessors') and 'DG native phase checks' not in line:
   lines.append(' set List of postprocessors = particles')
  else:lines.append(line)
 return lines
text='\n'.join(expand(Path(__file__).with_name('growth.prm')))+'\n'
text+=f'\nset Output directory = {out}/output\nset End time = {args.end:.17g}\n'
(out/'input.prm').write_text(text)
exe=root/'build-particle-free/dgq2-phase/native/aspect-release'
plugin=root/'build-particle-free/dgq2-phase/operators/libdg_native_phase_checks.release.so'
command=['/opt/openmpi/5.0.6/bin/mpirun','-np',str(args.ranks),str(exe),str(out/'input.prm')]
record=dict(command=command,sources=sources,core_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
            checker_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),input_sha256=hashlib.sha256(text.encode()).hexdigest())
(out/'command.json').write_text(json.dumps(record,indent=2)+'\n')
env=os.environ.copy();env['ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION']='1'
record['environment']={'ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION':'1'}
start=time.monotonic()
with (out/'run.log').open('w') as log:
 process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:code=process.wait(timeout=1100)
 except subprocess.TimeoutExpired:
  os.killpg(process.pid,signal.SIGTERM)
  try:process.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
  code=124
record.update(exit_code=code,seconds=time.monotonic()-start)
(out/'command.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record|dict(sources='see command.json')),flush=True)
raise SystemExit(code)
