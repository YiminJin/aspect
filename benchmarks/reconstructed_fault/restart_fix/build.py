#!/usr/bin/env python3
"""Build the restart correction and a regression binary using the pre-fix manager."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,shlex,subprocess
root=Path(__file__).resolve().parent;repo=root.parents[2];base=repo/'build-refactor-r3a';build=repo/'build-restart-fix';e=root/'evidence'
build.mkdir(exist_ok=True); e.mkdir(exist_ok=True)
entries=json.loads((base/'compile_commands.json').read_text()); commands=[]; replacements={}
for unit,debug in [('46','-g1'),('1','-g1')]:
 entry=next(c for c in entries if c['file'].endswith(f'/unity_{unit}_cxx.cxx'))
 args=shlex.split(entry['command']);cmd=[];i=0
 while i<len(args):
  if args[i] in ('-include','-o'):i+=2;continue
  cmd.append(args[i]);i+=1
 output=build/f'unity_{unit}-fix.o';cmd += [debug,'-o',str(output)]
 commands.append((unit,cmd));replacements[f'CMakeFiles/aspect.exe.release.dir/Unity/unity_{unit}_cxx.cxx.o']=str(output)
(e/'fix-compile-commands.json').write_text(json.dumps(dict(commands),indent=2)+'\n')
def run(item):
 unit,cmd=item
 return subprocess.run(['python3',str(root/'run_logged.py'),f'compile-fix-{unit}','600',*cmd]).returncode
with ThreadPoolExecutor(max_workers=2) as pool:
 codes=list(pool.map(run,commands))
assert codes==[0,0],codes
args=shlex.split((base/'CMakeFiles/aspect.exe.release.dir/link.txt').read_text());cmd=[];i=0
while i<len(args):
 arg=args[i]
 if arg.startswith('-Wl,--dependency-file='):i+=1;continue
 if arg=='-o':cmd+=['-o',str(build/'aspect-fix')];i+=2;continue
 cmd.append(replacements.get(arg,arg));i+=1
(e/'fix-link-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
subprocess.run(['python3',str(root/'run_logged.py'),'link-fix','180',*cmd],cwd=base,check=True)
red = [str(build/'aspect-before-fix') if a == str(build/'aspect-fix') else ('CMakeFiles/aspect.exe.release.dir/Unity/unity_46_cxx.cxx.o' if a == str(build/'unity_46-fix.o') else a) for a in cmd]
subprocess.run(['python3',str(root/'run_logged.py'),'link-before-fix','180',*red],cwd=base,check=True)
