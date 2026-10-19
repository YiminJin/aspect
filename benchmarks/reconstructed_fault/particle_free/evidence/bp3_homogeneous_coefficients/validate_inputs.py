from pathlib import Path
import subprocess,sys,json,hashlib
root=Path.cwd();work=root/'build-particle-free/bp3-homogeneous-coefficients';stage=root/'build-particle-free/bp3-optimization/server-validation/bp3_dg'
parsed={}
for name in ('parameters','resume','startup','startup-resume'):
 command=[sys.executable,'benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py','bp3-coefficients-validate-'+name,'60','env','--chdir='+str(stage),'LD_LIBRARY_PATH=/opt/openmpi/5.0.6/lib','OMP_PROC_BIND=false','./plugin/aspect-release','--validate',name+'.prm']
 subprocess.run(command,check=True)
 stack=[];values={}
 for raw in (stage/(name+'.prm')).read_text().splitlines():
  line=raw.split('#',1)[0].strip()
  if line.startswith('subsection '):stack.append(line[11:].strip())
  elif line=='end':stack.pop()
  elif line.startswith('set '):
   key,value=line[4:].split('=',1);values['/'.join(stack+[key.strip()])]=value.strip()
 assert not stack;parsed[name]=values
allowed={'Resume computation','Postprocess/BP3/Last accepted step','Postprocess/BP3/Graceful wall seconds','Checkpointing/Steps between checkpoint'}
base=parsed['parameters'];differences={}
for name,values in parsed.items():
 changed={key:[base.get(key),values.get(key)] for key in base.keys()|values.keys() if base.get(key)!=values.get(key)}
 assert changed.keys()<=allowed,(name,changed)
 for key,value in {'Maximum first time step':'4e6','Maximum time step':'4e7','Material model/Phase field fault/Initial time step':'4e6','Time stepping/Reconstructed fault time step/Maximum logarithmic state change':'0.2','Material model/Phase field fault/Optimize stationary mature DG':'true'}.items():assert values[key]==value,(name,key)
 differences[name]=changed
for file in (stage/'plugin/build/native/CMakeFiles/bp3_restore_150x50.dir/flags.make',stage/'plugin/build/CMakeFiles/bp3_coupled_checks.dir/flags.make'):
 flags=file.read_text();assert 'ASPECT_BP3_LOCAL_BOTTOM_TEST' not in flags and 'ASPECT_BP3_TEST_NONROOT_WALL_STOP' not in flags
result=dict(all_four_inputs_valid=True,only_allowed_parameter_differences=differences,native_adaptive_controls_unchanged=True,ordinary_plugins_without_test_macros=True,full_mesh_constructed=False,sha256={name+'.prm':hashlib.sha256((stage/(name+'.prm')).read_bytes()).hexdigest() for name in parsed})
(work/'parameter-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
