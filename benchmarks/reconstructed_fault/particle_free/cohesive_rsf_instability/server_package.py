#!/usr/bin/env python3
"""Package current source and an accepted native checkpoint. Never advances state."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,os,re,shutil,subprocess,sys,tarfile,shlex
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
NATIVE=HERE.parent/'cohesive_rsf_native';sys.path.insert(0,str(NATIVE))
from package import source_files
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()

def flatten(path,active=None):
 path=path.resolve();active=set() if active is None else active
 if path in active:raise ValueError('Recursive parameter include')
 active=active|{path};lines=[]
 for line in path.read_text().splitlines():
  if line.strip().startswith('include '):
   target=Path(line.strip()[8:]);target=target if target.is_absolute() else ROOT/target
   lines.extend(flatten(target,active))
  else:lines.append(line)
 return lines

def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--build-dir',type=Path,default=ROOT/'build-particle-free/cohesive-rsf-native/native');p.add_argument('--dealii-dir',type=Path);p.add_argument('--compiler');p.add_argument('--mpiexec');args=p.parse_args()
 meta=json.loads((args.run/'execution.json').read_text());assert meta['returncode']==0,'Only accepted, successfully stopped runs may be packaged'
 rows=list(csv.DictReader((args.run/'output/cohesive-index.csv').open()));last=rows[-1];step=int(last['step']);time=float(last['time_s'])
 reference=list(csv.DictReader((args.reference/'output/cohesive-index.csv').open()));assert len(reference)==1 and int(reference[0]['step'])==step,'Require zero-advance reference at the packaged endpoint'
 out=args.output;out.mkdir(parents=True,exist_ok=False)
 shutil.copytree(args.run/'output/restart',out/'initial-checkpoint/restart')
 refstem=Path(reference[0]['stem']);ref=out/'checkpoint-reference';ref.mkdir()
 for f in refstem.parent.glob(refstem.name+'-*.csv'):shutil.copy2(f,ref/f.name)
 shutil.copy2(args.reference/'output/cohesive-index.csv',ref/'cohesive-index.csv')
 # Flatten only maintained inputs; remove all local include/plugin/output paths.
 lines=flatten(HERE/'episode.prm');base='\n'.join(line for line in lines if not line.strip().startswith(('set Additional shared libraries =','set Output directory =','set Resume computation =')))+'\n'
 base+='set Additional shared libraries = ./plugin/build/libcohesive_rsf_native_checks.release.so\n'
 (out/'physics.prm').write_text(base)
 def prm(name,folder,endstep=None,endtime=None,resume=True,wall=1.5,extra=''):
  text='include physics.prm\nset Output directory = ./'+folder+'\nset Resume computation = '+('true' if resume else 'false')+'\n'
  if endtime is not None:text+=f'set End time = {endtime:.17g}\n'
  text+='subsection Termination criteria\n set Termination criteria = end time, end step, user request, wall time\n'+f' set Wall time = {wall}\n'
  if endstep is not None:text+=f' set End step = {endstep}\n'
  text+='end\n'+extra;(out/name).write_text(text)
 prm('parameters.prm','episode-output');prm('resume.prm','episode-output')
 prm('zero.prm','zero-output',endtime=time,wall=.25)
 prm('startup.prm','startup-output',endstep=step+4,wall=.25)
 prm('startup-resume.prm','startup-output',endstep=step+8,wall=.25)
 prm('control.prm','control-output',endstep=step+8,wall=.25)
 prm('temporal.prm','temporal-output',extra='''subsection Time stepping
 subsection Cohesive RSF
  set Maximum continuous history error = .000125
  set Maximum slip increment in Dc = .025
  set Maximum logarithmic state change = .025
  set Maximum damage increment = .005
 end
end
set Maximum time step = 5
''')
 prm('fresh-coarse.prm','fresh-coarse-output',resume=False)
 prm('fresh-fine.prm','fresh-fine-output',resume=False,extra='''subsection Geometry model
 subsection Box
  set X repetitions = 256
  set Y repetitions = 192
 end
end
subsection Cohesive RSF pilot
 set Seed cells = 192
 set Future surface spacing = .03125
end
''')
 (out/'plugin').mkdir();shutil.copy2(HERE/'SERVER_README.md',out/'README.md');shutil.copy2(HERE/'server_verify.py',out/'verify_startup.py')
 analysis=out/'analysis';analysis.mkdir()
 for name in ['analyze.py','event_window.py']:
  shutil.copy2(HERE/name,analysis/name)
 prefix=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/rsf_episode/event'
 if prefix.exists():
  for name in ['event-gauss.csv.gz','event-history.csv','event-summary.json','event-window.png','event-window.gif']:
   if (prefix/name).exists():shutil.copy2(prefix/name,analysis/name)
 qualification=out/'local-qualification';qualification.mkdir()
 for name in ['activation-retry.json','lifecycle.json','continuation.json','provenance.json']:
  if (prefix.parent/name).exists():shutil.copy2(prefix.parent/name,qualification/name)
 (out/'submit.slurm').write_text('''#!/bin/bash
#SBATCH --job-name=rsf-episode
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=02:00:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
set -e
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
ibrun ./plugin/aspect-release parameters.prm
''')
 files=set(source_files());files.update(HERE.glob('*.py'));files.update(HERE.glob('*.prm'));files.update(HERE.glob('*.md'));files.add(HERE/'parameters.json')
 files.update(ROOT/'doc/reconstructed_fault/particle_free'/name for name in ['RSF_EPISODE_ESTIMATOR.md','RSF_EPISODE_REPORT.md'])
 manifest={str(f.relative_to(ROOT)):sha(f) for f in sorted(files) if f.is_file()}
 (out/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 with tarfile.open(out/'source.tar.gz','w:gz',compresslevel=6,dereference=True) as archive:
  for name in manifest:archive.add(ROOT/name,arcname=name,recursive=False)
 cache=args.build_dir/'CMakeCache.txt'
 cache_lines=cache.read_text().splitlines()
 setting=lambda name:next(line.split('=',1)[1] for line in cache_lines if line.startswith(name+':'))
 dealii=args.dealii_dir or Path(setting('DEAL_II_DIR'))
 compiler=args.compiler or setting('CMAKE_CXX_COMPILER')
 mpi=args.mpiexec or os.environ.get('MPIEXEC') or meta['command'][0]
 config=(dealii/'include/deal.II/base/config.h').read_text()
 versions=[line for line in config.splitlines() if re.match(r'#\s*define DEAL_II_(PACKAGE_VERSION|BOOST_VERSION_|P4EST_VERSION_|WITH_SUNDIALS)',line)]
 env=dict(local_dealii=versions,build_cache=[line for line in cache.read_text().splitlines() if line.startswith(('ASPECT_','DEAL_II_DIR:','CMAKE_CXX_COMPILER:','CMAKE_BUILD_TYPE:'))],
  local_compiler=subprocess.check_output(shlex.split(compiler)+['--version'],text=True),local_MPI=subprocess.check_output(shlex.split(mpi)+['--version'],text=True),
  linked_libraries=subprocess.check_output(['ldd',str(args.build_dir/'aspect-release')],text=True))
 (out/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
 checkpoint_hashes={str(f.relative_to(out)):sha(f) for f in (out/'initial-checkpoint').rglob('*') if f.is_file()}
 (out/'checkpoint-manifest.json').write_text(json.dumps(checkpoint_hashes,indent=2)+'\n')
 (out/'manifest.json').write_text(json.dumps(dict(baseline=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_step=step,accepted_time_s=time,ranks=int(meta['actual_ranks']),source_run=str(args.run),source_reference=str(args.reference),source_binary_sha256=meta['binary_sha256'],native_job_submitted=False,first_episode_complete=False,physical_parameters_unchanged=True,checkpoints=checkpoint_hashes),indent=2)+'\n')
 (out/'SHA256SUMS').write_text(''.join(sha(f)+'  '+str(f.relative_to(out))+'\n' for f in sorted(out.rglob('*')) if f.is_file() and f.name!='SHA256SUMS'))
 print(json.dumps(dict(package=str(out),accepted_step=step,accepted_time_s=time,source_files=len(manifest)),indent=2))
if __name__=='__main__':main()
