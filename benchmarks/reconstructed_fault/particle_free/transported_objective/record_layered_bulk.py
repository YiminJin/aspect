#!/usr/bin/env python3
"""Compact reproducibility bundle; never prune raw campaigns, binaries or checkpoints."""
from pathlib import Path
import hashlib,json,subprocess,shutil
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/layered_bulk_campaign'
BUILD=ROOT/'build-particle-free/transported-objective/stage-c'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 entry=json.loads((OUT/'entry.json').read_text());candidate=json.loads((OUT/'candidate.json').read_text())
 protected=entry['binaries']|candidate['binaries']
 supplemental={}
 for p in sorted(OUT.glob('candidate-*.json')):
  d=json.loads(p.read_text());supplemental[p.name]=d;protected.update(d['binaries'])
 bookmark=OUT/'adaptive-intermediate-checkpoint.json'
 if bookmark.exists():
  d=json.loads(bookmark.read_text())
  protected.update({str(Path(d['checkpoint_directory'])/p):h for p,h in d['files_sha256'].items()})
 old=json.loads((OUT.parent/'exterior_stress/final-provenance.json').read_text())
 # Old pre-correction source hashes are not this task's starting state.
 # Protect retained binaries/raw evidence; current core source is recorded in candidate.json.
 protected.update({p:h for p,h in old['preserved'].items()
                   if p.startswith(('build-particle-free/','benchmarks/reconstructed_fault/particle_free/evidence/')) or Path(p).is_absolute()})
 failures=[p for p,h in protected.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
 assert not failures,failures
 commands={};hashes={}
 runs=sorted((BUILD/'material-runs').glob('layered-bulk-*'))+[BUILD/'material-runs/exterior-compatible0125-bulk']
 for run in runs:
  if not (run/'command.json').exists():continue
  cmd=json.loads((run/'command.json').read_text());commands[run.name]=cmd
  dest=OUT/'runs'/run.name;dest.mkdir(parents=True,exist_ok=True)
  files=[p for p in run.iterdir() if p.is_file()]+[run/'output'/n for n in ['material-summary.csv','material-model.csv','layered-background.csv','exterior-stress.csv']]
  for p in files:
   if p.is_file():shutil.copy2(p,dest/p.name)
  for p in sorted((run/'output').glob('*.csv')):hashes[str(p.relative_to(ROOT))]=sha(p)
 here=Path(__file__).resolve().parent
 source=list(here.glob('*layered*.py'))+[here/'run_exterior_stress.py',here/'native/material_checks.cc']+list((here/'layered_background').glob('*'))
 source+=list((ROOT/'doc/reconstructed_fault/particle_free').glob('LAYERED_*.md'))
 source+=[ROOT/p for p in subprocess.check_output(['git','diff','--name-only'],text=True).splitlines()]
 for p in source:
  if p.is_file():hashes[str(p.relative_to(ROOT))]=sha(p)
 dest=OUT/'commands';dest.mkdir(exist_ok=True)
 for p in (ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence').glob('layered-bulk-*'):
  if p.is_file():shutil.copy2(p,dest/p.name)
 (OUT/'final.diff').write_bytes(subprocess.check_output(['git','diff','--binary']))
 for p in sorted(OUT.iterdir()):
  if p.is_file() and p.name!='manifest.json':hashes[str(p.relative_to(ROOT))]=sha(p)
 record=dict(utc=datetime.now(timezone.utc).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),root=str(ROOT),preserved=protected,candidate=candidate,supplemental_candidates=supplemental,runs=commands,files_sha256=hashes,max_mpi_ranks=4,max_build_workers=4,reference_provenance_overlap="One ~.04-s scalar reference/provenance invocation overlapped the bridge. One .44-s JSON summary overlapped a serial pair reduction (two analysis workers, no MPI). Other builds/native runs/reductions are sequential.",server_jobs=False,cleanup_losses=['39/45 native-pilot frames','earlier temporary logs'])
 (OUT/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
 print('Protected',len(protected),'artifacts;',len(commands),'run records;',len(hashes),'hashes',flush=True)
if __name__=='__main__':main()
