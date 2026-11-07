#!/usr/bin/env python3
"""Preserve campaign commands, verify native identity, and hash retained evidence."""
from pathlib import Path
import hashlib,json,subprocess,shutil
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/layered_campaign'
BUILD=ROOT/'build-particle-free/transported-objective/stage-c'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    previous=json.loads((OUT.parent/'exterior_stress/final-provenance.json').read_text());source=previous['source_hashes']
    assert {p:sha(ROOT/p) for p in source}==source
    # Historical reports and outputs stay immutable; only maintained current
    # guide/status/spec pages are updated by this explicitly selected task.
    mutable={str(p.relative_to(ROOT)) for p in (ROOT/'doc/reconstructed_fault').rglob('*') if p.name in ['STATUS.md','SOURCE_AUDIT.md','REVIEW_PLAN.md','RESEARCH_HANDOFF.md','MATURE_MOVING_FAULT.md','current_design.md','specification.tex']}
    preserved=previous['preserved'] | {p:h for p,h in previous['files_sha256'].items() if p not in mutable}
    preserved={p:h for p,h in preserved.items() if p not in mutable}
    failures=[p for p,h in preserved.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h];assert not failures,failures
    commands={};files=[]
    runs=sorted((BUILD/'material-runs').glob('layered-*'))+[BUILD/'material-runs/exterior-closure00625']
    for run in runs:
        if not (run/'command.json').exists():continue
        cmd=json.loads((run/'command.json').read_text());commands[run.name]=cmd
        dest=OUT/'runs'/run.name;dest.mkdir(parents=True,exist_ok=True)
        paths=[p for p in run.iterdir() if p.is_file()]
        paths += [run/'output'/n for n in ['material-summary.csv','material-model.csv','layered-background.csv','exterior-stress.csv'] if (run/'output'/n).exists()]
        for p in paths:shutil.copy2(p,dest/p.name)
        files+=paths+list((run/'output').glob('material-*.csv'))+list((run/'output').glob('background-*.csv'))
    here=Path(__file__).resolve().parent
    files+=list(here.glob('*layered*.py'))+list((here/'layered_background').glob('*'))
    files+=list((BUILD/'layered-campaign').rglob('*.so'))+[BUILD/'layered-campaign/check-reference',BUILD/'layered-campaign/provider/CMakeCache.txt']
    for p in (ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence').glob('layered-*'):
        if p.is_file():dest=OUT/'commands';dest.mkdir(exist_ok=True);shutil.copy2(p,dest/p.name)
    files+=list((ROOT/'doc/reconstructed_fault/particle_free').glob('LAYERED_*.md'))
    files+=[ROOT/p for p in mutable if (ROOT/p).is_file()]
    files += [p for p in OUT.rglob('*') if p.is_file() and p.name!='final-provenance.json']
    result=dict(utc=datetime.now(timezone.utc).isoformat(),root=str(ROOT),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      native_source_hashes=source,native_core=sha(BUILD/'exterior-stress/aspect-release'),native_checker=sha(BUILD/'exterior-stress/checks/libtransported_material_energy.release.so'),
      preserved=preserved,commands=commands,max_workers=4,no_server_jobs=True,inherited_losses=['39/45 native-pilot frames','earlier temporary logs'],
      files_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files)) if p.is_file()})
    (OUT/'final-provenance.json').write_text(json.dumps(result,indent=2)+'\n');print('Preserved',len(preserved),'files; recorded',len(commands),'runs.')
if __name__=='__main__':main()
