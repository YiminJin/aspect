#!/usr/bin/env python3
"""Archive compact commands/results and verify retained experiment identities."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,subprocess
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/exterior_stress'
BUILD=ROOT/'build-particle-free/transported-objective/stage-c'
NAMES=['exterior-zero025','exterior-compatible025','exterior-compatible050','exterior-compatible0125','exterior-fine0125']
CHANGED=['include/aspect/material_model/phase_field_fault.h','source/material_model/phase_field_fault.cc','source/material_model/phase_field_fault/material_history.cc','source/material_model/phase_field_fault/stress_evolution.cc']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    entry=json.loads((OUT/'entry.json').read_text());preserved={p:h for p,h in entry['preserved'].items() if p not in CHANGED}
    assert {p:sha(ROOT/p) for p in preserved}==preserved
    assert sha(ROOT/'build-particle-free/native/aspect-release')==entry['baseline_binary_sha256']
    files=[];commands={}
    for name in NAMES:
        run=BUILD/'material-runs'/name;dest=OUT/'runs'/name;dest.mkdir(parents=True,exist_ok=True)
        command=json.loads((run/'command.json').read_text());assert command['exit_code']==0;commands[name]=command
        paths=[run/p for p in ['command.json','input.prm','runner.py','fault.txt','run.log']]
        paths+=[run/'output'/p for p in ['material-summary.csv','material-model.csv','exterior-stress.csv']]
        for p in paths:shutil.copy2(p,dest/p.name)
        files+=paths+list((run/'output').glob('material-*.csv'))
    guards=BUILD/'exterior-stress/guard-runs'
    for p in guards.rglob('*'):
        if p.is_file() and p.name in ['results.json','input.prm','run.log','tested-runner.py','material-summary.csv','exterior-stress.csv']:
            dest=OUT/'guards'/p.relative_to(guards);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for p in (ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence').glob('exterior-*'):
        if p.is_file():dest=OUT/'commands';dest.mkdir(exist_ok=True);shutil.copy2(p,dest/p.name)
    here=Path(__file__).resolve().parent
    files+=list(here.glob('*exterior*.py'))+list((here/'exterior_stress').glob('*'))+[ROOT/p for p in CHANGED]
    files+=[here/p for p in ['assess_corner_isolation.py','assess_sheared_moving_fault.py','compare_bounded_profile.py','diagnose_bounded_energy.py','assess_material_native.py','isolate_profile_projection.py','isolate_tensor_defect.py']]
    files+=[BUILD/'exterior-stress'/p for p in ['aspect-release','baseline-aspect-release','provider/libexterior_stress.release.so','checks/libtransported_material_energy.release.so','provider/CMakeCache.txt','checks/CMakeCache.txt']]
    files += list((BUILD/'exterior-stress/provider').glob('*.so'))+list((BUILD/'exterior-stress/provider').glob('pre-*.cc'))
    docs=ROOT/'doc/reconstructed_fault/particle_free'
    files+=[docs/p for p in ['EXTERIOR_STRESS_PROTOCOL.md','EXTERIOR_STRESS_REPORT.md','STATUS.md','SOURCE_AUDIT.md','REVIEW_PLAN.md','RESEARCH_HANDOFF.md','MATURE_MOVING_FAULT.md']]
    files += [ROOT/'doc/reconstructed_fault/current_design.md',ROOT/'doc/reconstructed_fault/specification.tex']
    files += list((BUILD/'exterior-stress/guard-runs/restart_match/output').glob('material-*.csv'))
    files += [p for p in OUT.rglob('*') if p.is_file() and p.name!='final-provenance.json']
    record=dict(utc=datetime.now(timezone.utc).isoformat(),root=str(ROOT),revision=entry['revision'],branch=entry['branch'],
                selected_source_files=CHANGED,source_hashes={p:sha(ROOT/p) for p in CHANGED},preserved=preserved,
                core_and_checker_rebuilt_separately=True,retained_core_restored=True,maximum_workers=4,maximum_MPI_ranks=4,
                commands=commands,guards=json.loads((guards/'results.json').read_text()),
                no_server_or_production_resume=True,inherited_losses=['39/45 native-pilot frames','earlier temporary logs'],
                files_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files)) if p.is_file()})
    (OUT/'final-provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Preserved',len(preserved),'prior files; recorded five coupled runs and guards.')
if __name__=='__main__':main()
