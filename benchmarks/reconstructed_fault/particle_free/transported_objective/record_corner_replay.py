#!/usr/bin/env python3
"""Verify preserved evidence and package the bounded non-affine replay."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/corner_replay'
BUILD=ROOT/'build-particle-free/transported-objective/stage-c'
RUN=BUILD/'material-runs/corner-replay32'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    entry=json.loads((OUT/'entry.json').read_text())
    assert {p:sha(ROOT/p) for p in entry['preserved']}==entry['preserved']
    previous=json.loads((OUT.parent/'corner_isolation/final-provenance.json').read_text())
    extra={p:h for p,h in previous['files_sha256'].items() if p.endswith(('CORNER_ISOLATION_REPORT.md','CORNER_ISOLATION_PROTOCOL.md','.py'))}
    assert {p:sha(ROOT/p) for p in extra}==extra
    assert not subprocess.check_output(['git','diff','--name-only','--','source','include','CMakeLists.txt','cmake','tests'],text=True).strip()
    command=json.loads((RUN/'command.json').read_text());assert command['exit_code']==0 and command['arguments']['ranks']==4
    for key,p in [('core_sha256','build-particle-free/native/aspect-release'),('plugin_sha256','build-particle-free/transported-objective/stage-c/checks/libtransported_material_energy.release.so')]:
        assert command[key]==entry['preserved'][p]
    dest=OUT/'run';dest.mkdir(exist_ok=True)
    selected=[RUN/p for p in ['command.json','runner.py','input.prm','fault.txt','run.log']]
    selected += [RUN/'output'/p for p in ['corner-replay.csv','replay-native-audit.txt','material-summary.csv','material-model.csv']]
    for p in selected:shutil.copy2(p,dest/p.name)
    evidence=ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence'
    for p in evidence.glob('corner-replay-*'):
        if p.is_file():shutil.copy2(p,dest/p.name)
    here=Path(__file__).resolve().parent
    files=list(here.glob('*corner_replay.py'))+list((here/'corner_replay').glob('*'))
    files += [here/'assess_material_native.py',ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py']
    files+=[BUILD/'corner-replay'/p for p in ['libcorner_replay.release.so','CMakeCache.txt']]
    docs=ROOT/'doc/reconstructed_fault/particle_free'
    files += [docs/p for p in ['CORNER_REPLAY_PROTOCOL.md','CORNER_REPLAY_REPORT.md','STATUS.md','SOURCE_AUDIT.md','REVIEW_PLAN.md','RESEARCH_HANDOFF.md','MATURE_MOVING_FAULT.md']]
    files += selected+[p for p in OUT.rglob('*') if p.is_file() and p.name!='final-provenance.json']
    record=dict(utc=datetime.now(timezone.utc).isoformat(),root=str(ROOT),branch=entry['branch'],revision=entry['revision'],
                native_source_and_binaries_unchanged=True,protected_artifacts_unchanged=True,
                previous_corner_and_shear_evidence_unchanged=True,maximum_MPI_ranks=4,build_workers=2,
                no_physical_steps_or_restart=True,initialization_only=True,command=command,
                inherited_data_losses=['39/45 native-pilot frames','earlier temporary logs'],
                preserved_count=len(entry['preserved']),preserved=entry['preserved'],
                additional_prior_report_and_script_checks=extra,
                files_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files)) if p.is_file()})
    (OUT/'final-provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Verified',len(entry['preserved']),'preserved files; recorded scratch replay and provenance.')
if __name__=='__main__':main()
