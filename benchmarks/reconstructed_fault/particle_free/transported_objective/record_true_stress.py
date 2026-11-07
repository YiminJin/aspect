#!/usr/bin/env python3
"""Record final experiment inputs, source and binary identities without cleanup."""
import hashlib,json,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/direct_true_stress'
RUNS=ROOT/'build-particle-free/transported-objective/stage-c/material-runs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    entry=json.loads((OUT/'entry.json').read_text());sources={p:sha(ROOT/p) for p in entry['native_source']}
    checks=ROOT/'benchmarks/reconstructed_fault/particle_free/transported_objective/native'
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in checks.glob('*') if p.is_file()})
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in checks.parent.glob('*true_stress*.py')})
    sources[str((checks.parent/'run_material_native.py').relative_to(ROOT))]=sha(checks.parent/'run_material_native.py')
    expected=dict(entry['saved_controls'])
    previous=json.loads((OUT.parent/'stage_c/entry.json').read_text())
    expected.update({p:v['sha256'] for p,v in previous['protected_files'].items()})
    expected.update({v['copy']:v['sha256'] for v in previous['controls']})
    for group in ['stage_c_coupled','bounded_profile']:
        expected.update(json.loads((OUT.parent/group/'entry.json').read_text())['saved_controls'])
    controls={p:sha(ROOT/p) for p in expected};assert controls==expected
    docs=ROOT/'doc/reconstructed_fault/particle_free'
    for name in ['DIRECT_TRUE_STRESS_REPORT.md','STATUS.md','SOURCE_AUDIT.md','IMPLEMENTATION_MAPS.md','TRANSPORTED_OBJECTIVE_SPEC.md','TRANSPORTED_OBJECTIVE_TASK_STATE.md','REVIEW_PLAN.md','RESEARCH_HANDOFF.md','CODEX_TRUE_STRESS_HANDOFF.md']:
        path=docs/name;sources[str(path.relative_to(ROOT))]=sha(path)
    for name in ['current_design.md','specification.tex','CURRENT_STATUS.md']:
        path=docs.parent/name;sources[str(path.relative_to(ROOT))]=sha(path)
    cases={};inputs={}
    for folder in sorted(RUNS.glob('true-*')):
        if not (folder/'command.json').exists():continue
        cases[folder.name]=json.loads((folder/'command.json').read_text())
        for f in [folder/'command.json',folder/'input.prm',folder/'runner.py',folder/'run.log',*folder.glob('output/material-*.csv'),*folder.glob('output/true-stress*.csv')]:
            if f.is_file():inputs[str(f.relative_to(ROOT))]=sha(f)
    binaries={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'build-particle-free/native/aspect-release',*ROOT.glob('build-particle-free/transported-objective/stage-c/checks/*.so')]}
    report=dict(root=str(ROOT),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources=sources,changed_native_inputs=[p for p in entry['native_source'] if sources[p]!=entry['native_source'][p]],binaries=binaries,controls=controls,cases=cases,input_sha256=inputs,
        limits='Local only, at most four build/test workers, native simulations sequential; no production checkpoints converted. Incoming manual work and all failures retained.')
    (OUT/'final-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    archive=OUT/'implemented-source.tar.xz'
    if archive.exists():raise RuntimeError('Refuse to overwrite final recovery archive')
    with tarfile.open(archive,'w:xz') as tar:
        for p in sources:tar.add(ROOT/p,arcname=p,recursive=False)
    print('Recorded',len(cases),'runs and',len(sources),'source/build/checker inputs; controls unchanged.')
if __name__=='__main__':main()
