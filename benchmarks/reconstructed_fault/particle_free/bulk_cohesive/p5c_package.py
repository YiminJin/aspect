#!/usr/bin/env python3
"""Review package for P5c; full native solver outputs/checkpoints stay separate."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[4]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(work,out):
    work=work.resolve();out=out.resolve();development=work/'development';development.mkdir(exist_ok=True)
    for p in list(out.glob('*progress*'))+[out/'mpi-comparison-progress.json',out/'radial-mpi-records.json',*out.glob('fields-0000*.png')]:
        if p.exists():shutil.move(p,development/p.name)
    sources=[ROOT/'source/material_model/phase_field_fault.cc',ROOT/'benchmarks/reconstructed_fault/particle_free/CMakeLists.txt',ROOT/'benchmarks/reconstructed_fault/particle_free/bulk_cohesive/parallel.cc']
    sources+=list((ROOT/'benchmarks/reconstructed_fault/particle_free/bulk_cohesive').glob('p5c_*'))
    sources=[p for p in sources if p.is_file() and p.suffix in ['.cc','.py','.prm','.txt']]
    for p in sources:
        target=out/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    docs=[ROOT/'doc/reconstructed_fault/particle_free'/name for name in ['P5C_REPORT.md','P5C_SPECIMEN_SELECTION.md','STATUS.md','SOURCE_AUDIT.md','P5B_NATIVE_REPORT.md','P5B_NATIVE_SELECTION.md']]
    docs +=[ROOT/'doc/reconstructed_fault/CURRENT_STATUS.md',ROOT/'doc/reconstructed_fault/propagation/STATUS.md']
    docs +=[ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/p5b_native'/name for name in ['provenance.json','README.md']]
    for p in docs:
        target=out/'documents'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    main_report=ROOT/'doc/reconstructed_fault/particle_free/P5C_REPORT.md'
    (out/'REPORT.md').write_text(main_report.read_text().replace('../../../benchmarks/reconstructed_fault/particle_free/evidence/p5c_native/',''))
    shutil.copy2(ROOT/'doc/reconstructed_fault/particle_free/P5C_SPECIMEN_SELECTION.md',out/'P5C_SPECIMEN_SELECTION.md')
    changed=[str(p.relative_to(ROOT)) for p in sources+docs]
    (out/'changed-files.txt').write_text('\n'.join(changed)+'\n')
    for p in sorted(work.glob('*.prm')):
        target=out/'inputs'/p.name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
    commands=[]
    for p in sorted((ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence').glob('p5c-*.json')):
        commands.append(dict(label=p.stem,**json.loads(p.read_text())))
    (out/'run-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    dependencies=[ROOT/p for p in ['include/aspect/material_model/phase_field_fault.h','include/aspect/material_model/phase_field_fault/bulk_cohesive_internal.h','include/aspect/material_model/phase_field_fault/bulk_cohesive_tensor.h','source/material_model/phase_field_fault/bulk_cohesive_tensor.cc','source/material_model/phase_field_fault/bulk_cohesive_history.cc','source/simulator/solver/bulk_cohesive_phase.cc']]
    references=dependencies+[ROOT/'benchmarks/reconstructed_fault/particle_free/bulk_cohesive'/n for n in ['propagation.cc','native.prm','parallel.prm','mixed_boundary.prm','fracture.prm']]
    for p in references:
        target=out/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    provenance={'revision':'f1f9f29cf40c32d1ddcea000b724375b64853532','dirty_entry_overlay':json.loads((work/'entry.json').read_text()),'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources+dependencies},'binary_sha256':sha(work/'aspect-release'),'plugin_sha256':sha(work/'plugin/libbulk_p5c_checks.release.so'),'build_scope':'P5b qualified executable/link chain, recompile material unity19 for parser eligibility only. Same solver/point objects; plugin rebuilt. No clean ASPECT build claimed.','compiler':'GCC12.4.0','dealII':'9.6.2','MPI':'OpenMPI5.0.6, 4 ranks; small controls also1','strict_reference_match_passed':False,'production_defaults_changed':False,'server_jobs_launched':False,'physical_scope':'bounded finite-body bulk cohesive fixed-geometry, constant-coefficient, no transport; current fixed/radial law only'}
    for p in ['build-commands.json','build-records.json']:
        shutil.copy2(work/p,out/p)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    archive_records=[]
    for p in (work/'archives').glob('*.json'):
        a=json.loads(p.read_text());archive_records.append({k:v for k,v in a.items() if k!='files'})
    (out/'full-outputs.json').write_text(json.dumps({'root':str(work.relative_to(ROOT)),'raw_archives':archive_records,'native_checkpoints':'runs/*/restart; retained and not in review package','analysis_views':'analysis/* are hardlinked common-prefix views; derived only, not simulator state'},indent=2)+'\n')
    for p in out.rglob('*.json'):json.loads(p.read_text())
    manifest={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=out.with_suffix('.zip');assert not archive.exists(),archive;temporary=archive.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(out)))
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
        for name,a in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==a['sha256']
    temporary.replace(archive);archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps({'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(manifest)+1},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('out',type=Path);a=p.parse_args();main(a.work,a.out)
