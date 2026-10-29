#!/usr/bin/env python3
"""Compact review package; native checkpoints remain outside the archive."""
from pathlib import Path
import hashlib,json,shutil,zipfile
from coulomb_analysis import ROOT,WORK,OUT

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2)+'\n')

entry=json.loads((WORK/'entry.json').read_text())
for name,expected in entry['protected'].items():
    assert digest(ROOT/name)==expected,name
changed=[ROOT/'benchmarks/reconstructed_fault/particle_free'/name for name in
         ['CMakeLists.txt','bulk_cohesive/CMakeLists.txt','bulk_cohesive/coulomb_point.h',
          'bulk_cohesive/coulomb_point.cc','bulk_cohesive/coulomb_points.cc',
          'bulk_cohesive/coulomb_archive.cc','bulk_cohesive/coulomb_analysis.py','bulk_cohesive/coulomb_package.py']]
changed += [ROOT/'doc/reconstructed_fault'/name for name in
            ['particle_free/COULOMB_DIRECTION_FORMULATION.md','particle_free/COULOMB_DIRECTION_REPORT.md',
             'particle_free/SOURCE_AUDIT.md','particle_free/STATUS.md','CURRENT_STATUS.md']]
context=[ROOT/name for name in
         ['source/material_model/phase_field_fault/bulk_cohesive.cc',
          'source/material_model/phase_field_fault/bulk_cohesive_tensor.cc',
          'include/aspect/material_model/phase_field_fault/bulk_cohesive.h',
          'include/aspect/material_model/phase_field_fault/bulk_cohesive_tensor.h',
          'benchmarks/reconstructed_fault/particle_free/bulk_cohesive/tensor_points.cc',
          'benchmarks/reconstructed_fault/particle_free/bulk_cohesive/p5c_campaign.cc',
          'benchmarks/reconstructed_fault/particle_free/bulk_cohesive/p5c_native.cc',
          'benchmarks/reconstructed_fault/particle_free/bulk_cohesive/propagation.cc',
          'benchmarks/reconstructed_fault/particle_free/bulk_cohesive/parallel.cc']]
for path in changed+context:
    target=OUT/'context'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
report=(ROOT/'doc/reconstructed_fault/particle_free/COULOMB_DIRECTION_REPORT.md').read_text()
report=report.replace('../../../benchmarks/reconstructed_fault/particle_free/evidence/coulomb_direction/','')
report=report.replace('(COULOMB_DIRECTION_FORMULATION.md)','(FORMULATION.md)')
report=report.replace('[review archive](../../../benchmarks/reconstructed_fault/particle_free/evidence/coulomb_direction.zip)','review archive (this package)')
(OUT/'REPORT.md').write_text(report)
shutil.copy2(ROOT/'doc/reconstructed_fault/particle_free/COULOMB_DIRECTION_FORMULATION.md',OUT/'FORMULATION.md')
commands={p.stem:json.loads(p.read_text()) for p in sorted((ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence').glob('coulomb-*.json'))}
commands['original_constant_control']={'command':['env','ASPECT_P5A_OUTPUT='+str(WORK/'radial-control'),str(WORK/'point/bulk_cohesive_tensor_points')],
                                      'passed':True,'scope':'unchanged original fixture'}
save('run-commands.json',commands)
save('entry.json',entry)
save('archive-inputs.json',json.loads((WORK/'archives.json').read_text()))
for path in WORK.glob('*.prm'):
    target=OUT/'inputs'/path.name;target.parent.mkdir(exist_ok=True);shutil.copy2(path,target)
save('provenance.json',{'revision':entry['revision'],'protected_hashes_verified':len(entry['protected']),
    'native_source_modified':False,'production_defaults_changed':False,'native_coulomb_mode_enabled':False,
    'no_native_physical_advancement':True,'strict_reference_match_passed':False,
    'native_executable_sha256':digest(ROOT/'build-particle-free/p5c-native/aspect-release'),
    'point_executable_sha256':digest(WORK/'point/bulk_cohesive_coulomb_points'),
    'plugin_sha256':digest(WORK/'plugin/libcoulomb_archive_checks.release.so'),
    'point_restart_sha256':digest(WORK/'points-release/restart-point.bin'),
    'build_commands':[['cmake','-S','benchmarks/reconstructed_fault/particle_free/bulk_cohesive','-B',str(WORK/'point'),'-DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++','-DCMAKE_BUILD_TYPE=Release'],
                      ['cmake','--build',str(WORK/'point'),'--target','bulk_cohesive_coulomb_points','bulk_cohesive_tensor_points','-j2'],
                      ['cmake','-S','benchmarks/reconstructed_fault/particle_free','-B',str(WORK/'plugin'),'-DAspect_DIR='+str(ROOT/'build-particle-free/evolving-phase/native'),'-DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++'],
                      ['cmake','--build',str(WORK/'plugin'),'--target','coulomb_archive_checks','-j2']],
    'full_copied_checkpoints_and_point_restarts':str(WORK.relative_to(ROOT)),
    'context':'Review context requires the retained worktree and installed ASPECT/deal.II; not a new server package.'})
(OUT/'changed-files.txt').write_text('\n'.join(str(p.relative_to(ROOT)) for p in changed)+'\n'+str(OUT.relative_to(ROOT))+'/ (compact evidence)\n')
for path in OUT.rglob('*.json'):json.loads(path.read_text())
manifest={str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':digest(p)}
          for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'}
save('manifest.json',manifest)
archive=OUT.with_suffix('.zip');assert not archive.exists(),'Preserve completed evidence'
partial=archive.with_suffix('.zip.partial')
with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as target:
    for path in sorted(OUT.rglob('*')):
        if path.is_file():target.write(path,str(path.relative_to(OUT)))
with zipfile.ZipFile(partial) as target:
    assert target.testzip() is None
    for name,record in manifest.items():assert hashlib.sha256(target.read(name)).hexdigest()==record['sha256']
partial.replace(archive)
archive.with_suffix('.zip.sha256').write_text(digest(archive)+'  '+archive.name+'\n')
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':digest(archive)},indent=2))
