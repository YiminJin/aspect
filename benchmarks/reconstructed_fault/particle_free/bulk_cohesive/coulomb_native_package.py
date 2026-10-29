#!/usr/bin/env python3
"""Compact verified native-block review package; preserve full solver archives."""
from pathlib import Path
import difflib,hashlib,json,shutil,zipfile
from coulomb_native_checks import ROOT,WORK,OUT,read
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2)+'\n')

entry=json.loads((WORK/'entry.json').read_text())
for name,digest in entry['protected'].items():assert sha(ROOT/name)==digest,name
verification=json.loads((OUT/'verification.json').read_text())
assert verification['MPI_1_4_max_scaled_QP_defect']<=1e-12
assert verification['homogeneous_mean_point_scaled_defect']<=1e-12
assert not verification['homogeneous_pointwise_1e12_identity_passed']
assert len(verification['legacy_native_bitwise_files'])==27
for record in verification['restart'].values():assert len(record['bitwise_files'])==48
for record in json.loads((OUT/'negative-restart.json').read_text()).values():assert record['no_publication'] and record['checkpoint_files_unchanged']

new=['include/aspect/material_model/phase_field_fault/bulk_cohesive_coulomb.h',
     'source/material_model/phase_field_fault/bulk_cohesive_coulomb.cc',
     'benchmarks/reconstructed_fault/particle_free/bulk_cohesive/coulomb_native.cc',
     'benchmarks/reconstructed_fault/particle_free/bulk_cohesive/coulomb_native.prm',
     'benchmarks/reconstructed_fault/particle_free/bulk_cohesive/coulomb_block_reference.cc',
     'benchmarks/reconstructed_fault/particle_free/bulk_cohesive/coulomb_native_checks.py',
     'benchmarks/reconstructed_fault/particle_free/bulk_cohesive/coulomb_native_plot.py',
     'benchmarks/reconstructed_fault/particle_free/bulk_cohesive/coulomb_native_package.py',
     'doc/reconstructed_fault/particle_free/COULOMB_NATIVE_SELECTION.md',
     'doc/reconstructed_fault/particle_free/COULOMB_NATIVE_REPORT.md']
changed=list(entry['baseline_files'])+new
context=['benchmarks/reconstructed_fault/particle_free/bulk_cohesive/native.prm',
         'benchmarks/reconstructed_fault/particle_free/bulk_cohesive/coulomb_points.cc',
         'source/material_model/phase_field_fault/bulk_cohesive.cc',
         'source/material_model/phase_field_fault/bulk_cohesive_tensor.cc',
         'include/aspect/material_model/phase_field_fault/bulk_cohesive.h',
         'include/aspect/material_model/phase_field_fault/bulk_cohesive_tensor.h',
         'doc/reconstructed_fault/particle_free/COULOMB_DIRECTION_FORMULATION.md']
patch=[]
with zipfile.ZipFile(WORK/'entry-files.zip') as baseline:
    for name in ['doc/reconstructed_fault/particle_free/STATUS.md','doc/reconstructed_fault/particle_free/SOURCE_AUDIT.md','doc/reconstructed_fault/CURRENT_STATUS.md']:
        assert (ROOT/name).read_bytes().endswith(baseline.read(name)),name
    for name in changed:
        current=(ROOT/name).read_text();old=baseline.read(name).decode() if name in entry['baseline_files'] else ''
        patch+=difflib.unified_diff(old.splitlines(True),current.splitlines(True),fromfile='a/'+name,tofile='b/'+name)
    old=baseline.read('benchmarks/reconstructed_fault/particle_free/bulk_cohesive/coulomb_point.cc').decode()
    promoted=(ROOT/new[1]).read_text().split('\n',1)[1].replace('#include <aspect/material_model/phase_field_fault/bulk_cohesive_coulomb.h>','#include "coulomb_point.h"')
    assert old==promoted
(OUT/'changes.patch').write_text(''.join(patch))
for name in changed+context:
    dest=OUT/'context'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
for path in WORK.glob('*.prm'):
    dest=OUT/'inputs'/path.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest)
for name in ['run_matrix.py','build.py','build-commands.json','build-records.json','unity_19.cc','unity_53.cc','entry.json','resume-r1-input.json','resume-r4-input.json','negative-inputs.json']:
    dest=OUT/'inputs'/name;shutil.copy2(WORK/name,dest)
report=(ROOT/new[-1]).read_text().replace('../../../benchmarks/reconstructed_fault/particle_free/evidence/coulomb_native/','')
report=report.replace('(COULOMB_NATIVE_SELECTION.md)','(SELECTION.md)')
(OUT/'REPORT.md').write_text(report);shutil.copy2(ROOT/new[-2],OUT/'SELECTION.md')
(OUT/'changed-files.txt').write_text('\n'.join(changed)+'\n'+str(OUT.relative_to(ROOT))+' (compact evidence)\n')

logs=ROOT/'benchmarks/reconstructed_fault/refactoring_boundary/evidence'
commands={p.stem:json.loads(p.read_text()) for p in sorted(logs.glob('coulomb-native-*.json'))}
save('commands.json',commands)
for label in ['coulomb-native-homogeneous-r1','coulomb-native-verification','coulomb-native-verification-data',
              'coulomb-native-release-phase-forbidden','coulomb-native-changed-background','coulomb-native-wrong-mode']:
    path=logs/(label+'.log');dest=OUT/'diagnostics'/path.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest)
failure=WORK/'pre-output/homogeneous-failed-FD'
for path in failure.glob('bulk-tangent*.csv'):
    shutil.copy2(path,OUT/'diagnostics'/path.name)
shutil.copy2(WORK/'points/checks.json',OUT/'promoted-point-checks.json')
save('provenance.json',{'revision':entry['revision'],'preserved_local_work':True,
    'protected_hashes_verified':len(entry['protected']),'qualified_point_body_unchanged':True,
    'native_executable_sha256':sha(WORK/'aspect-release'),
    'native_plugin_sha256':sha(WORK/'plugin/libcoulomb_native_checks.release.so'),
    'phase_active_Coulomb_enabled':False,'strict_reference_match_passed':False,
    'homogeneous_pointwise_1e_minus12_identity_passed':False,
    'legacy_native_control_bitwise_files':27,
    'scope':'Fixed-phase, fixed-mesh confined small block; not a server package or specimen accuracy qualification.',
    'build':'Two affected native unity groups recompiled; unchanged native objects reused. Compiler commands included; no fresh full-ASPECT build claim.',
    'full_native_archives':str(WORK.relative_to(ROOT))})
save('units.json',{'time':'s','coordinates':'m','stress_and_pressure':'Pa','plane_angles':'radians modulo pi',
    'energies_and_native_work_columns':'J/m (per unit out-of-plane thickness)',
    'system_slip_speed_and_W':'1/s','A_K_L':'dimensionless strains','physical_averages':'Gauss6 JxW area averages',
    'activity_display_threshold':'1e-10/s, diagnostic only; no phase mask or constitutive threshold change'})
for path in OUT.rglob('*.json'):json.loads(path.read_text())
manifest={str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'}
save('manifest.json',manifest)
archive=OUT.with_suffix('.zip');assert not archive.exists(),'Never overwrite completed evidence'
partial=archive.with_suffix('.zip.partial')
with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file():z.write(p,str(p.relative_to(OUT)))
with zipfile.ZipFile(partial) as z:
    assert z.testzip() is None
    for name,record in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==record['sha256']
partial.replace(archive)
archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive)},indent=2))
