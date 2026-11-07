#!/usr/bin/env python3
"""Record controls, failed setup attempts and unchanged protected evidence."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import shutil
import subprocess
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/corner_isolation'
BUILD=ROOT/'build-particle-free/transported-objective/stage-c'
CASES=['corner-prescribed32','corner-prescribed32-r1','corner-prescribed64',
       'corner-coupled050','corner-coupled050-r1','corner-coupled025']


def main():
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    entry=json.loads((OUT/'entry.json').read_text());preserved={}
    for key in ['source_hashes','binaries','protected_files','retained_shear_data']:
        actual={p:sha(ROOT/p) for p in entry[key]};assert actual==entry[key],key;preserved[key]=actual
    assert not subprocess.check_output(['git','diff','--name-only','--','source','include','CMakeLists.txt','cmake','tests'],text=True).strip()
    files=[];commands={}
    for name in CASES:
        run=BUILD/'material-runs'/name;dest=OUT/'runs'/name;dest.mkdir(parents=True,exist_ok=True)
        command=json.loads((run/'command.json').read_text());commands[name]=command
        assert command['arguments']['ranks']==4
        assert command['core_sha256']==entry['binaries']['build-particle-free/native/aspect-release']
        assert command['plugin_sha256']==entry['binaries']['build-particle-free/transported-objective/stage-c/checks/libtransported_material_energy.release.so']
        selected=[run/p for p in ['command.json','input.prm','runner.py','fault.txt','run.log']]
        selected+=[run/'output'/p for p in ['material-summary.csv','material-model.csv','corner-isolation.csv'] if (run/'output'/p).exists()]
        for path in selected:shutil.copy2(path,dest/path.name)
        files+=selected+list((run/'output').glob('material-*.csv'))
    # Verify the valid eight-step prefix of the retained extra-ulp attempt.
    prefix=[]
    old=BUILD/'material-runs/corner-coupled050/output';new=BUILD/'material-runs/corner-coupled050-r1/output'
    for step in range(9):
        for file in list(old.glob(f'material-{step}-rank*.csv'))+[old/f'material-{step}-surface.csv']:
            assert sha(file)==sha(new/file.name),file
            prefix.append(file.name)
    here=Path(__file__).resolve().parent
    files+=list(here.glob('*corner_isolation.py'))+list((here/'corner_inflow').glob('*'))
    files+=[here/p for p in ['assess_sheared_moving_fault.py','compare_bounded_profile.py','diagnose_bounded_energy.py',
                            'assess_material_native.py','isolate_profile_projection.py','isolate_tensor_defect.py']]
    files+=list((BUILD/'corner-inflow').glob('*.so'))+list((BUILD/'corner-inflow').glob('failed-*.cc'))
    files+=[BUILD/'corner-inflow/CMakeCache.txt']
    docs=ROOT/'doc/reconstructed_fault/particle_free'
    files+=[docs/p for p in ['CORNER_ISOLATION_PROTOCOL.md','CORNER_ISOLATION_REPORT.md','STATUS.md','SOURCE_AUDIT.md','REVIEW_PLAN.md','RESEARCH_HANDOFF.md','MATURE_MOVING_FAULT.md']]
    files+=[p for p in OUT.rglob('*') if p.is_file() and p.name!='final-provenance.json']
    record=dict(utc=datetime.now(timezone.utc).isoformat(),root=str(ROOT),revision=entry['revision'],
                branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
                native_source_unchanged=True,existing_binaries_unchanged=True,new_benchmark_DSO_only=True,
                maximum_MPI_ranks=4,native_runs_sequential=True,build_workers=2,commands=commands,
                preserved=preserved,extra_ulp_attempt_valid_prefix_bitwise_equal=prefix,
                failed_attempts={'corner-prescribed32':'Checker norm requested on a ghosted Trilinos vector before transport; fixed with owned-vector copies.',
                                 'corner-coupled050':'Eight valid steps followed by a 5.55e-17 s terminal interval; retained, rerun with existing representable-clock rule.'},
                files_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files)) if p.is_file()},
                data_losses=['Inherited 39/45 native-pilot frames lost.','Earlier temporary logs lost in inherited cleanup.'],
                no_production_or_server_run=True)
    (OUT/'final-provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Preserved native baseline, 63 protected artifacts and all original shear data; recorded new controls and failed setup attempts.')


if __name__=='__main__':main()
