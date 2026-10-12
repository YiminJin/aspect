#!/usr/bin/env python3
"""Fresh bounded extensions or the single selected native Q2 confirmation."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
from run_native_fe import flatten
from normal_resolution import E,OLD,load

ROOT=Path(__file__).resolve().parents[3]
BUILD=ROOT/'build-propagation-integration'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def native():
    selected=load(E/'selection.json')['selected_cells']
    assert selected and selected%16==0
    out=BUILD/'runs'/f'output-normal-resolution-q2-n{selected}'
    assert not out.exists(),'Preserve existing native confirmation'
    prm=E/'native.prm';prm.write_text(f'''include {OLD}/q2-refined.prm
set Output directory = {out}
subsection Geometry model
 subsection Box
  set Y repetitions = {selected//16}
 end
end
''')
    generated=E/'native-flat.prm';generated.write_text('\n'.join(flatten(prm))+'\n')
    files=['build-propagation-integration/aspect-release','build-propagation-integration/plugin/libnative_fe_strain.release.so','include/aspect/material_model/phase_field_fault.h','source/material_model/phase_field_fault.cc','source/material_model/phase_field_fault/fe_strain.cc','source/simulator/phase_field.cc','benchmarks/reconstructed_fault/propagation/native_fe_strain.cc','tests/phase_field_fault_test_access.h']
    (E/'native-build.json').write_text(json.dumps({n:sha(ROOT/n) for n in files},indent=2)+'\n')
    env=dict(os.environ,ASPECT_SOURCE_DIR=str(ROOT),ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    start=time.perf_counter()
    with (BUILD/'normal-resolution-native.log').open('w') as log:
        child=subprocess.Popen([str(BUILD/'aspect-release'),str(generated)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        _,status,usage=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
    record=dict(cells=selected,returncode=child.returncode,wall_seconds=time.perf_counter()-start,user_seconds=usage.ru_utime,system_seconds=usage.ru_stime,peak_memory_kib=usage.ru_maxrss,input_sha=sha(generated),executable_sha=sha(BUILD/'aspect-release'),plugin_sha=sha(BUILD/'plugin/libnative_fe_strain.release.so'),threads=1,cost_includes_diagnostics=True,resource_measurement='POSIX wait4, individual native child')
    (E/'native-run.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
    if child.returncode:raise SystemExit(child.returncode)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--native',action='store_true');p.add_argument('--cells',type=int,choices=[128,256]);a=p.parse_args()
    if a.native:
        assert a.cells is None;native()
    else:
        assert a.cells is not None
        import quadrature_1d
        quadrature_1d.E=E;quadrature_1d.case(a.cells,2,6)
