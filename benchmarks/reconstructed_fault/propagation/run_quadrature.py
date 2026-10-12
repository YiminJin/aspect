#!/usr/bin/env python3
"""Serial, uncontended native cases with per-process cost and provenance."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
E=ROOT/'benchmarks/reconstructed_fault/propagation/quadrature_evidence'
BUILD=ROOT/'build-propagation-integration'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
files=['build-propagation-integration/aspect-release','build-propagation-integration/plugin/libnative_fe_strain.release.so','include/aspect/material_model/phase_field_fault.h','source/material_model/phase_field_fault.cc','source/material_model/phase_field_fault/fe_strain.cc','source/simulator/phase_field.cc','benchmarks/reconstructed_fault/propagation/native_fe_strain.cc']
(E/'native-build.json').write_text(json.dumps({p:sha(ROOT/p) for p in files},indent=2)+'\n')
env=dict(os.environ,ASPECT_SOURCE_DIR=str(ROOT),ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
for label in ['q2','q3','q2-refined']:
    assert not (BUILD/'runs'/('output-quadrature-'+label)).exists(), 'Preserve existing run '+label
    start=time.perf_counter()
    with (BUILD/('quadrature-'+label+'.log')).open('w') as log:
        result=subprocess.Popen([str(BUILD/'aspect-release'),str(E/(label+'-native.prm'))],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        _,status,usage=os.wait4(result.pid,0)
        result.returncode=os.waitstatus_to_exitcode(status)
    record=dict(label=label,returncode=result.returncode,wall_seconds=time.perf_counter()-start,user_seconds=usage.ru_utime,system_seconds=usage.ru_stime,peak_memory_kib=usage.ru_maxrss,input_sha=sha(E/(label+'-native.prm')),executable_sha=sha(BUILD/'aspect-release'),plugin_sha=sha(BUILD/'plugin/libnative_fe_strain.release.so'),threads=1,cost_includes_diagnostics=True,resource_measurement='POSIX wait4, individual native child')
    (E/(label+'-run.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    if result.returncode:raise SystemExit(result.returncode)
