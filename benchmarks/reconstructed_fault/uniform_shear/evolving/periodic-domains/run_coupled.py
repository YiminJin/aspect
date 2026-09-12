"""One existing K3 fault32 replay after the periodic phase/action gates pass."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

here=Path(__file__).resolve().parent
root=here.parents[4]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case',default='spatial0375_n128_f32_periodic')
parser.add_argument('--cap',type=float,default=300)
parser.add_argument('--purpose',default='Does periodic-domain correction restore whole-fault homogeneity through the former 3-s failure?')
parser.add_argument('--library',type=Path,default=here/'coupled-build/libuniform_shear.release.so')
args=parser.parse_args()
name=args.case
assert Path(name).name==name and (here.parent/f'{name}.prm').is_file()
assert not (here.parent/name).exists(), 'Preserve prior coupled evidence.'
command=[str(root/'build-pf-cpdi/aspect-release'),str(here.parent/f'{name}.prm')]
start=time.monotonic()
with (here.parent/f'{name}.log').open('w') as log:
    process=subprocess.Popen(command,cwd=args.library.resolve().parent,stdout=log,stderr=subprocess.STDOUT,
                             start_new_session=True,
                             env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                                  'ASPECT_FAULT_PERFORMANCE':'1'})
    try: status=process.wait(timeout=args.cap)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL)
        process.wait()
        status=124
usage=resource.getrusage(resource.RUSAGE_CHILDREN)
report=dict(command=command,status=status,wall_seconds=time.monotonic()-start,
            peak_rss_KiB=usage.ru_maxrss,wall_cap_s=args.cap,
            purpose=args.purpose,
            sha256={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [root/'build-pf-cpdi/aspect-release',args.library.resolve(),
                                 root/'source/particle/particle_domain.cc',root/'source/reconstructed_fault/manager.cc',
                                 root/'source/simulator/phase_field.cc']})
(here.parent/f'{name}.resources.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
raise SystemExit(status)
