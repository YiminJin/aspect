"""Exercise the unpacked production launcher segments on a cropped, resolved box using localhost MPI."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("extracted",type=Path)
p.add_argument("destination",type=Path)
p.add_argument("--ranks",type=int,choices=[1,2],default=1)
p.add_argument("--linear-tolerance",type=float,help="Diagnostic override of only the Stokes linear tolerance.")
args=p.parse_args()
package=args.extracted.resolve()
dest=args.destination.resolve()
dest.mkdir()
sys.path.insert(0,str(package/"tools"))
import run as deployment
os.chdir(dest)
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","DEAL_II_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[k]="1"
# Only the scheduler launcher is substituted. The actual matched binary, generated
# input, termination criteria, checkpoints and startup checker run unchanged.
adapter=dest/"bin"
adapter.mkdir()
mpiexec=shutil.which("mpiexec")
(adapter/"ibrun").write_text(f'#!/bin/sh\nexec "{mpiexec}" -n {args.ranks} "$@"\n')
(adapter/"ibrun").chmod(0o755)
os.environ["PATH"]=str(adapter)+os.pathsep+os.environ["PATH"]
Path("plugins").symlink_to(package/"plugins",target_is_directory=True)
with Path("historical-vertices.txt").open("x") as out:
    subprocess.run([sys.executable,ROOT/"benchmarks/bp3_qd_thrust/make_historical_vertices.py","--height","125"],stdout=out,check=True)
original=(package/"production.prm").read_text()+"""
subsection Geometry model
  subsection Box
    set Box origin X coordinate = -62.5
    set X extent = 250
    set Y extent = 125
    set X repetitions = 4
    set Y repetitions = 2
  end
end
subsection Mesh refinement
  set Minimum refinement level = 0
  set Initial global refinement = 4
end
"""
if args.linear_tolerance is not None:
    original+=f"subsection Solver parameters\n subsection Stokes solver parameters\n set Linear solver tolerance = {args.linear_tolerance}\n end\nend\n"
Path("local-production.prm").write_text(original)
for step in (1,3):
    reasons=deployment.launch(dest,original,step==3,step,time.time()+7200)
    assert "end step" in reasons
    deployment.verify(step,domain=(-62.5,187.5,0.,125.),expected_ids=16)
# Restore production output cadences and postprocessors for the next allocation.
# End this small continuation after one accepted physical step.
from startup import checkpoint, read_statistics
case=Path("historical-output")
frames_before=(case/"solution.pvd").read_bytes()
profiles_before=sorted(case.glob("fault-profile-*.txt"))
bounded=original+"""
subsection Termination criteria
  set Termination criteria = end time, user request, BP3 first event complete, end step
  set End step = 4
end
"""
reasons=deployment.launch(dest,bounded,True,None,time.time()+7200)
assert reasons=={"end step"} and checkpoint()[0]==5
assert len(list(read_statistics(case/"statistics")))==5
assert (case/"solution.pvd").read_bytes()==frames_before and sorted(case.glob("fault-profile-*.txt"))==profiles_before
print("Step 4 continued with production output cadence, preserving the four startup frames.")
