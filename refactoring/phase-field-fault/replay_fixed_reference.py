"""Reduce the frozen mature input and run its original solver in a new directory."""
import argparse
import os
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference", type=Path, help="Unpacked closeout reference directory")
parser.add_argument("destination", type=Path)
args = parser.parse_args()
mature = args.reference.resolve() / "mature"
dest = args.destination.resolve()
dest.mkdir(parents=True, exist_ok=False)
fault = dest / "fault.txt"
fault.write_text("0 0 .4\n0 1 .4\n")
prm = (mature / "input.prm").read_text()
settings = {
    "Nonlinear solver tolerance": "1e-12",
    "Linear solver tolerance": "1e-13",
    "Prescribed faults file": fault,
    "Output directory": dest / "output",
    "Additional shared libraries": mature / "runtime/libtransported_material_energy.release.so",
    "Material tensor inflow": "accepted trace",
    "Material exterior stress identity": "",
    "X extent": 1,
    "X repetitions": 1,
    "Box origin X coordinate": 0,
    "Box origin Y coordinate": 0,
    "Y extent": 1,
    "Initial global refinement": 1,
    "Structural point spacing": 1,
    "Prescribed velocity boundary indicators": "left:function, top x:function, bottom x:function, right x:function",
    "Prescribed traction boundary indicators": "right y:function",
    "List of postprocessors": "material transaction checks",
}
for name, value in settings.items():
    pattern = rf"(?m)^(\s*set {re.escape(name)}\s*=)[^\n]*"
    prm, count = re.subn(pattern, lambda m: f"{m[1]} {value}", prm)
    if count == 0:
        raise ValueError(f"Missing setting {name}")
prm = prm.replace(".55+.45*tanh(y/.35)", "1")
prm = prm.replace(".35*max(0,1-(x/1.4)^2)^4*max(0,1-((y-.6*x)/.6)^2)^4", ".35")
prm = prm.replace("0.4+(0.5)*y; 0", "0;0")
prm = re.sub(r"subsection Postprocess\s+subsection Layered background\s+set Translation = 2.0\s+end\s+end", "", prm)
prm += """
subsection Boundary composition model
 subsection Function
  set Function expression = .35
 end
end
subsection Boundary traction model
 subsection Function
  set Function expression = 0; .02+t
 end
end
"""
parameter_file = dest / "input.prm"
parameter_file.write_text(prm)
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", DEAL_II_NUM_THREADS="1")
with (dest / "run.log").open("w") as log:
    subprocess.run([mature / "runtime/aspect-release", parameter_file], cwd=dest, env=env,
                   stdout=log, stderr=subprocess.STDOUT, check=True)
