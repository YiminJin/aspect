"""Run the four mesh policies at a common resolution, phase length and horizon.

The preventive variant starts with one deliberately short outer edge of the
fine disk, to put its 3-ell guard between the old and new supported tips.
"""
import argparse
from pathlib import Path
from shutil import copyfile, copytree
import subprocess
import sys


here = Path(__file__).resolve().parent


def run(executable, destination, mode, ranks, case, restart, observer):
    destination.mkdir(parents=True)
    for source in [here.parents[1]/"growing_fault"/"growing_seed.prm", here/(case+".prm")]:
        copyfile(source, destination/source.name)
    text = f"""include {case}.prm
set Output directory = output
set End time = .005
subsection Geometry model
 subsection Box
  set X repetitions = {4 if case=="shoulder" else 8}
  set Y repetitions = {2 if case=="shoulder" else 4}
 end
end
"""
    if mode == "uniform":
        text += """
subsection Mesh refinement
 set Initial global refinement = 1
end
"""
    else:
        coverage = {"sector":"forward sector", "disk":"tip disk", "preventive":"tip disk",
                    "reactive":"damage threshold"}[mode]
        # Limit only the deliberately near-guard case during initialization.
        cap = mode=="preventive"
        text += f"""
subsection Mesh refinement
 set Strategy = growing fault{", maximum refinement function" if cap else ""}
 set Initial adaptive refinement = 1
 set Skip solvers on initial refinement = true
 set Coarsening fraction = 0
 set Refinement fraction = .01
 subsection Growing fault
  set Coverage = {coverage}
 end
"""
        if cap:
            # First underresolved face at x=1.5 or 1.75 m, respectively.
            # Its distance to the physical tip straddles 3 ell only when
            # the accepted phase-supported station advances.
            x_limit=1.5 if case=="shoulder" else 1.75
            text += f"""
 subsection Maximum refinement function
  set Coordinate system = cartesian
  set Variable names = x,y,t
  set Function expression = if(t>0 | x<{x_limit},1,0)
 end
"""
        text += """end
subsection Time stepping
 set List of model names = growing fault refinement
end
"""
    if mode=="preventive":
        text += """
subsection Checkpointing
 set Steps between checkpoint = 1
 set Number of checkpoints to keep = 4
end
"""
    if observer:
        text+=f"""
set Additional shared libraries = {observer.resolve(strict=True)}
subsection Postprocess
 set List of postprocessors = growing fault history, growing remesh history
end
"""
    if restart:
        copytree(restart, destination/"output"/"restart"/restart.name)
        text += f"""
set Resume computation = true
subsection Checkpointing
 set Resume checkpoint = {int(restart.name)}
end
"""
    prm = destination/"run.prm"
    prm.write_text(text)
    subprocess.run([sys.executable, here.parent/"run.py", executable, prm, "timing",
                    "--ranks", str(ranks)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("mode", choices=("uniform","sector","disk","reactive","preventive"))
    parser.add_argument("--ranks",type=int,default=1, choices=(1,2))
    parser.add_argument("--case", choices=("shoulder","compact"), default="shoulder")
    parser.add_argument("--restart",type=Path,help="Copy a retained checkpoint generation to the fresh run")
    parser.add_argument("--observer",type=Path,help="Optional benchmark library for same-clock native observations")
    args=parser.parse_args()
    run(args.executable.resolve(),args.destination.resolve(),args.mode,args.ranks,args.case,args.restart,args.observer)
