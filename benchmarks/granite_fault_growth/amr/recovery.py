"""Run or replay the first missed extension on the 18-strip midpoint specimen.

'clean' takes the mesh actually returned by 'miss' and applies the matching
refinement at the accepted initial clock, before the first physical solve.
"""
import argparse
from pathlib import Path
import sys
from shutil import copytree
import subprocess
import csv

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[1] / "growing_fault"))
from parameter_file import public_nine_strip_input

def section(name, body):
    return f"\nsubsection {name}\n{body}\nend\n"


def run(args):
    folder = args.destination.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    text = public_nine_strip_input() + """
set Output directory = output
subsection Material model
  subsection Phase field fault
    set Target strip width = 0.125
  end
end
subsection Termination criteria
  set Termination criteria = end time, end step
  set End step = END_STEP
end
subsection Postprocess
  set List of postprocessors = growing fault history
end
subsection Checkpointing
  set Steps between checkpoint = 1
  set Number of checkpoints to keep = 4
end
""".replace("END_STEP", str(args.steps))
    if args.restart:
        generation = args.restart.resolve(strict=True)
        copytree(generation, folder/"output"/"restart"/generation.name)
        text += "set Resume computation = true\n"
        text += section("Checkpointing", f" set Resume checkpoint = {int(generation.name)}")
    mesh = " set Refinement fraction = 0.01\n set Coarsening fraction = 0\n"
    if args.mode == "miss":
        if args.levels == 2:
            text += section("Geometry model", section("Box", " set X repetitions = 4\n set Y repetitions = 2"))
        # Suppress the initial band only in this deliberate forecast miss.
        # At positive time all physically requested refinement is allowed.
        mesh += f""" set Initial adaptive refinement = {args.levels}
 set Skip solvers on initial refinement = true
 set Strategy = growing fault, maximum refinement function
 subsection Growing fault
  set Coverage = damage threshold
 end
 subsection Maximum refinement function
  set Coordinate system = cartesian
  set Variable names = x,y,t
  set Function expression = if(t>0,{0 if args.block_refinement else args.levels},0)
 end"""
        text += section("Time stepping", " set List of model names = "+
                        ("fault RSF, " if args.cut else "")+"growing fault refinement")
        if args.cut:
            text += section("Time stepping",section("Fault RSF", """ set Maximum slip fraction = 2e-14
 set Report timestep limits = true"""))
    else:
        original = args.missed/"output"
        with (original/"balance-0001.csv").open() as f:
            dt = float(next(csv.DictReader(f))["time"])
        text += f"set Maximum first time step = {dt:.17g}\n"
        # IDs are checked independently; never use the retry's accepted fields
        # or material checkpoint to initialize the clean reference.
        roots = set()
        for filename in original.glob("fields-0001-r*.csv"):
            with filename.open() as f:
                for row in csv.DictReader(f):
                    cell, path = row["cell"].split("_")
                    level = int(path.split(":")[0])
                    if level > 1:
                        raise ValueError("Expected one physical dyadic refinement level")
                    if level == 1:
                        roots.add(int(cell))
        if not roots:
            raise ValueError("The supposed missed prediction did not refine any cell")
        # Locate each root from original native samples; CellId ordering is
        # independent of the physical Cartesian ordering and MPI ownership.
        boxes = {}
        for filename in original.glob("points-0000-r*.csv"):
            with filename.open() as f:
                for row in csv.DictReader(f):
                    root = int(row["cell"].split("_")[0])
                    if root in roots and root not in boxes:
                        boxes[root] = tuple(int(float(row[d])/.25)*.25 for d in ("x", "y"))
        assert set(boxes) == roots
        condition = " | ".join(f"(x>{x} & x<{x+.25} & y>{y} & y<{y+.25})" for x,y in boxes.values())
        mesh += f""" set Initial adaptive refinement = 0
 set Additional refinement times = {dt/2:.17g}
 set Strategy = minimum refinement function
 subsection Minimum refinement function
  set Coordinate system = cartesian
  set Function expression = if({condition},1,0)
 end"""
    text += section("Mesh refinement", mesh)
    prm = folder/"run.prm"
    prm.write_text(text)
    subprocess.run([sys.executable, here.parent/"run.py", args.aspect, prm, "timing", "--ranks",str(args.ranks)], check=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("aspect", type=Path)
    p.add_argument("destination", type=Path)
    p.add_argument("mode", choices=("clean","miss"))
    p.add_argument("--ranks", type=int, default=1)
    p.add_argument("--missed", type=Path)
    p.add_argument("--cut",action="store_true")
    p.add_argument("--steps", type=int, choices=(1,2), default=1)
    p.add_argument("--restart", type=Path, help="Read only this generation in a fresh output tree")
    p.add_argument("--levels", type=int, choices=(1,2), default=1,
                   help="For the miss control, change the starting mesh; keep the target width fixed")
    p.add_argument("--block-refinement", action="store_true",
                   help="Negative control: leave the maximum level at zero even after the miss")
    a=p.parse_args()
    if a.mode=="clean" and not a.missed:
        p.error("clean requires --missed")
    run(a)
