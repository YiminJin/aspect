"""Start a fresh uniform or locally refined Example 2 calculation.

Use --steps 0 for the matched mesh/field measurement and --steps 1 to attempt
the first physical step. A failed physical solve retains the ASPECT exit status
and the complete timing and solver records in the new destination.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aspect", type=Path)
    parser.add_argument("mode", choices=("uniform", "sector", "disk"))
    parser.add_argument("destination", type=Path)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--ranks", type=int, default=2)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    dst = args.destination.resolve()
    dst.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(here.parent / "granite_fault_growth.prm", dst / "granite_fault_growth.prm")
    for overlay in ("granite_sector.prm", "granite_predictor.prm"):
        shutil.copyfile(here / overlay, dst / overlay)
    base = dict(uniform="granite_fault_growth.prm", sector="granite_sector.prm",
                disk="granite_predictor.prm")[args.mode]
    prm = dst / "startup.prm"
    prm.write_text(f"""include {base}
set Output directory = output
subsection Termination criteria
  set Termination criteria = end time, end step
  set End step = {args.steps}
end
subsection Postprocess
  set List of postprocessors = growing fault history
end
""")
    return subprocess.call([sys.executable, here.parent / "run.py", args.aspect.resolve(), prm,
                            "timed", "--ranks", str(args.ranks)])


if __name__ == "__main__":
    sys.exit(main())
