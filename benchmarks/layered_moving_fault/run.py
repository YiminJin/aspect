"""Run the two-step layered controls into a fresh local directory."""
import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("aspect", "plugin", "retry_plugin", "destination"):
        parser.add_argument(key, type=lambda p: Path(p).resolve())
    parser.add_argument("--mpiexec", default="mpirun")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    base = (here/"layered.prm").read_text()
    args.destination.mkdir(parents=True, exist_ok=False)
    env = os.environ | dict(DEAL_II_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                            ASPECT_SOURCE_DIR=str(here.parents[1]))
    for case in ("crossing-serial", "crossing-mpi", "interior", "principal", "retry", "mpi", "seed", "restart", "zero", "reverse", "constant",
                 "constant-base", "homogeneous-base", "homogeneous-wrapper"):
        run = args.destination/case
        run.mkdir()
        settings = base
        plugins = str(args.plugin)
        if case.startswith("crossing-"):
            # Place a single material ID close to the partition; retain the
            # exact same straight curve, phase profile, loading and age laws.
            x = -.00001
            for name, value, sep in (("Fault vertices", f"{x},{.6*x}", "; "),
                                     ("Initial label ages", f"{10+2*math.sin(2*x):.17g}", ", "),
                                     ("Initial label shear tractions", f"{3.7+.15*math.cos(3*x):.17g}", ", ")):
                line = next(row for row in settings.splitlines() if row.strip().startswith(f"set {name} = "))
                prefix, entries = line.split(" = ")
                data = entries.split(sep)
                data[9] = value
                settings = settings.replace(line, prefix+" = "+sep.join(data))
        elif case == "retry":
            plugins += ","+str(args.retry_plugin)
            settings += """
subsection Time stepping
  set List of model names = function, reject long first fault step
  subsection Function
    set Function expression = if(time==0,.025,.0125)
  end
end
"""
        elif case == "interior":
            settings += "\nsubsection Material model\n subsection Phase field fault\n  set Profile half width = 1.65\n end\nend\n"
        elif case == "seed":
            settings += "\nset End time = .0125\nsubsection Termination criteria\n set Checkpoint on termination = true\nend\n"
        elif case == "restart":
            shutil.copytree(args.destination/"seed/output", run/"output")
            settings += "\nset Resume computation = true\n"
        elif case in ("zero", "reverse", "constant", "constant-base"):
            translation = {"zero": 0, "reverse": -2, "constant": 2, "constant-base": 2}[case]
            settings += f"\nsubsection Postprocess\n subsection Layered background\n  set Translation = {translation}\n"
            settings += f" end\nend\nsubsection Material model\n subsection Layered mature fault\n  set Homogeneous = {str(case.startswith('constant')).lower()}\n end\nend\n"
            if case == "constant-base":
                # Both exterior parents are zero up to the first physical endpoint.
                settings = settings.replace("set Model name = layered mature fault","set Model name = phase field fault")
                settings += "\nset End time = .0125\n"
        elif case.startswith("homogeneous-"):
            # Same inherited Maxwell update in the earlier stationary donor control.
            settings = (here.parent/"fixed_fault/fixed_fault.prm").read_text()
            if case.endswith("wrapper"):
                settings = settings.replace("set Model name = phase field fault", "set Model name = layered mature fault")
            settings += "\nsubsection Material model\n subsection Layered mature fault\n set Homogeneous = true\n end\nend\nsubsection Postprocess\n set List of postprocessors = fixed fault probe\nend\n"
            plugins += ","+str(args.retry_plugin)
        settings += f"\nset Additional shared libraries = {plugins}\nset Output directory = {run/'output'}\n"
        prm = run/"input.prm"
        prm.write_text(settings)
        launch = [args.mpiexec, "-n", "2"] if case in ("mpi","crossing-mpi") else []
        with (run/"log").open("w") as log:
            subprocess.run([*launch, args.aspect, prm], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(case, "completed", flush=True)


if __name__ == "__main__":
    main()
