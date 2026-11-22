"""Run one stationary ridge control from its initial state in a fresh directory."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import resource
import time

import numpy as np


def mature_profile(ell):
    """AT1 first integral with the mature curvature-one degradation and peak 0.6."""
    peak = 0.6
    theta = np.linspace(0, np.pi / 2, 5000)
    phase = peak * np.cos(theta)**2
    h = lambda d: d * (1+d) / (1-d)**2
    hp = (1+3*peak) / (1-peak)**3
    integrand = np.empty_like(theta)
    interior = phase[1:-1]
    integrand[1:-1] = (2*ell*peak*np.sin(theta[1:-1])*np.cos(theta[1:-1])
                       / np.sqrt(interior-peak*h(interior)/h(peak)))
    integrand[0] = 2*ell*math.sqrt(peak/(peak*hp/h(peak)-1))
    integrand[-1] = 2*ell*math.sqrt(peak/(1-peak/h(peak)))
    distance = np.r_[0, np.cumsum(np.diff(theta)*(integrand[1:]+integrand[:-1])/2)]
    return distance, phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aspect", type=Path)
    parser.add_argument("plugin", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("case", choices=["vertical", "narrow", "translated", "inclined", "inclined-q1", "inclined-fit", "free"])
    parser.add_argument("grid", type=int, choices=[32, 64])
    parser.add_argument("--ranks", type=int, choices=[1, 2], default=1)
    parser.add_argument("--retry-plugin", type=Path, help="Existing fixed_fault test library for a rejected 25 ms attempt")
    args = parser.parse_args()
    if args.retry_plugin and args.case != "free":
        parser.error("--retry-plugin applies to the free-RSF trajectory")
    dest = args.directory.resolve()
    dest.mkdir(parents=True)
    vertical = args.case in ("vertical", "narrow", "translated")
    ell = 0.078125 if args.case == "narrow" else 0.15625
    center = 0.07 if args.case == "translated" else 0.
    offset = 0.015625 if args.case in ("inclined-fit", "translated") else 0.
    angle = 90 if vertical else 60
    distance, phase = mature_profile(ell)
    table = dest / "profile.txt"
    np.savetxt(table, np.column_stack((distance, phase)))
    ys = np.linspace(-0.5, 0.5, 8 if vertical else 38)
    vertices = "; ".join(f"{center+offset+(0 if vertical else y/math.sqrt(3)):.17g},{y:.17g}" for y in ys)
    free = args.case == "free"
    # Match the donor's mechanical interval, viscosity, modulus and prescribed
    # boundary components for the inclined transfer control. Leave bottom y free
    # for the production pressure datum. Its natural normal traction is zero.
    native = not vertical and not free
    extra = """
subsection Material model
  subsection Simple model
    set Viscosity = 1e20
  end
  subsection Phase field fault
    set Shear modulus = 1e6
    set Background compression = 50000
    set Background shear traction = 0
  end
end
subsection Boundary velocity model
  set Prescribed velocity boundary indicators = left:function, right:function, top:function, bottom x:function
  subsection Function
    set Function expression = 0.005*(-sqrt(3)/2*x+0.5*y);0.005*sqrt(3)*(-sqrt(3)/2*x+0.5*y)
  end
end
subsection Boundary traction model
  set Prescribed traction boundary indicators =
end
""" if native else ""
    dt = 0.1 if native else 0.0125
    end = 2*dt if free else dt
    libraries = str(args.plugin.resolve())
    if args.retry_plugin:
        libraries += ", " + str(args.retry_plugin.resolve())
        extra += """
subsection Time stepping
  set List of model names = function, reject long first fault step
  subsection Function
    set Function expression = if(time==0,0.025,0.0125)
  end
end
"""
    prm = dest / "input.prm"
    prm.write_text(f"""include $ASPECT_SOURCE_DIR/benchmarks/fixed_fault/stationary.prm
set Additional shared libraries = {libraries}
set Output directory = {dest / 'output'}
set End time = {end}
subsection Geometry model
  subsection Box
    set Box origin X coordinate = -0.5
    set Box origin Y coordinate = -0.5
    set X repetitions = 4
    set Y repetitions = 4
  end
end
subsection Mesh refinement
  set Initial global refinement = {3 if args.grid == 32 else 4}
end
subsection Time stepping
  subsection Function
    set Function expression = {dt}
  end
end
subsection Boundary traction model
  subsection Function
    set Function expression = 0;0.02+t+0.12*y
  end
end
subsection Initial composition model
  set List of model names = stationary ridge
  subsection Stationary ridge
    set Profile table = {table}
    set Grid cells = {args.grid}
    set Bilinear donor phase = {str(args.case == "inclined-q1").lower()}
    set Angle = {angle}
    set Center x = {center}
  end
end
subsection Material model
  set Model name = {'phase field fault' if free else 'prescribed ridge diagnostic'}
  subsection Phase field fault
    set Fault vertices = {vertices}
    set Fit trace to phase = {'false' if args.case in ('inclined', 'inclined-q1') else 'true'}
    set Degradation magnitude = {20/ell}
    set Profile half width = {distance[-1]:.17g}
  end
end
subsection Postprocess
  set List of postprocessors = fixed fault statistics, ridge probe
end
{extra}""")
    command = [str(args.aspect.resolve()), str(prm)]
    if args.ranks > 1:
        command = ["mpirun", "-np", str(args.ranks)] + command
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", DEAL_II_NUM_THREADS="1")
    started = time.monotonic()
    with (dest / "run.log").open("w") as log:
        subprocess.run(command, cwd=dest, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if args.ranks == 1:
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        (dest / "cost.txt").write_text(f"wall_seconds {time.monotonic()-started:.3f}\npeak_RSS_KiB {peak}\n")


if __name__ == "__main__":
    main()
