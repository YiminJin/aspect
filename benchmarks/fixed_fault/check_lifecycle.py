"""Compare clean and repeated steps from fresh physical initial conditions.

Build the tests.fixed_fault regression first to produce its observation plugin.
Each invocation uses a fresh destination. No donor files are required.
"""
import argparse
import math
import os
from pathlib import Path
import re
import subprocess

from compare_snapshots import compare_step

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("aspect", type=Path)
parser.add_argument("plugin", type=Path)
parser.add_argument("destination", type=Path)
parser.add_argument("--case", choices=("fixed_fault", "stationary_fault"), default="fixed_fault")
args = parser.parse_args()
binary, plugin, dest = (p.resolve() for p in (args.aspect, args.plugin, args.destination))
dest.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[2]
env = dict(os.environ, OMP_NUM_THREADS="1", DEAL_II_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


def run(name, settings="", expected_error=None):
    folder = dest / name
    folder.mkdir()
    output = folder / "output"
    prm = folder / "input.prm"
    prm.write_text(f"include {root / 'tests' / (args.case + '.prm')}\n"
                   f"set Output directory = {output}\nset Additional shared libraries = {plugin}\n"
                   + settings)
    with (folder / "run.log").open("w") as log:
        result = subprocess.run([binary, prm], cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
    log = (folder / "run.log").read_text()
    if expected_error:
        if not result.returncode or expected_error not in log:
            raise RuntimeError(f"{name}: missing expected diagnostic\n{log[-3000:]}")
    else:
        result.check_returncode()
    return output, log


clean, _ = run("clean")
# Check the file users visualize at the first nonzero-history endpoint.
graphics = clean / "solution/solution-00001.0000.gnuplot"
rows = graphics.read_text().splitlines()
names = next(re.findall(r"<([^>]+)>", row) for row in rows if "<fault_tau_xy>" in row)
samples = [list(map(float, row.split())) for row in rows if row.strip() and not row.startswith("#")]
if not samples or not any(abs(row[names.index("fault_tau_xy")]) > 1.e-3 for row in samples):
    raise ValueError("The physical-output check requires nonzero shear stress")
for row in samples:
    for component in ("xx", "yy", "xy"):
        saved = row[names.index("fault_stress_" + component)]
        physical = row[names.index("fault_tau_" + component)]
        if not math.isfinite(physical) or physical != saved:
            raise ValueError(f"{graphics}: physical {component} disagrees with its accepted DG field")
if args.case == "fixed_fault":
    # At this endpoint kappa*dv_y/dx is only the viscous part of tau_xy.
    viscous = -10.*math.expm1(-.0125)*.098036219175796038
    if any(abs(row[names.index("fault_tau_xy")] - .0325) > 1.e-10 or
           abs(row[names.index("fault_tau_xy")] - viscous) < .01 for row in samples):
        raise ValueError("Homogeneous physical stress failed the nonzero-history comparison")
print("Physical stress matches all three accepted DG fields at", len(samples), "graphical points")

retry, log = run("retry", """
subsection Time stepping
  set List of model names = function, reject long first fault step
  subsection Function
    set Function expression = if(time==0,0.025,0.0125)
  end
end
""")
if log.count("Repeating the current time step") != 1:
    raise RuntimeError("Expected exactly one rejected attempt")
for folder in (clean, retry):
    if (folder / "fault-checks").read_text() != "0 36\n1 36\n2 36\n" or (folder / "restart").exists():
        raise RuntimeError(f"{folder}: unexpected publication or checkpoint")
for step in range(3):
    print("Retry", step, "maximum absolute difference", compare_step(clean, retry, step))

run("resume", "set Resume computation = true\n", expected_error="Phase field fault requires a fresh run")
run("checkpoint", "subsection Checkpointing\n set Steps between checkpoint = 1\nend\n",
    expected_error="Phase field fault requires a fresh run")
run("auto_resume", "set Resume computation = auto\n", expected_error="Phase field fault requires a fresh run")
run("on_termination", "subsection Termination criteria\n set Checkpoint on termination = true\nend\n",
    expected_error="Phase field fault requires a fresh run")
print("Fault requests for continuation and checkpointing diagnosed")

# A loose bulk tolerance accepts a Picard iterate before rates agree. The
# section check must continue the solve inside the ordinary iteration limit.
_, log = run("coupled", "set Nonlinear solver tolerance = 0.05\n"
             "subsection Postprocess\n subsection Fixed fault probe\n"
             "  set Compare homogeneous reference = false\n end\nend\n")
if "Coupled Stokes unknowns require another Picard iteration" not in log:
    raise RuntimeError("The loose bulk stopping criterion did not exercise coupled convergence")

# Keep the donor inputs unchanged. In this independent probe the material is
# evaluated twice at 293 K with a nonzero EOS contrast and positive xx stresses
# of 1 and 1e6 Pa. Its single-lithology density must still be 3300 kg/m^3.
contrast, _ = run("density_contrast", """
subsection Material model
  subsection Simple model
    set Density differential for compositional field 1 = 123
  end
end
""")
for step in range(3):
    print("EOS contrast", step, "maximum absolute difference", compare_step(clean, contrast, step))

permuted, _ = run("permuted", """
subsection Compositional fields
  set Names of fields = phase_field, fault_stress_xy, fault_stress_xx, fault_stress_yy
  set Types of fields = generic, stress, stress, stress
end
subsection Initial composition model
  subsection Function
    set Function expression = 0.35;0;0;0
  end
end
""")
for step in range(3):
    print("Reordered fields", step, "maximum absolute difference", compare_step(clean, permuted, step))

if args.case == "fixed_fault":
    # The frozen homogeneous table is indexed by accepted step and time.
    clock_output, _ = run("wrong_clock", """
subsection Time stepping
  subsection Function
    set Function expression = 0.015
  end
end
""", expected_error="Unexpected homogeneous donor clock at step 1")
    if (clock_output / "fault-checks").read_text() != "0 36\n":
        raise RuntimeError("The off-clock endpoint was reported as checked")

    run("scaled_viscosity", """
subsection Material model
  set Model name = depth dependent
  subsection Depth dependent model
    set Base model = phase field fault
    set Reference viscosity = 10
    set Depth dependence method = Function
    subsection Viscosity depth function
      set Function expression = 20
    end
  end
end
""", expected_error="Phase field fault must be the top-level material model")

    # This wrapper parses its base but does not forward initialize().
    run("averaging_wrapper", """
subsection Material model
  set Model name = averaging
  subsection Averaging
    set Base model = phase field fault
    set Averaging operation = arithmetic average
  end
end
""", expected_error="Phase field fault must be the top-level material model")

if args.case == "stationary_fault":
    # Two equivalent clamped-wall problems. Prescribed velocity functions
    # force a fresh matrix even when zero. A zero-velocity boundary lets
    # ASPECT reuse it, exercising actual RHS-only iterations.
    full, full_log = run("full_matrix", """
subsection Boundary velocity model
  set Prescribed velocity boundary indicators = left:function
end
""")
    rhs_only, rhs_log = run("reuse_matrix", """
subsection Boundary velocity model
  set Prescribed velocity boundary indicators =
  set Zero velocity boundary indicators = left
end
""")
    marker = "Rebuilding Stokes preconditioner"
    if rhs_log.count(marker) != 3 or full_log.count(marker) <= rhs_log.count(marker):
        raise RuntimeError("The matrix-reuse run did not exercise both assembly paths")
    for step in range(3):
        print("RHS-only", step, "maximum absolute difference", compare_step(full, rhs_only, step))

    failed, failed_log = run("failure", "set Max nonlinear iterations = 1\n"
                             "set Nonlinear solver tolerance = 1\n",
                             expected_error="failed to converge")
    if "Coupled Stokes unknowns require another Picard iteration" not in failed_log or (failed / "fault-checks").exists():
        raise RuntimeError("Inconsistent rates at the iteration limit must fail before publication")

    closed = """
subsection Boundary velocity model
  set Prescribed velocity boundary indicators = left:function, right x:function, top y:function, bottom y:function
end
"""
    for pressure in (0, 0.1):
        run(f"closed_{pressure}", closed + f"set Surface pressure = {pressure}\n",
            expected_error="Fault friction requires a pressure datum")

    run("periodic_closed", """
subsection Geometry model
  subsection Box
    set Y periodic = true
  end
end
subsection Boundary velocity model
  set Prescribed velocity boundary indicators =
  set Zero velocity boundary indicators = left
  set Tangential velocity boundary indicators = right
end
subsection Boundary traction model
  set Prescribed traction boundary indicators =
end
""", expected_error="Fault friction requires a pressure datum")

    run("topography", """
subsection Geometry model
  subsection Initial topography model
    set Model name = function
    subsection Function
      set Function expression = 0.25
      set Maximum topography value = 0.25
    end
  end
end
subsection Material model
  subsection Phase field fault
    set Fault vertices = 0,0.5; 0.5,0.5; 1,0.5
  end
end
""", expected_error="The fault material requires the zero initial topography model")

    # Opposite tensile tractions on the horizontal faces shift the pressure
    # by -3 Pa at 12.5 ms. Extrapolating to 25 ms opens the trial fault.
    traction_boundaries = """
subsection Boundary traction model
  set Prescribed traction boundary indicators = right y:function, top y:function, bottom y:function
  subsection Function
    set Function expression = 0; if(x==1,0.02+t+0.12*(y-0.5),(2*y-1)*({tension}))
  end
end
"""
    opened, _ = run("open_initial_state", traction_boundaries.format(tension="6"),
                    expected_error="Fault trial requires finite, positive normal compression")
    if (opened / "fault-checks").exists():
        raise RuntimeError("Published an open initial fault")
    pulse = traction_boundaries.format(tension="3*max(0,1-abs(t/0.0125-1))")
    recovered, log = run("pulse_cutback", pulse + "set Nonlinear solver failure strategy = cut timestep size\n")
    if (log.count("Fault trial requires finite, positive normal compression") != 1 or
            log.count("Repeating the current time step") != 1):
        raise RuntimeError("Expected one recoverable closed-fault violation")
    split, _ = run("pulse_clean", pulse + """
subsection Time stepping
  subsection Function
    set Function expression = if(time==0,0.0125,0.00625)
  end
end
""")
    for output in (recovered, split):
        if (output / "fault-checks").read_text() != "0 36\n1 36\n2 36\n3 36\n":
            raise RuntimeError("Unexpected number of accepted pulse endpoints")
    for step, clock in enumerate((0., .0125, .01875, .025)):
        time = float((split / f"fault-{step}-0.txt").read_text().split()[0])
        if not math.isclose(time, clock, rel_tol=1.e-14):
            raise ValueError(f"Unexpected pulse reference time {time} at step {step}")
        print("Tensile pulse", step, "maximum absolute difference", compare_step(split, recovered, step))
