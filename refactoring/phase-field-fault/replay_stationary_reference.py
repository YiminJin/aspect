"""Build the observation plugin for donor 663937bed and run the stationary comparison.

Arguments are the reconstructed historical source, its Release build, the
homogeneous reduced donor input produced by replay_fixed_reference.py, and a
fresh destination. The observer selects stationary history and holds the phase fixed. The compiled
material model, geometry, bulk/section assembly and acceptance are unmodified.
"""
import argparse
import csv
import os
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
for name in ("source", "build", "homogeneous_input", "destination"):
    parser.add_argument(name, type=Path)
args = parser.parse_args()
src, build, parent, dest = (p.resolve() for p in vars(args).values())
fault = dest / "fault.txt"
text = parent.read_text()
settings = {"Output directory": dest/"output", "Prescribed faults file": fault,
            "Structural point spacing": ".5", "Stress rotation": "Jaumann DG Lie",
            "Material profile transport": "Q1 remap", "Material tensor inflow": "zero",
            "Material stress representation": "separated",
            "Additional shared libraries": dest/"build/libstationary_checks.release.so",
            "Composition polynomial degree": "1", "Use discontinuous composition discretization": "false"}
for key, value in settings.items():
    text, count = re.subn(rf"(?m)^([ \t]*set[ \t]+{re.escape(key)}[ \t]*=)[^\n]*", lambda m: f"{m[1]} {value}", text)
    if count == 0:
        raise ValueError(f"{parent}: missing inline setting {key}")
# The historical stationary selector retains its continuous Q1 phase.
text = re.sub(r"(?m)^\s*set Phase basis = Bernstein\s*$", "", text)
text, count = re.subn(r"(?m)^([ \t]*set Function expression[ \t]*=[ \t]*)0; \.02\+t([ \t]*(?:#.*)?)$",
                      r"\g<1>0; .02+t+.12*(y-.5)\2", text)
if count != 1:
    raise ValueError(f"{parent}: expected one homogeneous shear traction expression, found {count}")

dest.mkdir(parents=True, exist_ok=False)
fault.write_text(".5 0 .4\n.5 1 .4\n")
native = src / "benchmarks/reconstructed_fault/particle_free/transported_objective/native"
observer = (native / "material_checks.cc").read_text()
# The historical stationary route distributes a prescribed value to every
# phase DoF. This is the same post-constraint hook used by its BP3 fixture.
hook = """signals.post_constraints_creation.connect([](const SimulatorAccess<dim> &sim, AffineConstraints<double> &constraints)
      {
        const auto &fe=sim.get_fe();
        const auto phi=sim.introspection().phase_field().first_component_index;
        std::vector<types::global_dof_index> dofs(fe.n_dofs_per_cell());
        for(const auto &cell:sim.get_dof_handler().active_cell_iterators())
          if(!cell->is_artificial())
            {
              cell->get_dof_indices(dofs);
              for(unsigned int i=0;i<dofs.size();++i)
                if(fe.system_to_component_index(i).first==phi && constraints.can_store_line(dofs[i]) && !constraints.is_constrained(dofs[i]))
                  {
                    constraints.add_line(dofs[i]);
                    constraints.set_inhomogeneity(dofs[i],.35);
                  }
            }
      });
    """
for old, new in (
    ("model.uses_transported_mature_history()",
     "model.uses_jaumann_dg_stress() && !model.uses_transported_mature_history()"),
    ("s.accepted_publications==step && s.geometry_publications==step",
     "s.accepted_publications==step && s.geometry_publications==0"),
    ("signals.post_simulator_initialization.connect", hook+"signals.post_simulator_initialization.connect"),
):
    if observer.count(old) != 1:
        raise ValueError(f"Expected one occurrence of {old!r} in the historical observer")
    observer = observer.replace(old, new)
(dest / "stationary_checks.cc").write_text(observer)
(dest / "CMakeLists.txt").write_text(f"""cmake_minimum_required(VERSION 3.13.4)
find_package(Aspect REQUIRED)
DEAL_II_INITIALIZE_CACHED_VARIABLES()
project(stationary_reference)
add_library(stationary_checks SHARED stationary_checks.cc)
target_include_directories(stationary_checks PRIVATE "{native}")
target_compile_definitions(stationary_checks PRIVATE MATERIAL_ENERGY_CHECKS)
ASPECT_SETUP_PLUGIN(stationary_checks)
""")
subprocess.run(["cmake", "-S", dest, "-B", dest/"build", f"-DAspect_DIR={build}"], check=True)
subprocess.run(["cmake", "--build", dest/"build", "-j1"], check=True)
(dest/"input.prm").write_text(text)
env = dict(os.environ, DEAL_II_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
with (dest/"run.log").open("w") as log:
    subprocess.run([build/"aspect", dest/"input.prm"], env=env, cwd=dest,
                   stdout=log, stderr=subprocess.STDOUT, check=True)

# Convert donor observations into the native regression probe's column order.
# The donor exports rate and age. Cumulative physical slip is the sum of dt * V.
def records(name):
    with (dest/"output"/name).open() as stream:
        return list(csv.DictReader(stream))


comparison = dest / "comparison"
comparison.mkdir()
slip = {}
for state in records("material-summary.csv"):
    step = state["step"]
    header = [state["time"], state["dt"]]
    for label in records(f"material-{step}-surface.csv"):
        j = int(label["i"])
        slip[j] = slip.get(j, 0.) + float(state["dt"])*float(label["V_m_s"])
        header.extend((label["x"], label["y"], label["Theta_s"], format(slip[j], ".17g"), label["V_m_s"]))
    with (comparison / f"fault-{step}-0.txt").open("w") as snapshot:
        print(*header, file=snapshot)
        for q in records(f"material-{step}-rank0.csv"):
            # Temperature is prescribed at 293 K in this reduction.
            print(q["cell"], q["q"], q["x"], q["y"], q["vx"], q["vy"], q["pressure"], 293,
                  q["tauxx"], q["tauyy"], q["tauxy"], q["phi"], file=snapshot)
