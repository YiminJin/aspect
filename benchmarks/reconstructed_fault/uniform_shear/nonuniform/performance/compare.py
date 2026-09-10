#!/usr/bin/env python3
"""Compare the short performance runs, requiring actual convergence."""
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

directory = Path(__file__).resolve().parent
sys.path.insert(0, str(directory.parent))
from measure_case import summarize

report = dict(cases={}, exported_fields={})
for case in ("baseline", "optimized", "accounted", "local", "local-mpi"):
    log = directory / (case+".log")
    steps = summarize(log)
    if len(steps) != 2:
        raise ValueError("The short replay must accept initialization and one real step")
    for step in steps:
        final = step["nonlinear"][-1]
        if not (final["bulk"] < final["bulk target"]
                and final["surface"] < 1e-8*final["surface scale"]
                and all(row["fresh"] <= row["target"] for row in step["linear"])):
            raise ValueError("A final nonlinear or fresh linear criterion failed")
    text = log.read_text()
    timings = {name.strip(): dict(calls=int(calls), seconds=float(seconds))
               for name, calls, seconds in re.findall(
                   r"\| ([^|]+)\|\s*(\d+)\s*\|\s*([0-9.eE+-]+)s", text)}
    elapsed = float(re.search(r"Total wallclock time elapsed since start\s*\|\s*([0-9.eE+-]+)s", text)[1])
    resources = json.loads((directory / (case+".resources.json")).read_text())
    if resources["exit_status"] != 0:
        raise ValueError("Replay did not finish cleanly: "+case)
    report["cases"][case] = dict(resources=resources, aspect_elapsed=elapsed,
        startup_shutdown_seconds=resources["wall_seconds"]-elapsed,
        timings=timings, converged_states=len(steps),
        fresh_linear_checks=sum(len(step["linear"]) for step in steps),
        final_residuals=[step["nonlinear"][-1] for step in steps])

for path in sorted((directory / "baseline").glob("*.csv")):
    other = directory / "optimized" / path.name
    if not other.exists():
        raise ValueError("Missing optimized export: "+path.name)
    identical = path.read_bytes() == other.read_bytes()
    fields = {}
    if not identical:
        old = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
        new = np.atleast_1d(np.genfromtxt(other, delimiter=",", names=True))
        if old.shape != new.shape or old.dtype.names != new.dtype.names:
            raise ValueError("Changed export layout: "+path.name)
        fields = {name: float(np.max(np.abs(old[name]-new[name]))) for name in old.dtype.names}
    report["exported_fields"][path.name] = dict(byte_identical=identical, max_absolute_errors=fields,
        baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if any(error != 0 for error in fields.values()):
        raise ValueError("Nonzero numeric change: "+path.name+" "+str(fields))

report["all_exports_numerically_identical"] = True
report["accounted_exports_identical"] = {
    path.name: path.read_bytes() == (directory/"accounted"/path.name).read_bytes()
    for path in sorted((directory / "baseline").glob("*.csv"))}
if not all(report["accounted_exports_identical"].values()):
    raise ValueError("Accounting-only replay changed a saved field")
report["local_timer_exports_identical"] = {
    path.name: path.read_bytes() == (directory/"local"/path.name).read_bytes()
    for path in sorted((directory / "baseline").glob("*.csv"))}
if not all(report["local_timer_exports_identical"].values()):
    raise ValueError("Local-timer replay changed a saved field")
report["mpi_surface_max_absolute_differences"] = {}
for step in (0, 1):
    for stem in ("surface", "surface_weak"):
        name = f"{stem}_{step}.csv"
        one = np.atleast_1d(np.genfromtxt(directory/"local"/name, delimiter=",", names=True))
        two = np.atleast_1d(np.genfromtxt(directory/"local-mpi"/f"{stem}_rank0_{step}.csv", delimiter=",", names=True))
        other_rank = np.atleast_1d(np.genfromtxt(directory/"local-mpi"/f"{stem}_rank1_{step}.csv", delimiter=",", names=True))
        if not np.array_equal(two, other_rank):
            raise ValueError("Replicated MPI surface fields disagree: "+name)
        one.sort(order=["fault", "node"])
        two.sort(order=["fault", "node"])
        if one.shape != two.shape or not np.array_equal(one[["fault", "node"]], two[["fault", "node"]]):
            raise ValueError("MPI surface layout changed: "+name)
        report["mpi_surface_max_absolute_differences"][name] = {
            field: float(np.max(np.abs(one[field]-two[field])))
            for field in one.dtype.names if field not in ("fault", "node")}
accepted = directory.parent / "domain-convergence/space32"
report["accepted_spatial_prefix_identical"] = {
    path.name: path.read_bytes() == (accepted/path.name).read_bytes()
    for path in sorted((directory / "baseline").glob("*.csv"))}
timings = report["cases"]["accounted"]["timings"]
# Disjoint work regions. In particular, omit Setup initial conditions because
# it contains the first domain generation, and omit inclusive nested subtimers.
regions = ("Fault: Property preparation", "Particles: Domains/CPDI",
           "Fault: Surface R/K total", "Fault: G total", "Fault: B action",
           "Fault: B linearization", "Fault: History commit",
           "Assemble Stokes system Picard", "Assemble Stokes system rhs",
           "Build Stokes preconditioner", "Postprocessing", "Initialization",
           "Setup dof systems", "Setup matrices")
assigned = sum(timings[name]["seconds"] for name in regions)
report["elapsed_reconciliation"] = dict(
    disjoint_regions=list(regions), assigned_seconds=assigned,
    remaining_aspect_seconds=report["cases"]["accounted"]["aspect_elapsed"]-assigned,
    startup_shutdown_seconds=report["cases"]["accounted"]["startup_shutdown_seconds"],
    remaining_scope="Other setup/phase-field solve, Krylov preconditioner applications, "
                    "linear algebra, constraints, scalar reductions and uninstrumented orchestration; "
                    "not attributed to surface nonlinear quadrature.")
local = report["cases"]["local"]
local_assigned = sum(local["timings"][name]["seconds"] for name in regions)
report["local_timer_reconciliation"] = dict(
    assigned_seconds=local_assigned,
    remaining_aspect_seconds=local["aspect_elapsed"]-local_assigned,
    startup_shutdown_seconds=local["startup_shutdown_seconds"],
    note="Subsystem timers are rank-local and unsynchronized. Their individual "
         "lifetime totals overlap and are not additive.")
(directory / "comparison.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(dict(states_per_case=2, exports_compared=len(report["exported_fields"]),
                      numerically_identical=True)))
