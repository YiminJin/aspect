# Bound interpolation and 30–33 km startup qualification

The interpolation repair and new initialization pass their focused checks.
**The new trajectory is not yet qualified for a server first-event run.**
Initialization and real step 1 are accepted; the second real solve stalls far
above the unchanged nonlinear criteria. No failed candidate is treated as an
accepted history. The startup evidence is retained under
`weakening30-dc010-ell100/startup/`; the historical 15–18 km cases are unchanged.

## 1. Velocity-bound repair

The captured coordinate `xi=0.07893139508566324` demonstrates the defect:
`(1-xi)*1e-20 + xi*1e-20` rounds below `1e-20`. The two admissible nodal
values can therefore produce a falsely inadmissible constitutive input.

The shared absolute-rate interpolation now starts from the smaller endpoint,
preserves exact endpoint/equal-contact values, and remains in the nodal convex
hull. It never clamps to `V_min`: truly inadmissible nodal values remain
inadmissible. The manager, bulk-source residual, particle/domain surface rule,
and bulk-work surface rule use this same helper. Signed perturbation actions
retain ordinary Q1 derivatives. No bound, active-set, Armijo or residual
tolerance was changed; no history/lifecycle change accompanies this repair.

Changed production files:

- `include/aspect/reconstructed_fault/utilities.h`: narrow interpolation API.
- `source/reconstructed_fault/utilities.cc`: stable convex-hull evaluation.
- `source/reconstructed_fault/manager.cc`: canonical fault interpolation.
- `source/reconstructed_fault/surface_system.cc`: both absolute surface paths.
- `source/simulator/assemblers/reconstructed_fault_stokes.cc`: bulk residual.

`unit_tests/reconstructed_fault.cc` adds the captured case, equal/near-bound
nodes, ascending/descending profiles, exact endpoints, invalid-input nonrepair
and MPI agreement. Existing contact/release, Armijo rejection/exhaustion and
rollback coverage remains. `doc/reconstructed_fault/current_design.md` records
the invariant; the equations and linear actions have not changed.

Commands/results (Release, rebuilt with `-j4`):

```sh
build-pf-cpdi/aspect-release --test 'Stage-I*'
mpirun -np 2 build-pf-cpdi/aspect-release --test 'Stage-I*'
```

Both pass: **20,124 assertions in 12 cases** (on each MPI rank). Logs and build
logs are in `weakening30-dc010-ell100/verification/`. The pure first-event
observer test also passes onset, equality/re-entry, five-state termination and
retained peak. No full integration suite was run.

## 2. Regenerated model

| Quantity | New fixture |
|---|---|
| Weakening/transition | 0–30 km / 30–33 km down dip |
| Bulk extension | Horizontal physical-depth interfaces |
| Friction | `a=0.004/0.04`, `b=0.03`, `Dc=0.1 m` |
| Geometry | 300 × 100 km box, 60-degree fully frictional fault |
| Regularization | Frozen AT1, `ell=100 m`, same physical profile |
| Fault grid | 1,156 vertices; 99.9637–100 m spacing, exact 30/33-km nodes |
| Bulk mesh | 114,984 cells; 24.4140625 m near fault, at most 12.5 km |
| Background | Uniform effective shear 26.546122365139291 MPa, compression 50 MPa |
| Cohesion/history | Mature `C=0`, zero initial Maxwell stress perturbation |
| Clock | Artificial 4e6 s; ordinary controller with 4e6-s real-step ceiling |
| Linear solver | AMG, sparse B/G, pivoted tridiagonal inverse, FGMRES |

The existing graded fine band covers the entire new transition. Regenerating
the mesh therefore reproduces the existing leaf-tree hash; there is no need
to move a refinement boundary or coarsen another region. This is not a new
spatial-convergence claim and does not erase the earlier 1% width-criterion
failure of the economical candidate.

Material is regenerated from the new physical-depth function and projected
normally. Positive initial Q1 state is recomputed after that projection with
the configured friction law and actual mechanical work weights. No old nodal
state is transferred. Stored background corrections are zeroed together with
setting the nominal shear, so the *effective* background is uniform.

Both endpoint completion inputs are regenerated at the new fault quadrature
coordinates. Independent order comparison differs by at most
`5.18412e-11 m`; no production quadrature/tail tolerance changes. Production
checks retain the paired source continuation, not just an integral correction.
Raw diagnostic windows follow the new transition (19–43 km, with primary
28–35 km window); endpoint and deeper controls are retained.

The new benchmark parameter is `Postprocess/BP3/Weakening region length`.
It defaults to 15 km for old inputs. The new fixture explicitly sets 30 km.
Implementation: `bp3/bp3_model.h`, `bp3/bp3.cc`, `bp3/work_replay.h`.
The separate `bp5_initialization` plugin preserves the previously accepted
weak-state initialization; ordinary historical plugin behavior is unchanged.

## 3. Accepted-prefix evidence and pending solve

| Check | Initialization | Real step 1, t=4e6 s |
|---|---:|---:|
| Final relative bulk residual | 9.473638e-15 | 6.526275e-10 |
| Final relative surface residual | 2.159330e-15 | 2.469598e-9 |
| Surface RMS, Pa | 5.429919e-9 | 0.006209779 |
| Free/lower-active nodes | 1156 / 0 | 1156 / 0 |
| State-update relative error | 0 (retained) | 2.220446e-16 |

All returned linear directions satisfy their fresh residual checks. Real step
1 consumes zero initial stress history, then publishes current stress once.
The independent first-Maxwell-update discrepancy is `3.03020e-7 Pa` on a
`26700.25 Pa` scale. Stable history IDs cover 1,034,856 particles. Phase and
fault geometry remain unchanged. Native work loads reconstructed from the raw
quadrature exports agree, including the continued endpoint source regions.
The lateral rigid-translation constraints have zero measured error.

The projected nodal inverse alone has a `9660.978 Pa` weak friction mismatch;
the one-time weak initialization reduces it to `1.19538e-7 Pa` in two Newton
updates. Its directional derivative error is `1.72642e-10` relative. Initial
`V/Vp=0.998988244…0.999972063`; the sharp initial pulses are absent. Raw total
normal traction is `49.986394…50.009761 MPa`.

The first accepted aging update changes shallow state from approximately
`25118.864 s` to `3945236…3945270 s`, a factor of `157.063…157.064`.
This is the implemented physical aging update, not repeated publication or
timestep-zero evolution. It is useful context for the subsequent difficult
solve, **not proof of the cause of its nonlinear stagnation**.

Step 2 at t=8e6 s **exhausted all 30 nonlinear iterations** and aborted on
all ranks with `ExcNonlinearSolverNoConvergence`. The final recorded bulk/fault
relative residuals were `0.7215125 / 0.3328721`, against `1e-8`. The final
recorded accepted alpha was `4.99064e-7` (minimum `8.51302e-8`). This is not a
precision-floor residual. The original interpolation exception did not recur.
Every printed fresh linear check passed, but the nonlinear solve did not.
See `startup_analysis.json`
for all recorded nonlinear residuals and accepted alphas. The final process
outcome is recorded separately in `startup/execution.json`.

The four-rank attempt took **1460.52 s (24.34 min)**, below its 1800-s cap;
exit status 1 is a numerical failure, not a timeout. Recorded peak child RSS
was `2,762,964 KiB` (2.64 GiB); this is a maximum-process statistic, not a
simultaneous four-rank memory sum. Keep a conservative node-memory allowance
(e.g. 16 GiB or more for this four-rank fixture) rather than interpreting it as
the whole job footprint. No automatic retry or additional trajectory followed.

`accepted_prefix_checks.json` explicitly checks only published states 0–1;
it is not a passing whole-run certificate. Matched half-timestep and restart
scripts are prepared but gated on a successful startup. In particular there
is not yet an accepted-step-2 checkpoint from which to qualify the first
resumed history update. No second trajectory is used to conceal this failure.

## 4. Reproduction and server handoff

```sh
python3 benchmarks/reconstructed_fault/bp5/startup_30km.py setup
python3 benchmarks/reconstructed_fault/bp5/startup_30km.py prepare startup
python3 benchmarks/reconstructed_fault/bp5/startup_30km.py run startup
python3 benchmarks/reconstructed_fault/bp5/check_startup_30km.py startup --accepted-prefix
python3 benchmarks/reconstructed_fault/bp5/analyze_startup_30km.py
```

The launcher refuses to overwrite existing labels, caps each simulation at
1800 s, and records exact input/library/binary hashes. It does not automatically
retry a failed run. Preparation/build time is separate. Reuse the saved result,
not the first three commands, in the existing working tree.

`server-30km/` is a portable **blocked** first-event input package: clean fresh
and resume parameter files, self-contained hashed fixture, source patch/overlay,
qualified local plugin artifact, environment file and Stampede3 template.
The local binary/plugin hashes plus HEAD and patch identify the actual source;
a HEAD string alone does not. Rebuild the plugin with the server ASPECT/MPI
stack rather than using the local binary artifact blindly.

The observer is configured to stop after the first event's five consecutive
accepted sub-threshold decay states, with an ordinary termination checkpoint;
1500 years is an independent no-event safety limit. Hourly wall-time checkpoints
and a graceful pre-wall-limit stop are also configured. No long run was launched.
The server launch guard remains closed pending startup, timestep and restart
qualification. Resolving the new stalled nonlinear solve is the next decision;
this task does not silently change the solver or initial physical state.
