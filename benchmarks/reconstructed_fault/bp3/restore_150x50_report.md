# BP3 restoration status — 2026-09-24

The approved shear-sense correction and new model inputs are implemented.
The restored chart preserves thrust slip with positive V. Focused one-/two-rank
coupling tests pass; the full requested mesh and paired completion inputs have
been generated. **The 10-step large-mesh startups have not been run**, because
the estimated simulation memory exceeds available local memory. Thus endpoint
loading behavior, filtered/raw growth, and the first-event model are **not yet
qualified**. No BP5 trajectory, constitutive equation, solver target, support
policy, transfer algorithm or existing case's default sign was changed.

## Resolved model

| Quantity | Value |
|---|---|
| Box | x = [-60,90] km; y = [0,50] km |
| Fault | right-dipping 60 degrees, trace (0,50) km, bottom (28.8675134595,0) km |
| Length / velocity representation | 57.735026919 km; 2889 continuous-Q1 nodes, all frictional |
| Exact breakpoints | 15, 18, 40 km |
| Fault spacing | 19.9818347–20.0000000 m |
| ell / minimum edge | 20 / 3.90625 m |
| a / b / Dc | .010 to .025 over 15–18 km / .015 / .008 m |
| mu0 / V0 / Vinit / Vp | .6 / 1e-6 / 1e-9 / 1e-9 m/s |
| G / viscosity / radiation damping | 32038120320 Pa / 1e26 Pa s / 4624440 Pa s/m |
| Normal / shear background | 50 MPa / uniform nominal 26.5461223651 MPa |
| Initial Theta | BP3 varying inverse, about 8000 s shallow and 8e6 s deep |
| Particle interpolation / field space | unlimited native LLS / continuous Q2, all five compositions |
| Phase / C / Maxwell initial history | frozen AT1 core .6 / mature C=0 / zero perturbation tensor |
| Normal input | primary Helmholtz 20 m; separate raw and 40 m inputs |
| Boundaries | smooth bottom and rigid sides; free perturbation top; no pressure normalization |
| Timesteps | first physical cap 100 s; predictor .02; global ceiling 4e6 s |

This is modified BP3: incompressible finite box, diffuse fixed fault, filtered
normal input, and a frictional deep extension with bottom Dirichlet loading.
It is not domain-converged or equivalent to official compressible BP3 with
prescribed deep slip.

## Geometry and normalization preparation

Production-consistent stationary support radius is **39.5291640201 m**, giving
the requested fine half-width **47.3416640201 m**. Using the provisional 40 m
band would not satisfy the specified support-plus-two-cells formula.
The reference full normal integral is 13053.8394454810 m. This is a boundary/
completion reference, **not a substitute for production FE I_h**.

Eight surface panels produce 69,312 profile origins, of which 56 need nonzero
outside completion. Cell-split adaptive order-8/order-16 completion integrals
differ by at most 5.59e-10 m (about 4.3e-14 of the full reference integral).
The independent profile agrees with the production stationary evaluator to
6.63e-14 in phi. Production also checks that agreement before mechanics.

Actual balanced mesh inventory:

| Level | Cells |
|---|---:|
| 1 | 6,958 |
| 2 | 1,078 |
| 3 | 2,167 |
| 4 | 4,328 |
| 5 | 8,646 |
| 6 | 17,312 |
| 7 | 34,613 |
| 8 | 69,216 |
| 9 | 398,640 |

Total **542,958 square cells**, edges 3.90625–1000 m; 319,324 intersect the
stationary support. The mixed FE inventory has **19,149,424 DoFs**: 2,252,838
per Q2 scalar (two velocity components, temperature and five compositions),
563,360 each for pressure and phase. Planned particles: **4,886,622** at nine
per cell; particles were not instantiated in the mesh-only tool.
Mesh/DoF preparation RSS was 584,612 KiB, not full simulation memory.
See `fixtures/bp3_150x50/mesh_inventory.txt`, `mesh.csv`, and `mesh.png`.

The saved 352,062-cell four-rank reference used about 24.8 GiB aggregate RSS at
one sampled time, with individual high-water marks up to 6.8 GiB. Scaling to
this larger mesh suggests roughly **40–60 GiB aggregate**, uncertain until run.
Only about 20 GiB is locally available. A server allocation with at least
128 GiB usable aggregate RAM leaves a reasonable initial margin. No smaller
mesh was substituted to obtain a local pass.

## Sign and focused verification

For down-dip s=(.5,-sqrt(3)/2), n=(sqrt(3)/2,.5), use
`u=-Vp*(F(r)-.5)*s`. Its derivative matches the signed source S. Left velocity
is (+2.5e-10,-4.3301270189e-10) m/s; right is opposite. The smooth bottom uses
the same complete stationary profile, not a truncated integral or tanh.
Far-field relative rate is Vp, not 2Vp. Realized FE boundary/flux checks are
implemented but **not measured yet on the full fixture**.

The new manager setting is immutable, serialized (archive version 1), and
reattached explicitly by this plugin. Old version-0 checkpoints imply +1.
V remains positive; normal tensor N and all geometry normals remain unchanged.
Bulk source/B, surface R/K/G, history subtraction and diagnostic shear use
the same signed S. Default +1 retains the existing convention.

| Focused check | Result |
|---|---|
| One-rank sign/restart and filter units | 100 assertions / 2 cases passed |
| Two-rank same units | 100 assertions / 2 cases passed per rank |
| One-rank plus production-profile comparison | 5103 assertions / 3 cases passed |
| Final units including existing manager lifecycle/restart checks | 5155 assertions / 7 cases passed |
| Existing Stage-I bound/line-search/lifecycle units | 20136 assertions / 14 cases passed |
| Reflected B versus bulk residual derivative, one/two ranks | 1.60e-14 / 1.58e-14 relative error |
| Reflected shear virtual work, one/two ranks | 2.92e-15 / 2.15e-15 relative error |
| Filtered G, one/two ranks | about 6.81e-14 / 1.19e-14 relative error |
| Coupled filtered K/G centered differences | second-order contraction to 3.13e-7 |
| Rejected-trial linearization/history preservation | existing checks passed |
| PRM syntax validation | raw/filter20/filter40/continuation valid |

The one-/two-rank saved filtered residual, normal input and K/G directions
agree to within 1.40e-14 relative difference across the reported fields.

An initial added B test failed because its **test result vector lacked the
required owned layout**. Initializing that vector corrected the harness;
no production criterion was changed. MPI sandbox interface restrictions were
handled by running the two-rank tests outside the sandbox. No full ASPECT suite
or historical campaign was run.

## Files and remaining gates

Core changes: `include/aspect/reconstructed_fault/manager.h`,
`source/reconstructed_fault/manager.cc`,
`source/simulator/assemblers/reconstructed_fault_stokes.cc`,
`source/reconstructed_fault/surface_system.cc`, and
`source/material_model/phase_field_fault.cc`.
Tests: `unit_tests/reconstructed_fault.cc` and
`tests/phase_field_fault_surface_system.cc`.
The working tree already contained unrelated qualified diagnostic/filter changes
in several of these files; those were preserved, not attributed to this task.

Benchmark changes: separate `bp3_restore_150x50` build target and
`restore_150x50.h`; narrowly selected chart/loading/prestress branches in
`bp3.cc`, `bp3_model.h`, `mature_fault.h`; signed common work observer in
`work_replay.h`; `restore_mesh.cc`, preparation/analysis scripts, complete
raw/filter20/filter40 PRMs, gated first-event continuation and generated inputs.
The ordinary BP3/BP5 targets retain their previous chart and initialization.

Still required: actual full-mesh initialization and boundary source closure,
10 accepted loading steps for raw/filter20, measured simulation RSS/time,
matched-time endpoint/interior growth, and full fixture checkpoint/resume.
The early-loading checkpoint does not yet exist. No first-event run has started.
Follow `README_150x50.md`; do not treat syntax checks or the small derivative
fixture as a substitute for these remaining gates.

The independent algebraic initialization check gives tau_bg =
26546122.365133367 Pa, shallow Theta = 7999.99999999994 s and deep Theta =
8000000.000000014 s. Reinserting these values into the regularized law leaves
force-balance errors below 5e-9 Pa. This checks the intended physical data, not
the Q1-interpolated finite-width initial mechanical solution.

Logs, base revision, tested executable/plugin hashes and the full tracked
working-tree patch are in `restore-150x50-verification/`. That patch deliberately
also preserves pre-existing working changes; it is not a claim that this task
authored all its hunks. No commit was made.
