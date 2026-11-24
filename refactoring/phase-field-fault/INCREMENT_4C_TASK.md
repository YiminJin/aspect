# Increment 4C · Layered moving-fault coupling

Work in the current ASPECT 3.1 `pf-rsf` worktree, starting from the reviewed
Increment 4B plus the completed three-dimensional extension of the shared DG
transport. Implement and verify the first **coupled layered moving-fault** case.
Stop for review when the bounded case below passes; do not turn this increment
into a BP3 campaign.

## Purpose and physical meaning

Increment 4B showed that the bounded phase ridge and physical DG stress can be
transported. Increment 4C must show that, after that transport, reconstructed-
fault mechanics acts on the **new fault location and the correct persistent
surface material**, while the bulk rheology is spatially layered.

Use the existing short reference problem:

\[
\eta(y)=5.5+4.5\tanh(y/0.35),
\]

in the units and coordinate convention of the frozen donor. This gives the
existing smooth 1-to-10 viscosity contrast. The mature fault is already formed,
and phase-field growth remains disabled. “Moving” means that the phase ridge,
reconstructed curve, fault-material labels, bulk stress and surface histories
change their Eulerian mesh location. It does **not** mean along-fault slip, which
remains a separate RSF kinematic quantity.

The principal reference is one rank, two accepted steps of 12.5 ms, with joint
publications at 12.5 and 25 ms. Compare the new implementation with the frozen
coupled donor at both times. Use common physical observations rather than raw
block-vector indices.

## Start by freezing the evidence

1. Read `AGENTS.md` and the current `PLAN.md`, `STATUS.md` and verification guide
   under `refactoring/phase-field-fault/`. Record HEAD, the worktree diff and the
   exact 3-D-extension commits and tests. Preserve all incoming work.
2. Confirm that all 4B review repairs are present: interior-action CFL, rejection
   of constrained explicitly transported fields, supported prescribed-solution
   setup, checkpoint phase-space compatibility, velocity-degree/quadrature
   contract, physical phase extrema, and an active partition face in MPI tests.
   Do not reopen them unless 4C exposes a concrete regression.
3. Locate the frozen layered moving-fault donor in the read-only sibling worktree
   and identify its input, executable/plugin identity, accepted-time convention,
   viscosity coordinate convention, translation velocity, phase profile, RSF
   parameters and output observations. Do not infer missing constants from old
   plots. Record hashes or commits sufficient to reproduce the two donor steps.
4. Before editing, write a data-flow trace from accepted phase/stress/surface
   state to the first Stokes assembly and to joint publication. Identify the one
   current owner of every state listed below.

## State-ownership contract

| Quantity | Meaning in 4C | Attempt/acceptance rule |
| --- | --- | --- |
| Native phase field | Single authoritative bounded DG field locating the mature fault | Transport once from the accepted parent before geometry binding; candidate only until acceptance |
| Physical bulk stress | Authoritative independent DG stress components | Transport once from the same accepted parent and accepted velocity; then use the existing spin/relaxation/slip correction |
| Curve and support | Geometry reconstructed/bound from the transported candidate phase | Build once per physical attempt; nonlinear iterations reuse it; discard on rejection |
| Fault labels/stations | Persistent surface material identity, not cell identity | Match deterministically across the moved curve; migrate ownership without resetting or duplicating history |
| RSF state, slip, age and memory | Lagrangian surface history | Follow persistent labels; trial updates are reversible; publish exactly once per accepted step |
| Viscosity and other layered material coefficients | Eulerian material response evaluated at the current physical point | Re-evaluate at the candidate quadrature/section location; never carry them as label history |
| Boundary/reference history | Existing benchmark-plugin state, if stateful | Candidate evaluation must be retry-safe and checkpoint-safe; advance only with the accepted time sequence |

The decisive distinction is: **history follows the surface material label;
environmental material coefficients are sampled at its current position.** Do
not make CellId, MPI owner or nearest quadrature point a persistent fault label.

## Required implementation sequence

### 1. Bind mechanics to the transported candidate

Use one ordinary simulator attempt:

1. Snapshot the accepted parent.
2. Advance selected phase and physical-stress fields over the attempted interval
   with the existing shared SSPRK3/DG operator and the accepted bulk velocity.
3. Reconstruct or rebind the fault curve, support, tangents/normals, bulk
   associations and surface labels from the transported candidate phase.
4. Prepare the existing Maxwell/RSF coupling from that candidate geometry and
   the accepted surface parent.
5. Run the ordinary nonlinear Stokes/RSF solve. Every repeat evaluation must see
   the same transported candidate and the same accepted parent.
6. On rejection, destroy all candidate transport, geometry, label matching,
   surface updates and stateful boundary changes, then recompute for the new
   duration. On acceptance, publish phase, stress, geometry and surface state
   jointly and exactly once.

Do not transport again inside nonlinear iterations. Do not let the first
iteration see the old curve while later iterations see the moved curve. The
Stokes load, section balance, pressure-inclusive tractions, localization and
surface update must all use one candidate geometry.

### 2. Move an already formed curve, without adding crack growth

Reuse the current reconstructed-fault manager and phase observation machinery.
Add only the missing attempt-local rebinding/matching needed for a translated
mature curve. Keep the existing section ordering and orientation convention.
Persistent label matching must be deterministic under cell crossings and MPI
ownership changes. Use physical station/geometry plus the established tie-break,
not iteration order.

The short fixture must keep the localization support away from external
boundaries and must not create or delete tips. Reject unsupported topology
changes with a clear diagnostic. No new tip law, nucleation rule, healing rule
or cohesive growth belongs here.

### 3. Add the layered material response at the normal material-model level

Implement the smooth viscosity through the smallest ASPECT-style material-model
or benchmark wrapper that supplies the existing Maxwell path. Keep constant
shear modulus and the donor's other physical inputs. Evaluate `eta(y)` at the
same physical points used by the constitutive update. Do not pre-tabulate
viscosity by fault label or cell index.

First verify the wrapper independently:

- values and derivatives/sampling on both sides and through the transition;
- the constant-viscosity limit;
- one- and two-rank equality at common physical points;
- no change to feature-disabled or existing homogeneous inputs.

Then verify that an oblique/dipping fault samples the intended range of
relaxation times while moving across cells. The 4C fixture is an algorithmic
layered benchmark; it is not yet a lower-crust parameter claim.

### 4. Keep the 3-D boundary explicit

The shared DG transport stays dimension-generic and all new 3-D transport tests
must remain green. Reconstructed-fault mechanics in 4C remains deliberately
two-dimensional. Keep this boundary in types/registration where possible and
give a direct diagnostic for a 3-D reconstructed-fault input. Do not introduce
a nominal 3-D curve/surface implementation to make templates compile.

## Verification matrix

Use fresh output directories and at most four local workers/ranks. Register a
compact production-path test; keep larger comparison artifacts outside CTest.

### A. Two-step layered moving reference — principal gate

Run the exact donor setup for two accepted `dt=0.0125 s` steps on one rank. At
`t=0.0125 s` and `t=0.025 s`, compare:

- phase mass/mean, physical extrema, normal localization integral and ridge
  position;
- curve coordinates, station ordering, tangents/normals and bulk support;
- persistent label identity, age and ownership;
- physical bulk-stress components at common physical observations;
- velocity, pressure and the actually assembled stress load or an equivalent
  independently checked weak action;
- pressure-inclusive normal traction, work-adjoint shear traction, slip rate,
  cumulative slip, RSF state and any retained surface memory;
- Maxwell relaxation/work terms, bulk-to-surface work defect and nonlinear cost;
- exactly one joint publication per accepted step.

Use the donor's serialized precision plus independently evaluated FE norms to
set tolerances. Do not impose a universal 1% rule, and do not require raw
coefficient equality when basis/storage changed. Any difference larger than the
representation/output bound must be traced to phase transport, geometry/label
matching, coefficient sampling, stress evolution or RSF coupling; do not tune
friction or viscosity to hide it.

### B. Discriminating controls

1. **Zero-translation control:** with the same layered material and zero added
   translation, recover the stationary-geometry path to its established
   tolerance. State clearly that fault-induced flow may still move material;
   use a genuinely frozen geometry/profile control only if the existing code
   provides one without changing constitutive algorithms.
2. **Homogeneous limit:** replace the layered wrapper with its constant-viscosity
   limit. The moving result must agree with the existing homogeneous 4B/earlier
   coupled path at common observations.
3. **Stale-geometry discriminator:** establish one observable for which feeding
   the old curve/support into the new Stokes solve gives a measurably different
   result. This may be an offline negative control; do not add a runtime option.
4. **Label-following check:** choose at least one label that crosses a cell and,
   if possible, an MPI ownership boundary. Its age, RSF state and accumulated
   slip must remain continuous while material coefficients are evaluated from
   its current physical location.
5. **Reversal check:** run the short translation with the sign reversed if the
   donor contains that case; otherwise use a one-step geometry/transport control
   to expose orientation- or inflow-side mistakes without expanding the coupled
   reference matrix.

### C. Retry, MPI and restart/lifecycle

- Force a rejected attempt across a cell-crossing interval, then accept the
  shorter interval. Compare with a clean run at identical accepted times. Phase,
  stress, curve, labels, surface history, layered-boundary state and publication
  counts must agree at the strongest reproducible level; explain anything not
  bitwise identical.
- Repeat the compact crossing case on two ranks with an active partition face.
  Compare owned bulk data by CellId/component and surface data by persistent
  label/station. Demonstrate that at least one relevant datum actually migrates
  or receives a remote contribution.
- If 4C changes checkpointed state or the layered provider is stateful, restart
  between the two publications and compare the second publication at the same
  physical time. Otherwise record restart as deferred rather than adding a new
  checkpoint format in this increment.

### D. Regression gates

Rerun:

- the 4B bounded moving-phase and physical-stress tests;
- the new three-dimensional scalar/stress transport tests;
- stationary reconstructed-fault, tangent, work/energy and forced-retry tests;
- the feature-disabled release control and any ordinary ASPECT DG control whose
  shared assembler was touched.

Do not repair unrelated stock reference-file discrepancies under this increment.

## Architecture constraints

- The shared DG transport must remain free of reconstructed-fault and RSF
  concepts. Fault motion is a consumer of transported fields.
- Keep accepted-parent immutability and the established joint transaction. Do
  not add a second timestep driver, hidden publication path or mutable singleton.
- Continue using the one authoritative native phase field and the one
  authoritative physical-stress representation. Do not revive proxy fields,
  particle authority or upstream elasticity-output structures.
- Use the same localization/normalization in forward coupling and surface-to-
  bulk work transfer. A moving curve is not permission to renormalize only one
  side.
- Keep benchmark-specific layered boundary/reference algebra in the benchmark
  plugin. It must not become a general production solver.
- Prefer small extensions to the existing manager/lifecycle interfaces over a
  new moving-fault driver.

## Explicit exclusions

Do not add phase-field growth, nucleation, healing, tip propagation, arbitrary
topology changes, AMR/history transfer, ALE, 3-D reconstructed-fault mechanics,
gravity, poroelasticity, a production BP3 input, long earthquake cycles or
server jobs. Do not change the RSF law, pressure convention, endpoint scheme,
spin law, limiter, timestep controller or nonlinear tolerances unless a proved
4C defect requires it; report such a change separately.

## Handoff

Update `PLAN.md`, `STATUS.md` and the verification record. Provide:

1. a source walkthrough from accepted phase/stress through transport, geometry
   rebinding, layered constitutive sampling, Stokes/RSF assembly and publication;
2. exact build/run commands, binary/plugin identities, rank/thread counts and
   accepted timestep sequence;
3. a table at 12.5 and 25 ms comparing donor and new physical observations with
   tolerance rationales;
4. a plot or short animation overlaying phase ridge, reconstructed curve, layer
   transition and persistent labels during the two accepted steps, including
   any cell crossing;
5. retry/MPI evidence and all regression results;
6. a concise list of remaining differences and their classified source.

Stop for author review after this bounded 4C result. A dimensional BP3-like
fixed/moving pair, moving AMR and long variable-timestep campaigns are later
increments.
