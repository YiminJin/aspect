# PF5 progress: accepted native history, rollback and restart

16 October 2026. Worktree `aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, HEAD `3232056cde12dbae17ef4265ecb804f9098b88f1` plus
preserved incoming PF0–PF4/manual work. The selected fixed-mesh/fixed-geometry
subset is implemented and measured. Unlimited upwind DGQ2 remains the
experimental baseline. Both WENO paths, limiter settings and production defaults
are unchanged. This is progress within PF5, not completion of its original
AMR/MPI/transfer plan or authorization for PF6.

Final native core SHA256 is
`21cf5d8589be2194a7a02e40e518f3d2f3cabe3528b6a22490e53be7063adfdd`;
the five main runs use observer SHA256
`c8db51c3c583a0d74c04e28882df3292c682fb90229e2920cdec62676395b4ce`.
All final selected native runs match 880 current production source hashes.
Pre-edit hashes/patch and all prior reports/evidence are preserved separately.

## Composition and ownership

Write `E_h=D3(symgrad u_mid)`, `r=G/eta`, `beta_a=exp(-r a)` and
`kappa_a=-eta expm1(-r a)`. Each trial reads immutable accepted T/Z/I:

```
Ta = beta_(dt/2) Tn + 2 kappa_(dt/2) E_h
Tb = SSPRK3(dt, L_transport, Ta)
Tnew = beta_(dt/2) Tb + 2 kappa_(dt/2) E_h
Znew = beta_dt Zn + kappa_dt V
s_elas = Znew/G;  qnew = Znew/I;  tau = Tnew - 2 chi Znew S
```

The native FE-source projection and Maxwell coefficients are reused, including
the full 3D plane-strain deviator, `zz=-(xx+yy)`, and the audited non-objective
component rate. Transport uses existing Gauss3 mass, Gauss4 volume/face rules,
upwind material flux and unlimited SSPRK3. The transport matrix must contain
**zero reaction**, receives **zero source**, and must have no boundary inflow
loads. Closed boundaries avoid an unqualified split-inflow correction. Native
forcing is supplied by the caller's midpoint bulk field; the measured fixture
is stationary in time. The two half steps compose exact Maxwell/source
integration, not two full source increments. Surface Z uses the existing
full-step cohesive candidate once. Supplying the completed T to existing FE
commit preparation bypasses an additional bulk update; publication only writes.

This Strang scheme is second order overall. SSPRK3's third-order transport
substep does not make the source/transport composition third order. The measured
temporal result below is for constant coefficients and prescribed stationary
forcing, not arbitrary time dependence or a coupled mechanics integrator.

Material-owned DG coefficients remain authoritative. The existing FE container
retains phase, geometry and Z/I; its velocity-history block stays zero. The
existing `HistoryCandidates` validator/publisher and noexcept accepted-state
swaps are reused. Parent, step, time, dt and prescribed-rate checks reject stale,
duplicate or changed-context candidates. All finite/layout/cohesive checks and
derived particle sampling occur before persistent writes. Particle stresses are
derived outputs, never incoming tensor data or fault weak-load feedback. H is
unchanged under the established `Evolve phase field=false` law.

## Native persistence and source corrections

`Enable prescribed DG accepted history=false` is the new default-off selector.
The experiment reuses the Simulator checkpoint archive, triangulation/native FE
solution transfer, fault manager's property/committed-V serialization,
postprocessor save/load and native particle/RNG checkpoints. No parallel history
backend or second checkpoint file format is introduced. A manager registry marker
gates a version-1 material payload: CellId-keyed local DGQ2 coefficients, exact
accepted Z/I, phase anchor, committed step/publication count, and frozen
mesh/geometry/material/quadrature metadata. The unselected path adds no archive
bytes. Native/particle or selected/unselected checkpoint conversion is rejected.

Restoration rebuilds the native FE/DG layout and transient association,
normalization and constitutive caches, then restores accepted coefficients
without applying any evolution. I is recomputed for consistency but its exact
saved values are retained. It checks CellId uniqueness/layout, accepted clock,
fixed vertices/profile/orientation, G/eta/localization/quadrature and agreement
between material Z/I and manager cohesive history. Restore runs **after all
native userdata loading**; the material's earlier callback precedes particle
loading. The optional post-resume timestep-reduction hook is not used to mutate
histories.

Preserved preliminary failures led to two corrections: bitwise comparison is
required for intentionally unused NaN registry-marker values, and the restored
surface-temperature cache must be rebuilt before native cohesive evaluation.
The first trial run's NaN-equality assertion and the first restart's missing-cache
segmentation fault remain under `build-particle-free/runs/`; final runs include
both corrections. No physical values are sanitized or tolerances relaxed.
The native manager load also makes an omitted default shear sense explicitly
`+1`. Raw manager archives differ by 13 bytes for that representation; complete
manager archives after the **same existing native load** agree bitwise. Raw and
canonical archives are both retained; no manager serialization policy changed.

## Bounded fixture and independent reference

The unit square has 4×8 fixed Cartesian cells, 288 scalar DG DoFs / 864 tensor
DoFs, and one fixed horizontal fault with 17 surface nodes. G=1e6 Pa,
eta=1e8 Pa s, V=.001 m/s, stationary surface labels and the inherited native
fixed cohesive phase/profile/normalization are used. Particles stay fixed.
An unloaded native preload of .1 s uses the PF4 Gaussian-x Q2 FE source to
generate a non-proportional tensor and nonzero physical Z. Its tensor coefficient
matrix has singular values 2431.14, 979.39 and 220.287 Pa; this is not one scalar
pulse times a fixed tensor.

For the next .2 s, transport and native source use the prescribed Q2 velocity

```
vx = .4 x(1-x)(1-2y)
vy = -.4 (1-2x)y(1-y)
```

It is divergence free with zero boundary-normal velocity. Bulk normal motion
crosses the fixed localization. With uniform Z and fixed surface labels,
reconstruction preserves
`D tau + r tau = 2G E_h - 2G chi V S - 2 Z (v.grad chi) S`.
The last term is nonzero: its final native-quadrature full-tensor L2 norm is
947.746 Pa m/s (RMS 947.746 Pa/s on this 1 m² domain). It is the cohesive
fixed-profile commutator, not an omitted
profile derivative or a mature `Q_geo=0` law. This fixture does not solve
nonuniform surface-history transport.

`assess_pf5.py` independently differentiates each target cell's Q2 preload
interpolant (including one-sided DG face values), assembles the upwind spatial
matrix with analytic velocity and independent Gauss5, and evaluates the exact
affine semidiscrete ODE `Tdot=(L-rI)T+2G E_h` by an augmented matrix exponential.
It also independently composes native half steps and the SSPRK3 polynomial.
No C++ mass matrix, source evaluator or time integrator is called. T errors
use the exact Q2 mass norm
`|A|_3²=xx²+yy²+(xx+yy)²+2xy²`. Physical-stress errors use the actual shared
native mechanical quadrature; source associations, phase and I are exported.
Independent reconstruction and Q1 weak contractions compare history-induced
shear and normal fault loads at that same quadrature. Prescribed pressure and
external loads are not added to those history contributions. This isolates temporal error on one grid;
it is not a spatial or nonlinear-quadrature convergence claim.

|Steps over .2 s|dt, s|Relative T L2|Relative physical stress L2|Relative weak-load l2|
|---:|---:|---:|---:|---:|
|8|.025|3.75525e-6|3.75951e-6|5.71235e-6|
|16|.0125|9.38753e-7|9.39819e-7|1.42790e-6|
|32|.00625|2.34681e-7|2.34948e-7|3.56953e-7|
|64|.003125|5.86693e-8|5.87360e-8|8.92354e-8|

Successive observed orders are 2.00002–2.00009 for T/physical stress and
2.00005–2.00019 for weak loads. Independent/native spatial matrix maximum
difference is 1.25e-14 s^-1; preload relative error is 4.52e-16. Independent
composed/native evolution agrees within 3e-14 relative L2. Native zero-transport
relaxation and source controls agree with one full native update within 1e-9 Pa.
Deliberately doubling transport, relaxation or forcing yields relative errors
about .02550, .000998 and 1.00062, respectively; omitting forcing gives 1.00062.
These negative controls distinguish counting errors from convergence.

The analytic final Z is 299.550449662703 Pa m and elastic slip
.000299550449662703 m. Maximum Z error over all stepsizes is 9.10e-13 Pa m;
cohesive and physical-slip errors remain roundoff sized. Independent physical
stress and weak-load contractions agree within 3e-11 Pa and 3e-12 Pa m respectively.
Localization agrees exactly with its exported native phase/I formula.

## Acceptance, rejection, restart and retained paths

All four runs check repeated identical trials without accepted mutation, a
discarded larger-interval trial and identical retry, rejection of a nonfinite
candidate before writes, rejection of double relaxation, changed-rate rejection
followed by the manager's existing rollback, and exactly one publication per
accepted step. A second commit of a sibling candidate is rejected without writes.
Checkpoint creation also leaves persistent history unchanged.

The 16-step case writes a **real native filesystem checkpoint after eight
accepted transport steps**, then continues uninterrupted. A separate process
loads that snapshot and advances the remaining eight steps. At restore and at
the final endpoint, differences are exactly zero in all DG components, Z/I,
physical slip/q, every manager property and current/committed V, phase/FE anchor,
native solution/old/older vectors, clocks/counters, particle IDs/positions/all
properties, reconstructed stress and both weak loads. Persistent-vector and RNG
archives and complete canonically loaded manager archives agree bitwise.
There are 280 successful fresh-run gates and 18 restart gates.

The prescribed observer deliberately drives accepted Simulator clocks and
invokes native publication/checkpoint APIs; restart uses `no Advection, no
Stokes` to prevent an intervening coupled solve. Thus this verifies actual
native history ownership and filesystem persistence, **not** the production
outer-loop pending-interval, coupled solver or rejection controller.

The opt-in path rejects initial AMR, changed checkpoint G/quadrature and mode
conversion in four measured negative controls. Existing serial/fixed-grid,
unchanged phase/geometry/particles, periodic/hanging/moving/coupled restrictions
remain. The rebuilt default-off FE control matches the retained incoming-core
FE control exactly; same-build FE/DG fields differ by at most 3.20e-14 scaled and
weak loads by 7.32e-13 scaled. Existing particle/FE stress-poison, work-adjoint,
repeat/publication and full-gradient-source controls pass. All 23 rebuilt PF4
transport/operator-control CSVs reproduce the retained outputs byte for byte;
its independent weak-matrix/material-rate checks pass. Native tests pass 628
assertions in 8 cases, and all 94 PF1 mathematical checks pass. The four fresh
PF5 runs take 4.777–4.878 s and restart 3.975 s, including startup, extra trial
controls and checkpoint work. Maximum RSS is 214432 KiB. These are process costs,
not isolated throughput or coupled-trajectory timings. Every native run is capped
at 180 s, one rank/thread, with at most two concurrent. No production jobs were
started. The evidence package records exact commands and failed preliminaries.

## Remaining dependencies and concrete next task

CPDI/domain/H startup, fixed-particle validation, material-mixture projection,
support/association, derived particle stress publication and native particle/RNG
checkpoint services remain. H is frozen here; RSF Theta, nonuniform surface Z,
phase evolution/lower-bound transport, surface resampling and particle movement
are not qualified by this fixture. No filter, component clipping or benchmark
envelope is applied to native history.

The next proposed task is a **fixed-geometry PF5 transfer qualification**, before
any PF6 coupled fixture:

1. Retain this unit-square cohesive fault, fixed surface labels and fixed
   limiter settings. Add one prescribed bulk refinement/coarsening cycle and
   serial versus two-rank advancement/restart. Extend the existing native
   SolutionTransfer/ownership lifecycle for the three material-owned DG fields;
   use exact Q2 prolongation and conservative L2 restriction, including mixed
   modes. Resolve hanging/subface upwind fluxes and MPI ghost/owner publication.
   Check constants/global complete Q2, cell/global moments, non-proportional
   history, remapping norm/energy defects, physical stress and weak loads before
   and after transfer. Do not call remapping dissipation physical relaxation.
2. Keep the surface mesh unchanged for the first bulk-AMR fixture: persistent
   nodal Z, committed V, properties and shear sense then need no surface remap.
   Rebuild normalization/association data from transferred native phase and
   quantify resulting physical-stress/load differences. If the intended PF6
   fixed-geometry fixture also resamples the same physical surface, select an
   explicit conservative material-coordinate Q1 transfer for **Z**, not q;
   recompute I and recover q=Z/I. Verify work/measure consistency and transfer
   other selected surface histories, including Theta if RSF is selected.
3. Specify the intended PF6 fixture as a small fixed-geometry cohesive
   phase/mechanics test with no birth or propagation. First qualify transfer
   of native accepted phase and its **physical lower bound** through the existing
   FE solution-transfer infrastructure. Begin with refinement and coarsening
   regions where these fields are representable; reject incompatible coarsening
   rather than inventing a cap/floor. Verify bounds, irreversibility, accepted
   rollback and energy/work defects separately from physical phase evolution.
   Extend phase-history/normalization transfers only under a stated admissible
   rule; PF3's noncommitting phase operator is not already a coupled integrator.

Exit that next task only with measured serial/two-rank AMR/restart equivalence
and a precise list of transfers needed by the chosen PF6 fixture. Moving
geometry, fault birth, coupled propagation and full BP3 remain subsequent tasks.
This report proposes that work; it does not execute it.

Commands, exact artifacts, process costs, tolerances, preservation checks and
portable replay are in
[PF5 evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf5/README.md).
