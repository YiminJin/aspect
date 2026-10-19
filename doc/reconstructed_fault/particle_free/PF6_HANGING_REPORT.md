# PF6: bounded fixed-mesh hanging-node phase evolution

Scope selected 19 October 2026: serial, 2D, homogeneous domain-free accepted
DGQ2 cohesive history, fixed surface and a locally refined mesh fixed before
physical evolution. Unlimited history, Gauss12 physical/source3, seed, loading,
constitutive equations and native acceptance criteria are retained. This is a
constrained-solver qualification, not a spatial-convergence or propagation claim.
The incoming worktree consolidation is preserved separately in
`build-particle-free/hanging-phase/entry.patch`; base revision is
`612b75d3570d81763407875f3a12c12999b9fa8e`.

## Formulation

Closed native homogeneous convex Q1 hanging rows define a sparse prolongation
`phi=C psi`, with independent physical nodes as masters. Native
`AffineConstraints::distribute` supplies the primal expansion. Accepted physical
phase must satisfy precisely the same rows. The reduced functional is the
existing consistent-endpoint energy, including the unchanged normalization and
its first/second derivatives, with mechanics solved at each physical trial.

All raw physical phase duals are reduced exactly once:
`g_psi=C^T(g_elastic+g_fracture)` and
`H_psi dpsi=C^T(H_endpoint(C dpsi)+H_fracture C dpsi)`. The endpoint action
includes the native mechanical/slip response. Already constrained mechanical
loads are not reduced again. On the admitted local route fracture derivatives
come from the raw cell functional; system-matrix artificial constrained-row
diagonals never enter the master space. Uniform `C=Id` retains the former
assembly and operation order, including the legacy fracture path.

The obstacle is `psi>=psi_n`. Every master is a physical node and every dependent
row is a convex partition of unity. Therefore this is equivalent to physical
nodal irreversibility for a consistent accepted anchor, and convexity preserves
the physical upper bound. No dependent node is clipped independently. Final
physical feasibility is checked explicitly. Unsupported rows, MPI evolution,
nonhomogeneous/legacy local paths remain guarded.

KKT, active sets, phase linear accuracy and stability directions use only
masters. Positive roundoff scales are pulled back by `|C|^T=C^T`; no tolerance
multiplier is introduced. The search metric is the raw fracture pullback plus
`scale*C^T diag(m_physical) C`, where `m_physical` is the existing nodal
cell-volume lump. Physical stability Rayleigh denominators evaluate `C dpsi`.
This changes coordinates, not the physical energy or acceptance law.

Each trial retains the same accepted bulk/surface histories and physical lower
obstacle. The existing outer transaction owns staging, rejection and once-only
publication. Checkpoint history format is unchanged. The optional benchmark
observer stores an extra independent completion flag without changing its
previous serialized record.

## Declared bounded checks

Use the existing 64-cell preparation and one-cell 64→67 refinement before
positive physical steps. Move that single patch to
`0.5<x<0.75, 0.1875<y<0.25`: its upper interface is a growing flank, and the
lower interface is obstacle-active in the retained short event. Surface and
144 diagnostic points stay fixed; no particle domains are configured.

Require exact uniform-control field identity and native constraint/accepted
state identities; the existing source-derived publication allowance is
unchanged. Independent constrained FD gates use the established `1e-6` relative
operator criterion, reference energies/gradients/actions use `1e-9`, and physical
Gauss24/32 controls retain the prior integration checks. Native solver entry,
energy, mechanical and phase accuracy criteria are not modified. Diagnose any
failure before changing implementation or interpreting the result.

Require full native master expansion/adjoint/Hessian unit tests, independent
geometric reconstruction of C, normalization-sensitive complex-step checks,
re-equilibrated endpoint action finite differences, master KKT and stability,
short evolution, actual failure/retry and native timestep cutback, once-only
publication and restart against an uninterrupted accepted-sequence control.
The measured results and their limits follow.

## Measured qualification

Final native core SHA256:
`686288cbbda7cb9793249a5d9dc31cd6f5ea8414737896c376bf71aefa090f80`.
Final transfer observer SHA256:
`e6e95e83467c7623298af436353b7d64f4a75b68373aad8d5686e32ad9b297a3`.
Native and both observer targets build in the isolated evolving-phase tree.
The final complete native unit suite passes **129 cases / 165144 assertions**
(50 new assertions in two independent constraint tests).

| Check | Measured result |
|---|---|
| Uniform identity | All **162** non-timing CSVs byte-identical to the incoming qualified consolidation. No new parameter is needed in that control. |
| Actual local coordinates | 67 fixed cells, 90 physical phase nodes, **86 independent masters**, four native phase hanging rows. Physical phase expansion is exact. Maximum native hanging defect across all mixed fields/time levels is 5.56e-16. |
| Initialization | Original sampled physical prescription and unloaded T/Z retained. Refinement exactly prolongs that field; no reseeding. Original-node seed error zero; relative L2 representation error **0.0801966379666**, sampled Linf **0.0470205583862** against the supplied profile. |
| Independent constrained gradient | Scaled error **1.261e-14**, with C independently recovered from physical coarse-cell edges and normal integration recomputed. |
| Independent complete fixed-endpoint Hessian actions | Complex-step errors **1.043e-14**, **3.455e-15** for two master directions, including normalization pullbacks/second derivatives and FE fracture. |
| Mechanically re-equilibrated endpoint action | Best central-FD errors **2.321e-9**, **1.607e-9**; fixed-endpoint energy directional errors **4.485e-11**, **3.230e-12** at eps in {1e-4,1e-5,1e-6}. Report the full table, not just its minimum. Error scaling is max(1, norm of the analytic quantity), as in the retained checks. |
| Admissible stability | Complete **10×10** free-master Hessian: eigenvalues **7.35720506–25.10325151 J/m** in nodal coordinates. Scaled nonsymmetry **9.285e-17**. The other 76 masters are strictly active; there are no weakly active coordinates requiring an extra cone classification in this state. Independent eigensolution agrees. No constrained-row or active-row artificial diagonal enters this matrix. |
| Native stability probe | Positive physical translation Rayleigh quotients **2525.48**, **2519.87**; action uncertainty included by the existing acceptance check. |
| Histories/physical stress | All 27 DG coefficients per cell, Z/I, phase, native time levels and all surface properties exported. Independent full embedded-tensor reconstruction error <=**3.060e-15** at common points. Final Z range **6867.73–6871.56 Pa m**, I range **91.8552–91.9030**, V range **0.0471335–0.0471599 m/s**. |
| Quadrature | Native vs common Gauss32 energy/load scaled errors <=**5.042e-16 / 2.962e-15** on primary accepted snapshots. Normalization bulk-integration control <=**2.351e-14**; independent normal-ray reference <=**1.392e-15**. Gauss24/32 energy uncertainty <=**3.553e-15 J/m**. No quadrature input changed. |
| Accepted mechanics/fault | Native true bulk defects **1.040e-11**, **3.870e-11**; fault KKT RMS **1.827e-12**, **4.738e-12 Pa**. Independent weak-shear/cohesive functional errors <=**3.322e-14 / 1.331e-15**. |

Positive steps of the primary trajectory:

| t (s) | dt (s) | maximum incremental phase growth | growing hanging-node increment | active hanging rows | native master KKT (J/m) | publications |
|---|---|---|---|---|---|---|
| 0.125 | 0.125 | 0.01919493242 | 0.01916875154 | 1 | 2.405e-13 | 1 |
| 0.25 | 0.125 | 0.00078746456 | 0.00063824412 | 1 | 8.030e-14 | 2 |

Cumulative nodal growth is **0.01982827268**; healing is exactly zero. Independent
master KKT values are 2.397e-13 and 7.301e-14 J/m; higher normal quadrature gives
2.411e-13 and 7.214e-14. The retained stopping target is computed in master
coordinates with its original relative/operand-based rule. This is not a test
on all 90 physical rows treated as free variables.

## Transactions, restart and timestep control

The existing observer exercises the actual outer `solve_timestep` transaction:
a prepublication failure, rejected accepted-state/RNG/vector/clock/bookkeeping
identities, a successful staged retry, rejection of a transfer while pending,
discard without publication, and a terminal matching-cell sampling failure.
All these negative checks pass with their guards active. The real loop then
publishes each accepted physical step exactly once.

The native controller rejects step 2 at `(t,dt)=(0.25,0.125)`, retries at
`(0.1875,0.0625)`, and retains publication count one throughout rejection.
Accepted steps use `[0.125,0.0625,0.0625]`. Every one of these three accepted
states is **byte-identical** to the no-rejection control with that same sequence.
Restart from the accepted step-one checkpoint continues to step two and is
**byte-identical in 20 flat/reference exports**, including every persistent
history/time level, native loads, stress, particle properties and RNG. The
step-one parent prefix is likewise identical. The final core additionally
replays a checkpoint written before the final guard-only correction; native
history format/ownership are unchanged.

The dt/2 control reaches t=0.25 in four accepted steps. On common physical points,
relative full-tensor T/stress L2 differences are **8.604e-5 / 4.128e-5**; physical
phase Linf/L2 differences are **1.428e-7 / 3.809e-8**, and phase-gradient Linf is
**3.737e-6**. Old/older levels at different physical times are excluded from
this temporal field comparison. This single halving is a bounded sensitivity
control, not temporal-order qualification. The first half-step sticks at the
native minimum V: its **500.156 Pa unprojected** surface RMS is a permitted
negative contact reaction, with **zero projected fault KKT violation**. Raw
loads and the raw defect remain exported. Subsequent half-step fault residuals
are below 4.42e-12 Pa.

Independent reuse of the existing PF6a work ledger gives energy-identity
remainders <=**1.45e-15 J/m**, with 12/24-point time-integration uncertainty
<=**4.99e-15 J/m**. The ledger follows linear physical phase and exact Maxwell
T/Z at fixed endpoint rates; it is accounting, not an alternative integrator.
The **external** storage/work/dissipation remainders remain **0.4563337032** and
**4.046947583e-5 J/m**, explained in that ledger by interior mechanical work,
contact work and phase work along this nonequilibrated interval path. Do not
interpret algebraic closure as a closed physical energy balance or first-event
work qualification. Original endpoint temporal findings remain open.

## Evidence and reproduction

[Compact verification](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6_hanging/verification.json)
contains all formulas' numerical checks, full FD table, eigenvalues, physical
quadrature separation, native gates, exact lifecycle comparisons and BP3 hashes.
[Evidence instructions](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6_hanging/README.md)
describe the replay archive and manifest. Local labels under
`build-particle-free/runs/`, for the qualified controls:

| Label suffix `-20261019` | Accepted steps / native gates | Budget / measured elapsed (s) |
|---|---|---|
| `hanging-final` (operators + recovery) | 2 / 112 | 300 / 47.22 |
| `hanging-uniform-final` | 2 / 100 | 180 / 13.54 |
| `hanging-parent` | 1 / 84 | 180 / 28.52 |
| `hanging-resumed-final` | 1 continuation / 37 | 180 / 10.59 |
| `hanging-cutback` | 3 / 127 | 180 / 46.85 |
| `hanging-matched` | 3 / 120 | 180 / 41.34 |
| `hanging-half` | 4 / 138 | 180 / 42.90 |

Run order in raw records places the parent/cutback/half controls before the final
source repeat. Every record preserves actual argv, environment, input, core,
observer and source identities. The only final differences from those earlier
passed controls are an explicit periodic exclusion, ordinary header includes,
and diagnostic units/contact-KKT exports; equations/arithmetic are unchanged.
Durations include extensive diagnostics and deliberate retries; they are not
production performance measurements. Final unit budget 120 s, elapsed 5.67 s.
No native timing runs overlapped.

Prepare a fresh primary input with the existing thin wrappers, then launch its
recorded argv directly:

```sh
python3 benchmarks/reconstructed_fault/particle_free/run_pf6_hanging.py prepare LABEL --binary build-particle-free/evolving-phase/native/aspect-release --plugin build-particle-free/evolving-phase/plugin/libpf6_transfer_checks.release.so --operators
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py LABEL 300 env ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1 OMP_PROC_BIND=false /opt/openmpi/5.0.6/bin/mpirun --map-by core --bind-to core --rank-by slot --report-bindings -n 1 build-particle-free/runs/LABEL/aspect-release build-particle-free/runs/LABEL/input.prm
python3 benchmarks/reconstructed_fault/particle_free/run_pf6_hanging.py finalize LABEL
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/evidence/pf6_hanging/verify.py --protected build-particle-free/hanging-phase/protected-bp3.json --output benchmarks/reconstructed_fault/particle_free/evidence/pf6_hanging/verification.json
```

Use `--end .125` for the parent, `--restart .../run/restart` for continuation,
and `--schedule cutback|matched|half` for the recorded controls. Record the argv
in `command.json`; these wrappers prepare/log inputs and do not control the
native phase, mechanics, time integration or acceptance loop. Build commands
and raw logs are preserved under `refactoring_boundary/evidence/hanging-*`.
Only one observer DSO is loaded in each run.

Two failed diagnostic-development runs are retained: `hanging-coupled` had an
inadmissible untapered central perturbation in the tiny phase tail;
`hanging-coupled2` reused a frozen tangent after FD mechanics changed its
linearization generation. Taper only diagnostic directions by `phi*(1-phi)`
and compute every frozen action before FD relinearization. Both native guards
remain intact; physical trajectories/criteria were not corrected to fit a test.
The final observer also corrects the initial raw `KKT_Pa_m` column's dimensional
label to `KKT_J_per_m`; replay normalizes that label without changing raw values.

## Limits and retained dependencies

The coarse sampled seed is not an equilibrium phase profile (initial projected
master gradient **0.516495 J/m**). Early growth includes its relaxation. These
results qualify constrained coordinates and native lifecycle on this fixed
mesh; they do not turn this coarse fixture into a mechanical/profile spatial
reference or qualify physical onset. Preserve **strict_reference_match_passed=false**,
all mirror/branch discrepancies, original Gauss6 failure and spatial/endpoint
accuracy findings. All **34 protected BP3/package/build files** match their
entry hashes; no server job, package regeneration or BP3 edit occurred.

All 144 particles remain stationary, with unchanged IDs/coordinates, no domains
configured/constructed/read, and no additional population. Remaining reads
support fixed-population/admission checks, independent matching-cell native
publication checks, diagnostic stress/H properties, transaction assertions and
native persistence/RNG. The selected phase/material/mechanical/fracture data
come from the prescribed seed, configured homogeneous values, bulk mesh and
fixed fault, with authoritative DG history; no particle stress feeds mechanics.

Nonhomogeneous/periodic phase constraints, MPI local-phase evolution, runtime
AMR and lossy obstacles, transport, surface changes, moving geometry, birth and
propagation remain outside this qualification. The existing legacy/default
routes and limiter/filter settings remain unchanged. This task stops here.
