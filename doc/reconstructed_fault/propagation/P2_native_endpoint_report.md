# Optional native consistent-endpoint P2 verification

The default-off native endpoint solver is implemented. Its fixed-state operators
pass, and a fresh native trajectory resolves 5.63e−6 flank growth, agreeing with
the matching prototype in phase to 3.28e−13. The one-hour cost gate stops before
the first core-changing entry converges: accepted native evidence ends at
0.5283203125 s, not the planned 0.53125 s. Core growth, native saddle escape
and full native evolution remain unqualified. The matching 512-cell Q2 prototype
completes 0–0.625 s at both 1/1024 and 1/2048 s and passes all seven unaligned
temporal targets. This is P2 with fixed sharp geometry; neither propagation nor
a unique physical band position is qualified. Updated 2026-10-11.

## Fixed model and initialization

| Setting | Value / units |
|---|---|
| Domain / sharp geometry | [0,1]×[0,1] m; one horizontal through-going fault at y=0.5 m |
| Bulk FE / mesh | Q2 velocity, Q1 pressure, Q1 phase; 16×512 Cartesian cells |
| Bulk quadrature | Six normal Gauss points; tangential bulk/fault intersection panels with Q3 |
| Surface / particles | 64 Q1 segments; 3×3 fixed particles per bulk cell, 73,728 total |
| Association | Fixed entire normal cross-section, half-width 0.50000000000001588 m |
| G / eta / lambda | 10^6 Pa / 10^8 Pa s / 100 s; identical homogeneous constituents |
| Viscosity bounds | 10^6–10^10 Pa s, applied to material viscosity before Maxwell coefficients |
| c / Gc / ell / p | 1000 Pa / 26.6666666667 J/m² / 0.15625 m / 1 |
| c0 / Hc / m | 8/3 / 0.5 Pa / 128.00000000016001 |
| Friction / pressure | Fixed mu=0.1; prescribed background normal pressure 10,000 Pa |
| Radiation / body force | Zero / zero; no RSF state evolution |
| Native slip-rate floor | Existing 10^−20 m/s; prototype zero-rate contact limit, no new floor |
| Seed | Shared continuum initialization, nominal core 0.6; relaxed core 0.6000019377630632 |
| Initial histories | Zero accepted FE strain, zero elastic surface history Z, zero stress and slip |
| Physical phase bounds | Immutable previous accepted phase ≤ phi ≤ 1; a trial at exact saturation has undefined normalization and is rejected |
| Boundary velocity | ux=0.256 t(y−0.5), uy=0 at top/bottom, applied as each interval's mean |
| Lateral conditions | uy=0, ux natural; no phase-essential boundary condition |
| T / reference dt | 0.625 s / 1/1024 and 1/2048 s; displacement difference 0.05 m |
| Native tolerances | Phase relative nonlinear 10^−12, linear 2×10^−7; mechanics 10^−8; unchanged arithmetic allowance |
| Prototype tolerance | Maximum scaled projected phase residual 10^−11; unchanged |
| Consistent endpoint phase solve | Enabled explicitly in fixture; production default false |
| FE initial phase profile file | Shared y/phi seed data; production default empty (native seeded-H relaxation) |

G=10^6 Pa remains a **dimensional synthetic verification value**. There is no
material optimization in this task. The rich 1D mechanical reference has 1023
free Q2 ux(y) coefficients, not the original one-free-mode approximation.
Native mechanics retains its full velocity/pressure space and tangential modes.

The shared seed is prescribed pre-existing damage protected by the obstacle.
It is not an unconstrained unloaded equilibrium of total energy. The accurately
relaxed physical seed is accepted before associations and cohesive/normalization
data are initialized. H is evaluated from its independent physical seed
prescription at the native particle coordinates, never regenerated from phi.
The fresh benchmark's maximum correction to the original sampled native H is
0.006267193664825754 Pa; the actual seed maximum and initial-field equivalence
are recorded in the evidence. Production shared-profile initialization itself
does not modify particle H. The native one-second initialization setting is
retained; unloaded cohesive initialization produces q=Z=0.

## Implemented algorithm, unchanged governing energy

Accepted physical state is `(d_R^a, Z^a, phi^a, slip^a)`, with
R=D3(symgrad d_R). For every trial phi, including Newton/tangent/line-search
trials, start mechanics from the same accepted d_R and Z. Previous physical
phi is the fixed obstacle. Predictor velocities may be initial solver guesses;
predictor phase is never the physical reference.

| Equation / operation | Previous restricted FE sequential step | Optional consistent endpoint |
|---|---|---|
| Phase condition | E_phi(phi, R_a, Z_a) with previous-phase obstacle | E_phi(phi, R_b(phi), Z_b(phi)) with the same obstacle |
| Maxwell endpoint | beta R_a + (kappa/G) D3 epsilon(u); beta Z_a + kappa V | Identical law; recomputed from R_a/Z_a for every trial |
| Mechanics / contact | Solved after accepting phase | Solved at every explicit phase trial |
| Reduced search functional | Incoming-history phase energy | J=E+(kappa/G) integral(mu P V ds); same physical E |
| Publication | Once after sequential physical solves | Once after complete endpoint convergence |

This comparison concerns the previously selected FE option. The older
particle-authoritative/default path retains its own established history and
phase law. The new switch does not redefine that path.

1. Solve endpoint momentum, weak incompressibility and surface contact at phi,
   without publishing constitutive or surface histories.
2. Construct d_R^b=beta d_R^a+(kappa/G)u and Z^b=beta Z^a+kappa V,
   with beta=exp(−dt G/eta), kappa=eta(1−beta).
3. Evaluate bulk/cohesive/CPDI fracture energy and its phase gradient holding
   these explicit endpoint internal variables fixed. Differentiate the actual
   consistent normalization I=M^−1 b and include its adjoint.
4. Minimize J=E+(kappa/G)∫mu P V ds with projected Newton descent and a
   true-energy line search. At mechanical stationarity its gradient is the
   fixed-endpoint phase gradient. The reduced tangent is the mechanical Schur
   complement; native actions currently use central differences of that gradient.
5. Check KKT and sampled admissible translation curvature. If a resolved
   negative translation curvature permits resolved descent, try both signs
   within the original obstacle. No physical Hessian, energy or bound is changed.
6. Restore normalization caches to the selected endpoint, validate the complete
   state and commit once. Publish accepted phase/current linearization, accepted
   FE coefficients, surface Z/I/q, slip and diagnostic particle samples/H together.
   Preserve simulator old/old-old rotation. Failed trials do not advance histories.

The source checks accepted FE strain/phase, committed slip, surface q/previous I,
and every particle H after each non-committing trial. Pressure-normalization
bookkeeping follows the selected mechanical endpoint and is restored on failure.
The tested energy evaluations ignore poisoned unrelated old/older/predictor
phase blocks and restore them normally and exceptionally.

Sparse CPDI fracture storage, with active rows/columns eliminated before
factorization by the existing serial sparse direct solver, provides a positive
search preconditioner;
no dense native phase Hessian is allocated. Fixed particle domains, homogeneous
fracture coefficients and AT1 make the native fracture functional exactly
quadratic. Each physical step assembles its actual sparse matrix, reference
gradient and reference energy once. Trial fracture energies and gradients reuse
this operator; tangent actions combine its exact action with the mechanically
reduced elastic-gradient difference. Every stationary candidate independently
reassembles the original CPDI residual and must pass the original entry target.
The reference energy offset stays fixed throughout the solve: replacing it by
a differently rounded particle sum would contaminate tiny energy differences.
The component check measures at most 8.74e−13 J/m energy-level difference and
2.17e−14 gradient difference. The energy-level offset is not claimed to be below
the line-search arithmetic allowance; consistent differences use the same
offset. No projection, physical energy or quadrature is changed.
A mass shift changes only this
preconditioner. Matrix-free actions at phase amplitudes 10^−5 and 5×10^−6
quantify differentiation uncertainty. Inner search uses the configured linear
tolerance; acceptance/convergence use the actual gradient and energy, never
the approximate action alone. The homogeneous bulk Stokes block and its AMG/pressure preconditioners
are rebuilt at the first trial of each physical step and reused within that
step: kappa, pressure scaling and homogeneous Newton constraints are fixed.
The phase-dependent B/G/K blocks and every actual coupled residual are reassembled.
Private velocity and slip-rate starting guesses are retained, while physical
R/Z/phase references stay immutable.
At the first entry only, a sparse primal-dual solve of
`min_{s>=0} g.s + s.P.s/2` identifies a feasible search direction using that
positive preconditioner. The true J line search accepts it; subsequent Newton
and curvature actions use the unchanged reduced Hessian. Four retained
prototype entries reproduce the original accepted phases within 8.54e−11.
The first native phase-changing state is unchanged within 3.17e−14 by this
search prediction. It reduces obstacle-release search cost without changing
the energy, bounds, physical history or stopping tolerance. Both configured
box/irreversibility flags are required; the configured 35-trial line-search
budget is retained.
The legacy mechanical path continues rebuilding as before. Translation probing is a sampled stability check,
not an exhaustive native critical-cone spectrum. Saddle escape in a full fresh
native continuation is not established by the fixed-state operator checks.

The new option requires serial 2D, homogeneous matching coefficients, fixed
Cartesian mesh/geometry, prescribed pressure, fixed friction, zero body force,
zero background shear and radiation, and no particle transport/replenishment.
Fresh initialization only: checkpoint conversion and native resume remain
rejected. Defaults and the particle-authoritative path remain unchanged.

## Complete discrete reference and schedule gate

No alignment or reflection is used for these criteria. Initial phase, H, FE
coefficients and Z are bitwise identical across the timestep family. Stress is
sampled independently with Q24 normal evaluation, away from assembly points.

| Quantity | 1/1024 vs 1/2048 maximum | Target |
|---|---:|---:|
| Full-field phase | 5.35773e−5 | .001 |
| Cohesive traction | 2.56624e−6 Pa | 1 Pa |
| Slip / reference final slip | 2.94551e−10 | .001 |
| Common-interval rate / fixed peak rate | 7.70734e−8 | .005 |
| Whole-section stress / 1332.521975 Pa | 3.44843e−6 | .01 |
| Localized stress / same scale | 9.69971e−6 | .01 |
| Bulk storage / .887807407 J/m | 1.49252e−8 | .01 |

Both paths pass KKT, contact, force, bounds and exact FE recurrence checks;
there is no measured healing. First full-field growth above 10^−6 is bracketed
by [.52734375,.5283203125] s on the coarse path. The largest increment drops
.001573902→.000787278. Peak centroid excursions are .7389/.7415 mm. A single
consistently followed branch is retained; positional uncertainty remains a
limitation. Two levels demonstrate a small halving difference, not a measured
convergence order or an exact continuum time solution.

The cheaper schedule uses one interval to .5 s, one to .52734375 s, then
1/1024-s intervals. Its full .625-s prototype also passes all seven targets:
phase .000317351, q .281976 Pa, slip fraction 3.87502e−5, common-rate fraction
.00332647, whole/localized stress fractions .000211675/.000211911 and storage
fraction .000417666. Loading is always the continuous ramp's interval mean;
total imposed displacement remains .05 m. The fresh native comparison uses
the first six intervals, ending .53125 s, with exactly its matching prototype.

This is a **field/storage accuracy gate, not a whole-path energy gate**.
The initial 0.5-s interval starts with zero R/Z in stick, but its reconstructed
constant interval-rate path slips before satisfying contact. The prototype
records 15.3446022 J/m friction-residual work and −15.3446022 J/m external
remainder on that interval. Full-prefix signed friction-residual work is
15.3400741 J/m, compared with −.001132625 J/m in the uniform 1/2048-s path.
The coarse prefix therefore cannot establish small cumulative physical energy
imbalance, despite accurate endpoint fields. Fine growing-step comparisons
are restricted to identical reported intervals. No compensating loss is added.
Independent Q8/Q16/Q32 time integration of five frozen reference paths,
including the coarse prefix and strongest final state, differs by at most
2.36e−12/8.25e−12 J/m from Q32. The large initial contact-path error is therefore
not a time-quadrature uncertainty. This checks the frozen prototype paths;
it does not replace native growing-state spatial or work checks.

## Native operators and energy accounting

Fifteen fixed states (unloaded, growing .546875 s and asymmetric .59375 s, plus
two symmetric perturbation spacings) exercise actual native normalization,
energy, CPDI fracture, point constitutive functions and weak FE quadrature.
These tests temporarily supply explicit FE fields and restore all physical
history; they are not imported checkpoints or fresh evolution.

| Maximum native–prototype operator error | Measured |
|---|---:|
| Bulk / cohesive / fracture storage | 2.47e−12 / 2.31e−14 / 1.75e−12 J/m |
| Scaled phase gradient, collapsed / full 2D | 5.37e−14 / 3.72e−15 |
| Normalization relative / directional derivative | 2.66e−15 / 5.91e−12 m |
| Weak interior force / contact | 3.03e−10 N/m / 5.55e−10 Pa |
| Point stress | 4.88e−10 Pa |
| Reduced-tangent action, scaled | 5.44e−9; half-spacing uncertainty ≤8.49e−9 |

Tangentially varying phase directions exercise the surface projection adjoint.
Energy finite-difference error decreases under halving; maximum at 10^−5
amplitude is 1.98e−6 J/m per phase unit. Normalization derivative relative
error ≤1.79e−9. These are operator consistency checks, not surface-resolution
accuracy estimates or a substitute for a native coupled solve.

Prototype work is integrated along the same documented diagnostic path:
linear phase, exponential R and Z from immutable incoming histories, and fixed
interval velocity. Intermediate mechanical, contact and phase residual powers
are retained separately. There is no added D_relax or compensating dissipation.

| Full-interval residual work | dt=1/1024 | dt=1/2048 |
|---|---:|---:|
| Mechanical, signed / absolute-step sum | .01294946 / .01294957 | .003237535 / .003237562 |
| Friction, signed / absolute-step sum | −.004480813 / .004601520 | −.001132625 / .001137972 |
| Phase, signed / absolute-step sum | −.008401129 / .008401267 | −.002097895 / .002105486 |
| External remainder, signed / absolute-step sum | −6.75176e−5 / 7.80320e−5 | −7.01418e−6 / 9.65581e−6 |

Cancellation makes the external remainder much smaller than its individual
residual-work terms. Each of those terms decreases about fourfold. Fracture
expenditure is 4.40543 J/m; normalization-profile work remains about −4.40547
J/m. This finite term is accounted for by the energy derivative, not discarded.
Incoming-state profile cost is positive (4.28988→2.14877 J/m cumulative); it is
not relabeled as viscous dissipation or a physically prescribed phase jump.

The native observer uses weak boundary-reaction work from the same FE operator
and quadrature, separately from lifting and bulk-rate work. It retains viscous,
friction, profile, storage and three residual-work contributions and compares
Q8/Q16 time integration at the final selected growing state. Its raw external
balance column uses `Delta E − W_reaction + losses`, opposite the prototype's
`W_reaction − Delta E − losses`; comparison must account for this stated sign.

## Fresh native accepted interval

The fresh `onset-obstacle-search` trajectory accepted through .5283203125 s
before its explicit wall budget interrupted the next, uncommitted entry.
Actual phase evolution occurs at the two outer flanks of the seeded band:
maximum cumulative growth is 5.62566e−6; core phase remains .600001937763.
No reflection or alignment is used. All 8721 phase nodes, 65 surface nodes and
three independent stress sections are compared with the identical schedule.

| Maximum native–matching-prototype error over accepted times | Measured |
|---|---:|
| Full phase | 3.28e−13 |
| Traction | 1.50e−8 Pa |
| Cumulative slip / common-interval rate | 1.63e−12 m / 6.10e−11 m/s |
| Whole / localized embedded stress RMS | 6.83e−7 / 8.37e−7 Pa |
| Bulk / cohesive / fracture storage | 3.01e−10 / 6.62e−11 / 7.82e−14 J/m |
| Normalization relative | 5.38e−14 |
| Tangential phase / traction variation | 6.50e−13 / 2.33e−8 Pa |
| Native normal-stress maximum (sum of component maxima) | 4.37e−6 Pa |

Every accepted row passes the original entry KKT target. Maximum published
physical phase residual is 5.12e−14, against entry targets 2.44–2.65e−11;
the scaled stopping allowance is 3.81–4.13e−13. Healing is zero and FE
recurrence error is at most 1.74e−18 m. The bounds and once-only publication
assertions pass. Translation curvature at .52832 s is positive in the sampled
direction, L2 Rayleigh quotient 48090.15; this is not global native stability.

For the fine growing interval [.52734375,.5283203125], native signed mechanical,
friction and phase residual works are respectively −9.57161e−9, +9.59639e−9
and +8.95066e−9 J/m. Matching prototype values are −9.56987e−9, +9.59637e−9
and +8.95066e−9 J/m. Native external remainder is +8.97537e−9 J/m (the
opposite prototype convention); the independent identity remainder is
−7.09e−14 J/m. Bulk/cohesive viscous losses are 1.68948e−5/1.05482e−4 J/m,
frictional loss .1307451 J/m, fracture expenditure 4.06995e−6 J/m and cohesive
normalization-profile work −4.06099e−6 J/m. The separate residual magnitudes
are retained rather than qualified through cancellation. This narrow, tiny
tail-growth interval does not establish core-growth or late-branch energetics.

## Source changes and measured cost

| Cost / completion gate | Wall time | Peak RSS | Outcome |
|---|---:|---:|---|
| Matching FE 640-step prototype | 279.94 s | 169 MiB | Complete .625 s |
| Matching FE 1280-step prototype | 383.92 s | 185 MiB | Complete .625 s |
| Native fixed-state operators | 267.98 s | 1296 MiB | Pass |
| Fresh native onset, final build | 3600.69 s | 1314 MiB | Three physical steps accepted; wall-budget stop during step four |

The final native cost includes 3534.31 s user and 21.80 s system CPU. The signal
exit is a deliberate cost stop, not a completed six-step run or evidence of an
inadmissible physical state. At .529296875 s the last completed phase correction
has scaled KKT 7.27823e−7 against target 4.14454e−13. Its inner Krylov residual
falls from 2.66647e−5 to 7.83396e−7 after five actions, against 5.33293e−12.
It has not converged; this log does not establish an arithmetic-floor stall or
a physical instability. No tolerance is loosened and no trial is published.

| Production file / function | Purpose and behavior |
|---|---|
| phase_field_fault.cc / declare/parse_parameters | Default-false endpoint flag and empty shared-profile filename; enforce restricted evolving FE/fixed-friction configuration |
| phase_field_fault.h | Explicit FEEndpointTrial, supplied-reference energy/normalization action and initialization API |
| phase_field_fault/fe_strain.cc / initialize_fe_shared_phase | Accept supplied fresh pre-existing phase before dependent data; never regenerate H |
| Same / make_fe_endpoint_trial | General-beta endpoint from the one accepted anchor, bounded native rate interpolation, fixed-friction potential |
| Same / evaluate_fe_phase_energy_at and wrappers | Reuse native energy and normalization adjoint with explicit R/Z; retain old wrapper/default behavior |
| phase_field.h, simulator/phase_field.cc / evaluate_fe_fracture, assembly, sparsity, initialization | Fracture-only scratch assembly; skip uninitialized Jacobian in residual-only calls; sparse optional phase graph; shared fresh seed |
| simulator.h, solver/reconstructed_fault_stokes.cc / solve_reconstructed_fault_stokes | Non-committing mechanical result with scratch pressure bookkeeping; reuse fixed homogeneous bulk block/preconditioner and private guesses within endpoint step; default terminal publication retained |
| simulator/assembly.cc / assemble_stokes_system | Permit residual-only assembly with the explicitly checked homogeneous Newton constraints in this optional fault path |
| solver_schemes.cc / selected scheme | Default-off route to consistent endpoint after advection, retain timestep-zero path |
| solver/reconstructed_fault_endpoint_phase.cc / solve_reconstructed_fault_endpoint_phase | Immutable anchors/obstacle; exact sparse CPDI quadratic reuse with fresh stationary-residual verification; matrix-free elastic tangent, projected Newton, actual-energy descent and translation escape; cache refresh and one complete publication |

Benchmark additions are native_endpoint.py, assess_native_endpoint.py,
prepare_native_endpoint.py, native_endpoint_verification.cc and its CMake
target, run/monitor/check_native_endpoint.py and assess_native_endpoint_run.py.
check_native_endpoint_search.py verifies the positive-operator search prediction
against unchanged physical minimization on four retained entries.
check_native_endpoint_path_quadrature.py isolates time integration on frozen
matching paths without changing or advancing physical histories.
They implement references, provenance, fixed-state tests, compact field/energy
observation and charts; they do not replace mechanical H or constitutive laws.
tests/phase_field_test_access.h extends its existing assembly test accessor with
an optional fracture-only flag, default false, for the independent CPDI quadratic
check. This is test infrastructure, not a new generic projection operator.
The cumulative earlier production inventory remains in
[P2_implementation_checkpoint.md](P2_implementation_checkpoint.md).

Prototype 640/1280-step runs take 279.94/383.92 s and 169/185 MiB peak RSS.
The successful full-gradient/quadratic native operator control takes 267.98 s
and 1296 MiB peak RSS (earlier full-gradient: 299.42 s /1283 MiB;
collapsed-gradient: 435.30 s /1235 MiB),
including fresh initialization, one unloaded physical step and all fixed-state
tests. The final default-path regression subset passes 21,105 assertions in
54 cases. A residual-only Jacobian access and a friction check at an inadmissible
zero rate were fixed as routine implementation defects; failures are preserved.
The initially unoptimized matrix-free onset runs were interrupted after
949.84/1062.94 s without a committed phase-changing state, to address repeated
assembly and free-set preconditioner cost. Those interruptions are cost controls,
not physical failures; CPU/RSS were not captured and are not inferred.
A subsequent bulk/preconditioner-reuse control stopped after 1093.26 s wall,
1058.97 s user CPU and 1297 MiB peak RSS, before a phase-changing commit, to
restart fresh with exact fracture-operator reuse. Its first projected Newton
correction releases additional obstacle nodes and increases the maximum KKT
residual; the matching prototype reproduces that increase. It is not evidence
of negative curvature or failure of the physical phase equation.
The first fresh prefix was interrupted to correct work diagnostics before
continuing growth. It accepted .5 s and agreed in q to 1.49e−8 Pa, in slip to
1.62e−12 m and sampled shear stress to 4.53e−7 Pa. This is a prefix control,
not a completed growing trajectory.

Build: GCC 12.4, deal.II 9.6.2, Trilinos 14.2, p4est 2.8.7, release, one rank/thread;
executable/plugin/input hashes and timing logs are in native_endpoint_evidence.
HEAD remains de7beaa65846c7372711f7922abf1c5d9a745ee6 on fault-propagation;
incoming dirty work and historical failures are retained. No VTUs are generated.

The anisotropic mesh made approximate fracture-AMG application expensive:
the fracture-reuse control spent many native tangent actions on each active-set
expansion. The optional serial search preconditioner now factors the same
restricted sparse fracture-plus-mass operator with ASPECT's existing
Trilinos/Amesos KLU solver. This is a preconditioner accuracy/cost correction;
the reduced Hessian, energy, physical bounds and stopping tolerances are
unchanged. The default phase and mechanical paths are unaffected. Each new
free set refactors this sparse search operator; no dense reduced Hessian is
formed. A fresh comparison measures its effect before any long run.

The KLU-only cost control accepted the first phase-changing state using 34
mechanical trials; its complete commit occurred 1020.8 s after run start.
The obstacle-search version accepts the same state with 11 trials at 489.3 s.
The following accepted entry needs 20 trials and about 1449 s, dominated by
repeated mechanical residual/Jacobian and bulk–fault assembly. At this cost,
640 growing steps would be of order 250 hours on one core, rather than the
minutes required by the 1D prototype; this extrapolation is not a measured
full-run cost. No full native trajectory is launched on this evidence.

## Evidence and remaining qualification

[Full reference charts](../../../benchmarks/reconstructed_fault/propagation/native_endpoint_evidence/temporal-comparison.png),
[temporal gate](../../../benchmarks/reconstructed_fault/propagation/native_endpoint_evidence/temporal-comparison.json),
[operator gate](../../../benchmarks/reconstructed_fault/propagation/native_endpoint_evidence/operator-checks.json),
[selected schedule gate](../../../benchmarks/reconstructed_fault/propagation/native_endpoint_evidence/prefix-2-gate-onset.json).
[Fresh accepted native charts](../../../benchmarks/reconstructed_fault/propagation/native_endpoint_evidence/onset-obstacle-search-comparison.png),
[native field/KKT/work comparison](../../../benchmarks/reconstructed_fault/propagation/native_endpoint_evidence/onset-obstacle-search-comparison.json),
[search-direction check](../../../benchmarks/reconstructed_fault/propagation/native_endpoint_evidence/search-prediction-check.json),
[frozen-path quadrature check](../../../benchmarks/reconstructed_fault/propagation/native_endpoint_evidence/path-time-quadrature.json),
[source/build identity](../../../benchmarks/reconstructed_fault/propagation/native_endpoint_evidence/final-source-provenance.json).

Reproduction commands (fresh output tags only):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/native_endpoint.py --denominator 1024
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/native_endpoint.py --denominator 2048
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/check_native_endpoint.py --prefix 2 --onset
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/check_native_endpoint.py --operators build-propagation-integration/runs/output-native-endpoint-operators-quadratic
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/check_native_endpoint_search.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/check_native_endpoint_path_quadrature.py
python3 benchmarks/reconstructed_fault/propagation/run_native_endpoint.py onset --tag NEW_TAG --max-wall-seconds 3600
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/p2-native-mpl python3 benchmarks/reconstructed_fault/propagation/assess_native_endpoint_run.py onset-obstacle-search
```

Successful reference destinations are protected against overwrite; reuse the
read-only assessments for retained runs. Individual native tags preserve their
logs, inputs, hashes and measured costs. A named harness stop-request file or
wall budget terminates a cost control without converting histories.

The bounded operator/early-tail equivalence assessment is complete, with the
native cost gate failing before substantial core growth. Keep the option
experimental and default off. The next focused numerical task is reduced-tangent
conditioning and cost: test an action based on the existing linearized mechanical
operator rather than two full nonlinear mechanical solves per difference, with
the same energy, history anchors, true KKT criterion and additional native modes.
That work and a new fresh core-growth check precede any 640-step launch.
Full native .625-s evolution, native saddle escape and global admissible stability,
tangential/spatial selection, physical rejected-step rollback, native coupled
restart and transported/birth histories remain unqualified. No P3 or physical
fault-propagation claim follows. All earlier failed trajectories and incoming
work remain preserved: 1306 historical files unchanged, zero missing files.
