# PF0/PF1 results and PF2 proposal — 15 October 2026

PF0/PF1 are complete. **Production source and defaults are unchanged.**
The independent checks validate restricted mathematics, not a particle-free
production model. [SOURCE_AUDIT.md](SOURCE_AUDIT.md) records actual equations,
ownership, hidden dependencies, instruction/path reconciliation and provenance.
PF2 has not begun.

Historical boundary: the subsequent user selected PF2, now completed and
reported in [PF2_REPORT.md](PF2_REPORT.md). The PF1 results and proposal below
remain unchanged evidence of the preceding stage.

## Runs and measured gates

| Check family | Measured result | Meaning |
|---|---|---|
| Retained native history fixture | Exit0, 60.274 s within180 s; 12 synthetic paths,106496 stress samples/path | Native fixed-profile recurrence/commit/rejection diagnostic; no coupled trajectory |
| Native normalization | min I=4841.791606m; relative difference to independent Q32=4.8504e-16; Q16/Q32=8.5331e-16 | Positive completed profile on this fixed dipping fixture |
| Native stress recurrence | max error28.1476Pa at enormous artificial history scales; max pointwise error/retained allowance=0.00992856 | Passes existing arithmetic gate; absolute error alone is misleading here |
| Native Z recurrence | max absolute16Pa m; max scaled error1.7766e-16 | Mature auxiliary memory retained independently of zero cohesive q |
| PF1 V1–V4 | **94/94 assertions pass**,0.0168 s numerical work | Fresh mathematical evidence; no production calls |
| Local identity gates | max scaled defect1.0103e-14, target1e-12 | Includes full 3D plane-strain tensor norm |
| Two-profile mature closure | compact-only L2 defect0.494094542952Pa sqrt(m); residual representation scaled defect1.3119e-16 | Current profile and one scalar memory alone do not retain general mature history |

Native fixture details: a 2000x1000m translated Box,2048 Cartesian cells,
Q2 velocity/Q1 pressure and phase,32768 fixed particles, six-point tensor rule,
one60-degree fault with paired automatic completion, ell20m, p=1,
G=32038120320Pa and eta=1e26Pa s. Theta initialization/material support remain
particle-based. The diagnostic injects two nonuniform Q2 velocity modes
(including nonzero divergence) and six dt/lambda ratios
`1e-12,1e-6,.1,1,4,.25`; the last is a zero-strain/minimum-V relaxation.
It exercises candidate rejection, unchanged anchors, duplicate-commit rejection,
mature q=0 and transport rejection, then restores its synthetic histories.
These are not twelve physical BP3 trajectories or a native restart test.

The copied core's embedded revision is `c33edaaeb`, not the audit HEAD
`3232056c`. Compilation included historical dirty work; exact correspondence
to every current source file is not proved. The manifest distinguishes the
checkout from the actual binary. No new native compilation, coupled-growth
run, MPI test or historical numerical campaign was needed for this test-only
task. Nonfatal sandbox socket warnings did not prevent serial completion.

## Reproduction and evidence

From the repository root, using new output names:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/reference_checks.py --output build-particle-free/pf1-reproduction.json
python3 benchmarks/reconstructed_fault/particle_free/run_baseline.py --retained-build /home/ein/repository/aspect-fault-propagation/build-bp3-native --output build-particle-free/pf0-reproduction --seconds 180
```

Actual outputs: `build-particle-free/pf1.json` and `build-particle-free/pf0/`.
The exact baseline command used the same runner with `--output
build-particle-free/pf0`. The runner copies the core/two matching plugins,
expands paths and redirects all writes locally. It does not change the old
build. Serial execution and OMP/deal.II/OpenBLAS thread counts are1. Local
budget:180s native plus10s reference checks, one process at a time; observed
reference process wall time was about0.17s, including imports.

Python 3.14.7, NumPy 2.5.2 and SciPy 1.18.1 were used (machine-readable identity
is authoritative). No additional dependency installation was required.
Compact results, hashes, input and native CSVs are in
[`evidence/`](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf1.json);
large parameters, complete logs and copied binaries remain in ignored output.
Earlier89/93-check development successes are preserved locally; final94-check
evidence supersedes them after adding explicit narrow compatible initialization.

Every check asserts or exits nonzero; expected failure of a proposed
simplification is an explicit *nonzero-defect assertion*, not a silently passing
bad approximation. `source-manifest.json` identifies audited source/reference
files. Syntax, evidence JSON and Git whitespace checks were also run. No
production C++ tests/build were rerun; the native observer uses its retained
compiled assertions.

## References, norms and findings

All tests use independent formulas/characteristic integrals; they import no
ASPECT code or old prototype. G=3Pa, eta=6Pa s and lambda=2s are unit-test
inputs, not changes to production material. Local tensor norms are full3x3
Frobenius norms. Batched identities use the Euclidean norm over the stated
samples; their negative-control magnitudes are sample norms, not continuum
spatial errors. Scaled error is `||q-q_ref||/max(q_scale,||q_ref||)`.
The predeclared unit scale is1 in the listed SI units, except Maxwell kappa
uses the predetermined input increment scale `eta*min(dt/lambda,1)` to expose
tiny-step cancellation. Identity tolerance is1e-12; no gate was relaxed.
The separate finite-difference/truncated-integration checks use their declared
1e-5/1e-6 discretization targets and second-order tests.

**V1: Maxwell, initialization and tensor convention.** Stable double
`expm1` coefficients are checked against independently rounded70-digit Decimal
exponentials at dt/lambda=`1e-16,1e-12,1e-6,.1,1,50`. Both nonzero compatible
initial stress/slip and zero/nonzero forcing are included. Direct physical
Maxwell evolution agrees with trial-minus-localization reconstruction;
traction is `G s/I`, `S:S=1/2`, trace and reconstructed zz are checked.
Naive `1-exp(-r)` at r=1e-16 loses11.0223% of the increment; stable evaluation
passes. A zero-physical-stress/nonzero-slip initial state explicitly constructs
a narrow trial tensor from chi (width.01m). Its gradient exceeds50Pa/m.
Smooth independent history is therefore not a universal consequence of the
change of variables.

**V2: product rule, coefficients and energy.** A normalized Gaussian widens
with time while elastic slip is nonzero. Complex differentiation of the
reconstructed tensor gives the stress-rate equation with
`Q_geo=-2G s chi_dot S`: scaled defect4.0878e-16. Omitting this term leaves
sample norm3.32827Pa/s. A time-dependent multiplicative h amplitude with fixed
width changes I proportionally and cancels from chi (defect5.2843e-16).
A finite cohesive physical phase change changes stress, sample norm1.12324Pa;
the distinct mature residual remap preserves physical stress to1.0974e-16
scaled. These two updates must not be conflated.

Normally varying bulk relaxation at constant surface rate and a separate
time/materially varying modulus use an analytic manufactured stress solution,
differentiated independently by complex step. `eq:variableG` defects are
1.4324e-16 and1.5674e-16; omitted extra terms have sample norms0.151044 and
0.885977Pa/s. The explicitly stated test law has no extra varying-spring
`(D_t G/G)tau` term. Which law to use outside constant G must be selected
physically; these tests do not silently choose one for production.

The fixed-profile cohesive power identity uses the normal integral, unit
tangential length, adjoint driving traction and a defined positive remainder
resistance4.37812Pa. Storage derivative equals mechanical input minus separate
bulk viscous, surface viscous and resistance losses (measured defect0 at double
precision). This is algebraic energy balance, not RSF or moving-area energetics.

For `eq:phasevar`, an interior profile uses the source's Lorentz-p1 rational h,
I integrated from that profile, fixed incoming trial tensor/slip, and explicit
AT1/gradient energy. The derivative includes both local h and normalization
terms. Complex-step defect is3.3654e-16. A separate centered-difference sweep
with increments.01/.005/.0025 gives errors
`1.17389e-5,2.93455e-6,7.33626e-7 J/m`, orders2.00009/2.00002.
This tests the **unprojected continuum normal integral** on a fixed resolved
rule, not the native projected-I adjoint or CPDI derivative. Active bounds,
material phase transport and stability remain separate gates.

**V3: commutator.** Characteristic stress integrals are compared with the
transformed stress whose residual is integrated from the analytic commutator
forcing. Separate physical cohesive stress integrals include Q_geo. Frozen
profiles without transport and rigidly co-translating material/profile/labels
commute (scaled errors<=1.26e-16). Material crossing a fixed profile does not:
the correct residual agrees to2.66e-16, while omitting it gives sample tensor
error1.14366Pa. Different bulk/surface tangential speeds with varying surface
memory give1.89e-16 agreement and0.103214Pa omission error. Thus stationary
fault geometry alone does not justify zero C. Including cohesive redistribution
instead gives the compact cohesive stress, defects<=3.27e-16.

A separate **counterfactual objective-law** test rotates the full profile and
slip tensor rigidly. Jaumann `A_dot+A W-W A` vanishes to1.76e-16 and the
relaxing reconstructed tensor matches the rotating characteristic solution.
The non-objective derivative is nonzero (norm0.989949/s). This verifies the
sign convention in the proposed extension; it does not add spin to the
source's nonrotational constitutive law.

**V4: two-profile mature memory.** Slip V=.2m/s acts on normalized Gaussians
of width.12m centered at-.35m then+.35m; switch time.4s, final time1s.
Along a fixed characteristic the exact exponentially weighted slip moments
are

```
w1=lambda V [exp(-(T-t_switch)/lambda)-exp(-T/lambda)]
w2=lambda V [1-exp(-(T-t_switch)/lambda)]
tau_direct=-2G(w1 chi_old+w2 chi_current) S.
```

The residual's finite jump is propagated independently from the switch.
`tau_tr-2G chi_current s_mem S-tau_res` matches direct history to1.3119e-16
scaled. The missing-history L2 error on the normal line is known analytically:

```
2G w1 sqrt([1-exp(-separation^2/(4 width^2))]/(2 sqrt(pi) width))
 = 0.494094542951848 Pa sqrt(m).
```

Normal integration on[-2,2]m has negligible Gaussian tail error and agrees
with that value to1.01e-14 absolute. A separately discretized midpoint
characteristic integral gives:

| Panels per time interval | Direct L2 error (Pa sqrt(m)) | Compact-only error (Pa sqrt(m)) |
|---|---:|---:|
|16|1.0137563e-5|0.494099919|
|32|2.5344096e-6|0.494095887|
|64|6.3360359e-7|0.494094879|

Direct-integration orders are1.99999 and2.00000; compact-model error remains
finite. Zero initial residual does not license dropping its subsequent
evolution. Auxiliary memory has no cohesive energy/resistance. This is a
closure counterexample, not evidence favoring particles over DG.

## Concrete PF2 proposal (implementation not started)

Select a **default-off fixed-profile cohesive DG trial-stress history
prototype**, with the existing physical elastic-slip surface state. Use the
audited no-spin, no-advection law, homogeneous matched G/eta, affine Cartesian
Q2 velocity, one fixed straight horizontal fault, the same accepted seed,
normalization, mechanics and quadrature as the small P2 control. Freeze phi,
geometry and population after initialization. This is an exact history
transformation under these assumptions plus an explicit storage replacement;
it is not full particle removal. No new redistribution law is needed.

1. In `include/aspect/material_model/phase_field_fault.h` and
   `source/material_model/phase_field_fault.cc`, add an explicit experimental
   bulk-history selection, preserving both existing selections/defaults.
   Enforce fixed mesh/profile/geometry/population, constant matched
   coefficients, no transport, no resume/conversion. Select the existing
   three-dimensional FE strain convention; do not change the legacy default.
2. In one material-owned implementation file, proposed
   `source/material_model/phase_field_fault/dg_history.cc`, store a three-component
   symmetric in-plane DGQ2 **trial stress** using deal.II DoFHandler/FE_DGQ and
   distributed vectors on the existing mesh. Reconstruct zz=-xx-yy. No new
   manager/backend hierarchy. On affine Cartesian cells DGQ2 retains mixed
   Q2 velocity-gradient degrees; DGQ1 cannot generally do so. Use at least
   QGauss3 per direction for the DG mass/source projection (nine scalar DoFs);
   retain the existing shared mechanical six-point rule. Reject or quantify
   mapped-cell errors before generalization.
3. Initialize compatible trial stress `tau_tr0=tau0+2chi Z0 S`, checking
   representability; the existing unloaded seed gives zero initially.
   Read immutable accepted state during every trial, form
   `tau_tr+=beta tau_tr-+2kappa D3(symgrad u+)`, and reuse the existing Q1
   `Z+=beta Z-+kappa V+` physical slip update. Evaluate **reconstructed physical
   stress** at mechanics integration points. Do not project chi or the narrow
   reconstructed correction into the independent DG field.
4. Reuse `retained_history_stress` in `fe_strain.cc` as the existing mechanical
   history hook, routing the experimental selection to its authoritative DG
   field. Inspect its consumers in
   `source/simulator/assemblers/reconstructed_fault_stokes.cc`,
   `source/reconstructed_fault/surface_system_bulk_work.cc` and
   `source/material_model/phase_field_fault/constitutive.cc` together so Stokes,
   fault loads and accepted history use the same source and tensor convention.
   The fixed-profile physical recurrence remains
   `beta(tau_tr- -2chi Z- S)+2kappa(E+-chi V+ S)`.
5. In `history.cc`, stage and validate the complete DG/surface candidate before
   publication; particles may still receive derived stress samples for existing
   consumers. Keep the independent native recurrence check, exactly-once
   publication and rejected-candidate restoration. Make no generic projector,
   checkpoint schema or CPDI phase change. Do not route this option through
   incoming-FE total-energy phase evaluation: phase is frozen in PF2; new phase
   derivatives belong to PF3.
6. Extend `tests/phase_field_fault_test_access.h` narrowly and use existing
   `unit_tests/phase_field_fault_maxwell.cc`,
   `unit_tests/phase_field_fault_cohesive.cc` and the native-history observer
   conventions. Add one small selectable fixed-profile fixture/observer under
   `benchmarks/reconstructed_fault/particle_free/`, deriving its physical inputs
   from `propagation/native_fe.prm`/`recoverable.prm`, with both DG and accepted-FE
   runs consuming the **same saved accepted initial phase**. Compare a bounded
   two-step loading/hold path, not the full native endpoint trajectory.

Minimal PF2 checks: V5 source representation with cross-term Q2 velocities,
an independent DGQ1-loss negative control, DG mass rank, V6 full weak-load/work
adjoint checks (target1e-10 scaled), and fixed-profile V9 repeated trial,
failed candidate, accept once and duplicate commit. Compare DG source/stress
with analytic/local references at1e-12 when well conditioned; compare to native
FE on matched quadrature, including whole-bulk weak loads, q,V,slip and work.
Re-run a disabled-option control and existing cohesive/Maxwell units. Native
restart remains deliberately rejected rather than claimed qualified.

After implementation, proposed commands (new fixture names are deliverables,
not files claimed to exist now):

```sh
cmake -S . -B build-particle-free/native -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ -DCMAKE_C_COMPILER=/opt/gcc/12.4.0/bin/gcc -DDEAL_II_DIR=/opt/dealii/9.6-local -DASPECT_WITH_VORO=ON -DVORO_DIR=/home/ein/local/voro++/0.4.6
cmake --build build-particle-free/native -j2
build-particle-free/native/aspect --test '[phase_field_fault_cohesive]'
build-particle-free/native/aspect --test '*Maxwell*'
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/reference_checks.py --output build-particle-free/pf2-math-control.json
build-particle-free/native/aspect benchmarks/reconstructed_fault/particle_free/fixed_profile_fe.prm
build-particle-free/native/aspect benchmarks/reconstructed_fault/particle_free/fixed_profile_dg.prm
```

Configure the new observer with the existing ASPECT plugin convention and this
new build's `Aspect_DIR`; resolve all fixture includes/library/output paths into
this worktree. Commands above are a concrete starting configuration from the
retained cache, not evidence of a PF2 build. Begin with a small serial fixture,
180s per control, extending only for a specific unresolved gate. The old
builds/outputs remain read-only. A mature direct-stress control can be proposed
separately; extending compact mature memory beyond C=0 requires residual/direct
transport, not a smoother interpolation or a silent cohesive resistance.

PF3 projected-normalization/FE fracture replacement, PF4 transport/limiting,
geometry birth/transfer, restart/AMR/MPI expansion and propagation remain
unselected. Moving-surface energetics, heterogeneous laws and physical branch
selection remain open. PF0/PF1 has not removed any of those requirements.
