# Particle-free status — selected PF5 persistence subset complete

Updated 16 October 2026. User accepted the PF4 method recommendation and selected
fixed-mesh/fixed-geometry native history advancement, rollback and restart.
Implemented default-off `Enable prescribed DG accepted history`; retain
**unlimited upwind DGQ2** as the experimental baseline. Production defaults and
limiter settings are unchanged; both WENO paths remain experimental.

- Native exact Maxwell/source half steps surround pure unlimited SSPRK3
  transport; surface Z advances once with the existing cohesive law. Independent
  exact semidiscrete reference establishes **second-order combined accuracy**:
  T relative L2 falls from 3.75525e-6 to 5.86693e-8 over four step counts
  on one 4×8 grid. Finest physical stress/weak-load errors are 5.87360e-8/8.92354e-8.
- Non-proportional tensor and nonzero physical elastic slip are included.
  The fixed-profile cohesive commutator is retained; final Z=299.550449662703 Pa m,
  maximum analytic Z error 9.10e-13. No mature Qgeo=0 substitution or spin change.
- 280 fresh-run and 18 restart gates pass: repeated trials, rejection/retry,
  nonfinite/changed-context rejection, single publication and checkpoint creation
  without accepted mutation. A real native filesystem restart matches all checked
  persistent histories, physical stress and both fault weak loads exactly.
  Persistent/RNG and canonically loaded complete manager archives match bitwise;
  raw manager archives retain the existing implicit-to-explicit +1 shear-sense
  representation difference. Failed preliminary evidence is preserved.
- Existing checkpoint/solution/manager/particle infrastructure is reused.
  Four measured guards reject initial AMR, coefficient/quadrature mismatch and
  mode conversion. Native 628 assertions/8 cases and 94 PF1 checks pass;
  default-off FE matches the retained incoming-core result exactly, same-build
  FE/DG operator/publication checks pass, and all 23 PF4 operator-control CSVs
  reproduce byte for byte. No further WENO sweeps or threshold tuning.

This completes the **selected bounded subset**, not general PF5 transfers or
coupled stepping. The observer prescribes accepted clocks; surface V/Z are
uniform, phase/geometry/particles fixed and transport boundaries closed.
Remaining particle dependencies include startup H/CPDI/domains, surface material
projection, support/association and fixed-particle checks, derived samples and
particle/RNG persistence. RSF Theta, phase-bound transport, nonuniform surface
transport, AMR/MPI/periodic/open-boundary splitting, moving geometry and birth
remain unsupported or unqualified.

**Proposed next task:** fixed-geometry PF5 bulk AMR/MPI transfer and restart
qualification, reusing native ownership/SolutionTransfer, Q2 prolongation and
conservative restriction. First keep the surface grid fixed, then qualify any
material-coordinate Z/Theta transfer and native phase/physical-lower-bound
transfer actually required by a separately selected small fixed-geometry PF6
phase/mechanics fixture. Measure remapping work/energy defects separately from
physical evolution. No PF6, moving geometry, fault birth, coupled propagation or
BP3 is automatically selected. Stop here.

[PF5 progress, measurements and concrete next task](PF5_REPORT.md),
[source audit](SOURCE_AUDIT.md),
[portable evidence/replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf5/README.md).

---

The prior PF4 method comparison and earlier evidence below are historical and
retained unchanged.

# Particle-free status — bounded transport-method comparison complete

Updated 16 October 2026 UTC. The selected comparison is implemented and measured.
**Retain unlimited DGQ2 as the resolved smooth/prescribed-history baseline.**
Both WENO paths remain experimental; no threshold tuning or further WENO
sweeps were performed. Scaling remains a scalar benchmark reference.

|32×64, common base step|Unlimited DGQ2|Scaling|Native scalar EV|Native scalar SUPG|
|---|---:|---:|---:|---:|
|Sharp relative tensor L2|.177548|.226747|.225369|.211070|
|Finite-width relative tensor L2|.000427638|.000427638|.001275466|.000545821|
|Scalar DoFs|18,432|18,432|8,385|8,385|

Independent exact split-cell projection gives **.149862170276557** for the sharp
DGQ2 representation floor, confirming the predicted .1498621703. The report
separates representation, numerical evolution and native interpolation errors;
CG sharp initialization alone is .223607. Fixed-mesh time differences, extrema,
all three front widths, signed tensor-integral errors, norm changes and measured
costs are retained. Exact projection itself can overshoot substantially.

A native FE-strain-generated independent tensor is transported without
publication, with nonzero shear and normal fault weak loads. At half the base
step, weak-load relative errors are DG **2.272e-5**, EV **1.140e-4**, SUPG
**3.648e-5**. DG and SUPG tensor errors are close (.004595/.004629), but DG has
lower measured fixed-coefficient cost and better resolved-pulse accuracy. SUPG
weak-load temporal uncertainty is comparable to its gap from DG; no converged
native-load ranking is claimed.
Native EV/SUPG are actual continuous-Q2 scalar solves with native BE/BDF2 and
unchanged stabilization defaults, not reimplementations labeled native.

Production sources/defaults/laws/backend and all PF0–PF4 evidence are preserved.
All fourteen comparison records match the same core and 879 current production
source hashes; ten new successful controls are bounded to 180 s, one rank/thread,
at most two simultaneous. DG/scaling fields reuse unchanged-source evidence.
Projection/error/load checks, 628 native assertions and 94 PF1 checks pass.

Remaining particle dependencies: H/CPDI/domain startup, fixed-particle checks,
support/association, surface material projection and derived samples. Transport
reads no particle properties. No production adoption, coupled evolution, restart,
AMR/MPI/periodic/moving geometry, BP3, propagation or PF5 is selected. Stop here;
any next experiment needs a separate user selection.

[Measured comparison and recommendation](PF4_METHOD_COMPARISON_REPORT.md),
[source audit](SOURCE_AUDIT.md),
[portable evidence/replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf4_methods/README.md).

---

The prior PF4 reports and evidence below are retained unchanged.

# Particle-free status — selected PF4 follow-up complete

Updated16 October2026 UTC. All four user-selected tasks are implemented and
measured: fixed-mesh timestep sensitivity, direct norm changes at limiter calls,
common scaling on the signed pulse, and a physically scaled finite-width control.
The user confirmed the benchmark envelope, largest common factor and
`delta=ell/ln9`, with errors against the actual overlapping analytic profile.
**Keep unlimited as the smooth baseline; retain both WENO paths and scaling as
experiments. No more elaborate stabilization is justified by this resolved
finite-width control. Stop after bounded PF4; no production adoption or PF5.**

|32x64, original dt|Unlimited|KXRCF/WENO|Fu–Shu/WENO|Scaling|
|---|---:|---:|---:|---:|
|Sharp relative tensor L2|.17755|.37629|.37658|.22675|
|Sharp Q loss|3.574%|28.948%|28.903%|12.549%|
|First sharp front width, m|.02353|.09967|.09686|.02655|
|Finite-width relative tensor L2|.00042764|.00055178|.00044103|.00042764|

Fixed32x64 dt/dt2/dt4: sharp WENO dt2-to-dt4 field changes12.98–13.16%; scaling
.02812% (largest common-time change.03615%). Unlimited behaves at third order.
KXRCF finite-width L2 grows to.051565 at dt4. The WENO family is not temporally
resolved; no threshold retuning or unbounded refinement was attempted.

Direct-call Q measurements distinguish raw sums, RK-weighted contributions
(1/6,2/3,1) and remaining transport/SSP changes. Sharp RK-weighted scaling loss
9.595%, versus WENO6.30%, despite smaller total scaling loss. Finite-width KXRCF
increases Q in257/309 calls: WENO is not a norm-contraction guarantee. Held-field
reconstruction changes sharp WENO fields by5.06–5.51% after12 passes without
transport, versus exactly0 for scaling. These are not mechanical-energy balances.

The fixed analytic ell=.15625m pulse spans5 tangential cells on32x64. Unlimited
error.04276%, positive peak attenuation.003715%, first-front broadening.02889%
at dt4; scaling is idle and byte-identical to unlimited. Actual analytic peak
918.505516Pa and reference widths include transition overlap. All actual scaling
means stay inside the envelope (zero excess); deliberate mean1.1 is rejected.
GL9 envelope enforcement is distinguished from small between-point polynomial
overshoot; no production component/stress bounds or chi filtering were added.

149 transport/ledger,45 local/algebra and77 retained WENO gates pass;628 native
assertions/8 cases,94 PF1 checks and standalone production DG compilation pass.
Original operator/publication controls, prior base-step fields and the final
instrumentation-nonintrusiveness comparisons are byte-identical. Accepted
DG/phase/Z/I and incoming states are unchanged. All four final runs are within
180s (12.087/39.292/118.492/164.762s), at most two simultaneous, one rank/thread;
core SHA prefix`3ee53a8645b2`. Raw preliminaries/binaries remain preserved.

Remaining particle dependencies: fixed-particle startup checks, H/CPDI/domain
initialization, support/association, surface material projection and derived
sampling. Limiting/advancement read no particle properties. Native accepted
publication, surface-history/phase-bound transport, restart, MPI/AMR/periodic/
moving/coupled paths remain guarded or unimplemented. Defaults/physical laws,
manual instructions/derivation and all PF0–PF4 evidence are preserved. No commit,
push, BP3, PF5 or propagation work.

[Full report](PF4_STABILIZATION_REPORT.md), [source audit](SOURCE_AUDIT.md),
[portable evidence and replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf4_stabilization/README.md).

---

The prior limiter comparison and PF4 closeouts below are retained unchanged.

# Particle-free status — bounded PF4 limiter comparison complete

Updated16 October2026 UTC. Latest user selection is fully implemented and run:
**serial fixed-grid prescribed-transport tensor limiter comparison only**.
Evidence favors **neither limiter for adoption**; production/default remains
unlimited. Stop after this report; no PF5, coupled BP3 or propagation campaign.

- Same worktree/branch/HEAD as the preserved status below. Incoming manual
  instructions/derivation and PF0–PF4 source/evidence preserved; no commit/push.
- New private `dg_limiter.cc`: complete DGQ2 immediate-neighbor compact WENO,
  target-cell means, shared full-plane-strain tensor weights, stable dimensionless
  smoothness/zero treatment and immutable stage reads. Optional SSPRK3 hook only;
  no physical-law/default/accepted ownership change. No initial limiting.
- Safeguarded Fu–Shu experimental RMS/mean-plus-polynomial decision usesC=.1,
  fixed before validation from saved .05/.1/.2 screening. KXRCF-style threshold1
  uses pointwise inflow and the same WENO. Source quadrature audited; Gauss3 is
  full rank for Q2. This is a port of `limiters.cc`, not `elasticity.cc`.

| Measured gate /32×64 tradeoff | Unlimited | KXRCF+WENO | Fu–Shu RMS+safeguard+WENO |
|---|---:|---:|---:|
| Signed-jump over/undershoot |16.50/15.82%|.016/.024%|.049/.022%|
| Relative full-tensor L2 error |.17755|.37629|.37658|
| Quadratic-norm loss |3.574%|28.948%|28.903%|
| First10–90% front width, m |.02353|.09967|.09686|
| Finest smooth L2 error |.00026275|.00026515|.00029650|
| Finest smooth convergence rate |2.997|3.342|4.658|
| Jump advance time, s |.405|2.513|5.089|

77 local/snapshot and66 transport checks pass. Zero/constants, forced piecewise
constants, global affine/full Q2 includingx²y², means, finite weights, traversal,
in-plane tensor covariance, amplitude rescaling, zero crossings, non-proportional
components, direction changes and mixed boundary inflow pass. The locally
centered mixed-xy mean-only blind spot is independently demonstrated; Ipoly=.288
flags it, while actual forced reconstruction changes it only.0375%. Constant
background sensitivity is measured and remains a limitation. Maximum stage-local
mean error1.07e-15 scaled; incoming and accepted bulk/phase/Z/I remain unchanged.

Fresh operator/source/publication fields/metrics are byte-identical to PF4;
628 native assertions/8 cases and94 PF1 checks pass. New limiter/transport
translation units compile independently without unity/PCH. Validation4/8/16/32
runs take4.024/4.522/10.983/42.143s, all within180s. Same core SHA prefix
`a54c7f7b2e47`; plugin/input/source hashes and pinned raw binaries remain under
`build-particle-free/runs/pf4-limiter-*`. Supplementary local covariance checks
use a separately pinned observer; governing kernel is identical.

No particle reads enter limiting/advancement. Startup fixed-particle checks,
H/CPDI/domain initialization, support/association, surface material projection
and derived sampling still need particles. Native surface-history/phase-bound
transport, accepted publication/restart, MPI/AMR/periodic/moving-geometry/coupled
paths remain guarded or unimplemented. No chi limiting, stress clipping, adaptive
quadrature, new spin or constitutive law.

Details, source corrections, exact formulas and recommendation:
[PF4_LIMITER_REPORT.md](PF4_LIMITER_REPORT.md), [SOURCE_AUDIT.md](SOURCE_AUDIT.md).
[Portable evidence/replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf4_limiter/README.md).
A further selected task could examine reconstruction-induced mode damping and
spatial accuracy; this comparison does not authorize it or coupled deployment.

---

The earlier PF4 closeout below is retained as historical evidence.

# Particle-free work status

Updated 15 October 2026. **PF4 prescribed transport scope complete; no limiter
enabled. Stop before further integration or PF5.** Latest authorization:
“Proceed with PF4 as proposed.” A noncommitting DGQ2 material transport operator
uses native Q2 velocity, upwind faces, explicit inflow/source and the material
rather than density divergence correction. All284 final transport gates pass.
Defaults, native Maxwell/cohesive/friction/H laws and physical outer-step guards
remain unchanged. No coupled trajectory or nonsmooth-history qualification.

## Baseline and preservation

- Worktree `/home/ein/repository/aspect-pf-rsf-particle-free`, branch
  `pf-rsf-particle-free`, HEAD `3232056cde12dbae17ef4265ecb804f9098b88f1` plus
  uncommitted PF2–PF4 source/tests/docs. Nothing staged, committed or pushed.
- Root `AGENTS.md` applies; no override. Incoming instructions, derivation,
  identical installed plans, PF0–PF3 evidence and other checkouts/builds preserved.
  Derivation: [../particle_free_history.tex](../particle_free_history.tex), SHA256
  `bb05702d5c712bc51a87f0b2a75d881d56601d86106bd600d305815b00927539`.
- Native/plugin builds: `build-particle-free/native` and `/plugin`, GCC12.4,
  deal.II9.6.2, Trilinos14.2, OpenMPI5.0.10, Release, serial/one thread. Each
  run bounded180s; at most two native simulations overlap.
- Final core SHA256 `964340e5af68fcff87ef065a151f5247fd343e4b2c1e88acb3376aae3ad0e30d`.
  Pinned binaries, raw/preliminary states/logs and incoming manifest remain under
  `build-particle-free/`. No task process remains active after verification.

## Measured results

| Gate | Result | Record |
|---|---|---|
| PF0–PF3 retained | Audited laws;94 independent checks; native history and phase operator/bounds verified; historical failures retained | [SOURCE_AUDIT.md](SOURCE_AUDIT.md), [PF3_REPORT.md](PF3_REPORT.md) |
| PF4 independent matrix | Matrix/divergence/inflow errors≤1.554e-15 scaled; polynomial derivative4.330e-15 | [PF4_REPORT.md](PF4_REPORT.md), [assessment](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf4/assessment.json) |
| Material/density distinction | Constant tensor error≤8.28e-16; removing divergence correction gives3.921% finite error | Same report/assessment |
| Smooth spatial refinement |4x8→32x64; finest rates2.972–3.017. Translation/divergence pulse errors .02628%/.02212% on32x64 | Same report/assessment |
| Fixed-space time refinement | SSPRK3 orders3.0013–3.0030;512/1024 uncertainty1.37% of128-step error. Finest pulse dt/2 effect .00322% of spatial error | Same report/assessment |
| Commutators | Cohesive/mature/residual converge; omitted normal/tangential source errors40.78%/5.45%. Rotating-profile objective/non-objective distinction verified separately | Same report/assessment |
| Source quadrature |8x16 Gauss4→8 difference .004770Pa m;8→12 difference6.397e-12Pa m | Same report/assessment |
| Signed nonsmooth data | Integrated errors decrease, but32x64 retains16.50%/15.82% sampled over/undershoot and3.574% quadratic-norm loss | Same report/assessment |
| Trial state | Repeated trials identical; invalid source/unsafe dt rejected; accepted DG/phase/Z/I unchanged | Same report/assessment |
| Compatibility | PF3 assessment byte-identical,67 gates; frozen FE baseline fields byte-identical; same-core FE/DG max field/load differences3.195e-14/7.319e-13 | [evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf4/README.md) |
| Builds/tests | Full native/plugin, both DG translation units without unity/PCH;628 native assertions/8 cases and94 PF1 checks pass | Same evidence |

Final transport runs: `pf4-final-4/8/16/32`, wall times4.122/5.273/11.687/40.393s,
peak child RSS214360/241216/308028/651508KiB. Native phase and frozen controls
use the same final core. Compact provenance and replayable fixed-entry archive
are under `benchmarks/reconstructed_fault/particle_free/evidence/pf4/`.

## Boundaries and remaining dependencies

- `dg_transport.cc` owns the new private operator. Existing DG definitions move
  to shared `dg_history_internal.h`; accepted ownership/layout and updates stay
  in the material. No new public selector, manager or physical parameter.
- The observer projects analytic scratch tensors in the actual native space.
  Transport time T=.2s is controlled; the outer simulator clock, particles,
  native surface histories and accepted field are not advanced. Published-state
  transport, full rollback/restart and coupled physical stepping remain guarded.
- Surface histories/localization are prescribed characteristic controls.
  Co-moving/rotating fixed-amplitude templates do not qualify native moving
  normalization or geometry. No production spin or mature residual state added.
- Direct/residual DG closure is convergent, not roundoff exact: continuum source
  projection and DG trace differentiation do not commute algebraically.
- No limiter is needed for the selected resolved smooth checks. The jump case
  establishes a significant oscillation issue for future nonsmooth use, whose
  acceptable mechanical effect and limiter benefit remain unqualified. No fixed
  stress-component bounds, chi filtering or historical WENO port.
- Advancement consumes no particle properties/domains. Construction still
  checks fixed particles; startup H/CPDI/domains, surface material projection,
  support/association and derived sampling require managers. No end-to-end
  particle-free claim.
- Native Z/theta and material phase-bound transport, geometry/birth/AMR, MPI,
  persistence, coupled time/energy/work and propagation remain subsequent stages.
  No PF5 or broader campaign has started.

Proposed next selection: a **bounded PF4 tensor-limiter comparison** using the
saved jump and smooth controls, after reviewing a mean-preserving common-tensor
reconstruction. Verify constants/zero, affine/smooth accuracy, means, tensor-basis
covariance and degenerate weights; measure actual overshoot/norm-loss benefit.
Keep coupled adoption and PF5 persistence/transfer separate. Full equations,
commands, costs and limitations are in [PF4_REPORT.md](PF4_REPORT.md).
