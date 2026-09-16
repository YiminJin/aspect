# K5: completed normalization and bottom-wedge source continuation

## Decision

The finite-tip source exclusion is a demonstrated dominant cause of the
remaining bottom artifact in this test. Restoring the source consistently in
assembly and Maxwell-history updates reduces bottom all-QP strain mismatch
by **15.24 times**, to the same-resolution interior level. Raw and weak
normal-traction errors also decrease substantially. The paired completed-Ih
and continued-source treatment is now an explicit supported option for this
**frozen mature, straight BP3 through-bottom configuration**.

This does not establish an exactly stress-free discrete solution or resolve
every BP3 boundary issue. A roughly 1.33% strain mismatch remains, comparable
to 1.37% in the interior. The top was unchanged. The earlier small
remote/full-column integration discrepancy remains recorded. No long RSF
trajectory, refinement study, or new restart campaign was run.

## What changed, and what did not

Write `b` for the bottom intersection, `s` for the upward fault tangent,
`n` for its normal, and `lambda=(x-b).s`. The additional source is defined only
for physical points inside the Box with `lambda<0` that had no ordinary fault
association. It uses

\[
 \dot\epsilon^{\rm crack}(x)=
 \frac{h(\phi_h(x),f_{\Gamma,0})}{\widehat I_{h,0}}V_p S,
 \qquad S=\operatorname{sym}(s\otimes n).
\]

The numerator is the **actual physical FE phase**, not a plotted or analytic
replacement. Endpoint surface fields, including the completed I_h, are
continued constantly. The source is zero when physical phase is zero. The
ordinary normal-width cutoff is not reapplied in this wedge: Q1 phase can
remain positive beyond it. Outside the wedge the original admission is
unchanged; all previously admitted sources retain their existing evaluation.

The manager's bulk-source map is distinct from particle/surface admission.
The Stokes QP cache uses it for the absolute residual and B action, including
explicit B. The particle Maxwell commit uses the same map for its local
crack-strain subtraction. It does not add those particles to surface
quadrature or change C/Theta projection, domains, ownership, G's surface
quadrature, or open endpoint connectivity. Constitutive diagnostics use the
same source and old working FE stress as assembly. Candidate particle stress
is still validated before the unchanged terminal publication phase.

I_h completion remains the earlier preprojection addition of outside profile
integrals to the physical-profile RHS, with the original surface mass matrix.
No mechanical cell, quadrature point, or force is added outside the Box.
The mesh, profile, material parameters, boundary data, timestep clock,
solver criteria and initialization semantics are unchanged.

## Paired replay and safeguards

The existing completed-denominator run exported only associated bulk QPs;
its vertex-only graphical output could not recover exact Q2 gradients in
the missing wedge. An observer-only fresh control was therefore necessary.
It retains all positive-phase QPs in the requested windows and reproduces
the saved initialization and step 1 exactly in the checked bulk exports.
Step-2 differences are roundoff; the complete-field repeatability check is
`max|difference| <= 1e-12 max|field|`, four orders tighter than the unchanged
nonlinear relative target. A pointwise-relative comparison near zero was
inappropriate: the flagged small stress components differed by only about
3.65e-12 Pa. This reporting check does not alter any solver criterion.

An initial continuation retained the old normal-width cutoff. It restored
165 positive-phase QPs but missed 22 more Q1-positive wedge points. That
intermediate run and its comparison are preserved separately. The final run
restores all **187** missing positive-phase wedge QPs, using the paired BP3
parameter rather than either legacy diagnostic selector.

All runs use four ranks, 42,880 cells and 1236 fault vertices, fresh zero
perturbation-stress history, and initialization plus exactly two real steps:

| State | Physical time (s) | Maxwell/real interval (s) |
|---|---:|---:|
| 0 | 0 | artificial initialization: 4,000,000 |
| 1 | 2,487,214.2056652424 | 2,487,214.2056652424 |
| 2 | 4,970,143.2669182401 | 2,482,929.0612529977 |

Final candidate relative bulk residuals: `5.34944e-10`, `7.46280e-10`,
`5.46926e-10`. All six fresh linear checks pass; worst fresh/target ratio is
`0.830874`. Each solve accepts alpha=1. Surface convergence residual is zero
because all V rows are prescribed, **not** because unconstrained friction
equilibrium was demonstrated. V remains exactly Vp, Theta obeys the split
aging update, initial stress is retained at zero, and all 385,920 stable-ID
particle H histories remain unchanged. Coordinates and I_h are unchanged.

Wall times: control **131.02 s**, width-limited intermediate **134.35 s**,
final complete-wedge **124.51 s**. Each had a 600-s cap and no retry after a
simulation failure. The intermediate was superseded to complete the requested
source coverage, not to retune physics or convergence.

## Source closure and history consistency

The final comparison uses identical physical QPs and JxW, including formerly
unassociated points: 351 in bottom 0–200 m, 1521 in 200–1000 m, 1824 in
1000–2000 m, and 3189 in the 59–61 km down-dip interior control.

New-source amplitude differs from the continued reference by at most
`4.04e-28 /s` at every accepted state. Original admitted chi values are
bitwise identical at initialization and differ by at most `1.30e-18` later;
evaluating the identical degradation through advected/projected mixtures
changes last bits, not the support or constitutive rule.

| Region | Final source RMS error / reference | Missing integrated amplitude fraction |
|---|---:|---:|
| Bottom 0–200 m | 4.23532e-5 | 1.04796e-5 |
| Bottom 200–1000 m | 6.70910e-5 | 2.00201e-5 |
| Bottom 1000–2000 m | 6.67676e-5 | 2.11783e-5 |
| Interior 59–61 km | 1.59160e-4 | 4.49101e-5 |

The first-layer source error was 0.340702 before continuation, with missing
integrated fraction 0.175494. Remaining differences are normal-width tail
exclusion **outside** the restored wedge. These regional source fractions
are not the fraction of a clipped physical column or a global slip-normalization
test; a physically clipped column is still not renormalized to unity.

The commit audit checks 466 real parent particles in the continued coordinate
region at each step, including 187 with positive phase. Independently evaluate
`tau_new = 2*kappa*(eps-crack) + beta*tau_old` from the exported precommit
inputs and compare to both the candidate and stable-ID committed particle
properties. Maximum formula discrepancy is zero at step 1 and `2.27e-13 Pa`
at step 2; candidate/publication agreement is bitwise. Initial stress retention
is verified separately. This rules out a plotted-source-only or bulk-only
change: the continued source is consumed by the next history transfer too.

## Mechanical improvement on the common all-QP set

At the final time, strain mismatch is `RMS|sym grad u-actual crack source|`.
Relative values below use the **same continued-reference source scale** in
both runs, avoiding a changed denominator when missing points are restored.

| Region | Mismatch RMS before -> after (/s) | Relative mismatch before -> after | Velocity-reference RMS / Vp before -> after |
|---|---:|---:|---:|
| Bottom 0–200 m | 1.41107e-13 -> 9.26083e-15 | 20.3275% -> 1.33409% | 2.01213% -> 0.313428% |
| Bottom 200–1000 m | 2.09152e-14 -> 9.66767e-15 | 2.96018% -> 1.36829% | 0.915931% -> 0.317967% |
| Bottom 1000–2000 m | 1.08521e-14 -> 9.65765e-15 | 1.53602% -> 1.36695% | 0.538201% -> 0.319085% |
| Interior 59–61 km | 9.52270e-15 -> 9.52219e-15 | 1.36681% -> 1.36673% | 0.317548% -> 0.317262% |

For bottom 0–200 m the corresponding initialization RMS is
`1.39419e-13 -> about 9.08e-15 /s`; the improvement persists rather than being
lost when stress history updates. The velocity reference is the independent
full stationary-profile sliding primitive used in the previous report.

Raw final bulk-QP extrema, over that same bottom-200-m set:

| Quantity | Completed denominator only | Plus complete wedge |
|---|---:|---:|
| Delta p (kPa) | -56.7971 to 125.0834 | -0.45548 to 0.88514 |
| tau_xx (kPa) | -114.5786 to 36.5492 | -2.4763 to 2.9582 |
| tau_yy (kPa) | -36.3074 to 130.7502 | -4.4273 to 5.1002 |
| tau_xy (kPa) | -71.7637 to 25.4420 | -2.4943 to 3.5737 |
| tau:N (kPa) | -75.9666 to 54.3189 | -1.4539 to 2.3993 |
| Total sigma_n (MPa) | 49.921983 to 50.196568 | 49.997787 to 50.001652 |

Extrema of separate components need not occur at the same QP. The actual
normal traction is evaluated jointly as `50 MPa + delta p - tau:N`, not
inferred from pressure or one deviatoric component alone.

For the existing bottom surface window (last 2 km of fault coordinate), raw
**constitutive surface** extrema change from `49.922826–50.085605 MPa` to
`49.997133–50.002586 MPa`. Consistent weak row means change from
`49.977194–49.999836 MPa` to `50.0000047–50.0000577 MPa`; projected Q1 extrema
are `50.0000036–50.0000599 MPa`. Raw surface, weak surface and raw bulk-QP
statistics are deliberately distinct. The weak bottom bias is reduced from
about 22.8 kPa to under 58 Pa, without smoothing.

## Supported option, checks and remaining limits

`Postprocess/BP3/Bottom normalization completion file` selects **both** the
fixed preprojection completion and the in-box source continuation. The generic
manager offers the bulk-only coordinate map; the material selector invalidates
its completed-value cache when initially attached and rejects a changed path
during a run. The BP3 plugin reattaches it before preparation, also on restart.
The completion file is an immutable external input for the exact mesh, phase,
materials and fault, not a universal analytic constant or a checkpointed cache.
Bottom V must be prescribed to Vp; curved and non-mature uses are rejected.

The actual final two-step run used this parameter, with neither legacy
completion/source environment selector. Restart reconstruction is implemented
but was **not** independently rerun in this task. The prepared
`bottom-source-supported-50-local4/run.prm` also passed `--validate`; it was
not launched. Do not infer a restart, fine-mesh, top-boundary, or free-RSF
qualification from these checks.

Release ASPECT/plugin builds passed with `-j4`. Final focused tests:
35 assertions/5 cases on one rank; 29 assertions/4 cases on **each** of two
ranks. Python source/geometry/history checks and compilation passed. No full
ASPECT suite was run. Reporting scripts first used overly strict bitwise/
near-zero relative repeatability checks; their failures are preserved, along
with the measured roundoff and corrected field-scaled checks. Production
convergence/accuracy criteria never changed.

The remaining local strain/velocity errors are comparable to the interior
discretization baseline, not evidence of exact continuum compatibility.
The earlier I_h remote/full-column discrepancy (approximately 8.4e-6 relative)
is still separate. The unchanged top concentration and the earlier 40-km RSF
junction questions are outside this result. Do not resume a long trajectory
or claim the entire bottom/finite-domain problem solved on this evidence.

## Artifacts and changed files

- Final run: `benchmarks/reconstructed_fault/bp3/bottom-source-complete-wedge-50-local4/`.
- Observer control: `bottom-source-control-50-local4/`.
- Preserved width-limited intermediate: `bottom-source-continued-50-local4/`
  and `bottom-source-width-limited-comparison/`.
- Final matched evidence: `bottom-source-comparison/all_qp_comparison.csv`,
  `all_qp_0.csv` through `all_qp_2.csv`, `checks.json`, and `comparison.png`.
- Each run contains exact commands, environment and source hashes in
  `provenance.json`, its execution record, ordinary outputs, raw/weak surface
  data, all-QP exports, and precommit source-history CSVs. Detailed lifecycle
  checks are in each `analysis/summary.json`.

Implementation changed `include/aspect/reconstructed_fault/manager.h`,
`source/reconstructed_fault/manager.cc`,
`include/aspect/material_model/phase_field_fault.h`, and
`source/material_model/phase_field_fault.cc`. Benchmark configuration/observers
are in `bp3.cc` and `uniform_sliding.h`; new drivers are `run_bottom_source.py`
and `analyze_bottom_source.py`. `unit_tests/reconstructed_fault.cc` contains
the focused geometry/ownership-boundary regression. The authoritative design
and specification record this limited exception. Usage is in
`benchmarks/reconstructed_fault/bp3/bottom_continuation.md`.
Unrelated working-tree changes were retained; no commit was made.
