# K5 bottom source-support audit

Follow-up: the approved source continuation was subsequently tested in
[the paired mechanics report](stage_K5_bottom_source_continuation_report.md).
It substantially reduces this artifact and is now an explicit limited BP3
treatment; remaining discretization and normalization errors remain separate.

## Decision

The bottom hotspot is **not solved**. Completing the denominator leaves a
substantial missing crack-strain source inside the physical box, beyond the
finite fault's bottom tangent plane. In the first 200 m above the bottom, the
actual source has a 34.07% weighted tensor-RMS difference from a continued
straight-fault reference and omits 17.55% of its integrated amplitude.
Associated QPs agree with that reference near roundoff. This is a finite-tip
support limitation, not an incorrectly clamped bulk endpoint coordinate.

No mechanics, state update, or simulation was run for this audit. The current
source follows the documented finite-segment admission rule. A continuation
of the in-box source would therefore be a separately reviewed support-policy
experiment, not a correction to a source/specification discrepancy. Its effect
on the remaining stress hotspot has not yet been measured.

## Frozen data and production-path verification

Inputs are the accepted `bottom-completion-50-local4` and
`uniform-sliding-50-local4` outputs. These use the same fixed straight geometry,
fixed phase, mature model, and prescribed uniform `Vp=1e-9 m/s`. In this case
the phase-history localization term vanishes; the source under investigation
is exactly `chi*Vp*S`. Evolving Maxwell stress is not part of this source.

The audit reconstructs the actual 3-by-3 bulk Gauss quadrature from all 42,880
saved physical cells, then selects all positive-profile QPs in the bottom
0–2 km and a 59–61 km down-dip interior control. Admission is **not** a selection
condition. The resulting union contains 6,885 QPs, including points where only
the uniform-grid comparison profile is positive.

The hidden test `Saved bulk quadrature support audit` builds the saved fault
in `ReconstructedFaultManager` and calls its production normal-profile
projection on every point. This calls the same internal projection routine
as the Stokes QP cache. Active/inactive membership agrees exactly with the
saved assembly exports in both cases; active segment and coordinate agree.
The assembler skips the crack-strain contribution at inactive QPs. Thus their
actual source is zero, not an unavailable diagnostic value.

Validation maxima:

| Quantity | Difference |
|---|---:|
| Reconstructed versus saved QP coordinates | 5.33e-15 m |
| Relative quadrature weight | 4.44e-16 |
| Reconstructed Q1 phase versus saved phase | 2.74e-8 |
| Associated scalar `chi*Vp` versus reference | 2.02e-27 /s |
| Associated saved crack tensor versus reference | 2.59e-25 /s |
| Unexplained inactive QPs | 0 |
| QPs mapped exactly to a segment endpoint | 0 |

The newly visible wedge includes both 97.65625-m and 195.3125-m cells. Its phase
is reconstructed from the actual saved Q1 corner values, including constrained
values, rather than assuming the near-fault fine lattice continues everywhere.
The ordinary VTU corner data have Float32 precision. Full-precision phase from
the saved bulk-QP export replaces these values at associated points. This is
an offline field-precision limitation, not a production evaluation change.

## Reference and geometry

Let `b=(21132.486540518701,0)` m be the bottom fault intersection,
`s=(1/2,sqrt(3)/2)`, `n=(-sqrt(3)/2,1/2)`, and
`S=sym(s tensor n)`. Define `lambda=(x-b).s` and `r=(x-b).n`.
The geometry-only reference is

\[
 E_{\rm cont}(x)=\frac{h(\phi_h(x))}{\widehat I_h(
 \operatorname{clamp}(\lambda,0,L))}V_p S.
\]

This uses the **actual physical Q1 phase** and the completed production Q1
denominator, with a constant endpoint extension of that denominator only for
this offline comparison. It does not install a new mapping, nodal field, or
source in production. On associated QPs it is the existing source.

The production finite-segment rule rejects `lambda<0`, even when the point is
inside the box and `h(phi_h)>0`. The missing wedge satisfies

\[
 y\ge0,\qquad x-b_x<-\sqrt{3}y,
 \qquad r>2y.
\]

It is distinct from the **outside-box** normal interval completed in the
previous denominator experiment. An integral added to the denominator cannot
create a source in this physical wedge. The largest sampled height of the
missing tip source is 401.631 m; the furthest projected foot is 477.581 m beyond
the tip. The Q1 profile slightly extends beyond its nodal compact support.

The current design explicitly excludes tangent extensions beyond true open
tips. Surface-domain integration of already admitted particles has a different
corner/tip coordinate rule; that must not be confused with bulk-QP admission.
No cyclic identification, particle-domain modification, or endpoint clamp was
introduced here.

## All-QP results

The tensor RMS uses actual bulk `JxW` and the Frobenius norm of `S`. The relative
error is `||E_actual-E_cont||_JxW / ||E_cont||_JxW`. The integrated missing
fraction is the signed scalar-amplitude difference divided by the reference
amplitude integral over the same selected QPs. It is not a global constrained
Stokes weak-load norm or the whole-fault slip-normalization diagnostic.

| Region | QPs / active | Tip-excluded / width-excluded | Relative source RMS error | Missing integrated amplitude |
|---|---:|---:|---:|---:|
| Bottom 0–200 m | 351 / 209 | 129 / 13 | 34.0702% | 17.5494% |
| Bottom 200–1000 m | 1521 / 1360 | 58 / 103 | 0.671739% | 0.118381% |
| Bottom 1000–2000 m | 1824 / 1683 | 0 / 141 | 0.00667676% | 0.00211783% |
| Interior, down dip 59–61 km | 3189 / 2946 | 0 / 243 | 0.0159160% | 0.00449101% |

In the first layer, the reference tensor RMS is `6.94167e-13 /s`, and the
source-error RMS is `2.36504e-13 /s`. Normal-width exclusion contributes only
`0.00104796%` of the integrated reference there; the tip wedge dominates.
Its selected physical quadrature measure is `165480.155 m^2` out of
`400543.213 m^2`. These are QP-weighted region measures, not exact polygon areas.

The strongest missing sample is at `(21104.756022,11.006022)` m, with
`lambda=-4.33376460 m`, `r=29.5183445 m`, and `phi=0.590675555`.
The production source is zero. The reference amplitude is `2.65436076e-12 /s`,
and its `(xx,yy,xy)` tensor components are
`(-1.14937192e-12,+1.14937192e-12,-6.63590190e-13) /s`.
The detailed CSV retains the owning rank, actual CellId, and QP number.

Two separately labelled references check that this is not merely a remaining
denominator or local-mesh artifact: using the fixed complete same-resolution
column integral `I*=12691.748894691811 m` gives 34.0674% in the first layer;
also replacing the phase by the continued uniform-97.65625-m Q1 profile gives
34.0674%. Neither eliminates the missing-source wedge.

Against the same continued completed reference, the original truncated-Ih
case has 43.3138% source RMS error in the first layer, versus 34.0702% after
completion. This is consistent with partial improvement, not a cure. The
previous associated-only **strain mismatch** of 14.11% -> 10.03% uses different
fields and an admission-restricted sample set; these percentages must not be
added or interpreted as a quantitative partition of the stress error.

## Remaining decision

The evidence identifies a substantial source-support/compatibility error
relative to uniform continued sliding. It does not establish how much of the
remaining mechanical stress concentration it causes. No new all-QP strain
evaluation or counterfactual equilibrium solve was performed.

The smallest useful next experiment is a separately approved, benchmark-local
continuation of the straight-fault crack-strain source **inside the physical
box** beyond this through-boundary tip. It should keep the completed
denominator, profile, `Vp`, mesh, physical boundary conditions and open surface
topology fixed, and compare the same source and stress diagnostics. Mapping
such bulk source points must not implicitly add surface connectivity or change
particle/history ownership. Generalizing this to ordinary internal open tips
or curved faults would be a separate policy decision. Do not call the bottom
hotspot solved before this distinction is tested.

The previous remote-versus-completed-column precision discrepancy remains
documented in the normalization report; it was neither reopened nor hidden by
this audit.

## Reproduction, artifacts and verification

New benchmark analysis: `benchmarks/reconstructed_fault/bp3/bottom_source_support.py`.
Test-only exporter: `unit_tests/reconstructed_fault.cc`. Production source,
specification, numerical parameters and saved trajectories were not changed
by this task.

```sh
MPLCONFIGDIR=/tmp/bottom-support-mpl python3 benchmarks/reconstructed_fault/bp3/bottom_source_support.py prepare
cmake --build build-pf-cpdi --target aspect.exe.release -j4
ASPECT_SAVED_FAULT_SUPPORT_AUDIT="$PWD/benchmarks/reconstructed_fault/bp3/bottom-source-support-audit" build-pf-cpdi/aspect-release --test '[.fault_saved_support]'
MPLCONFIGDIR=/tmp/bottom-support-mpl python3 benchmarks/reconstructed_fault/bp3/bottom_source_support.py analyze
python3 -m py_compile benchmarks/reconstructed_fault/bp3/bottom_source_support.py
```

Preparation refuses to overwrite completed inputs. Release build passed;
the hidden test passed **1242 assertions in one test case**; the Python
association/source/geometry checks and Python compilation passed. No new
one-/two-rank mechanics or lifecycle campaign was run: saved four-rank
cell/QP data were checked against the replicated geometry's production
projection, offline in one process.

Evidence under `benchmarks/reconstructed_fault/bp3/bottom-source-support-audit/`:
`all_qp.csv`, `production_projection.csv`, `source_comparison.csv`,
`region_summary.csv`, `summary.json`, and `source_coverage.png`.
The initial incomplete uniform-grid preflight is preserved separately under
`bottom-source-support-audit-preflight-incomplete/`; its assumption was rejected
before producing the final physical-FE comparison. Build/test/analysis logs
are preserved in the evidence directory as `bottom-support-build-final.log`,
`bottom-support-projection-test.log`, and `bottom-support-analysis.log`.
