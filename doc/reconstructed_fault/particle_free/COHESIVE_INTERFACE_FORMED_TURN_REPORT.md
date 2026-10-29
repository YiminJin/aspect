# Full loading turn with future-material refinement

Baseline `f1f9f29cf40c32d1ddcea000b724375b64853532` plus preserved local work.
[Declared controls](COHESIVE_INTERFACE_FORMED_TURN_SELECTION.md) ·
[PNG/GIF and indexed CSV/JSON](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_formed_turn/INDEX.md).

**Completed the bounded turn and future-cell refinement.** New material responds
to the changed accepted stress without rotating old geometry or tensor history.
Response differences satisfy the declared spacing/late-time criteria. This is
a serial experimental native/JSON result, not production propagation adoption.
The normal-reaction accuracy target remains borderline and raw point tractions
remain unqualified.

## Exact experiment and result

Both trajectories resume the **same verified .425 s half-step checkpoint**.
Its ten formed pieces, 132 Gauss identities and all physical histories remain
unchanged as incoming material. The earlier .410–.425 s history is reused;
neither the old seed nor already formed cells are refined or remapped. Continue
the existing physical boundary-strain turn to .440 s (α=−20°) using
Δt=.000125 s. This completes the original turn, **not a displacement hold**.

Keep Cartesian 64² bulk Q2/P1disc, Gauss6 mechanics, surface Gauss3, Gauss8
selection, physical R=.75 m, numerical support half-width .75 m, current-pressure
contact and all cohesive–RSF/Maxwell/radiation parameters. Only **future** Q1
commit spacing changes: .09375→.046875 m. Initially 2/4 temporary cells give the
same .1875 m extent. New spacing also refines future jump modes, Gauss ownership
and cellwise normal-work tests; it is not merely a metadata comparison.

| At common .440 s | .09375 m | .046875 m |
|---|---:|---:|
| Positive accepted steps / publications | 120 / 120 | 120 / 120 |
| New formed cells; physical length | 6; .5625 m | 12; .5625 m |
| Incremental ∫a ds beyond original seed | .589141 m | .588796 m |
| ∫a ds on material added **after .425** | .0559505 m | .0554292 m |
| Signed-slip integral increment | .00147948 m² | .00147929 m² |
| Connected a=.05 front beyond original seed | 1.47131 m | 1.46749 m |
| Connected a=1 front beyond original seed | .49744 m | .49744 m |
| Accepted-stress preferred direction | .48062° | .47933° |
| Absolute work remainder / sliding | .09616% | .09620% |
| Absolute work remainder / added fracture | .40852% | .40895% |
| Wall time, including on-demand kernels | 383 s | 766 s |

Total incremental damage changes **−.0586%**; damage and slip confined to newly
admitted material change **−.9317%**, so agreement is not inferred solely from
the larger old-material response. Over **all** common accepted times, maximum
increment-relative differences are .8533% for damage and .2835% for signed slip;
preferred direction differs ≤.0358°. Maximum process-front difference is .03631 m,
within the predeclared .09375 m target. Final estimate separation is 3.82 mm;
the actual Gauss brackets overlap. Matching brackets are not millimetric accuracy.
Two spacings do not establish a convergence order.

The a=1 onset is bracketed [.42875,.428875] s in both cases. At .440 s its
distance bracket is [.47930,.51559] m beyond the original seed. It advances on
the **older** formed geometry; it has not reached material admitted after .425 s.
That new material has peak native-Gauss damage .2500/.2529 and forms a real
cohesive process zone, not a fully decohered extension. No detached surface
damage is observed; bulk secondary nucleation is outside this surface model.

The preferred direction changes ~10.24°→.48°, while the final new-material
tangent is 7.368°/7.336°. Persistence over R and irreversible old geometry
explain that difference. The new curves differ in y by ≤.128 mm at common x;
the actual process-front tangent differs .0219°. This supports bounded
stress-responsive formation, not a new direction-selection law or a claim that
the fully decohered crack has already followed the latest preference.

## Temporal and physical-work checks

Replay .4375–.440 s from the coarse run's **own identical** incoming state at
Δt=.0000625 s: 40 accepted steps, 74 s. Suffix growth changes −.2128%
(−.0001924 m), slip increment −.0342%, and both front estimates retain their
brackets. The late temporal change is small but comparable to the tiny spacing
effect in absolute terms; do not interpret their difference as pure spatial
error or extrapolate this suffix to a full-history temporal order.

The 1% physical screen passes relative to both sliding and added fracture.
For coarse spacing, trapezoidal reaction–displacement work is **29.582259 J/m**;
bulk storage increases 26.295113, cohesive storage changes −.011321, fracture
costs .600630, friction dissipates 1.715831, radiation .835735 and Maxwell
.148725 J/m. The sum of absolute physical remainders is .00245370 J/m. Every
step remainder has the same sign: a small final sum is not cancellation between
steps. Endpoint work, **30.307427 J/m**, remains separately recorded and includes
the retained BE and surface endpoint remainders; these are not physical loss.
Maximum exact BE ledger defect is 6.85e−13 J/m. Half-step suffix remainders are
.02857% of sliding / .13452% of fracture. Historical coarse-energy failures stay
unchanged. Geometry metadata adds zero fracture, storage or work.

## Independent bulk/reference and lifecycle checks

Freeze the actual coarse .440 s geometry, physical jump and Maxwell field.
Reconstruct independently on Cartesian64, fitted64 and fitted96 (4096/5824/10560
cells; 44545/63445/115337 bulk masters). No reconstructed reference is published
as a history transfer. Same surface nodes, Gauss records, current normal work,
confinement and mean-zero p′ convention apply.

Against fitted96, the selection-disk full plane-strain deviatoric tensor discrepancy is
**.000710%** and angle difference **.000203°**. Fitted64/96 angle uncertainty is
.0000194°. Common regional weak shear differs ≤1.03% (≤25.45 Pa); fitted pair
shear uncertainty is ≤.192 Pa. Jump-induced storage differs .263%; the fitted
pair differs .258%, unlike normalization by the much larger background storage.

**Normal accuracy is not robustly below 2%.** At the bend the perturbation error
is 16.45 Pa / 1.937% against fitted96, but 2.006% against fitted64. The retained
2% target therefore remains unresolved at reference uncertainty; no tolerance
is changed. Raw normal-jump RMS is ~494 Pa versus 15.2 Pa fitted96 there. Earlier
point-traction, ahead-region 2.85%/2.29 Pa and old-region failures are preserved;
the new tests use explicitly exported regions and do not replace them.
Mixed residuals are <2.7e−15; strong divergence remains nonzero (whole-domain
RMS 1.31e−5 Cartesian versus 4.88e−6 fitted96), so weak incompressibility is not
claimed as pointwise accuracy. All sided traces remain exported.

Both loaded late-state tests exercise actual contact-loss rejection, immutable
input, deterministic retry, one publication, duplicate rejection and rebuilt
native-context JSON restart. Complete pressure-coupled tangent FD errors are
4.47e−10 / 5.39e−10 against unchanged 1e−6. All geometry replacements have zero
measured physical stress/storage defect. Native weak errors remain ≤1e−8 Pa,
contact stays positive and histories remain irreversible. These are experimental
owner checks, not Simulator/MPI restart or a global RSF stability theorem.

## Scope and next action

The selected short full-turn/refinement task is complete. Existing native code,
production/BP3, defaults and historical evidence are unchanged; four focused
experiment/check/analysis scripts were added. Trajectories ran sequentially;
frozen setups cost 20/26/53 s. Raw data remain within the 1.1 GiB budget; only
new unused cell-mean diagnostic bases were omitted, with hashes and regeneration
inputs retained. Checkpoints before/after each admission and at .440 s are saved.

This supports the adopted causal formation hypothesis within measured resolution.
It does not qualify quantitative adaptive propagation: the developed front is
still behind the newly selected zone, contact accuracy is borderline, and point
traction remains unresolved. Before a longer directional claim, resolve the
specific bend normal-work uncertainty with one frozen common-field control;
then a bounded catch-up/hold test could ask whether the developed front enters
the newly selected curve. Neither follow-up is started. No calibration of R,
new physical law, free branching, AMR, transport or MPI expansion.
