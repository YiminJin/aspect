# Server 3610007 accepted-work failure

**Resolved within the demonstrated observer scope:** the signed packet confirms
the frame explanation and the observer is repaired with its original guard.
[Final repair, bounded verification and server handoff](BP3_DG_WORK_OBSERVER_FIX.md).
The earlier candidate/follow-up accounts below remain historical evidence;
full-grid repaired advancement and first-event accuracy are not claimed.

## Replay follow-up — 19 October 2026

Server replay **3613912** reproduces the original failure at step 816 with the
same logged mechanical solves/publication values and accepted 810–815 time/dt
sequence. The original normal observer error is **1.001357615619641e-5 Pa** at
node **4918**, versus the unchanged 1e-5 Pa bound. The directly integrated frame
contribution is **9.931623371162834e-6 Pa** at that node. The native-frame shadow
passes all seven steps, with maximum normal/shear errors at 816 of
**2.0867824647698363e-7 / 5.0678682972525474e-7 Pa**. This establishes that matching
the mechanical frame removes the observed guard violation for the replayed state;
the earlier candidate explanation is now supported by native measurements.

[Replay log reduction and requested compact artifacts](../../../benchmarks/reconstructed_fault/particle_free/evidence/server_3613912/README.md).
The nodal failure CSV is requested for signed error reconciliation and a repair
regression. Effective parameters, replay acceptance records and checkpoint labels
complete the provenance. Stdout records a completed 815/slot-03 checkpoint;
the metadata still needs verification. Preserve original/replay checkpoints on
the server; no large transfer is requested. No runtime code changed in this
follow-up, no repaired observer has yet been tested, and strict matching remains
false. The original diagnostic account below is retained as dated evidence.

---

18 October 2026. Source: reviewed `b2dfdebe43437d6028b06d4f9f862cc80a00cc71`
plus preserved optimization/Maxwell-repair work. Supplied files remain unchanged
in `benchmarks/reconstructed_fault/particle_free/evidence/server_3610007/`;
original scheduler logs remain in `bp3_dg_first_event/`.
[Reduction, hashes, diagnostic patch and bounded replay input](../../../benchmarks/reconstructed_fault/particle_free/evidence/server_3610007_analysis/README.md).
`strict_reference_match_passed=false` remains.

The termination is confirmed; the exact underlying discrepancy is not yet
confirmed. The supplied exports stop before the failing observer writes its
numeric error. A concrete observer-frame inconsistency is demonstrated below,
and a tested diagnostic replay preserves the original guard to distinguish
that inconsistency from publication/evaluation arithmetic.

## Confirmed termination and state

The 160-rank job exits with code 1 through MPI abort in
`server/plugin/work_audit.cc`, original line 295:
`consistency<1e-5`, "Accepted work observer does not reproduce frozen mechanical
traction." It fails during postprocessing of step **816**, at
`t=5172072493.9057245 s`, `dt=0.0074540672088498705 s`.
This is after mechanics and accepted-history publication, before the step's
accepted summary and coupled-check exports. It is not a graceful wall/event
stop. The latest complete checkpoint is **step 809**, slot **03**:
`809 5172072493.8510532`. CSVs through 815 cannot substitute for that checkpoint.

| Check | Measured result |
|---|---:|
| Step-816 final bulk / fault relative residual | 6.162430e-12 / 1.087297e-10 |
| Step-816 independent DG publication error | 9.8217983434e-6 Pa |
| Step-816 maximum error / operand-derived allowance | 0.00601013896 |
| Step-815 maximum slip rate | 0.2293039255 m/s, at 9000 m down dip |
| Step-815 minimum filtered normal stress | 49.460574 MPa |
| Step-815 maximum auxiliary Z | 1.6610474787e11 Pa m |
| Step-815 minimum state | 0.04821791224 s |
| Publication counts in supplied rank-0 checks, steps 0–815 | exactly one each |

All supplied trajectory values are finite. The largest normalized nonlinear
residual is 9.94065319e-9. Rank-0 sampled-history checks have maximum
error/allowance 0.00349464375; these samples are not the global publication check.
Independent multiplication of the supplied surface mass matrix reproduces all
six step-815 projected load exports within 1.79e-7 Pa m. The normal-load split
and residual assembly identities differ by at most 4.18e-6 and 3.32e-6 Pa m,
respectively. These integrated units differ from the observer's normalized Pa.
These checks establish neither full-event accuracy nor the absent step-816
observer error. There is no observed solver divergence or double publication.

## Demonstrated observer inconsistency

Mechanics in `surface_system_bulk_work.cc` uses each Stokes-QP association's
segment tangent and normal, defining `S_a=sense*sym(t_a tensor n_a)` and
`N_a=n_a tensor n_a`. The work observer instead uses the first segment's `S_0`
and `N_0` at every quadrature point. Both calculate physical stress first, but
project it onto different frames. The guard compares shear and **raw**, not
Helmholtz-filtered, normal loads:

```
D_i = sum_q JxW_q chi_q shape_i(q)
E = max_i { |Lshear_observer-Lshear_mechanics|/D_i,
            |Lnormal_observer-Lraw_normal_mechanics|/D_i }
require E < 1e-5 Pa
```

Independent scalar-double reconstruction of the supplied package's five anchors
and native per-anchor-interval 10 m resampling gives 5776 nodes, matching the
exports. Although the fault is geometrically straight, subtraction of nearby
coordinates produces segment-frame differences. The maximum Frobenius norm of
`N_a-N_0` (also `S_a-S_0`) is **8.38876778e-13**. At segment 5285, a constant
trace-free tensor `tau=2*(15 MPa)*S_a` has `tau:N_a=0` but
`tau:N_0=1.77957118e-5 Pa`. Its zz is zero; both xy contractions are included.
Thus the two implemented point-traction calculations can disagree by more than
the guard at an event-relevant shear amplitude, without any history error.
This is a manufactured point calculation, not a reproduction of the failing
native weak integral.

A Q1/equal-adjacent-weight estimate using the actual step-815 projected shear
perturbation predicts a maximum normal frame contribution **9.91008277e-6 Pa**
at node 4918 (8570 m), already close to the guard. This is explicitly a
**surrogate**, not a measured observer residual: the native quadrature stresses
and failing component/node are absent. It supports testing the frame explanation
but does not prove the failure at 816.

Publication/reconstruction cancellation remains a second possible contribution.
The supplied peak phase and nodal Z/I imply a localization-correction scale of
about 1.32e10 Pa, much larger than physical stress. This is only a scale estimate,
not a measured failing operand or a derived error bound. The independent point
publication test passed its existing operand-derived allowance; that does not
establish the stricter integrated observer check. Neither allowance is changed.

## Diagnostic implementation and bounded local verification

Only `benchmarks/reconstructed_fault/bp3_native/server/plugin/work_audit.cc`
changes executable behavior for this diagnosis. It retains the original
integration, guard, threshold and error message. A separate shadow integration
uses the actual cached QP segment frames. It also integrates the direct frame
contributions `tau:(S_0-S_a)` and `-tau:(N_0-N_a)` separately, avoiding subtraction
of large tractions to estimate those contributions. The original arithmetic is
unchanged. A root trace prints original and matching-frame shear/normal maxima
and nodes before the assertion, including when detailed output is disabled.
On failure, a collective decision writes all nodal loads, mass rows and errors
to `work_consistency_failure_STEP.csv` before the unchanged assertion aborts.
No production parameter, constitutive, core solver, history or default changes.

The two-rank 4136-cell mature fixture advances through the same adaptive
4e6 / 7.64e6 s steps. Twelve selected outputs are **byte-identical** to the
pre-diagnostic control: both ranks' history coefficients, authoritative stress
and checks; accepted summaries, surface weak loads, profile and work checks.
The maximum original observer discrepancy is 8.984e-8 Pa including initialization.
This small fixture does not reproduce the large-geometry frame failure.

A separate compile-only test macro deliberately perturbs one diagnostic shear
load by 2e-5 Pa at step 2; mechanics/history are unchanged. Both ranks reach the
original assertion, a finite 117-row failure CSV is written, and the run exits 1
in 21.17 s under a 180 s timeout. Measured original error is 2.00015082e-5 Pa;
matching-frame shear/normal errors are 1.20e-8 / 1.80e-8 Pa. The macro is absent
from both ordinary and normal local builds. This checks export/rejection, not
the physical cause of the server failure.

The bounded replay input validates using rebuilt ordinary plugins and the local
matching core. Its parsed parameters differ from the supplied resolved input
only in output directory, resume flag, checkpoint cadence, step stop and graceful
wall budget. The first parser attempt is retained: printed inline default
comments confused ASPECT's shared-library pre-parser. The replay now removes
comments without changing parameter values; validation passes. No full-grid
simulation or server job was launched.

## Next confirmation

Use the preserved **809 / slot 03** checkpoint in a fresh output branch with
the diagnostic plugin and supplied `replay-3610007.prm`, stopping by step 816
with checkpoint-every-step and a 7200 s accepted-state wall budget. Direct build,
launch and evidence instructions are in the linked README. Preserve original
outputs and checkpoint; do not append a replay to summaries already through 815.

If the original load fails while the matching-frame load passes and the direct
frame contribution accounts for the difference, correct the observer to use
the mechanical frame, then verify against the original guarded path. If the
matching-frame check also fails, investigate the remaining accepted/frozen stress
arithmetic with the saved replay state before selecting a repair. Do not remove
the guard, enlarge its threshold, or claim the first event is qualified.
Existing spatial/endpoint temporal failures remain.
