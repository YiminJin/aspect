# Temporal work and Maxwell consistency on the activated append

**The unchanged 1% physical-energy screens now pass on the retained spatial
problem.** The earlier failure was predominantly backward-Euler Maxwell time
error, not a missing physical dissipation. No correction is subtracted from the
acceptance metric. Sliding loss and the growth/hold response also pass the focused
half-step comparison. This does not qualify spatial accuracy or free-path growth.

[Selection and equations](COHESIVE_INTERFACE_ENERGY_SELECTION.md) ·
[PNGs, CSVs, checkpoints and commands](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_energy/INDEX.md).
Baseline: `pf-rsf-particle-free`, `f1f9f29cf40c32d1ddcea000b724375b64853532`
plus preserved local work. Production source, BP3, calibration and tolerances are
unchanged. Only two independent reference/analysis scripts and documentation are
added; the existing admission solver is reused unchanged.

## What was held fixed and checked

Reuse the native fitted 16×8 Q2/P1disc mixed operators, full current sided pressure,
Gauss-owned cohesive/RSF histories and 26.565° prescribed append. Start at .2 s
with prepared strain/memory .0015, zero jump/damage and age 1.001. Insert tied at
.25 s. Apply axial strain rate .001/s to .65 s, then hold displacement to .66 s.
G=1e6 Pa, eta=1e8 Pa s, Gc=1 J/m², peak=3000 Pa and p0=10000 Pa retain the
previous RSF/radiation parameters. Initial bulk storage is 4.5 J/m.

Reuse the archived Δt=.005/.0025 s runs. The measured first-order error selected
one full run at Δt=.0025/128=1.953125e−5 s before execution. Its .58 s accepted
checkpoint starts one half-step replay through .66 s. The replay retains all
incoming histories; its energy sums cover only .58–.66 s. Loading/hold boundaries
are exactly aligned. No mesh sweep or constitutive change was made.

Let c contain relaxed affine/jump coefficients, x the imposed strain/jump, and
K the retained mixed Schur energy matrix. With alpha=G Δt/eta,
Δx=Δc+alpha c_new. Define R_s=q_new·Δjump−Δbond−Gc Δa, with actual surface
quadrature. The raw physical remainder is trapezoidal boundary work minus storage
changes, fracture, sliding and Maxwell loss. Independently,

    R_phys = .5 ΔcᵀKΔc + R_s − .5 Δreaction Δgamma
           = R_M + R_Gamma,
    R_M = −.5 alpha ΔcᵀK c_new,
    R_Gamma = .5 (KΔc)_jump·Δjump + R_s.

These are numerical remainder terms, not added dissipation. Long-double
coefficient evaluation reproduces the raw balance to 6.38e−15 J/m per step.
The earlier .0025 s run has signed R_M=−3.78137e−5 and
R_Gamma=−2.06925e−6 J/m: 94.8% of the signed error is the Maxwell term.
The original failed results remain unchanged.

## Measured temporal results

All remainders below sum **absolute per-step** errors, preventing cancellation.

| Interval / Δt [s] | Absolute remainder [J/m] | / new fracture | / sliding loss |
|---|---:|---:|---:|
| .2–.66 / .005, retained | 7.98671e−5 | 1.429% | 122.29% |
| .2–.66 / .0025, retained | 4.00062e−5 | .7140% | 62.97% |
| .2–.66 / 1.953125e−5 | 3.13557e−7 | .005578% | **.5070%** |
| .58–.66 / 9.765625e−6 replay | 3.36985e−8 | .0005995% | **.05449%** |

For the same .58–.66 window the selected-step absolute error is 6.73794e−8 J/m;
the replay halves it. Total sliding loss changes from 6.1846763e−5 to
6.1840428e−5 J/m (**.01024%**, below 1%). At .66 s integrated damage and slip
change **.001528%**, peak damage **.001608%**, below the selected 2% response
criterion. Surface-weighted damage/jump L2 difference is .001550%; age L2
difference is 1.69e−7 relative. Common-time loading differences are much smaller
than the final hold difference; they are not interpreted as a convergence order.

The selected run ends with peak a=.0270373, integrated damage .00562122 m,
and peak jump 1.80249e−5 m. No fully decohered a=1 region forms. The discrete
tied-loss bracket is [.5814372444,.5814372540] s in both focused runs; a numerical
bracket is not a spatially qualified physical activation time.

The selected final energy accounting [J/m] is:

| Quantity | Value |
|---|---:|
| Trapezoidal boundary work | 3.0981279516 |
| Bulk storage increase | 3.0318264848 |
| Cohesive storage increase | .00549444736 |
| New fracture | .00562122083 |
| Sliding loss (friction + radiation) | .00006184676 |
| Maxwell loss | .05512426404 |
| Signed physical remainder | −3.12222345e−7 |

During the displacement hold external work is exactly zero, yet sliding loss is
3.59691e−6 and fracture increases 1.72602e−4 J/m, supplied by decreasing storage.
Stopping boundary motion does not remove stress. Loading contributes signed
−3.12890e−7 J/m error and hold +6.67182e−10; their small cancellation does not
explain the pass. Endpoint work and the exact BE ledger remain separate CSV
columns; neither replaces physical trapezoidal work.

An independent tied homogeneous control uses w'=rate−w/(eta/G), integrating the
exponential exactly over the complete ramp and hold. Its exact balance closes
to 1.31e−17 J/m. At the selected Δt, final memory error is 2.22e−8 relative,
trapezoidal work error −3.88414e−8 J/m, and Maxwell-loss error +5.92914e−7 J/m.
The last is .001075% of exact Maxwell loss, but **.959% of the append's sliding
loss**. Thus a small balance remainder is not evidence that every contribution
is equally accurate. The analytic half-step control halves these errors; the
actual append replay separately verifies sliding loss. The analytic tied control
is diagnostic, not a substitute for the activated solution.

## Verification, cost and scope

The full run took **106.51 s / 23,553 accepted steps**; the focused replay
**46.39 s / 8,193 steps**, sequentially with one BLAS/OpenMP thread and bounded
600/300 s timeouts. Every step retains contact, weak equilibrium, irreversibility,
BE ledger, admissible symmetric-tangent and publication checks. All post-event
steps retain the coupled finite-difference check. Across both runs:

- Maximum weak residual 2.93e−11 Pa versus 1e−8 Pa allowance.
- Maximum derivative relative error 9.71e−7 versus 1e−6 allowance. Tiny-increment
  finite differences approach their numerical resolution; no tolerance was changed.
- Maximum BE ledger error 6.38e−15 J/m versus existing 1e−10 scaled allowance.
- Minimum sided contact pressure 11800.02 Pa; minimum free symmetric tangent
  7.32573e6 Pa/m. This is the retained local admissible test, not a new general
  nonsymmetric stability theorem.
- Deterministic repeated trial, rejected invalid publication, unchanged incoming
  state, serialized restart, stale/duplicate-publication rejection and exactly
  one accepted publication are verified at activation in each run.

This task evolves the independent reference with actual native mixed operators.
It does **not** turn the prior native checkpoint/publication replay adapter into
an outer native nonlinear time solver. That existing native lifecycle evidence
is preserved, not reclassified as a new full native trajectory.

The energy/temporal question is closed within this fixed spatial component test.
The earlier **65.4% early spatial discrepancy**, closed-corner limitation,
pointwise tip sensitivity, provisional phi, prescribed-path restriction and
absence of fully decohered advance remain unresolved. Recommend next a bounded
native outer-loop signed-admission integration on this same append, using these
states as references, before longer growth or direction-selection claims. Do
not start that task or another propagation campaign automatically.
