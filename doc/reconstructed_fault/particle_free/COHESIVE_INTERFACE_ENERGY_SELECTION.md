# Temporal work / Maxwell consistency: bounded selection

Reuse the same 16×8 fitted mixed kernel, initial prestress, candidate geometry,
cohesive–RSF calibration, insertion clock and .2–.66 s loading/hold interval.
Do not change the constitutive law, BE evolution, physical controls or tolerances.
The existing .005/.0025 s results and failed sliding-energy screen remain evidence.

First decompose each saved physical remainder without subtracting it from the
acceptance metric. Write c for relaxed affine/jump coefficients, x for imposed
strain/jump coefficients, K for the native mixed Schur energy matrix, alpha=dt/tau,
and tau=eta/G. BE gives Delta x=Delta c+alpha c_new. Let R_s be the retained
surface remainder q_new Delta delta−Delta bond−Gc Delta a. Then, up to the
measured weak/ledger residual,

 R_phys = .5 Delta c^T K Delta c + R_s − .5 Delta reaction Delta gamma
        = R_M + R_Gamma,
 R_M = −.5 alpha Delta c^T K c_new,
 R_Gamma = .5 (K Delta c)_delta · Delta delta + R_s.

This is a diagnosis of time-discretization/work-quadrature terms, not an added
physical dissipation or an error correction. Check it independently against raw
storage/work and the analytic homogeneous Maxwell solution, including the hold.

Select ONE full finer run from the measured first-order error estimate. At
.0025 s the raw absolute remainder/sliding loss is .6297; the unchanged 1% screen
predicts dt ≲3.97e−5 s. Choose dt=.0025/128=1.953125e−5 s, giving a factor-two
margin (predicted ratio .00492). All physical observation times and the .65 s
hold align. About 23553 accepted steps are expected; estimated reference cost is
1–3 minutes using the existing small matrices. Preserve the old accepted-step,
contact, source, derivative and local admissible-tangent checks. Stop and save
state on a real solver/check failure; do not loosen acceptance.

If this run passes, use one focused half-step replay of .58–.66 s from its saved
accepted .58 s state. The replay has identical incoming histories and exactly
aligned loading/hold. This checks event/growth/hold and sliding dissipation without
another full-interval sweep. Expected ~8193 steps, under two minutes. No spatial
or constitutive campaign follows.

Retain the 1% raw absolute physical-work remainder screens normalized separately
by new fracture and sliding loss; retain signed and absolute histories. Require
<1% difference in sliding loss over the common replay window and <2% integrated
damage/slip response differences. Compare final state and hold response as well
as the loading interval. A passing energy balance does not by itself establish
accuracy of every individual contribution. Check exact Maxwell response errors
and actual sliding-loss sensitivity independently. Keep the earlier spatial and
closed-corner restrictions explicit. Deliver PNG/CSV/JSON and a concise report;
no longer propagation test starts automatically.
