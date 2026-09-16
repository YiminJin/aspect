## Stage K5-A — Construct the actual BP3 fixture, not another review.

 Use the official BP3 document as the physical authority, Erickson et al. as numerical/comparison context, the current reconstructed-fault implementation as the software authority, and your old bp3.prm / bp3(6).cc only as implementation hints. Start with the 60° thrust case. Keep a very short bp3_notes.md listing only substantive intentional deviations; do not produce a full model-match essay. Use true normal stress rather than silently reverting to prescribed 50 MPa, because BP3 explicitly includes \(\Delta\sigma_n\). The unavoidable incompressible-vs-\(\nu=0.25\), finite-width/cohesive, and Maxwell-vs-elastic differences should simply be recorded as reasons this is a modified BP3 unless they can be made negligible without changing the architecture.

## Stage K5-B — Implement the minimum missing capabilities needed for a clean BP3 input.

 Most benchmark-specific formulas should live in the BP3 plugin, but Codex may modify core source when a generic capability is genuinely missing. In particular, initialize the fixed Q1 phase field directly from the prescribed straight fault and stationary distance profile, including at the free surface and bottom; do not solve the homogeneous-Neumann phase problem and let it bend the fault vertically at the boundaries. Keep the phase field frozen for this BP3 mechanical benchmark. Initialize \(H\) consistently for bookkeeping. The reconstructed fault should extend from the free surface through the finite box rather than terminate at \(W_f\); below \(W_f\), implement an exact prescribed surface slip rate \(V=V_p\) if the current manager cannot yet impose it. This is a worthwhile generic reconstructed-fault capability rather than a BP3 hack.

For initial RSF state, I would prioritize reproducing the official \(V_{\rm init}\) and \(\Theta_0\). Because our model has an additional surface cohesive traction \(C_0\), if the current residual has the form

$$ q-C-\sigma_n\mu-\eta^dV=0, $$

the natural mapping is to construct the Airy shear prestress so that

$$ q_0=\tau_0^{\rm BP3}+C_0, $$

rather than changing \(\Theta_0\) to compensate for \(C_0\). Codex should verify the actual signs/equations before implementing this, but do not automatically inherit the old strategy of modifying the initial state to absorb cohesion.

Reuse the old Airy construction only after checking its current equations. Its role is an equilibrated ASPECT bulk extension of the specified BP3 fault prestress, not part of the official benchmark. Require equilibrium, top free traction and the intended fault tractions to be checked directly.

## Stage K5-C — Produce two usable parameter files and initialize them completely.

 I suggest bp3_smoke.prm and bp3_pilot.prm. Both should use the same physical model; the smoke case is simply coarser. Use \(\ell=400\) m as the starting regularization already motivated by your previous BP3/K4 experience. A reasonable smoke spacing near the fault is around \(100\) m; the pilot can target roughly \(40\)–\(50\) m if memory permits. The paper shows that volume-based BP3 implementations used a wide range of resolutions; FDCycle's published 60° case used a \(100\times100\) km domain with roughly 100 m resolution, while the benchmark itself recommends 25 m and emphasizes strong domain/resolution sensitivity.

I would therefore start the actual BP3 geometry at approximately 100 km × 100 km, with the 60° fault centered geometrically in the box and the high resolution localized to the full diffuse fault strip. This is a starting finite-domain realization, not a claim that 100 km is converged.

Initialization should not be considered successful merely because ASPECT starts. Codex should verify the straight phase ridge through both boundaries, reconstructed-fault/phase alignment, \(V_{\rm init}\), \(\Theta_0\), the \(a(x_d)\) transition, \(V=V_p\) below 40 km, \(50\) MPa initial effective normal stress, intended initial shear traction, free-surface traction, \(I_h\), support/normalization and finite pressure. Pay special attention to the first few fault nodes at the free surface because the diffuse profile is physically truncated by the domain there.

## Stage K5-D — Run a real short coupled smoke and let it continue far enough to expose performance.

 After initialization passes, run roughly 3–5 accepted real timesteps. Do not artificially shrink the timestep simply to make the test cheap; use the real reconstructed-fault timestep controller so the run exercises particle advection, CPDI-domain rebuilding, \(I_h\), surface coupling, Stokes solves, history publication and prescribed deep creep. The K3 phase-nonlinear residual-floor issue is out of scope because phase evolution is disabled here.

The first goal is not agreement with a full earthquake cycle. Check that slip remains physically signed, no spurious hotspot appears at the free-surface/bottom intersections, true normal stress remains compressive where expected, surface residuals and fresh linear residuals pass, and no fault/particle lifecycle failure appears.

Include the official on-fault sample locations from the beginning—0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 25, 30 and 35 km down dip—so we do not later redesign output around a finished run. For now, \(V,\Theta\), accumulated slip, shear traction and effective normal stress are sufficient; full official output formatting and off-fault displacement accumulation need not block the smoke.

## Stage K5-E — Profile the actual wall-time bottleneck.

 Add opt-in timers using existing ASPECT timing infrastructure rather than scattered permanent chrono calls. At minimum separate: particle advection/exchange; Voronoi/CPDI domain generation; particle→fault association/cache rebuild; \(I_h\) preparation/profile integration; surface \(R_\Gamma/K_V/G\) assembly and condensed fault work; bulk Stokes assembly; Stokes preconditioner construction; linear iterations; mechanical-history commit/particle transfer; and postprocessing/I/O.

Run the same few steps once with expensive output disabled and, if useful, once with normal BP3 output enabled. Report both absolute seconds and fraction of total wall time. The timing table should reconcile most of the measured runtime rather than merely naming the slowest log interval.

Give Codex freedom to perform one or two targeted optimizations if a component clearly dominates. It may cache invariant geometry/material data, avoid repeated point searches, remove duplicate assembly, or reuse already valid data. It must not gain speed by lowering fault/bulk resolution, reducing particle density, weakening quadrature, changing solver tolerances, dropping true normal stress or changing the physical model. Any performance modification must reproduce the pre-optimization numerical trajectory to the existing strict tolerances.

## Stage K5-F — Deliver a server-ready package and stop before the long earthquake sequence. 

The stage finishes with a current BP3 plugin, bp3_smoke.prm, bp3_pilot.prm, prescribed-fault input/sample files, benchmark postprocessor pieces that are already useful, and a short report containing initialization checks, 3–5-step trajectory, measured runtime/memory, top timing contributors, any performance correction, and the recommended MPI/memory/walltime configuration for the first server pilot. Do not spend this stage running to the first earthquake or 1500 years.

A particularly important rule for Codex is that additional source changes are allowed without stopping for your permission, provided they are either a generic missing capability required by BP3 or a verified performance correction. It should stop only if it reaches a genuine modeling choice—for example, it cannot implement the official deep \(V_p\) condition without changing the coupled equations, or the Airy/free-surface initialization proves incompatible with the desired true-normal-stress state.