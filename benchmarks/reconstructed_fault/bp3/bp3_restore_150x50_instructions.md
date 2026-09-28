# Codex handoff: restore BP3 parameters in a 150 x 50 km phase-field model

Implement a clean-start BP3-parameter pilot with a smaller domain, bottom Dirichlet loading, graded mesh refinement, and the tested normal-input filter enabled from initialization. Preserve the existing BP5 configurations and results. Complete the model files, required plugins, mesh generation, focused qualification, and a reproducible first-event run workflow.

This is a development experiment using BP3 friction parameters. Record the finite-domain, incompressible-bulk, diffuse-fault, filtered-normal-input, and deep-loading differences from the standard SEAS benchmark. Do not add compressible elasticity or redesign stress-history transfer in this task.

## 1. Configuration to build

| Item | Primary configuration |
| --- | --- |
| Domain width x height | 150,000 x 50,000 m |
| Cartesian box | x in [-60,000, 90,000] m; y in [0, 50,000] m, y upward |
| Fault | Straight, dipping 60 degrees downward to the right |
| Surface trace | (0, 50,000) m |
| Bottom intersection | (28,867.5134594813, 0) m |
| Along-fault length | 57,735.0269189626 m |
| Phase-field length ell | 20 m |
| Smallest bulk cell edge | 3.90625 m; ell/h = 5.12 |
| Coarsest bulk cell edge | At most 1,000 m |
| Fault-element target spacing | 20 m |
| Primary normal-input filter length | 20 m, enabled from the initial mechanical solve |
| Reference configurations | Raw pointwise friction input; optional 40 m filter sensitivity PRM |
| Bottom boundary | Prescribed two-component velocity |
| Top boundary | Existing free-surface traction convention |
| Stress history | Existing production particle update, advection, and FE transfer |

Use explicit names such as `bp3_150x50_raw.prm`, `bp3_150x50_filter20.prm`, and `bp3_150x50_filter40.prm`. These are requested deliverables, not existing filenames or parameter names to assume. Reuse the actual tested filter options from the checkout.

All cases start from t=0 with freshly generated geometry, phase field, fault state, and background properties. Do not resize, rescale, or continue a BP5 checkpoint. A restart of the primary BP3 run must retain the same model and filter settings.

## 2. Restore the BP3 friction parameters and initialization

Use the official BP3-QD specification, especially its parameter table and initialization equations (24)–(25), as the numerical reference:

https://strike.scec.org/cvws/seas/download/SEAS_BP3-QD-FD.pdf

| Parameter | Value |
| --- | --- |
| a, shallow | 0.010 |
| a, deep | 0.025 |
| b | 0.015 |
| Critical slip distance Dc | 0.008 m |
| Reference coefficient mu0 | 0.6 |
| Reference rate V0 | 1e-6 m/s |
| Plate-rate magnitude Vp and initial-rate magnitude Vinit | 1e-9 m/s |
| Initial effective normal stress | 50 MPa |
| Shear modulus | 32,038,120,320 Pa |
| Radiation damping | 4,624,440 Pa s/m |

For down-dip arc length s from the surface trace, set a=0.010 for s<15 km, increase it linearly to 0.025 over 15–18 km, and keep it 0.025 below 18 km. These are along-fault distances, not vertical depths. Use the existing regularized rate-and-state friction and aging law. Keep the mature-frictional mode, frozen phase field, and 1e26 Pa s bulk viscosity unless a documented existing implementation requirement makes a mechanical adjustment necessary.

Initialize zero accumulated slip. Use the BP3 constant background shear prestress and spatially varying initial state consistent with Vinit. Remove the BP5-specific 30 km weakening region, a/b/Dc settings, and weakening-state-ratio override (previously 0.8). Do not set Theta=Dc/Vinit everywhere: that would be a different prestress/state configuration.

For the positive-slip convention, an independent calculation gives the following useful checks:

- Background driving shear, including the small radiation-damping offset: approximately 26,546,122.3651 Pa.
- Initial Theta in the uniform shallow region: approximately 8,000 s.
- Initial Theta in the uniform deep region: approximately 8,000,000 s.

Verify the implemented friction law reproduces Vinit algebraically from these prestress/state data, with the correct signs and background split. Preserve the current physical sense of faulting and report it; reverse and normal slip must not be interchanged while changing coordinates or boundary functions. Apply signed shear/rate conventions consistently, while Theta remains positive.

Do not store the same prestress both in a fixed fault background property and in the incremental particle tensor. Initialize the incremental mechanical history consistently with the chosen prestressed reference configuration. The first finite-width mechanical solution need not reproduce the ideal continuum initialization to roundoff; report its departure instead of tuning a spatial shear correction to conceal it.

The current incompressible bulk is retained. Standard BP3 uses Poisson's ratio 0.25, so the pilot is not an exact reproduction of that benchmark.

## 3. Fault extent, properties, and endpoint data

Generate the entire straight fault from the surface to the bottom intersection. Place fault vertices exactly at s=15,000, 18,000, and 40,000 m, with segment spacings no greater than approximately 20 m between the breakpoints. Preserve the prescribed straight geometry rather than fitting it to noisy phase-field samples.

Retain the successful **fully frictional fault**, extending the velocity-strengthening properties down to the bottom. The standard BP3 Wf=40 km position remains a useful comparison/diagnostic location, but do not restore an internal prescribed-V/free-V junction there: that was a source of artifacts in the earlier model. The deep extension beyond 40 km is approximately 17.735 km long. Document this loading difference from standard BP3, which prescribes slip below Wf.

Regenerate every geometry-dependent asset: fault coordinates, initial phase/history data, mesh targets, particle initialization, fault mappings, normalizations, and endpoint-completion files. Keep the established top normalization/compensation algorithm; regenerate its values for the new geometry and ell. Recompute the bottom completion too. Never reuse tables keyed to the old 300 x 100 km geometry or ell=100 m.

## 4. Bottom Dirichlet loading

Replace the bottom traction prescription with a **smooth velocity profile across the diffuse fault**. Prescribe both velocity components. Keep left/right prescribed velocities, using the same loading convention and reference frame as the new bottom profile. Keep the top free surface. Remove the old bottom traction contribution and any obsolete loading correction derived for that traction condition; retain the geometric normalization compensation still required at the endpoint.

Use one consistent signed-distance description. With the coordinates above, a convenient down-dip unit vector and normal are

\[
\mathbf s=(1/2,-\sqrt3/2),\qquad
\mathbf n=(\sqrt3/2,1/2),\qquad
r=\mathbf n\cdot(\mathbf x-\mathbf x_{\rm trace}).
\]

Construct a normalized cumulative profile from the same stationary cross-fault localization used by the model:

\[
F(r)=\frac{\int_{-\infty}^{r}\chi_{\rm ref}(\rho)\,d\rho}
{\int_{-\infty}^{\infty}\chi_{\rm ref}(\rho)\,d\rho},
\qquad
\mathbf u_{\rm load}(\mathbf x)
=\mathbf u_{\rm frame}+V_{\rm src}\,[F(r)-1/2]\,\mathbf s.
\]

Here u denotes velocity, as in the Stokes implementation. Choose the sign of Vsrc, of magnitude Vp, from the existing slip/source convention. In particular, verify the derivative of this reference profile produces the same signed localized shear strain as the implemented chi*V*S source. Do not assume a normal orientation or a factor of two from another plugin. If the existing boundary plugin already implements this compatible profile, adapt it instead of introducing a competing function.

Use the same untruncated reference profile/continuation underlying the endpoint normalization; do not derive F from only the part of a normal cross-section left inside the bottom boundary. Keep the bulk chi and I_h definitions unchanged. If the reference integral differs slightly from one, report its normalization and the resulting match to the actual source.

Required properties:

- Far from the fault, the two prescribed velocities differ tangentially by the signed plate rate; each side receives half that rate in the symmetric frame. Prescribing +/-Vp would incorrectly double the loading.
- Their relative fault-normal velocity is zero. Do not replace the vertical component by zero or prescribe horizontal motion alone on a 60-degree fault.
- Use a common velocity frame on bottom and side boundaries; the symmetric choice is u_frame=0. Ensure shared corners receive compatible values.
- F must be continuous and resolved on the bottom mesh. Do not use a discontinuous block-velocity jump or an independently chosen tanh width.
- Check the applied FE boundary values, not only the analytic function. Sample outside both localization tails and recover the tangential rate difference in m/s.
- Audit global incompressibility/flux balance with the free top included. The sum of fluxes on only the three velocity-prescribed sides need not vanish separately. Do not introduce a pressure source to compensate for a boundary implementation error.

The reference field above is divergence-free in the continuum because s dot n=0. This is a compatibility check for the loading construction, not a claim that the evolving heterogeneous-slip model has zero elastic strain.

Retain the existing pressure normalization/prestress convention unless the actual nullspace analysis requires a change. The top remains a traction boundary; converting the bottom does not automatically make pressure freely shiftable. Do not impose volume-mean-zero pressure on top of a physically fixed pressure reference merely by copying a toy fixture.

**This fixes the relative velocity imposed at the bottom. It does not prescribe V=Vp at every interior deep fault node.** Record both quantities separately, especially at 40 km and close to the bottom, so an interior loading drift is not hidden by a successful boundary check.

## 5. Mesh with a resolved fault band and gentler grading

Use a stationary distance-based initial mesh; freeze it during the first-event pilot. Preserve square Cartesian cells, refine through the full fault length and both endpoint neighborhoods, and retain the same near-fault resolution in the strengthening extension. Do not coarsen abruptly at 15, 18, or 40 km.

One concrete grid hierarchy is:

- Box repetitions 75 x 25: square root cells of edge 2,000 m.
- Minimum active refinement level 1: maximum edge 1,000 m.
- Maximum level 9 relative to those roots: minimum edge 3.90625 m.
- A possible ASPECT setup is initial global refinement 1 plus up to 8 further adaptive passes. Verify actual generated edge sizes; do not assume plugin pass counts equal levels.

An equivalent hierarchy is acceptable if it meets these physical sizes and is materially easier to maintain. Do not use the diagonal cell diameter as h when checking ell/h.

Determine the actual support of the initialized localization chi. Let r_chi be its cross-fault support radius, or a tail radius consistent with the existing integration tolerance. Define the fine-band half-width

\[
r_f=\max(2\ell,\;r_\chi+2h_{\min}).
\]

This will be approximately 40 m if the existing AT1 profile fits inside that margin, but verify the profile convention rather than assuming its support. All cells intersecting |r|<=r_f must have the finest edge size.

Outside the fine band, use a reproducible grading rule. For the minimum distance d_K from a cell to the straight fault/required endpoint continuation, define

\[
h_{\rm target}(d_K)=
\begin{cases}
h_{\min},&d_K\le r_f,\\
\min\{1000\ {\rm m},\;2h_{\min}+(d_K-r_f)/4\},&d_K>r_f.
\end{cases}
\]

Refine a cell when its edge exceeds this target, subject to the finest level. Use minimum cell distance or intersection tests, not just the centre distance; otherwise cut cells can miss the fine band. Enforce the supported face/corner level-balance constraints, with at most a 2:1 size jump between neighboring levels. Outside the core, the rule gives approximately four cells across each dyadic grading band before integer-cell alignment/balancing.

The factor-two transition out of the finest region is the normal discrete refinement step. Keep the localized deformation support wholly inside the fine region; spread subsequent levels over the surrounding bulk. This is mesh grading, not smoothing of velocity or material properties. Quadtree stair steps remain, so do not promise that this removes the separate subcell stress mode.

Before mechanics, report cell counts by level, minimum/maximum edge, cells intersecting chi support, aspect ratios, fault spacing, DoFs by field, particle count, and estimated/observed memory per rank. Export one mesh overview and one near-fault/endpoint close-up.

A simple geometric calculation of this rule with r_f=40 m gives about **487,389 leaf cells and 4,386,501 particles at nine per cell**, before implementation-specific balancing and support adjustments. Treat this as a planning estimate, not an ASPECT result. The smaller box does not guarantee fewer DoFs than the old BP5 model: the finer fault band dominates cost. Use actual generated counts and existing server memory limits before the long solve. Reduce avoidable refinement outside the supported band before considering any change to its resolution; never silently underresolve ell to meet a budget.

Start with the graded mesh and 1 km coarse cap above. If velocity corrugations remain, locate them relative to refinement boundaries using identical physical samples. A further mesh variant should address a measured problem; do not launch a broad mesh or filter sweep as part of restoration.

## 6. Normal-input filter from initialization

Reuse the tested filter implementation and its consistent nonlinear coupling. The primary model uses Ls=20 m from the first mechanical solve, with a 20 m fault discretization. Keep ell, fault spacing, and filter length as separately documented parameters even though their nominal values coincide here.

The raw reference must use the original pointwise normal input. Ls=0 in the filter formula is a Q1 projection, not the raw reference. Save projection-only diagnostics alongside the raw and filtered stress where inexpensive. Provide a 40 m sensitivity PRM starting from the same clean physical initialization; it need not become a second long trajectory in this task.

Check constant 50 MPa reproduction, weighted-mean preservation, minimum compression, and behavior near the physical endpoints on this new mesh. Friction must use the filtered total normal input; raw pressure/current stress and retained particle tensors remain mechanical diagnostics. Do not overwrite the history with the smoothed field or switch to adiabatic friction pressure.

Do not assume that filtering from t=0 will prevent raw-history accumulation. Track raw history and filtered input separately from startup onward. Do not enlarge Ls in response to a rough plot without checking its effect on slip and resolved physical stress gradients.

## 7. Time integration, outputs, and qualification

Keep the existing reliable coupled solver, state-update lifecycle, linear/nonlinear tolerances, accepted-step checks, and production particle scheme. Use the server-qualified AMG path unless the current machine has a separately demonstrated working alternative. Reuse filter and history qualifications; do not reopen the previous transfer investigation.

Restore a state-based startup timestep appropriate to the new Theta. With shallow Theta around 8,000 s, the existing weighted logarithmic-state bound of 0.02 permits only order-100-second initial increments. A provisional first-step cap of 100 s is sensible; let the existing controller reduce it. Audit the separate material/artificial initialization timestep too, so a legacy 1e6-second artificial update is not silently retained as initial stress. Preserve its established no-double-publication semantics. Later steps may grow under the normal controllers; retain the existing large maximum-step ceiling rather than forcing large steps through the startup bound.

BP3's smaller Dc can still require short event timesteps. Do not relax the 0.02 state-change limit or other stability checks to obtain larger steps. Log the actual limiting condition and actual constitutive dt.

Use compact diagnostics at startup, representative loading times, nucleation, event peak, and arrest:

- V, incoming/committed Theta labels, Omega=|V|*Theta/Dc, and accumulated slip versus down-dip distance.
- Raw current normal/shear traction, actual filtered friction normal input, and separately sampled working/particle history roughness.
- Prescribed bottom relative rate, actual interior deep V/Vp, near-bottom normal stress, and loading reactions/flux checks.
- Refinement-interface velocity variation, using the same component, physical sampling, and color range. Distinguish discontinuous stress representation or plotting interpolation from oscillation in the solved velocity.
- Solver convergence, phase/I_h drift, active bounds, actual dt, wall time, and memory.

Use slip-triggered major output (the existing approximately 0.1 m increment is acceptable) plus a few physical-time checkpoints. Save an early loading checkpoint and a pre-nucleation checkpoint so future sensitivity work does not depend on a checkpoint already in rapid slip. Avoid full-volume output on every small event step.

Complete these focused checks:

1. Verify coordinates, along-fault property transitions, signed loading, regularized boundary trace, and the algebraic BP3 prestress/state initialization.
2. Build the real mesh and regenerate its normalization assets. Check endpoint closure, MPI ownership/consistency, and actual memory footprint. Reuse small one-/two-rank filter checks; add only a focused boundary/loading check if needed.
3. Run a bounded startup of the primary 20 m-filter model and raw reference, approximately 10 accepted steps each, using identical physical parameters and comparable accepted times. Report the first response and any developing grid-scale variation; a short startup does not qualify the interseismic loading history.
4. Prepare the primary first-event continuation from its own accepted BP3 checkpoint. Once startup and resource checks pass, use the established run workflow and available server budget to progress toward the first event. Retain the raw startup control and the clean-start sensitivity PRM for later comparisons. A changed geometry/filter/material model must never be applied to that continuation checkpoint.

Verify that the event termination criterion means a completed event (acceleration, peak, and subsequent decay/arrest), not merely the first crossing of the seismic-rate threshold. Stop gracefully and checkpoint at the existing wall-time limit if the event is not complete. Do not claim completion from a wall-time exit or a successful few-step test.

Investigate a concrete failure before continuing if the prescribed bottom rate is wrong, normal-input safeguards activate unexpectedly, solver checks fail, or grid-scale slip-rate variation grows strongly relative to the resolved slip. Ordinary stress accumulation during loading is physical; do not flag the entire normal-stress RMS as numerical noise.

## 8. Deliverables

Provide the code/plugin diff, complete raw/filter20/filter40 model PRMs, fault and normalization files or their generators, and mesh-generation instructions. Include exact build, bounded-startup, first-event, and restart commands with the actual executable/library paths.

Return a compact report containing:

- The resolved parameter table and explicit differences from standard BP3.
- The actual mesh/DoF/particle/memory inventory and refinement plots.
- The bottom loading definition, sign/frame check, and measured boundary rate.
- Startup comparisons and the status/location of checkpoints and the first-event run.
- Whether the remaining velocity variation follows mesh transitions, the fault discretization, or the original stress mode, where the data support that distinction.

The scientific goal is a usable, reproducible first-event pilot. Do not claim that the 150 x 50 km box is domain-converged, that Dirichlet loading forces every deep friction node to plate speed, that the filter cures history accumulation, or that an incompressible filtered model exactly reproduces the standard BP3 benchmark.
