# Instructions for Codex: BP3 with larger characteristic slip distance and a narrower band

## Authorized scope and proposed configuration

Implement a separately named modified-BP3 research case with `Dc = 0.024 m` and phase-field length `ell = 50 m`, and qualify its initialization and spatial resolution with bounded tests. Preserve the existing `Dc = 0.008 m, ell = 400 m` inputs and evidence as a reproducible reference. Do not launch another full-cycle run in this task.

Read `AGENTS.md`, the relevant sections of `doc/reconstructed_fault/current_design.md` and `specification.tex`, and current callers before editing. This instruction changes benchmark parameters and the mesh; it does not authorize changing the RSF law, work measure, aging update, constitutive coupling, or boundary formulation. Report an actual specification/implementation conflict rather than silently resolving it by redesign.

The source observations below were made at revision `359ea223cd0fa088ba7c8b03338cc7fe19d4acdc`; inspect the current `pf-rsf` checkout and record the actual starting revision. Adapt to current APIs and reuse existing diagnostic infrastructure.

| Quantity | Candidate | Purpose |
| --- | ---: | --- |
| Characteristic slip distance Dc | 0.024 m | Three times the old value |
| Phase-field length ell | 50 m | Working finite-band accuracy candidate |
| Narrower reference ell | 25 m | Prepare a reference configuration; evolution is deferred |
| Fault nodal spacing | 100 m | Retain initially; measure actual spacing |
| Candidate finest bulk-cell side | 12.20703125 m | Test whether this is sufficient |
| Local mesh-reference cell side | 6.103515625 m | Compare at the same physical ell |
| a, b, friction transition | Existing values; 15–18 km | Preserve the frictional contrast |
| Other physical settings | Existing fully frictional, fixed-profile, mature case | Preserve geometry, loading, G, viscosity and damping |

Keep the current 300-by-100-km geometry, 60-degree dip, core phase value 0.6, AT1 geometry and degradation curvature p=1, unless the current maintained case documents otherwise. Retain the working solver backend on the machine used; solver changes are outside this task. No extra refinement centered on the former 40-km junction is needed.

## Why these values

For the current incompressible plane-strain calculation, use mu* = 2G, with G = 32.03812032 GPa. The characteristic process-zone scale is

    L_b = mu* Dc / (b sigma_n).

For Dc=0.024 m and b=0.015, L_b is 2.050 km at 50 MPa and 1.709 km at 60 MPa. The corresponding homogeneous shallow velocity-weakening nucleation-size estimate is

    h* = (2/pi) mu* b Dc / [(b-a)^2 sigma_n],

giving 11.748 km at 50 MPa and 9.790 km at 60 MPa for a=0.010. These are screening scales, not predictions of a rupture width or recurrence time in the heterogeneous finite model. The increase in nucleation size is substantial relative to the shallow weakening region. Do not increase Dc further automatically if a test fails, and do not interpret smooth stable sliding by itself as successful earthquake-cycle qualification.

Use a provisional shortest wavelength for a mechanical-accuracy target of 1.8 km at 50 MPa, scaled from the earlier 600-m target by the threefold increase in Dc. Include 1.5 km as a conservative screen corresponding to 60 MPa. This wavelength choice is an engineering criterion, not an identity equating Fourier wavelength and L_b; later fronts may demand a shorter wavelength. The 60-MPa value is a test condition, not a stress cap.

For the already verified nominal profile and pure Fourier input:

| Wavelength | K_band/K_sharp at ell=50 m | At ell=25 m |
| ---: | ---: | ---: |
| 200 m | 0.27470 | 0.49675 |
| 600 m | 0.61887 | 0.78110 |
| 1500 m | 0.81970 | 0.90428 |
| 1800 m | 0.84675 | 0.91942 |
| 3000 m | 0.90428 | 0.95065 |

Thus ell=50 m gives about 15–18% band error at the provisional target, while ell=25 m gives about 8–10%. Neither is an exact sharp-fault model. Enlarging Dc does not change the mechanical response at a fixed wavelength and ell; it increases the frictional characteristic scales against which the width is judged.

The theoretical kernel is

    K_rate(k) = kappa/(2*pi) * integral[
        4*k^2*m^2/(k^2+m^2)^2 * |chi_hat(m)|^2 dm ],
    K_elastic(k) = G*K_rate(k)/kappa.

Use the actual input spectrum, including taper and Q1 interpolation, when comparing a numerical probe with this formula. A nodal alternating input is not a pure continuum sinusoid. Separate finite-band error from bulk-discretization error.

## 1. Make the configuration internally consistent

Use the existing material parameter `Characteristic slip distance` as the authoritative Dc. In the inspected source, `bp3_model.h` independently hard-codes `BP3::Dc = .008`, used by `theta0`, `aging_state_reference`, and normalized-step diagnostics in `bp3.cc`. Audit all current uses, helper scripts, fixture builders and timestep criteria. Pass the configured value through existing material/friction accessors and benchmark helpers as needed; do not add a second independently configurable Dc or change only the PRM. Preserve old-case reproducibility using its input value.

Generate a fresh initial state for the new case. With unchanged initial friction parameters, normal/shear prestress and intended initial V, the same regularized law requires

    Theta_0,new(s) = 3 * Theta_0,old(s).

Recompute this using the actual initialization formula and configured Dc. Check both reconstructed-fault state and any initial particle/composition copy. Theta_0/Dc and the friction evaluated at the intended initial V must remain unchanged. Do not replace the existing nonsteady initial profile with Theta=Dc/V everywhere: that would change the initial loading condition. Never restart the altered physical model from the old accepted checkpoint.

Keep the physical prestress target unchanged. Regenerate mesh-dependent initial weak projections/corrections with the existing initialization procedure, document their differences, and freeze them afterward. Do not retune prestress to manufacture a desired transient.

Add a focused consistency check that exercises a nondefault Dc in initialization, the production aging update and its independent benchmark check. Verify that the original case still uses 0.008 m. Record resolved parameters and input hashes.

## 2. Regenerate the physical profile and paired inputs

At ell=50 m the nominal full support width is about 197.65 m and chi RMS width is about 22.11 m. Generate the new stationary profile through the existing phase-field API. Recompute H metadata where required, the realized Q1 field, I_h and all associated caches from this profile.

Regenerate the paired boundary-normalization/source-support data for each exact mesh/profile/fault combination. In particular, an old `completion.txt` is not valid after changing ell or the mesh. Retain the currently qualified endpoint treatment; this task does not require a new general boundary method. Verify its existing in-box/outside integral and source-balance identities, including both endpoint neighborhoods. Do not enforce integral(chi)=1 over a deliberately truncated in-box column when the formulation uses a completed column.

Preserve the original fixtures. Generate new scenario-specific fixtures, manifests and launch files. Prepare the ell=25-m reference separately. At initialization, report intended and realized core phase, support, RMS width, normalization, geometry and endpoint balances. Check reconstruction coverage and geometry tolerances for the narrower profile.

## 3. Test mesh feasibility and mechanics before evolution

First build a graded candidate mesh with approximately 12.207 m cells wherever required to resolve the narrow physical band. Include the entire active band and endpoint neighborhoods in the eventual candidate mesh. Report cells, DoFs, particles, memory and assembly cost before proposing a long run. Do not refine the whole box to this size.

The 6.104-m mechanical reference can be a bounded interior refinement patch around the existing diagnostic region, with a sufficiently large graded halo. It need not cover the whole fault for these probes. Keep the physical profile, fault grid, boundary conditions, input wavelengths, kappa and histories fixed. Sample the same intended continuous profile on each mesh and report the realized-profile differences. If a patch edge could affect the measurement, enlarge its halo once for the target wavelength; do not launch a global refinement sweep.

Reuse the frozen mechanical probe for wavelengths 1500 m, 600 m and the 200-m alternating fault-grid mode, at both bulk resolutions. This is six linear response solves, with no history evolution. Use common physical coordinates and actual Q1/tapered inputs. Report:

- Projected work-conjugate shear stiffness and its sign, using the native work weights.
- Relative change on bulk refinement and agreement with the continuum prediction for the actual input/profile.
- Profile normalization and width errors, linear residuals and work closure.
- Elastic stiffness, obtained by G/kappa conversion rather than treating the timestep coefficient as G.

Use a practical 5% stiffness-change target for choosing the bulk mesh, and a 1% target for interior profile normalization/RMS-width error. These are provisional acceptance criteria, not universal resolution laws. Compare against the established solve/work tolerances. If 12.207 m passes, retain it as the economical candidate. If it fails, report the reason and cost of 6.104 m; one finer bounded reference is warranted only to resolve that failure. Do not assume either ell/h=4 or ell/h=8 is sufficient without the comparison.

Also evaluate the cheap uniform-Q1 steady-sliding stiffness screen over the short represented wavelengths, at 50 and 60 MPa. For Dc=0.024 m, a=.010, b=.015, the critical elastic stiffness is approximately 10.417 and 12.5 MPa/m, respectively. The nominal ell=50-m Q1 alternating mode on a 100-m fault grid is about 274.46 MPa/m, well above this threshold. Check the interval, not just this one mode. This is a screening calculation, not a stability proof for the evolving front. Do not demand stability of the intended long-wave nucleation modes, or require 200-m sharp-fault accuracy as the acceptance criterion.

## 4. Run a short production-path qualification

Only after the inputs and mechanics pass, run the fresh candidate for 8–12 accepted real timesteps. Use the production controller, history updates and working solver; report the actual elapsed physical time. Include one restart comparison and one short half-timestep comparison:

- Compare uninterrupted evolution with restart from the same new-case checkpoint over two further steps. Use existing restart tolerances and verify histories, cumulative slip and output metadata.
- From a common new-case checkpoint, replace two accepted real intervals with half-size intervals, reaching the same physical end time. Recompute all production timestep-dependent coefficients normally. Compare V, Theta, slip and tractions at matched times, not equal step numbers.

Log accepted/rejected steps, nonlinear/linear convergence, timestep size, maximum V*dt/Dc, and dt/Theta where relevant. Report weighted profile differences and any local discrepancy near floors; do not hide them with smoothing or rely solely on global maxima. Aim for percent-level agreement in resolved short-run profiles; explain failures rather than changing tolerances or damping until they disappear.

Keep heavy visualization slip-triggered and checkpointing sparse; use compact scalar/profile diagnostics for these tests. No full state snapshot every step except a specific checkpoint needed for the bounded comparison.

Default aggregate simulation wall-time budget for this task: two hours, including the frozen probes and short qualification. Estimate costs early, use graceful stopping, and report unfinished checks if the budget is reached. Do not automatically extend the physical end time until an event or oscillation appears. A quiet short startup does not establish long-time stability, and an unresolved test does not qualify the candidate.

## Deliverables and decision

Deliver the new input case and reproducible fixture-generation/launch commands, the narrowly scoped source changes needed for parameter consistency, a concise parameter/scale table, and a test report with actual costs and pass/fail/unrun status. Retain the old case and its results.

State which bulk resolution passed, what finite-band error is intentionally accepted, and whether the candidate is ready for a bounded longer evolution. Prepare the next launch but do not execute a full-cycle run. The ell=25-m case is a later width-sensitivity reference; do not run a full parameter matrix now.

Larger Dc changes the physical model: nucleation dimensions and cycle behavior may change even when initialization is consistent. The current estimate of h* is already a large fraction of the shallow weakening region. If later behavior is stable sliding, distinguish genuine consequences of the new parameters from unresolved numerical errors rather than treating any absence of oscillations as success.

Reference for the aging-law nucleation scaling and its regime dependence: Rubin and Ampuero (2005), https://doi.org/10.1029/2005JB003686. The wavelength accuracy target and test tolerances above are explicit choices for this modified research case.
