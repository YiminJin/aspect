# Increment 5A · Couple one growing fault front to Stokes and phase evolution

**Finish line.** Starting from the repaired ASPECT 3.1 `pf-rsf` branch (reviewed through `01575dd`), run the frozen two-rank, 8 × 4 oblique `localized-onecut` control through its three accepted steps to 15 ms. An ordinary ASPECT parameter file must exercise the production mechanics, phase solve, direction selection and publication together. Prove the operators and the accepted history independently, then stop for review. The proposed 5B assignment is the two 18-strip trajectories at their original durations. Scientific qualification of propagation speed or earthquake nucleation comes later.

## Establish the reference before editing

Read `AGENTS.md`, then `PLAN.md`, `VERIFICATION.md` and the top of `STATUS.md` under `refactoring/phase-field-fault/`. Record the actual target HEAD and incoming changes. Read the 4C review closeout, particularly the failure in which the solver returned a different stress response from the one it had tested. In the donor, start with these files relative to its root:

~~~text
doc/reconstructed_fault/particle_free/MIDPOINT_UPDATE_CLOSEOUT.md
doc/reconstructed_fault/particle_free/tip_treatment/reference_sections.md
doc/reconstructed_fault/particle_free/LOCALIZED_TIP_DIRECTION.md
~~~

The reference archive is `../aspect-pf-rsf-reference-20261121/reference.tar.gz`, SHA-256 `a82d362c3504be188e9602cf10c93cf51a26544be5e6d4a7677016cc52981c57`. Follow the archive README, verify its internal manifest, and use a fresh extraction outside the source trees. The numerical **tip** source is donor commit `c407e612bf7c9aa1a9c0fc60fd623208dfd53b39` plus `tip/qualified.patch` and `tip/qualified-additions.tgz`, in that order. Inspect this source alongside the actual input, command, runtime and saved outputs in the archive. Use the original initialization, quadrature, coefficients, prescribed compression, loading and step sequence. In particular, start from the archived seed with its original zero-clock equilibration setting.

With `R` the extracted `reference/` and `D` a fresh working directory, the donor commands are:

~~~sh
python3 "$R/analysis/replay_midpoint_reference.py" "$R" localized-onecut "$D/donor-onecut"
python3 "$R/analysis/check_midpoint_reference.py" "$R" localized-onecut "$D/donor-onecut/output"
~~~

The historical checker uses Python assertions. Run it with assertions enabled. Resolve local executable and MPI paths from the archive guide. If an artifact or binary is unavailable, identify it precisely, proceed with independent operator work, and report the external comparison as pending. The installed target and its ordinary tests must run without the donor.

## Implement three reviewable changes

Work through all three. Keep the diffs separable and exercise each through the code the coupled run will call. Before introducing interfaces, add a short owner map to the working record with the normal caller, accepted inputs, trial outputs, ownership and reason for each required core hook. Update it as the implementation settles.

### 1. Fixed material sections, native history and section mechanics

A **section** in this discretization is a strip spanning the specimen transverse to the initial seed tangent. Its material membership stays fixed while a prospective fault plane turns. Use the projected material coordinate ξ along the initial tangent. In each intersected cell split only at its projected vertices within the strip and integrate the resulting chords with the selected Gauss rule. Each native point has a physical area weight Jq and reproducible identity. For strip k with projected width ak and structural angle βk relative to the seed, the arclength weight is dsₖ = aₖ / cos βₖ. Rotating the prospective plane changes its shear tensor S = sym(t ⊗ n) and its arclength weight. The point locations and areas stay fixed.

Keep full plane-strain physical deviatoric stress τ and recoverable cohesive strain A at these native points, including the zz components. Preserve the signed and absolute shear inventories used for admission and publication. Give each strip its age θ, signed slip, and formed-plane history. Use existing material, fault and simulator interfaces where their contracts match. The 4C `MaterialFaultTrace` represents an ordered material curve. Here the phase field can reach a prospective section before a curve segment exists. Give fixed section membership its own explicit meaning. A fitted curve, a projected DG stress field or an output table cannot supply the native tip history.

Let P map a section value to its contributing points, J be the diagonal matrix of physical areas and Dₛ the diagonal matrix of arclength weights. For signed local fault shear rate w in s⁻¹, slip rate V in m/s and signed resistance f in Pa, implement

    V = Dₛ⁻¹ Pᵀ J w,       f_point = P f,       Vᵀ Dₛ f = wᵀ J P f.

The last equality is power per unit out-of-plane thickness. Multiply by the full physical duration for step work. The point law at fixed parent and plane gives w = w_free − m f. Integration gives V = V_free − M f(V). Recompute the signed RSF plus radiation root for the actual candidate, including its stage age and prescribed normal compression. In the tangent, with other direct law inputs fixed, δf = [f′/(1 + M f′)] (δV_free − f δM). Include the phase dependence of mobility and the age dependence in f′. Evaluate the analytic intact and zero-rate limits. Check reversed shear and locked response. The archived tip law and its clocks govern any reuse of the mature rheology.

**Verify before coupling.** Independently integrate an oblique cell/strip intersection and an area moment. Check the total area, section assignment and a strip receiving contributions from both ranks. Use nonuniform signed point rates and unequal measures to check the actual reduction and return against the work identity. Perturb a point on one rank and measure an induced resistance or residual on the other rank in the same strip. A cell-local Jacobian misses this coupling even when its diagonal blocks look correct.

### 2. Coupled midpoint mechanics and irreversible phase

Trace the ordinary simulator timestep, including the repeat decision. Add the smallest opt-in dispatch needed for the coupled solve. The phase-field module owns its fracture energy, spatial derivative, SIPG face terms, obstacle and phase-field operations. Fault geometry supplies membership and measures. Local constitutive evaluation supplies point response and derivatives without committing history. The coupled solver coordinates the global residual, linearization, active set and convergence. Postprocessors observe the solved state. Trace field dispatch for one full step. The tip phase receives exactly one physical update, through the coupled solve. Generic composition transport and the mature stress update must not advance the same tip state a second time. Preserve ordinary dispatch for other selected fields. Reuse the branch’s phase parameters and Bernstein FE. Keep the original ASPECT elasticity rheology and its additional outputs out of the fault path.

For full attempted duration Δ and h = Δ/2 the only global nonlinear unknowns are a = h vₘ, scaled pressure pₘ/G and DGQ2 Bernstein stage coefficients bₘ. Eliminate point and section variables within each trial. Apply the archived pressure gauge and use h times prescribed velocity for wall increments. With D₃(a) the plane-strain deviatoric symmetric displacement gradient and S:S = 1/2, check the implemented point response against

    (τ₁ − τₙ)/(2GΔ) + τₘ/(2η) = D₃(a)/h − wₘ S,
    (A₁ − Aₙ)/Δ + (G/η) Aₘ = wₘ S.

The stage Maxwell factor is 1/(1 + hG/η). Use the donor’s cohesive compliance, driving force and intact limit. Assemble the AT1 fracture-energy gradient minus the mechanical driving term on the stage, including the derivative of the eliminated section resistance. Retain the phase operator’s own volume and face quadrature. Assemble or apply every required block of the physical coupled tangent, including cross-cell and cross-rank contributions. State how the solver applies these blocks, how constraints and pressure scaling enter, and which approximations are confined to preconditioning.

| Quantity | Stage | Accepted publication |
|---|---|---|
| Phase coefficients | bₙ ≤ bₘ ≤ (1+bₙ)/2 | b₁ = 2bₘ − bₙ |
| Stress and cohesive strain | τₘ, Aₘ from the same immutable parent | τ₁ = 2τₘ − τₙ and A₁ = 2Aₘ − Aₙ |
| Section age | For k = abs(Vₘ)/Dc, use θ(s) = θₙ exp(−ks) − expm1(−ks)/k at s = h; at k = 0 use θₙ + s | Evaluate the same function from θₙ at s = Δ, including for unformed sections |
| Slip, rates and work | Solve for Vₘ and fₘ | Add ΔVₘ to signed slip. Report Vₘ, fₘ, a/h and pₘ as stage quantities. Wall work uses 2a. |

In stage coordinates each stage phase column has chain factor one. Endpoint phase publication has derivative two. An admissible incoming DG polynomial can have a negative Bernstein coefficient. Preserve the incoming representation. Verify physical bounds with whole-cell polynomial enclosures and irreversibility with the coefficient increment. Test lower, free and upper inequality rows on controlled inputs. The canonical seed need only exercise the rows it actually reaches.

**Verify before orientation changes.** On a smooth fixed-plane branch, difference the production residual at fixed accepted state against the operator used by the linear solver. Use several perturbation sizes, separate displacement and phase perturbations, and a mixed direction. Exercise Stokes-to-phase and phase-to-Stokes feedback. Verify the sign of active-row optimality and differentiate only across admissible smooth directions. Check at least one velocity test function against independently integrated physical stress work in the assembled mechanics. Isolate the deviatoric contribution or explicitly account for the pressure, boundary and constraint terms. Record how a wrong sign, measure or stage factor makes a check fail. This distinguishes a correct tangent of the wrong residual from the intended weak form.

### 3. Select the direction, admit a section, and commit the returned state

Use the archived localized integrated-onset rule. The initial seed registers its incoming plane on the contiguous positive support specified by the reference. Determine the forward disk center from the **accepted** formed graph and phase, with the archived anchor, DG trace convention and 0.05 tracking threshold. Freeze the center and kernel for all trial evaluations from that parent. Use R = 2ℓ and ρq = max(0, 1 − |Xq − Xtip|²/R²)². An unformed strip is eligible when it has a sample inside the open disk. Preserve the formed angles, open sector |β| < π/4, and the donor’s tie and edge handling.

Only the direct squared-traction direction score carries Jq ρq. Its angle-dependent sum uses (1−Fq)² (p_predictor,q:S(β) − fₖ(β))², where Fq is the relaxed fraction and p_predictor,q is the point’s Maxwell stress predictor. Preserve the archived normalization and handling of cancellation in weakly driven sections. For **each candidate angle**, recompute the candidate resistance over the entire strip. Slip, returned resistance, mechanical work and their tangent also use the full strip. Demonstrate this by perturbing a remote band in the same strip outside the disk. Its direct weight must be zero while the candidate resistance responds. The resolved local score can change through that resistance. Preserve the separate eligibility, positive-phase registration and exterior-history checks. A detached point with positive phase alone must not acquire a formed plane. Handle unsupported exterior cohesive history before publication.

Re-evaluate the mechanical residual and phase optimality on the **returned** converged fields and final plane map. Bind the reported tangent and candidate history to that same state. An updated map that invalidates convergence requires another solve or a recoverable failure. Commit once after the simulator’s repeat decision. Rejection discards trial angles, obstacle changes, stresses, shear, slip and age. Rebuild duration-dependent factors on retry. Exercise continuation from a checkpoint written by this target through its normal restart mechanism, with stable point/strip identity and full accepted history.

## End-to-end acceptance and review handoff

| Evidence | Passing result |
|---|---|
| Ordinary two-rank run | The target starts from the archived physical initial conditions, uses nine complete strips, and publishes steps 1, 2 and 3 exactly once through 15 ms. Compare accepted clocks, identities, formed flags, angles, section age/slip/rate/resistance, point τ/A/phase and integrated slip. The donor gives **1.693023 × 10⁻⁴ m²** integrated slip at 15 ms. Identify units and discrepancies at the first diverging step. |
| Residual, tangent and work | Use the frozen normalizations. Mechanical residual ≤ 10⁻⁹, phase KKT residual ≤ 10⁻⁷, integrated work-adjoint defect < 10⁻¹² J/m, relative tangent-action error < 2 × 10⁻⁵. The donor measured 6.94 × 10⁻¹⁸ J/m and 7.53 × 10⁻⁹ respectively. Compare bulk and cohesive energy, added fracture energy, wall work, dissipation and phase remainder. Report the physical energy defect and the remainder separately. |
| Retry and restart | Show an immutable accepted parent during trial evaluation, restoration after rollback, deterministic retry, unchanged second rollback, and single publication. Compare clean and retried **fields and histories** at common accepted times. Resume from the target equivalent of the parent of step 3 and compare the last accepted endpoint with the uninterrupted target. The donor continuation is `localized-oneresume`, from `onecut` generation 03. |
| Strength of the checks | Use the archived column tolerances in `check_midpoint_reference.py` without adjustment. Compare added fracture energy at its own scale. Use independent observers of actual target fields and operator actions. If target names or ordering differ, document the identity/unit mapping in the comparison script. Keep frozen values in test evidence. Corrupt a temporary output and confirm the new checker fails, including under `python3 -O` if written in Python. |
| Regressions and cost | Exercise a Debug coupled solve, the canonical two-rank Release control, the 4C mature path and focused phase tests. Run a relevant unchanged ASPECT feature-off test for each substantive shared-core edit, including generic transport if touched. Use one numerical thread per rank and at most four local workers. Record timers, iterations, number of material points, peak per-rank memory, and replicated history or matrix storage. |

At handoff, update the active item in `STATUS.md` and the source-linked verification guide with exact commands, input identities, measured results and unresolved differences. Show the ordinary call path from the parameter choice through assembly, section elimination, phase solve and exactly-once acceptance. Give the maintainer the few headers and implementations to read first, with copyable worktree-relative paths. Define “material section,” “structural plane,” “prospective,” “formed” and “stage” in their owning interfaces. Explain each substantive upstream core edit and identify its feature-off regression. Keep permanent production code independent of specimen dimensions, reference CSVs, benchmark plugins and donor paths.

Stop after the nine-strip control and its review evidence. Retain any failed comparison and explain it before claiming completion. The 18-strip timestep pair, endpoint reformulation, general fault networks, 3-D fault surfaces and publication-scale experiments are separate work.
