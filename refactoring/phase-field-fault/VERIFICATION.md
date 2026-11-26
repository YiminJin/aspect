# Scientific references and verification

The completed `MIDPOINT_UPDATE_CLOSEOUT.md` freezes the implemented midpoint update. The audit found that the irreversible phase equation equilibrates at the stage. Extrapolating that solution to the endpoint leaves further admissible downhill directions for the next solve. A fracture-only calculation reproduced the native second phase increment within 0.014% in spatial L². Preserve this update and its known timestep sensitivity. The proposed endpoint-equilibrated alternative belongs to a separate numerical-method study.

## 18-strip midpoint timestep comparison

Use the repaired 5A point and strip budgets before comparing schedules. Keys are (CellId, native quadrature index), (CellId, component, local FE index) and the material strip index. Require exact identities, membership, active sets, admissions and publication counts. The physical clocks use 1e−15 s absolute. Geometry and native areas use 1e−14 absolute. Use absolute floors of 1e−9 for phase, 1e−6 Pa for physical stress, pressure and resistance, 1e−10 for strain and shear inventories, 1e−11 for speed (m/s), slip and arclength (m), and age (s), and 1e−7 rad for angles, plus 1e−7 relative for physical fields. The velocity coefficient floor is 1e−10 m/s on the 0.02 m/s loading scale. The phase stage gates remain 1e−9 for mechanics and 1e−7 for projected phase. Compare the physical phase row residual in J/m with 1e−8 absolute (the 1e−7 projected gate times the physical row scale Kf·mass is 6.94e−9 J/m on these cells), plus 1e−7 relative. Scale the whole-window and individual interval reductions by 1e−5 relative with the archived floors of 1e−12 m² for signed slip, 1e−10 m for phase L², and 1e−11 J/m for incremental fracture, work and compliance. Compare ledger terms at 1e−11 J/m plus 1e−6 relative. Keep residual work distinct from the compliance remainder.

Increment 5 closed with an exception for the first turn of strip 15 in this fixture. Retain the four strict endpoint failures at the original tolerance. The completion audit also checks the virgin strip table by material ID against the donor's incoming formation, age, slip, reference width and zero-angle initialization. For strip 15, read both reduced buffers and the first angle store from the same two-rank selector. Match that store to its own acceptance and to the ordinary run. The common-input debugger replay supplies both original compiled selectors with the same twelve numbers and stops before publishing its trial. Compare both boundaries and all scan-resolved peaks across the forward sector with a bracketed signed RSF solve at each angle. The numerical interpretation of this fixture leaves the original thresholds in force. Reproduction commands and the two figures are at the top of the working record.

Compare conditional endpoint loads and residuals on all 1,152 (CellId, local Bernstein index) rows at each physical acceptance using the stage row's physical budget. Use archived endpoint driving in Pa on damaged donor points. On intact donor points resolve endpoint shear on the recorded plane and subtract the RSF resistance from the retained signed root and accepted age. Project each donor and target field with its own native areas. The donor audit supplies measured endpoint fracture gradients on the first update of each branch. On the preparation and final fine interval reconstruct these from the donor FE coefficients. Check the strict downhill mask and the feasible increment above 1e−7 by row, and compare their counts and the maximum load to the archived independent closeout. For solver effort read the donor solver table and separately match the target balance count to the completion signal.

The benchmark observer exports the ordinary solution with its mapped Cartesian corners, component and local basis identity on both executables. Its replay uses the archived solver, checker, MPI arguments, environment and donor restart. The target uses the normal function timestep model. `Simulator::run` advances the clock before writing its checkpoint, so generation 02 already carries the pending interval. Resume the coarse branch from that generation; reconstruct the fine parent from time zero and require byte equality of all physical parent tables before comparing. Check the pending duration printed on entry to each solve against the accepted parent and endpoint clock.

In the independent audit, use physical areas for the native weak products and section areas divided by current arclength for rate transfer. The velocity gradient, trace-free plane-strain tensor including zz, and boundary-only test come from Q2 shape gradients at native points using each cell's actual widths. Pressure uses its Q1 basis. A boundary velocity coefficient stays on the lift only if its support point lies on a prescribed face. Contract midpoint stress with the symmetric gradient of that lift, subtract pressure times divergence, and multiply by the attempted duration to recover prescribed external work. Use both endpoint tensors to reconstruct elastic and cohesive storage, both viscous terms, point friction work and the cohesive chain-rule remainder. Compare the friction work with the independently reduced signed strip work. Check stage-aged RSF and full-duration accepted aging separately.

For the phase rows, integrate the AT1 volume and symmetric interior-penalty terms in independent Gauss(4) quadrature. Visit each interior face once with the outward normal of its left/bottom cell; obtain measure and penalty from its physical dimensions. Project native driving onto each Bernstein test. With accepted obstacle bₙ, stage b and stage residual g in J/m, classify the projection of b−g/(Kf·mass) onto [bₙ,(1+bₙ)/2] before extrapolating the endpoint. For conditional endpoint downhill rows, evaluate driving with the published tensors and treat intact points with the endpoint stress, rate and age. The archived stage rows directly validate both first updates after 5 ms. The other two stages are reconstructed independently from donor fields and native loads. The scratch row recorder reads the actual returned residual after the coupled convergence and native storage checks. It uses the stored phase block offset and local FE component, and records h and B before FE publication. The audit requires byte-identical accepted data from recorded and ordinary runs, matches each physical returned row to independent assembly within 1e−11 J/m, and checks the normalized KKT against production to 2e−12. The storage/work cross-checks use 1e−11 J/m absolute, transfer work 1e−12 J/m, Maxwell stress 1e−9 Pa, signed RSF 1e−10 Pa and accepted age 1e−13 s. The 5B audit returns failure for any exceeded frame or aggregate budget.

~~~text
benchmarks/growing_fault/timestep/run.py
benchmarks/growing_fault/timestep/midpoint_fields.cc
benchmarks/growing_fault/timestep/frames.py
benchmarks/growing_fault/timestep/audit.py
source/simulator/solver/midpoint_fault.cc
source/simulator/solver/midpoint_direction.cc
source/phase_field/midpoint_fracture.cc
~~~

## Fault ownership and constitutive substitution

The two paths use the same material-law interface with different time levels. On the mature trace, `solve_rates` interpolates trial speed at the Q8 sample. The material interpolates the incoming RSF ages at that sample before evaluating traction and its fixed-age slope. The weak row contracts the supplied response with the Q1 test and the same normalized transfer used by Stokes. After convergence, the post-repeat hook commits slip and speed on the trace, law history at the material IDs, and Maxwell memory in the material. The section/profile pair moves by owning pointer, keeping the profile bound to its own trace.

For growing strips the simulator constructs the midpoint system with separate references to simulator services and to the active growing material. The strip transfer reduces free rate and mobility over physical area divided by current arclength. Its resistance law eliminates the signed speed and returns force and condensed derivative. Each prospective direction repeats that solve over its complete strip; the direct score alone has compact support. At acceptance the physical owner advances the law, including unformed strips. A new checkpoint contains an independent law identity and exactly its declared state. The repaired v1 reader splits interleaved RSF ages from its kinematic records and preserves balance availability.

For the state-free control, run the parameter-file regression in the [working record](STATUS.md#ownership-cleanup-before-5b). It differentiates the pointwise direction score, reconstructs the signed linear root and condensed slope from FE gradients and old tensors, compares stored native transfer, and differences the actual returned matrix. Reversing the loading changes the sign of the physical rates. A mature Q1 dashpot has a separate uniform-traction solution. Both checkpoint probes read the physical wire records.

~~~text
include/aspect/material_model/rheology/fault_resistance.h
include/aspect/material_model/growing_fault.h
include/aspect/reconstructed_fault/material_sections.h
source/simulator/solver/midpoint_fault.cc
source/reconstructed_fault/stokes_coupling.cc
benchmarks/growing_fault/check_state_free.py
~~~

## Growing fault review repairs

Use the [repair matrix and reproduction commands](STATUS.md#increment-5a-review-repairs) to check the published RSF pair at the stage age, integrate stored point increments with a cancellation-aware bound, and exercise the Stokes constraints, automatic cutbacks and restore-before-observation lifecycle. For signed friction work, compare both the native work integral and the section root product. Evaluate prescribed-velocity work over every constrained velocity reaction, including an interior actuator when present.

## Growing fault midpoint control (5A)

The default nine-strip comparator takes strip 8's beta (rad) and ds (m) at 5, 10 and 15 ms from the adjacent installed `nine-strip-geometry.csv`. Its six reviewed entries are the candidate rows in `results/direction-c1/frozen-arclength.csv` at `bffffca07`, obtained on the frozen 8 × 4 two-rank input. The original donor columns in the table identify the corresponding `localized-onecut` frames. All other quantities and comparison tolerances come from the existing donor comparison. Pass `--historical-donor` to `benchmarks/growing_fault/compare.py` to reproduce the donor comparison in full, including with `--archive` for the portable final frame.

Start with the [5A working record](STATUS.md#increment-5a--single-forward-front) for the input identities, reproduced clocks, exact commands and first failing points. The canonical layout is two MPI processes, one thread each. The nine-strip test qualifies one fixed 2-D mesh, constitutive law and short trajectory. The [strip 8 evidence and replay instructions](strip8/README.md) compare the pre-turn stage by CellId and quadrature index, replay both selectors on live and ordered moments, and check every endpoint column through 15 ms. The independent control solves the whole-strip resistance and sums the disk-weighted squared traction point by point.

### Follow one step in source order

1. The input selects `phase field fault` with RSF and the `midpoint fault phase` composition method for the configured AT1 phase. The parameter parser checks the 2-D DGQ2 role. Introspection allocates a Bernstein element. The single-advection iterated Newton scheme solves temperature and dispatches each composition first. It defers this phase row to the simulator's coupled midpoint solve. The mature model and original ASPECT elasticity are off.
2. `advance_midpoint` begins an attempt on the active growing material and borrows its accepted native tensors, strip history and ASPECT's old solution. Initialization constructs the fixed material sections, applies whole-polynomial bounds to the seed, and returns at zero elapsed time. On a physical attempt `solve_midpoint` forms the half-duration, scales wall increments by h, fixes one pressure coefficient, and restores discrete continuity. It freezes the disk from the connected accepted graph, the maximum incident DG trace and the 0.05 threshold. Every residual evaluation gets that same parent.
3. At native chord points the stage unknowns are evaluated as Q2 displacement increment a, Q1 pressure p/G and Q2 Bernstein coefficients b. The 3-D deviatoric plane-strain predictor is B(τₙ + 2G dev ε(a)), B = 1/(1+hG/η). For each strip the collective free rate and mobility use physical area divided by arclength. Solving V = Vfree − M f(V,θ(h)) supplies the signed resistance f to every native point. The physical rows integrate τₘ:dev ε(test)/G − (p/G) div(test) and −q div(a). On the phase rows the residual is the derivative of the discrete AT1 energy divided by Kf, less the physical-area integral of NᵢY/Kf. FractureRows integrates volume and two-sided interior face derivatives by its own quadrature. Exterior phase flux is natural.
4. The tangent applies the exact direct derivatives and df = [f′/(1+Mf′)](dVfree − f dM). Each sparse cell/strip border is accumulated on its owner. The root's linear system has one auxiliary increment df per strip. Eliminating those linear increments recovers cross-rank and cross-cell couplings. Wall and pressure constraints are eliminated on the increment, and a primal active set enforces bₙ ≤ b ≤ (1+bₙ)/2. The early positive phase search uses fracture stiffness plus mass. The tested physical Jacobian and convergence residual always include the full coupling. The nonlinear coordinates do not include independent force, age or point history.
5. At convergence `select_prospective_planes` sums localized squared predictor tractions with weight Jρ(1−F)². Every angle evaluation uses the *whole-strip* signed resistance root and angle-dependent arclength. Formed angles are fixed. A changed trial map requires another assembly. The final call to `assemble_stage` computes the physical residual, tangent, ledger and native history together. ASPECT receives a/h, p, and 2b−bₙ. The simulator's repeat decision runs while the material still owns the old accepted pair. `post_accept_timestep` moves the returned history into place exactly once. The normal checkpoint hooks archive owned native histories by CellId and quadrature index, strip history and clock. A restart checks the fixed physical vertices, quadrature order, material signature and strip edges and obtains phase from the simulator's checked FE checkpoint.

~~~text
benchmarks/growing_fault/onecut.prm
include/aspect/reconstructed_fault/material_sections.h
include/aspect/material_model/rheology/cohesive_maxwell.h
source/simulator/solver_schemes.cc
source/simulator/solver/midpoint_fault.cc
source/phase_field/midpoint_fracture.cc
source/simulator/solver/midpoint_direction.cc
source/material_model/growing_fault.cc
~~~

### Verify a change to a row or transfer

Use the distributed oblique integral and remote-force unit tests first. The benchmark observer then applies the same physical matrix used by the solver, eliminates its force borders, and differences independently reassembled residuals at fixed history and geometry. It separately contracts a real velocity test against physical stress. The time-step observer repeats the tangent test on the returned solution before acceptance. The offline material observer reconstructs ∇v from the emitted nodal coefficients, forms the full plane-strain tensor and checks native Maxwell, cohesive and aging increments. Its independent fracture-energy integral reads the endpoint Bernstein coefficients and counts each shared face once. Run both the direct retained-archive comparison and the all-frame replay comparison. Matching one export to another cannot establish the physical weak form on its own.

~~~text
unit_tests/material_sections.cc
unit_tests/phase_field.cc
benchmarks/growing_fault/growing_fault_checks.cc
benchmarks/growing_fault/check_history.py
benchmarks/growing_fault/check_admission.py
benchmarks/growing_fault/compare.py
benchmarks/growing_fault/run.py
~~~

### Shared ASPECT edits

| Release symbol or file | Why the ordinary extension point was insufficient | Feature-off coverage |
| --- | --- | --- |
| Parameters::AdvectionFieldMethod, parameter declaration and parse | A dedicated composition method declares the phase's sole time integrator. A material evaluate callback cannot prevent a generic second update. The new enum value is appended. | Ordinary parameter parsing, fixed_fault, explicit_dg_compositions and both stock solver schemes. |
| Introspection FE selection | The existing branch's Bernstein field selection depended on the SSPRK3 selector. The same native element is required for the midpoint obstacle when SSPRK3 is off. | 4C moving phase, independently selected DG tracers, affine AT2. Existing basis metadata is used unchanged. |
| compositional_field_needs_matrix_block | The dedicated phase contributes to the coupled sparse matrix. Allocating an ordinary transport block would leave an unused solver matrix. | matrix_nonzeros_1 and explicit_dg_compositions. |
| Simulator::assemble_and_solve_composition and solve_single_advection_iterated_newton_stokes | A local material or Stokes assembly callback cannot own a simultaneous irreversible phase solve, full-strip derivatives and native point recovery. The opt-in dispatch occurs after normal temperature and companion field handling. The existing repeat and acceptance hooks suffice. | Stock prescribed_dilation_newton_singleadvection and iterated_advection_and_stokes_DG compared directly with 4C, plus the layered mature trajectory and mixed explicit/implicit dispatch. |
| Test registration | The ordinary regression depends on the benchmark parameter file it includes. The compiler tracks the included C++ observer. | growing_fault CTest exercises the parameter-to-publication path without donor data. |

Material sections, the cohesive point law, SIPG fracture rows, direction selection and the midpoint solver are new branch code. None of these operations reads benchmark outputs. The mature MaterialFaultTrace, the generic SSPRK3 kernel and the original elasticity rheology are unchanged. A future phase integration change would touch the coupled time discretization and phase rows. It would leave material-strip membership intact.

## Layered moving mature fault

The trace unit tests rotate a unit segment through 105° and reject collapsed and self-crossing edges. The corrected 38-label fit exercises the initial boundary-spanning reconstruction independently of moving the donor curve.

The [4C closeout regressions](STATUS.md#increment-4c-review-closeout) check the returned DG projection with a nonzero parent, full and finite moving normal supports, the fixed/advected interior weak action, and the later Picard section tangent. Keep the prescribed donor radius explicit in the frozen case. The zero-width translation exercises full in-box admission separately.

For the projected section at a fixed Picard iterate, the assembled rate row is
`r_i = Σ_q JxW χ N_i [τ_drive − κ χ V − R(V,θ,σ)]`, with Q1 trial speed `V = Σ_j N_j V_j`. Applying the negative rate derivative to a direction `d` gives `Σ_q JxW χ N_i [κ χ + ∂R/∂V] Σ_j N_j d_j`. Physical bulk area times χ gives a length, the bracket in the residual is a traction in Pa, and the residual and matrix action have units Pa m. The centered difference keeps age, the returned bulk iterate, candidate χ, pressure-inclusive compression and the current projected stress defect fixed. Sample κ where the Q8 residual is integrated. The closeout capture reads the globally reduced diagonal and upper band used by `solve_rates` before elimination.

Use the [4C fixture, immutable donor identity, commands and measured comparisons](STATUS.md#review-of-the-bounded-4c-calculation). The canonical run compares both accepted physical endpoints and twenty material IDs. Reconstruct the Cartesian DG polynomials independently, check both signed transfer contractions, and compare the actual Q3 Stokes vector action with Q8 physical stress work. Initialization tests the direct section law. Physical steps test the projected response shared by mechanics and publication.

The target-only matrix covers fresh starts, rejection of a long first attempt, checkpoint continuation, an actual change of supporting MPI rank, pointwise layer sampling, zero and reversed translations, a compact support disjoint from the walls, and the homogeneous limit. Keep sampled physical extrema and exact geometric clearance distinct. The plot, parameter-file CTest and comparison script belong with their respective record, tests and benchmark. The frozen donor archive is needed only for the external scientific comparison.

## Stationary coupling repair

The fitted 32² free-fault regression compares 38 ordered material IDs at 0, 12.5 and 25 ms. It checks IDs and clocks exactly and uses 1e−10 absolute plus 1e−8 relative for the six physical columns. Both references ship with the benchmark. The Q3 and Q8 names refer to the measure of the boundary-spanning slip load. Ordinary ASPECT Stokes assembly uses Q3 in both cases.

| Free-32 label table | Provenance |
|---|---|
| `benchmarks/fixed_fault/phase_ridge/q8-free-32-labels.csv` | Active reference from the serial Release run of the Q8 implementation reviewed at `623bb1c57`. The strict checker's `--history` export reproduces the CSV. |
| `benchmarks/fixed_fault/phase_ridge/3a-repair-free-labels.csv` | Historical pointwise Q3 load. First recorded at `fb4b6b9b2` under `refactoring/phase-field-fault/results/3a-repair-free-labels.csv`. Commit `9dd92e321` moved the canonical copy into the benchmark and kept the original path as a symlink. |

The subsequent [3A repair specification](INCREMENT_3A_REPAIR.md) defines the fitted-contact, stress-moment, normalization, actual weak-load, support-complement and MPI acceptance checks. Use its fixture-specific numerical budgets for that assignment. Keep the Increment 2 reference values below.

Use the inputs documented for the three-label and homogeneous cases at `e91e801bab`. Both use a fresh physical initial state, explicit DG stress fields, the fault's Maxwell law, and upstream `Enable elasticity = false`. Record the resolved numerical values and commands. Compare at the frozen tolerances.

| Check | Evidence to keep |
| --- | --- |
| Both geometries | Both fixtures exercise the same ordinary timestep. Match label coordinates, rates, ages, slip, pressure, velocity and physical DG stress at the existing locations and times. Include the nonuniform three-label endpoint. |
| Virtual work and physical outputs | Compare the local stress RHS action against independent quadrature of the signed offset on a nontrivial test velocity. Exercise assembly with matrix rebuild and with RHS only. Verify the plane-strain trace correction exactly once and ordinary incompressible continuity. Check the supported physical stress output against accepted DG values at a nonzero-history endpoint where a purely viscous reconstruction differs. |
| EOS and field layout | At fixed temperature and lithology, vary positive stress. Check invariant density with a nonzero compositional density contrast or a clear diagnostic for an unsupported contrast. Declare field types explicitly. Verify the active field access and output use the intended components. |
| Nonlinear failure and retry | Run a clean smaller step and a deliberately repeated larger attempt from the same initial conditions. Compare at the same accepted times. Separately exercise a case where the bulk stopping condition is met while rates are inconsistent. The solve must iterate again or signal recoverable failure. Only the converged state is published. |
| Scope of the coupling | Run a release feature-off case and a release original-elasticity regression, e.g. `viscoelastic_numerical_elastic_time_step`. Check the normal fault run uses its own registration and no original elasticity symbols or conventions. Check that a request to resume a fault simulation is diagnosed while the current model is fresh-run only. |
| Checkers and parallel execution | A deliberately perturbed reference copy must fail under ordinary Python and `python -O`. Retain the friction/aging unit test and compare the nonuniform case on one and two ranks using material label and cell/quadrature identities. Run the scoped Debug and Release regressions from the recorded build. |
| Stationary review cases | Compare a 3 Pa tensile pulse with automatic cutback against a run prescribing the same accepted intervals. Check all four endpoints and require the open initial fault to fail before publication. Exercise closed normal velocity with two initial pressures, a closed periodic box, a viscosity multiplier of two, and a horizontal fault under +0.25 m initial topography. The unwrapped, flat, traction-anchored donor cases are the controls. |
| Clocks, coordinates and replay | A 15 ms homogeneous endpoint must fail its donor clock check. A 0.1 m origin plus 0.2 m extent must accept the 0.3 m face and reject a genuinely short section. Run `test_replay_stationary_reference.py` to check missing inline output and stress-rotation settings under both Python modes before any donor work. |

The frozen files are numerical comparison data. Prototype fault save/load, older-format input, and a disk-based restart comparison are outside this assignment. The donor can be inspected for provenance.

## Locate the reference from the target root

```sh
donor=../aspect-pf-rsf-particle-free
reference="$donor/build-particle-free/midpoint-closeout"
git rev-parse --show-toplevel
git rev-parse HEAD
git rev-parse --verify 'v3.1.0^{commit}'
git merge-base --is-ancestor 0d97d2d3311154ca3f01d801b18f3945dc8b8bad HEAD
git status --short
git --no-optional-locks -C "$donor" rev-parse HEAD
git --no-optional-locks -C "$donor" status --short --untracked-files=no
ls -lh "$reference/reference.tar.gz"
sha256sum "$reference/reference.tar.gz"
tar -tzf "$reference/reference.tar.gz" | head -40
```

The recorded peeled tag is `0d97d2d3311154ca3f01d801b18f3945dc8b8bad`. If the tag is missing locally, verify the full commit and ancestry and record the missing tag. A branch with the kit or later migration commits will have a different HEAD. A feature change already present at startup belongs in the working record.

Read `$reference/reference/README.md` or its copy in the archive. For original reports, use `rg --files "$donor/doc" "$donor/benchmarks" | rg 'MIDPOINT_UPDATE_CLOSEOUT|MATURE_FAULT_TEMPORARY_CLOSEOUT|LOCALIZED_TIP_DIRECTION|reference_sections\.md'`, then look at the archive index if a report lives elsewhere. Resolve links in donor reports from their original directories. Redirect all replays into new run directories. In particular, check fixture output settings, restart locations and plugin search paths before running an archived executable.

Record the two source bases, tracked patches, required untracked files, full executable/checker/plugin identities, build settings, input, parent checkpoint **generation**, parallel configuration, reducer command, units and stored acceptance tolerances. Check compact expected outputs against the manifest before using them. The donor revision reported at closeout was `c407e612bf7c9aa1a9c0fc60fd623208dfd53b39` plus local changes. Four documentation hashes postdated the 18-strip experiment. Use the archived numerical source identities. If the archive is missing, recover the same files from the indexed donor evidence where possible, recording each verified match and each remaining gap. An older public branch is insufficient.

### Three comparisons with different meanings

| Reference | Inputs and identity | Recorded outcome and its use |
| --- | --- | --- |
| Passing mature control | One rank, layered viscosity, moving fault, growth disabled, native Bernstein phase, direct DG physical stress, bulk-slip-subtracted accepted spin, prescribed exterior stress and compatible loading. Two 12.5 ms steps. Runtime plus three plugins. | Two joint publications, zero retries, terminal section rates 0.0533293–0.1601043 m/s. Compare the retained numeric control columns at both steps. Process peak memory was excluded from the old exact replay. |
| Passing tip control | Two ranks, 8 × 4 oblique single-front specimen, nine complete strips, localized onset with radius 2ℓ, physical point stresses and cohesive history. | Relative tangent-action error 7.53 × 10⁻⁹, maximum work-adjoint defect 6.94 × 10⁻¹⁸ J/m, integrated slip at 15 ms 1.693023 × 10⁻⁴ m². Reproduce the forced rejection, deterministic retry and the recorded endpoint from initial conditions. |
| Timestep-sensitive diagnostic | Two ranks, 16 × 8 specimen, localized law, 18 strips, same accepted 5 ms parent (generation 02). One 5 ms step and two 2.5 ms steps to 10 ms. Same physical loading, quadrature and strip width. | Compare the port with the donor **within each trajectory**. Include the 7.5 ms field, phase-row observers, endpoints, slip, energies and signed and absolute defects. This specimen diagnoses changes in the implementation. It cannot qualify propagation timing. |

The numbers above identify the controls. Set comparison tolerances from the archive's bounds and explain any platform-dependent differences using physical scales, absolute error near zero and relative error for resolved signals. The older full-section 80/40 μs propagation runs and the different-width binaries have their own identities. The four-run historical mature study documents spatial/temporal limitations of stable sliding. It cannot certify a BP3 first event. Older mature guides describe other spin, stress and phase choices. For the selected moving control the closeout and its frozen input take precedence.

### Reproduce the physical parent for the later 18-strip comparison

For each trajectory replay initialization and the first interval to 5 ms in the target. Compare that parent with the archived donor observations before following the two timestep schedules. Check phase coefficients and obstacle, material-point tensors, section ages, slip and formed planes, identities and measures, physical clock and interval. Account for permutations when point ordering changes. A disagreement at 5 ms should be localized before interpreting the later endpoints.

When a shared-core change shifts a comparison, first check feature-off behavior on the release, then compare the parent, stage inputs, residual and tangent on a frozen state, and finally the published endpoint. Comparing several full trajectories before locating the first differing operation usually obscures the cause.

## Numerical contracts easy to lose

**Mature path.** Store the selected full plane-strain physical stress in its DG representation and resolve the actual component convention, projection and material age update from the frozen source. Transport, rotation, geometry motion, reaction and nonlinear iterations have different clocks. State which velocity/gradient drives each. The selected control subtracts the slip contribution when constructing the accepted bulk spin. Surface age and slip follow material labels. Identify the same material labels before and after motion. Retain the mature law's time levels and any required projection correction. Implementing the tip law at mature points would change this control.

**Section coupling on the tip path.** For membership matrix $P$, material-area weights $J$ and reference arclength weights $D_s$, the reduction and return satisfy

$$
V = D_s^{-1}P^T Jw, \qquad f_{\mathrm{point}}=P f, \qquad
V^T D_s f = w^T J P f.
$$

Use the same measures in residual, condensed tangent and work. The localized kernel weights the prospective orientation score. The section rate, candidate resistance and resistance return use the entire section. Freeze the disk from the accepted parent through a trial and its retries. Respect formed-plane history, aging of virgin sections, the admitted angular sector, tie behavior and the single-forward-front eligibility rule. Check an oblique nonuniform case where a wrong weight or unsupported activation can be detected. A measured passing angle on one seed is insufficient to establish frame covariance or angular validity.

**Midpoint path.** With full duration $\Delta$ and $h=\Delta/2$, the nonlinear unknowns are the stage displacement $a=h v_m$, scaled pressure $p_m/G$ and stage Bernstein coefficients $b_m$. The accepted parent $b_n$ gives

$$
 b_n \le b_m \le (1+b_n)/2, \qquad b_1=2b_m-b_n.
$$

Phase columns differentiated with respect to $b_m$ have unit stage chain factor. Publication has derivative $2I$. Work uses the full displacement $2a$. The minimum incoming Bernstein coefficient in this physically admissible polynomial was −0.008669. Both native first stages had 947 lower-active, 205 free and zero upper-active rows. Preserve its representation and obstacle. Clipping all coefficients to zero from below would change this reference. Compare physical bounds using the retained polynomial enclosure and irreversibility using the actual increment constraint.

Point stress and recoverable cohesive strain publish from the stage as $2\tau_m-\tau_n$ and $2A_m-A_n$. Section slip accumulates $\Delta V_m$. Evaluate the analytic aging update from the same accepted age at $h$ for RSF and at $\Delta$ for publication, including its zero-rate limit. Published rates and forces on this path describe the last stage. The frozen loading is steady. It gives no evidence for stage accuracy under explicitly time-varying boundary velocities.

| 5–10 ms diagnostic | One 5 ms step | Two 2.5 ms steps |
| --- | ---: | ---: |
| Integrated signed slip (m²) | 3.748901 × 10⁻⁵ | 5.800755 × 10⁻⁵ |
| Added fracture energy (μJ/m) | 5.550475 | 15.041375 |
| Spatial L² norm of phase increment (m) | 0.0114891 | 0.0151855 |
| Signed physical-work residual (μJ/m) | +3.960087 | +6.782957 |
| Signed compliance remainder (μJ/m) | +3.960087 | +6.782315 |

For this specimen the physical-work residual is wall work less changes in stored elastic, cohesive and fracture energies and less bulk-viscous, cohesive-viscous and fault dissipation. Report the signed sum of step defects and the sum of their absolute values. Keep the compliance remainder and residual work in separate columns. The independent control reported a second phase increment of 0.00742635 m versus 0.00742678 m coupled. A tiny ledger error after subtracting the remainder can coexist with the physical-work defects above.

## Checks for the first implementation

| Question | Smallest discriminating check | Compare |
| --- | --- | --- |
| Does the chosen local law produce the intended response? | Independent point calculation on the fixed-phase route. Perturb slip/kinematics at fixed accepted state on a smooth branch and compare its derivative with the implemented linearization. | Sign, units, stress components, resistance, age time level and derivative. Use the reference's zero/near-zero treatment where relevant. |
| Is fault feedback present in the actual solve? | Launch a small ordinary `.prm` with an interior velocity unknown, active fault and nonzero loading. Select a matched donor case or document the reduction. | A solved bulk field or projected traction plus section slip rate, age and physical stress at common positions/times. Check residual or tangent action with section feedback included. |
| Does the accepted state advance exactly once? | Compare a clean shorter step with a forced reject/retry from the identical parent ending at the same time. Re-evaluate a trial before acceptance. | Accepted coefficients, section histories, clock, geometry identities and physical observables. No age/slip increment survives the discarded attempt. |
| Is upstream behavior preserved? | Run the same scoped upstream test before and after the common-code changes, with the feature off. | Existing expected output, supported default solver path and relevant assertions. |

Use the checked-out testing documentation. In a configured build, `ctest -N` lists registered cases. If a needed test is absent, register the release's test suite via its `setup_tests` target, then select the exact test with `ctest -R` and `--output-on-failure`. Check the selected count. A parameter parse or a zero-test exit cannot establish runtime behavior. Put deterministic local tests in the existing unit/integration framework. Larger donor reductions stay out of ordinary CI. At integration, run a relevant multi-rank check and the consolidated suite. A stage with specialized dimension/mesh support should diagnose unsupported configurations before using its data.

Use Debug for assertions and Release for matched cost. Match mesh, timestep sequence, ranks/threads, solver tolerances, outputs and build type. Record solver iterations, wall time, peak memory where available, and any additional full-state allocation or checkpoint gather. Investigate a sustained total slowdown of roughly 10%; noise and differing iterations need separate interpretation. The historical direction-selection cost budget measures a different operation. Ordinary increments need only the tests they can meaningfully exercise. Long event runs, new convergence campaigns and production cluster budgets are separate assignments.

[ASPECT 3.1.0 test instructions](https://aspect-documentation.readthedocs.io/en/v3.1.0/user/extending/testing/running-tests.html)
