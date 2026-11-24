# Scientific references and verification

The completed `MIDPOINT_UPDATE_CLOSEOUT.md` freezes the implemented midpoint update. The audit found that the irreversible phase equation equilibrates at the stage. Extrapolating that solution to the endpoint leaves further admissible downhill directions for the next solve. A fracture-only calculation reproduced the native second phase increment within 0.014% in spatial L². Preserve this update and its known timestep sensitivity. The proposed endpoint-equilibrated alternative belongs to a separate numerical-method study.

## Layered moving mature fault

Use the [4C fixture, immutable donor identity, commands and measured comparisons](STATUS.md#review-of-the-bounded-4c-calculation). The canonical run compares both accepted physical endpoints and twenty material IDs. Reconstruct the Cartesian DG polynomials independently, check both signed transfer contractions, and compare the actual Q3 Stokes vector action with Q8 physical stress work. Initialization tests the direct section law. Physical steps test the projected response shared by mechanics and publication.

The target-only matrix covers fresh starts, rejection of a long first attempt, checkpoint continuation, an actual change of supporting MPI rank, pointwise layer sampling, zero and reversed translations, a compact support disjoint from the walls, and the homogeneous limit. Keep sampled physical extrema and exact geometric clearance distinct. The plot, parameter-file CTest and comparison script belong with their respective record, tests and benchmark. The frozen donor archive is needed only for the external scientific comparison.

## Stationary coupling repair

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
