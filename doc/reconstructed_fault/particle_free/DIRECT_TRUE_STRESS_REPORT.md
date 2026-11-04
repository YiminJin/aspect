# Direct physical-stress history: implementation and native comparison

4 November 2026. **Recommendation: retain the direct path as an independent native
reference pending finite-curve-end stress/traction accuracy. Keep separated
history as the default.** The new representation works in the tested mature
scope and removes independent T/R ownership. It does not yet justify replacing
the analytic localization at finite ends, nor promise a runtime improvement.
Final unit/regression closeout is recorded below; production readiness is not claimed.

## Selection, preservation and identity

This follows `CODEX_TRUE_STRESS_HANDOFF.md` in this directory; the requested
parent-level path was absent. The broader positive-refinement accuracy/energy
continuation is paused. Work stayed in the existing
`/home/ein/repository/aspect-pf-rsf-particle-free` worktree, branch
`pf-rsf-particle-free`, HEAD `69e8a10a93a08718d31af961197899e320657c04`, with incoming
dirty documentation/manual work preserved. No checkout/reset, server job,
production replay, checkpoint conversion, or unrelated cleanup occurred.

[Evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/direct_true_stress/README.md)
contains entry identities, the incoming patch/source archive, compact reductions,
build/test logs and final provenance. Six verified `true-entry-` core/checker
copies preserve the separated controls and their required libraries. The core
still prints `27178813c`; that string is not its source identity. Every run's
`command.json` records the actual core/plugin/input hashes and command. The main
comparison uses the corrected final-gate build. The final source adds an optional
energy ledger; its two-step prefix reproduces the earlier direct trajectory to
5.6e-18 in bulk exports, with identical surface state. Matching checker DSOs
were rebuilt after the history-layout change. Protected libraries remain paired
with their preserved cores.

All native simulations ran sequentially, using two or four MPI ranks, with at
most four simultaneous build/test workers. The 32² pairs retain label spacing
0.1 m, dt=0.05 s, final t=0.2 s and the common 0.1 s initialization convention.
The 64² pair uses dt=0.025 s and the same labels. The earlier 0.05 m-label runs
supply only the baseline audit/time-error context. The old fine reference's
2.78e-17 s terminal interval is excluded from physical analysis but retained in
its recorded cost. Retained output/checkpoint data were not rewritten.

## Actual update and ownership

The only new scientific selector is
`Material stress representation = direct physical stress`, valid with the
mature `Jaumann DG material Lie` route. The default is `separated`.
Constant G, positive constant or prescribed spatial eta, fixed affine Cartesian
bulk mesh, full-3D plane strain, growth disabled and the same material labels
remain the tested scope. Cohesive/native/lifting histories are unchanged.

The continuum law remains

```
(partial_t + v.grad) tau - W tau + tau W + (G/eta_b) tau
    = 2 G (dev_3 sym(grad v) - chi V S),  W = skw(grad v).
```

The direct attempt, from the immutable accepted state, is

```
tau_adv = A_h(tau_old)                      # existing DG SSPRK3
tau_bar = P_h[Q tau_adv Q^T], Q = exp(dt W_old)
beta = exp(-G dt/eta_b), kappa = -eta_b expm1(-G dt/eta_b)
tau_loc = beta tau_bar + 2 kappa (D(v_new) - chi_new V_new S_new)
tau_candidate = P_h(tau_loc).
```

Products use endpoint bulk eta and the existing common Gauss rule, never surface
eta substituted into the bulk. Accepted FE velocity/gradient supplies transport,
curve/profile/support motion and spin. Normals come from geometry. Relative slip
V is not label velocity. Accepted-trace inflow freezes accepted physical tau;
the separated comparator freezes T and U=R+P B. This is the same stated finite-
domain boundary approximation, not exact infinite-domain continuation.

`DGHistory::Representation::physical_stress` owns one xx/yy/xy tensor and no
residual owner; zz=−xx−yy. Stress access returns that tensor directly. Z remains
only a diagnostic surface recurrence; changing Z at fixed physical data changes
neither direct reaction coefficients nor observed stress. I, age, V, profile and
geometry still enter their original physical roles. No inferred BP3 prestress
is introduced. The native exponential reaction and incoming-age friction lag
remain; the manuscript's backward-Euler/endpoint-age alternatives are not used.
Finite rotation does not make this a finite-strain hyperelastic model.

`uses_projected_material_reaction()` dispatches both constant- and variable-eta
direct cases to the existing correction outer loop. The frozen local Maxwell
matrix is an inner tangent; the full projected law additionally differentiates
`Delta = P tau_loc - tau_loc`. The unchanged 24-iteration limit and
128-epsilon scaled correction bound apply before the original residual/KKT and
publication gate. No acceptance tolerance, loading, profile cap or extra
physical dissipation was introduced.

A real common solver defect was exposed: the final publishing call discarded
the outer solve's private converged V and restarted from accepted-parent V.
It now receives the converged rate guess. The original failed direct sheared
run and publication mismatch remain in `true-native-check`. Rollback preparation
also permits accepted direct observations without rotating them again.

The old/new layered separated result is **not** an arithmetic identity after
this fix: its original 1e-11 bulk / 1e-10 surface comparison bound fails, at
1.0833e-11 / 7.0338e-10. That failed bound is retained, not relaxed. A fresh
corrected separated layered control supplies the A/B comparison. The unchanged
constant separated control reproduces surface state exactly and bulk exports
within 3.8e-19; the default Q1/zero-inflow regression is within 1.5e-16 with
identical surface state.

All histories/maps publish together through the existing material transaction.
Schema7 direct identity adds a representation suffix and carries an empty
residual-coefficient packet; mismatched separated/direct loads are rejected.
The diagnostic CSV retains its legacy Txx/Tyy/Txy column names for coefficients;
`material-model.csv: direct=1` explicitly identifies them as physical tau, and
R columns are zero placeholders, not stored histories. Readers dispatch on that
metadata. The physical tauxx/tauyy/tauxy columns are always unambiguous.

## Independent verification

The retained 64² accepted-state audit reconstructs the actual continuous Q2
velocity / Q1 pressure test spaces, removes left/right/bottom Dirichlet velocity
rows, retains the natural top, and integrates tau−pI. It checks every free
mechanical row, continuity row, constrained reaction components, net force,
raw per-face resultants and interior DG jumps. At dt=.025/.0125, free mechanical
L2 residuals are 4.2380e-15 / 2.1190e-15 Pa m (relative to absolute assembled
contributions, 2.0745e-12 / 1.0370e-12). Continuity L2 residuals are
7.0604e-17 / 7.0694e-17 m²/s. Constrained reaction distributions are nonzero,
while net weak force is roundoff. Raw net x-force remains −7.7867e-6 /
−1.0479e-5 Pa m. Thus weak equilibrium verifies; strong DG face reconstruction
remains unresolved and is not replaced by a zero-power argument.

Native nonpublishing checks exercise the actual DG transport and projected
reaction, with independent componentwise rotation/relaxation and forced discrete
recurrence references. Maximum full-3D tensor error is 3.2302e-15 Pa. A prescribed
shear/translation/layered-relaxation test uses nonzero spin and specified split
inflow, with an independent characteristic reference. At8², dt=.05/.025 gives relative
errors1.5739e-3/1.0352e-3; at16²,1.4189e-3/7.3331e-4. These measure PDE errors,
not a roundoff identity or coupled localization accuracy.

At a common frozen geometry/profile/coefficients and prescribed nonzero v,V,
`direct - separated = (direct - P separated) + (P separated - separated)`.
The within-DG L2 difference is about 1.1e-16; the coarse 8² analytic remainder is
6.50e-4 (constant eta) / 6.40e-4 (layered eta), using the full-3D area norm.
The decomposition identity is exact to displayed precision. It does not
attribute differences between evolving trajectories solely to projection.

Finite-difference actions perturb both velocity and slip. Full projected source
action errors are below 1.7e-10 and 1.2e-11 in area L2, respectively, against the
predeclared 1e-8 bound. Frozen-inner versus full slip action differs by about
6.5e-4; the layered velocity action differs by 8.87e-5. Existing native affine
bulk/continuity action audits also pass with nonzero slip perturbations. These
checks distinguish the full projected derivative from the inner local matrix.

Independent Legendre projections at q8/q12/q16 reproduce the native frozen q8
candidate norm within 4e-15. On the deliberately coarse 8² mesh, q12−q8 and
q16−q12 projection differences are approximately 8.6e-5 and 3.8e-5: quadrature
sensitivity at finite localization ends is visible. Native q8 orthogonal source
projection loses energy; evaluating that same polynomial at q12 can give a
negative energy difference. Only the native inner product supports a
nonnegative-loss claim. This is not negative physical dissipation.

## Coupled comparisons

All pairs use bounded MacCormack, accepted tensor trace, full normal support,
identical initial material labels/loading/friction and unchanged solver bounds.
Their initial physical stress, profile, surface state and mechanical solution
match exactly. Separated fine results are numerical comparators, not truth.
Full tensor norms include zz and double xy; area L2 norms have units Pa m.
The table uses common Gauss8 tensor norms; point traction uses segment Gauss3,
and work traction uses the common-volume adjoint mass norm. Full component,
interior/fault-band, v/p, phi/I, age/slip/geometry metrics remain in JSON.

| Pair | Tensor difference | Point traction | Work traction | V difference |
|---|---:|---:|---:|---:|
| constant +, 32² | 2.2424% | 1.6292% | 1.2635% | 0.001205% |
| constant −, 32² | 2.2095% | 1.2755% | 1.2230% | 0.001145% |
| layered +, 32² | 2.1773% | 1.5043% | 1.2216% | 0.001133% |
| constant +, 64² | 1.5917% | 1.1525% | 0.6365% | 0.000613% |

The positive pair's full stress difference falls from 2.24% to 1.59%; the
central fault-band difference falls from 0.1726% to 0.04142%. More than 99.5% of
the squared difference lies within two cell widths of the finite curve ends.
This is a measured location diagnostic, not proof that the evolved difference
has only one cause. It is distinct from the unresolved cohesive tip-loading test.
At 32² the positive pair's velocity difference is 4.97e-11 relative, pressure
1.32e-6, profile 3.94e-12 and normalization 2.05e-12; geometry differs by
1.49e-11 m. Age differs by 2.67e-8 relative and cumulative slip by 7.65e-6.

Joint h/dt refinement changes each representation: direct coarse-to-fine tensor
difference is 3.23%, separated 3.56%; V differences are both about 0.1184%.
This is not an isolated temporal/spatial order or a surface-continuum test.
No label-grid change is hidden in the comparison. Independent final weak
mechanical residuals remain about 2e-12 relative to absolute assembled terms.
The original publication bounds pass, and accepted-state readers agree with
native stress access. Profile bounds/source checks pass; nonconservative profile
mass drift is retained in the reductions rather than credited to stress storage.

## Work, flux and numerical projection

Pressure-aware reductions retain branch storage, bulk and slip work, Maxwell
loss, friction/radiation, imposed background work, material-energy flux and weak
boundary reaction work. Raw unequilibrated face traction is diagnostic only.
Signed and accumulated absolute errors use an active-work scale, not the large
cancelling background/friction inventory. Trapezoidal values below are a
bulk-work diagnostic, not a replacement time integrator.

| Case / representation | Endpoint / active | Trapezoidal / active | Signed endpoint (J/m) | Absolute sum (J/m) |
|---|---:|---:|---:|---:|
| constant +, 32², direct | 17.2043% | 0.2997% | -1.338253e-09 | 1.338253e-09 |
| constant +, 32², separated | 17.2041% | 0.2999% | -1.338913e-09 | 1.338913e-09 |
| constant −, 32², direct | 17.0598% | 0.4013% | -1.332252e-09 | 1.332252e-09 |
| constant −, 32², separated | 17.0608% | 0.4001% | -1.332996e-09 | 1.332996e-09 |
| layered +, 32², direct | 16.2897% | 0.4238% | -1.271592e-09 | 1.271592e-09 |
| layered +, 32², separated | 16.2890% | 0.4237% | -1.272182e-09 | 1.272182e-09 |
| constant +, 64², direct | 8.7410% | 0.0718% | -6.901550e-10 | 6.901550e-10 |
| constant +, 64², separated | 8.7412% | 0.0724% | -6.903486e-10 | 6.903486e-10 |

The representation change barely changes the endpoint remainder. It does not
repair its roughly first-order timestep contribution, finite-domain inflow,
profile mass drift or raw DG face traction. No new 1% energy gate is imposed.

The optional native stage ledger separately records accepted→advected,
advected→rotated/projected, and local reaction→projected reaction energies.
Accepted q8 field energy independently reproduces the final ledger value.
For the constant cutback-control sequence, source projection loses 2.4188e-12
J/m, 0.03281% of active work; for the complete layered prefix/restart sequence,
8.6205e-13 J/m, 0.01104%. Rotation/projection changes are at most about 4e-22
J/m in these nearly translational coupled cases. Transport net energy changes
are +4.10e-12 / +5.66e-12 J/m and include boundary exchange and numerical DG/time
error; they are not named physical dissipation. Existing independent native
transport volume/face/upwind identities are reused. These separately recorded
projection changes are much smaller than the endpoint-work remainder.

## Cost and storage

Timings are indicative short local runs, not a speed benchmark. Ranks/output
settings match within each pair (two ranks at 32², four at 64²). Native source
checks and the optional stage ledger are excluded from these timing pairs.

| Case / representation | Wall s | Linear solves / iterations | Tensor coefficients | Complete checkpoint MB |
|---|---:|---:|---:|---:|
| constant +, 32², direct | 58.29 | 21 / 775 | 27,648 | 6.207 |
| constant +, 32², separated | 35.33 | 17 / 635 | 55,296 | 6.424 |
| constant −, 32², direct | 58.62 | 21 / 775 | 27,648 | 6.207 |
| constant −, 32², separated | 29.93 | 17 / 635 | 55,296 | 6.424 |
| layered +, 32², direct | 58.80 | 21 / 778 | 27,648 | 6.208 |
| layered +, 32², separated | 64.50 | 17 / 638 | 55,296 | 6.451 |
| constant +, 64², direct | 243.22 | 33 / 1269 | 110,592 | 24.679 |
| constant +, 64², separated | 120.54 | 27 / 1056 | 221,184 | 25.996 |

Direct owns 27 coefficients/cell versus 54 for T+R. At 32²/two ranks, the
measured owned/relevant coefficient totals are 27,648/29,376 versus
55,296/58,752: ghost coefficients are counted separately. Vector-reported
allocations are 353,760 versus 707,520 bytes. Raw owned doubles alone are
221,184 versus 442,368 bytes. Surface Z/I entries are unchanged and replicated;
age, V, curve/support data, Q1 phase/obstacle and accepted bulk velocity remain
necessary. Fixed particles remain derived diagnostic samples, not incoming
stress owners. Hence neither total checkpoint size nor process memory halves.

The positive coarse pair's summed per-rank peak RSS is approximately 538/540 MiB
(direct/separated), including libraries, matrices, FE vectors, particles and
checker allocations; it is not simultaneous total RSS. Fine direct peak RSS is
about 1305 MiB over four ranks; the old fine comparator did not record this
metric. Do not infer a fine total-memory ratio. The final checker counts retained
sample/source caches separately; direct accepted snapshots have zero such
entries and no live trial anchor/reaction. Trial owners are additional scratch:
accepted tensor, transported/rotated anchor, and previous/new reaction candidates
can coexist. Standard FE spaces, maps and solver caches remain common.

For the positive coarse direct run, logged tensor advection is 0.322 s,
rotation/projection 0.0856 s, and reaction projection 0.313 s. The separated
rotation/projection timer also includes constructing projected R+B, so it is
not a pure rotation-kernel comparison. Extra native assembly/residual work,
not tensor transport, dominates constant-eta direct overhead. Layered separated
already needs a correction loop and direct is comparable in this short pair;
no universal speed advantage is claimed. Complete per-section timers and owner
counts are retained in the comparison JSON.

## Lifecycle, regressions and limits

A completed direct trial is rejected and cut back once. Its accepted sequence
0, .1, .15, .2 s matches the no-rejection schedule **bitwise** in exported bulk
and surface data; the native entry snapshot also verifies full rollback before
the retry. No rejected state is published. A layered two-rank checkpoint at
.1 s resumes on four ranks and continues through .15/.2 s, with five/four labels
crossing cells. Against uninterrupted evolution, maximum bulk/surface errors
are 3.16e-15 / 7.11e-15. Both representation-mismatch directions reject before
continuation. Default and unchanged constant controls pass as described above.
The changed layered final-gate comparison's failed arithmetic bound remains
explicit; its fresh corrected control is the paired reference.

Final full unit suite: **183170 assertions in152 test cases pass** (CTest `unit_tests`,8.40 s). This is the complete unit suite, not the deferred repository-wide numerical campaign.

Implemented and locally verified are distinct from physically qualified or
production-ready. This experiment supports the narrow transported mature owner,
its projected residual and lifecycle. It does not establish moving AMR, birth,
resampling, deformed cohesive shape work, cohesive energy/tip qualification,
production restart, or a full earthquake trajectory. BP3 production history and
its background-load conventions remain separate. The inherited cleanup losses
remain: 39 of45 original native-pilot frames and earlier temporary logs are unavailable;
no deleted campaign was recreated. BP3's accepted step 4237 versus latest
retained spatial step 4182, and the RSF marker2/step748 baseline, are unchanged.
The paper roadmap remains BP3, controlled viscoelastic crust and propagation;
no realistic subduction example is planned and no deferred campaign was launched.

The following recommendation was subsequently completed in
[FROZEN_FINITE_END_REPORT.md](FROZEN_FINITE_END_REPORT.md); it found that the
uncut DGQ2 approximation dominates quadrature error. Native adoption status
remains unchanged. The recommendation at this report's closeout was a
**common-frozen finite-curve-end spatial and
quadrature study**, using identical labels/profile/material/source data and
independent dense integration. Separate DG approximation of analytic localization
from cut-cell quadrature and test point/work traction near ends. Keep central
and end windows distinct. Do not infer a new tip model, add smoothing, or launch
a production run. The broader prior accuracy campaign remains paused until a
new user selection; the preserved separated controls and incoming source archive
provide its recovery boundary.
