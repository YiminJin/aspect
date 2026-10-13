# P2 phase-first single-pass versus consistent endpoint

The single-pass scheme is cheaper per run, but **does not achieve the required
physical accuracy at either tested timestep**. At 1/2048 s its final phase
maximum is .69025 instead of .75019, and traction is 225.68 instead of 188.83 Pa.
Both single-pass trajectories reach .625 s with converged incoming-history
phase subproblems, admissible monotone phase, converged endpoint mechanics and
correct once-per-step histories. Those checks do not remove the splitting error:
the completed-state phase defect reaches 89.1 Pa in the nodal-measure scale.

The timestep halving changes single-pass phase by .00412 and traction by 2.82 Pa,
but its differences from the coupled reference remain .05994 and 36.84 Pa.
Retain the coupled endpoint reference and reusable native MPI work. These two
levels do not establish the single-pass limit or its cost at a qualified timestep.
No native algorithm, physical equation, tolerance or constitutive parameter is
changed, and no native run, MPI optimization or P3 is undertaken.

## Configuration and evidence identity

| Quantity | Fixed setting / units |
|---|---|
| Domain and prescribed fault | Unit square, m; through-going y=.5 m fault; full-normal fixed association |
| Spatial model | Matching normal reduction of 16×512 Q2 velocity/Q1 pressure/Q1 phase; 1023 free ux(y) coefficients |
| Surface / particles | 64 consistent Q1 segments; 3×3 fixed particles/cell in the native fixture, 73728 total; no transport/birth |
| Integration | Six normal bulk points; unchanged AT1/CPDI fracture; analytic transverse normalization; independent 24-point stress sampling |
| G / eta / lambda | 10⁶ Pa / 10⁸ Pa s / 100 s |
| Viscosity bounds | 10⁶–10¹⁰ Pa s; unchanged |
| c / Gc / ell / p | 1000 Pa / 26.6666666667 J/m² / .15625 m / 1 |
| Calibration | c0=8/3, Hc=.5 Pa, m=128.00000000016; Gc/(c0 ell)=64.00000000008 Pa |
| Friction / pressure | Fixed mu=.1, prescribed normal pressure 10000 Pa; radiation damping and body force zero |
| Initial fields | Shared continuum-relaxed nominal-core-.6 seed, actual maximum .6000019377630632; zero FE d_R, Z, stress and slip |
| Particle H | Same independent seeded-H prescription and particle samples; never regenerated from relaxed phase; diagnostic in the total-energy option |
| Boundaries | Top/bottom ux=.256 t(y−.5), uy=0; compatible sides uy=0, ux natural; natural phase boundary condition |
| Time / loading | Fresh 0–.625 s, dt=1/1024 and 1/2048 s; interval-mean boundary velocity; imposed displacement difference .05 m in every run |
| Phase / histories | Previous physical phase is the immutable lower obstacle, upper bound 1; R_a/Z_a immutable through trials; commit once after the complete step |
| Prototype acceptance | Scaled projected phase residual ≤10⁻¹¹; mechanical/contact defect <10⁻⁷; existing −4.922×10⁻¹² healing allowance; no saturation |
| Parallelism | One BLAS/OpenMP thread; two new prototype trajectories, sequential; no native benchmark job |

G=10⁶ Pa remains a dimensional synthetic verification value. Along-fault fields
are uniform in this rich normal reduction: center and side-endpoint surface q/V/
slip are identical. It does not reproduce additional native tangential modes or
qualify fault propagation. The constant-along-fault normalization is exactly
represented by the retained consistent projection; this comparison does not
change projection type or constitute another general normalization-accuracy test.

Existing native_endpoint_evidence/dt-1-1024 and dt-1-2048 trajectories are reused.
All four phase, FE, Z, particle-coordinate and H initialization arrays are
numerically identical. Older retained NPZ files obtain their time labels from
their original accepted history records, without shifting a trajectory or
changing histories. The older 256-cell single-pass family is historical evidence,
not the primary comparator, because its mesh and sampled initialization differ.

Baseline is 33c0adba59132890b386f3e984b3e1650feffe44 on fault-propagation, with
the incoming uncommitted MPI implementation preserved. Evidence, exact hashes
and arrays: benchmarks/reconstructed_fault/propagation/single_pass_evidence/.

## What changes in the algorithm

Accepted state is (phi_a, d_R,a, Z_a, slip_a). In this shear reduction,
tau=G D d_R−Z chi, chi=h(B phi)/I, and

\[
 E_h=\frac{1}{2G}\tau^TW\tau+\frac{Z^2}{2GI(\phi)}+F_h(\phi).
\]

F_h is the same discrete fracture functional. I and DI use the same analytic
transverse integral, corresponding here to exact constant reproduction of the
consistent surface projection and its adjoint.

| Operation | Phase-first single pass | Coupled endpoint reference |
|---|---|---|
| Phase equation | KKT of E_phi(phi_b,d_R,a,Z_a), with phi_b≥phi_a | KKT of E_phi(phi_b,d_R,b(phi_b),Z_b(phi_b)), with the same bound |
| Mechanics | One endpoint solve after the fixed-history phase minimization | Endpoint mechanics recomputed from the same incoming state at every phase trial |
| Maxwell recurrence | d_R,b=beta d_R,a+(kappa/G)u; Z_b=beta Z_a+kappa V | Identical |
| Coefficients | beta=exp(−dt G/eta), kappa=eta(1−beta); no small-dt expansion | Identical |
| Search / publication | Existing fixed-history L-BFGS/true-energy Newton polish; commit once | Existing reduced minimizer and admissible saddle escape; commit once |

The cached Gram factor from the existing endpoint prototype is used for the
single mechanical solve. This avoids charging single-pass for redundant dense
factorizations while preserving its equation and endpoint. At .5/.5625/.625 s,
comparison with the original dense mechanical kernel gives errors ≤7.71×10⁻¹⁶ m
in FE history, 1.50×10⁻⁹ Pa m in Z, 1.07×10⁻¹¹ Pa in q and 1.54×10⁻¹² m/s in V.
The fixed-history phase problem's sampled enlarged-critical-space mass curvature
is positive: 719.4 and 673.7 Pa at .5625 and .625 s; at .5 s that space is empty.
This is distinct from mechanically reduced coupled stability.

## Physical accuracy and onset

The reference is the retained 1/2048-s endpoint trajectory. Its uncertainty is
assessed by the existing 1/1024→1/2048 endpoint comparison: maximum phase
difference 5.36×10⁻⁵, traction 2.57×10⁻⁶ Pa and relative slip 2.95×10⁻¹⁰.
All seven reference targets pass. No field is recentered or reflected. Coupled
centroid motion reaches .742 mm; single-pass remains symmetric. The .060 core
error cannot be explained by this positional difference or the reference error.

| Maximum common-time error | Target | Single 1/1024 | Single 1/2048 |
|---|---:|---:|---:|
| Full-field phase | .001 | .062605 | .059939 |
| Cohesive traction, Pa | 1 | 39.661 | 36.841 |
| Cumulative slip / final reference slip | .1% | .08130% | .07552% |
| Common-interval rate / fixed reference peak | .5% | 1.5226% | 1.3913% |
| Whole-section stress RMS / fixed stress scale | 1% | 3.0304% | 2.8217% |
| Localization-weighted stress RMS / fixed stress scale | 1% | 3.3138% | 3.1207% |
| Bulk storage / fixed storage scale | 1% | 5.3939% | 5.0041% |

Fixed scales are 1332.521975 Pa and .887807407 J/m, not nearly zero startup
values. Slip/rate use the finest endpoint's final slip and peak interval rate.
Each rate comparison uses [slip(b)−slip(a)]/(b−a) on the same physical interval.
Charts also use common 1/64-s windows. Returned V is an interval rate chosen to
satisfy endpoint contact, not an instantaneous endpoint velocity measurement.

| Scheme / dt | First full-field growth bracket, s | First core growth bracket, s |
|---|---|---|
| Endpoint 1/1024 | [.52734375,.52832031] | [.52832031,.52929688] |
| Single 1/1024 | [.52832031,.52929688] | [.52929688,.53027344] |
| Endpoint 1/2048 | [.52734375,.52783203] | [.52832031,.52880859] |
| Single 1/2048 | [.52783203,.52832031] | [.52880859,.52929688] |

The fixed onset threshold is 10⁻⁶ above each identical initial field. The onset
delay is one step, but later response differences are much larger than one
resolved phase increment. Halving single-pass reduces its largest increment
.00109686→.00054955; that alone does not establish temporal convergence.
Its full-field halving difference .004120 and q difference 2.820 Pa still exceed
their targets. Relative to endpoint, phase error falls only 4.3% and q error 7.1%.
Two levels do not justify extrapolating a convergence order or a practical smaller dt.

At T=.625 s the fine single-pass state has q=225.675 Pa, slip=.0487678733 m,
bulk/cohesive/fracture storage .751148/5.491840/18.311761 J/m. The endpoint state
has q=188.834 Pa, slip=.0488047301 m, storage .706721/4.598791/18.985557 J/m.
Single-pass has about 40% less core growth above the .600002 seed, despite a
small cumulative-slip difference. Its traction–slip curve remains materially stronger.

Incoming phase KKT maxima are 5.04/5.55×10⁻¹², below the unchanged 10⁻¹¹ target.
Endpoint phase defects are .002628/.002720 in the scaled nodal-gradient norm,
or 86.1/89.1 Pa divided by nodal measure; the final worst location is y=.5 m.
Those are splitting diagnostics, not failures of the converged fixed-history
phase subproblem. The completed mechanical/contact problems converge, bounds
hold, no healing occurs and I≥108.648896 m. FE recurrence errors are ≤10⁻¹⁶ m.

## Energy: phase update and subsequent fixed-profile mechanics

For the actual single-pass split, let

\[
 \Delta E_\phi=E_h(\phi_b,d_R^a,Z_a)-E_h(\phi_a,d_R^a,Z_a),
 \qquad D_{\rm relax}=-\Delta E_\phi.
\]

This measured minimization release is not an additional constitutive dissipation.
After redistribution, hold phi_b/chi_b/I_b fixed and integrate the general-beta
Maxwell path from the same incoming d_R,a,Z_a. Let F=D^T W tau and
r_Gamma=mu P+Z/I−chi^T W tau. Boundary-reaction work is external work. The exact
spatial chain rule is

\[
 \dot E_h=P_{\rm ext}-P_{\rm visc,b}-P_{\rm visc,\Gamma}-\mu PV
                 +u_f^TF_f+r_\Gamma V.
\]

Define M=integral(u_f^T F_f dt) and C=integral(r_Gamma V dt). Therefore

\[
 W_{\rm ext}-(\Delta E_h+D_{\rm visc,b}+D_{\rm visc,\Gamma}+D_f)
       =D_{\rm relax}-M-C.
\]

The unadjusted left side is retained. The fixed-profile mechanical remainder is
that quantity minus the explicitly measured D_relax, and equals −M−C. This is
bookkeeping of two separate update stages, not a compensation added to a loss law.
Bulk-rate work differs from external work by M. Affine-lifting work nearly equals
reaction work in this fixture (cumulative difference ≤2.4×10⁻¹⁰ J/m); equality
is not assumed for arbitrary interior lift extensions or other geometries.

| Actual split path, J/m | Single 1/1024 | Single 1/2048 |
|---|---:|---:|
| Bulk profile-energy change at fixed incoming history | +.010071 | +.015523 |
| Cohesive profile-energy change | −3.719062 | −3.769193 |
| Fracture expenditure | +3.685995 | +3.731634 |
| Phase release D_relax | .022997 | .022035 |
| Bulk / cohesive viscous loss | .007301 / .021744 | .007297 / .021691 |
| Frictional loss | 48.765051 | 48.767873 |
| Unadjusted external energy remainder, signed / sum abs | .046160 / .046280 | .044216 / .044222 |
| Fixed-profile mechanical remainder, signed / sum abs | .023163 / .023283 | .022181 / .022186 |
| Mechanical residual work M, signed / integral abs | −.132819 / .132870 | −.128912 / .128934 |
| Friction residual work C, signed / integral abs | +.109656 / .111314 | +.106731 / .107164 |

Maximum individual external remainders decrease .00073694→.00043505 J/m, but
cumulative magnitudes decrease only about 4%. Signed M and C cancel substantially;
their absolute values must remain visible. This does not establish vanishing
phase release or mechanical imbalance. No phase viscosity or compensating loss
is introduced. Stepwise identity closure is ≤1.4×10⁻¹³ J/m; doubling the selected
reaction/residual time-integration rule from 16 to 32 changes those terms by
≤8.4×10⁻¹⁷ J/m. Work/loss evaluation otherwise retains the existing 12-point
general-exponential Maxwell integration; no time-law approximation is introduced.

For a common comparison, both trajectories also receive the existing reconstructed
continuous-path audit: linear phi(t), exact Maxwell d_R(t),Z(t), and chi(phi(t))
with its complete normalization derivative. It includes −Z dot(chi) in stress
evolution and E_phi dot(phi) in the chain rule. Its intermediate states need not
satisfy equilibrium; mechanical, friction and phase residual works are measured.
This audit is diagnostic, not a substitute path or a second physical update.

| Continuous-path audit, J/m | Single 1/1024 | Single 1/2048 | Endpoint 1/1024 | Endpoint 1/2048 |
|---|---:|---:|---:|---:|
| External remainder, signed | +.046123 | +.044198 | −.00006752 | −.00000701 |
| Sum abs step external remainders | .046244 | .044203 | .00007803 | .00000966 |
| Integral abs mechanical residual work | .00078669 | .00023749 | .01294957 | .00323756 |
| Integral abs friction residual work | .00080088 | .00021569 | .00460152 | .00113797 |
| Integral abs phase residual work | .04622969 | .04422508 | .00840584 | .00210549 |

The single-pass common-path phase contribution dominates and barely decreases.
The endpoint reference's separate residual works decrease by about four on
halving, even though the signed remainder benefits from cancellation. No
D_relax is inserted into its continuous-path ledger. Positive incoming-state
profile cost for coupled trials is not relabelled as dissipative release.

## Cost, changes and decision

| Scheme / dt | Whole run, s | Peak MiB | Phase / mechanics / diagnostics, s |
|---|---:|---:|---|
| Single 1/1024 | 40.46 | 138.8 | 19.29 / 3.26 / 15.23 |
| Single 1/2048 | 71.75 | 168.4 | 34.20 / 5.97 / 28.39 |
| Endpoint 1/1024, reused | 279.94 | 169.2 | Detailed scopes not retained |
| Endpoint 1/2048, reused | 383.92 | 185.3 | Detailed scopes not retained |

Single-pass uses 640/1280 mechanical solves, 23630/43867 phase objective calls
and 101/202 analytic phase Hessians. Whole-run ratios are about 6.9/5.4, with
the qualification caveat: endpoint timings are retained historical measurements
including its stability checks, not a contemporaneous controlled speedup test.
Single-pass's extra two-path audit is charged to its measured cost. There is no
native cost claim. Cheaper converged subproblems do not supply the required
phase, traction, rate, stress or storage accuracy here. Only cumulative slip
passes all tested physical targets.

| Changed file / function | Purpose |
|---|---|
| single_pass_comparison.py: run | Reuse shared initialization, existing fixed-history phase solver and cached endpoint mechanics; immutable history and one publication |
| single_pass_comparison.py: split_ledger | Separate actual phase-energy change, fixed-profile work/losses and residual contributions; no changed constitutive law |
| assess_single_pass.py: comparison, controls | Unaligned common-time fields, identical-interval rates, bounds/KKT/recurrence and fixed reference scales |
| assess_single_pass.py: checks, plot | Dense/cached mechanical equivalence, selected fixed-history curvature and quadrature checks; compact scientific charts |
| This report, STATUS/PLAN, CURRENT_STATUS, design/specification, AGENTS and benchmark README | Record selected scope, algorithm/path distinction, accuracy failure and preservation |
| Production source / CMake | **No changes in this task**; incoming native/MPI code and reusable evidence preserved |

The first-run diagnostic release-roundoff guard was corrected to use the actual
incoming energy rather than a scratch energy left by the continuous-path audit.
Both saved trajectories independently pass the corrected guard. This changes
no solve, history, energy value or physical acceptance criterion. Assessment also
handles the retained NPZ format and converts NumPy booleans for JSON output.

Recommendation: retain the consistent-endpoint reference. This bounded comparison
does not justify replacing it or launching a cheaper native single-pass trajectory.
A sufficiently fine single-pass level might eventually qualify, but its timestep
and cost are unresolved by these two runs. Further timestep or solver work needs
a separately selected task; the MPI implementation remains available unchanged.

[Physical histories and traction–slip](../../../benchmarks/reconstructed_fault/propagation/single_pass_evidence/physical-comparison.png),
[phase profiles](../../../benchmarks/reconstructed_fault/propagation/single_pass_evidence/phase-profiles.png),
[onset and common-window rates](../../../benchmarks/reconstructed_fault/propagation/single_pass_evidence/onset-comparison.png),
[independently sampled stress](../../../benchmarks/reconstructed_fault/propagation/single_pass_evidence/stress-profiles.png),
[splitting defect and recorded cost](../../../benchmarks/reconstructed_fault/propagation/single_pass_evidence/defect-and-cost.png),
[actual split-path energy](../../../benchmarks/reconstructed_fault/propagation/single_pass_evidence/split-energy.png),
[common continuous-path audit](../../../benchmarks/reconstructed_fault/propagation/single_pass_evidence/continuous-energy.png).

Commands (outputs already exist and are protected):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/single_pass_comparison.py --denominator 1024
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/single_pass_comparison.py --denominator 2048
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/p2-native-mpl python3 benchmarks/reconstructed_fault/propagation/assess_single_pass.py
```

The complete accepted arrays, maximum/signed/absolute step diagnostics, initial
identity checks and hashes are retained. Native regressions are reused because
production and the existing observer are unchanged; no additional ASPECT run is
needed for this prototype-only comparison. P2 full-native closure and P3 remain
unqualified.
Use a fresh `--output` destination to repeat a single-pass case without
overwriting its retained trajectory.
