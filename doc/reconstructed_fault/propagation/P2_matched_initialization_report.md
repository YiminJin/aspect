# P2 matched initialization and positional accuracy

**The 512-cell Q2 case meets the retained practical spatial targets over
0–0.625 s, but band position is not uniquely qualified.** With a common
continuum initialization, refinement from 512 to 1024 phase cells reduces the
maximum centroid excursion from 0.739 to 0.262 mm. The ordinary phase difference
is 0.00208; aligning centroids for analysis reduces it to 6.85e-5. Traction,
slip, normalization and storage agree much more closely than position. The
translation-like curvature weakens with refinement. These results favor discrete
front/obstacle pinning over a resolved, persistent physical asymmetric branch;
they do not prove a globally admissible neutral translation family.

**Recommendation:** proceed to a separately selected, default-off native P2
integration of the consistent-endpoint prototype on 16×512 Q2 cells. Use the
shared initialization, immutable incoming histories/obstacle, single publication,
and a minimizer capable of rejecting/escaping admissible saddles. Verify the
native operators and matched fields first; accept either reflected diagnostic
branch without a centering constraint. Do not claim a unique position, general
localization, transport, or P3. This task implements no production integration.

## Actual model and initial state

| Setting | Value and units |
|---|---|
| Domain / fault | Unit square [0,1]² m; fixed through-going y=0.5 m; full normal association from initialization |
| Spatial controls | Q1 phase, 511 / 512 / 1024 normal cells; exact bulk mechanics; six normal Gauss points per cell |
| Retained FE comparison | Q2 velocity / Q1 pressure / Q1 phase, 512 normal cells; tangentially uniform reduction corresponding to 16 tangential cells |
| Surface / particles | 64 Q1 fault segments; native 3×3 particles/cell, frozen; controls inventory three transverse H samples/cell, without transport or birth |
| Elastic / viscous | G=10⁶ Pa, eta=10⁸ Pa s, lambda=100 s; viscosity bounds 10⁶–10¹⁰ Pa s |
| Cohesive / phase | c=1000 Pa, Gc=26.6666666667 J/m², ell=0.15625 m, p=1, Lorentz m=128; Hc=0.5 Pa, A≈64 Pa |
| Friction / pressure | mu=0.1, prescribed normal pressure 10000 Pa; background pressure and radiation damping zero; no RSF evolution |
| Loading / time | ux=.256 t(y−.5) m/s, uy=0; interval-mean boundary rate; dt=1/1024 s, 640 steps to .625 s, imposed displacement difference .05 m |
| Boundaries | Compatible sides: uy=0, ux natural; prescribed top/bottom velocities; natural phase boundary conditions |
| Initial mechanics | Unloaded R=Z=slip=0, bulk Maxwell stress and cohesive traction zero |
| Evolution | Instantaneous discrete total-energy phase driving; immutable previous-physical-phase lower bound, upper bound 1; H+ diagnostic only |
| Normalization / fracture | Existing consistent surface representation; analytic Q1 transverse integral and its adjoint; unchanged CPDI-equivalent fracture operator |
| Numerical gates | Scaled phase KKT ≤1e-11; force/contact ≤1e-7 Pa; healing allowance 4.922e-12; unchanged projected-Newton/line-search rules |

G remains a **dimensional synthetic verification value**, not a material
calibration. All fresh controls use the same continuous physical ramp and seed.
Reported solved velocities/slip rates are interval quantities, not endpoint
instantaneous velocities. General exponential Maxwell relaxation is retained.

### Reference initialization and its correction

[The equations and algorithm](P2_matched_initialization_equations.md) define one
continuous seed/H prescription, including its actual H branch transition at
normal distance **0.308492189967 m**. A split-domain stationary BVP evaluates
this prescription without particle sampling. It retains the outside Hc branch
and continuity of phase/slope; it does not extend geometry or regenerate H from
the relaxed phase. Particles are subsequently sampled from that same prescribed
H definition for the initialization inventory; H does not drive the new
instantaneous-energy evolution.

The relaxed core is **0.600001937763**, a +1.94e-6 change. The maximum change
from the physical starting profile is **6.774e-5**, near y=.191437 m, principally
the resolved support transition/tail. Phase at the H transition is 6.870e-5.
The minimum is 1.544e-10; **no box correction or clipping is needed**.
A tighter BVP changes phase by 8.57e-13; doubling seed integration changes the
stationary field by 1.14e-11. These are field uncertainty estimates, not a
particle/Q1 equality residual or a claim about pointwise second-derivative
accuracy. The scaled BVP collocation residual is ≤1e-10.

| Initialization | Maximum Q1/reference difference | Nodal maximum | Initial fracture (J/m) |
|---|---:|---:|---:|
| Retained particle-relaxed 512 | 2.171e-5 | .6000152908 | 14.58040559 |
| Shared reference, 511 | 3.309e-5 | .5999688522 | 14.58012639 |
| Shared reference, 512 | 3.294e-5 | .6000019378 | 14.58012728 |
| Shared reference, 1024 | 8.239e-6 | .6000019378 | 14.58028222 |

The 511 mesh samples the center between nodes; its lower nodal maximum is a Q1
representation effect. The shared field is interpolated, not independently
re-relaxed on each mesh. All unloaded phase KKT defects are **zero with this
seed as lower bound**, and mechanics/contact are admissible. This is a protected
pre-existing band, not an unconstrained unloaded energy equilibrium. Different
Q1 obstacles remain an unavoidable discretization difference. We compare the
actual evolving fields; subtracting initial offsets cannot remove this effect.

## Whole-interval comparisons: shape versus position

Every matched control completes 640 stable accepted steps. First full-field
increase >1e-6 above its own initial field is bracketed by
**[.52734375,.5283203125] s** in every control; no onset alignment is applied.
Incoming R,Z,slip and phase obstacles remain unchanged throughout trials.

Maxima over **all** common accepted times, including initialization:

| Comparison, exact bulk unless stated | Phase | Aligned phase | Centroid difference (mm) | q (Pa) | Slip (m) |
|---|---:|---:|---:|---:|---:|
| Retained vs shared initialization, 512 | .0003753 | 2.171e-5 | .1071 | .01799 | 1.802e-8 |
| Shared 511 vs 512 | .001779 | 6.295e-5 | .4742 | .000168 | 1.682e-10 |
| Shared 512 vs 1024 | .002079 | 6.851e-5 | .5652 | .01781 | 1.784e-8 |
| Retained 512 FE vs shared 1024 exact | .002424 | 8.019e-5 | .6603 | .02717 | 2.944e-8 |
| Retained 512 FE vs same-init exact | 1.602e-5 | 1.507e-5 | .002719 | .009800 | 1.206e-8 |

Alignment is a coordinate pullback in **postprocessing only**, using the phase
centroid. No accepted field, reference strain, slip, normalization or obstacle
is translated. Boundary extensions in this comparison are constant at the
original trace; these tiny tails do not define a new physical boundary condition.
I, traction, slip and energy remain physical scalar comparisons; none is
recomputed with shifted mechanical histories. Localization centroids are also
saved separately; they track phase centroids
closely. The complete trajectories are in the
[centroid CSV](../../../benchmarks/reconstructed_fault/propagation/matched_initialization_evidence/centroid-trajectories.csv).

For shared 512→1024, max |delta chi| is .06861 m⁻¹, decreasing to .006126 m⁻¹
after alignment; max |delta I|=.02363 m, or .00916%. Bulk/cohesive/fracture
storage differences are 2.342e-5 / 4.179e-4 / 1.549e-4 J/m. For the retained
512 FE→shared 1024 comparison, ordinary/aligned chi errors are .07992/.006781
m⁻¹ (about .765%/.065% of the fixed final reference peak 10.4417 m⁻¹).
This profile-position error is distinct from surface normalization projection
error. Along-fault fields are constant in these controls, so consistent
projection reproduces the transverse integral; direct 12/24-point checks agree
at about 3e-16 relative. Neither positivity nor this uniform projection check
qualifies a general along-fault normalization field.

### Complete centroid trajectories and continuum interpretation

Phase centroid offset from y=.5 in mm:

| Time (s) | Shared 511 | Shared 512 | Shared 1024 |
|---:|---:|---:|---:|
| .546875 | ≈0 | ≈0 | .05926 |
| .5625 | .42155 | .00647 | ≈0 |
| .578125 | .65646 | .47225 | ≈0 |
| .59375 | .24234 | .71443 | .26152 |
| .609375 | ≈0 | .33209 | ≈0 |
| .625 | ≈0 | .05785 | ≈0 |
| Maximum excursion | .71134 at .576171875 | .73863 at .5927734375 | .26152 at .59375 |

Thus the centered endpoint of the 511 case conceals a substantial transient;
1024 also undergoes transient displacements and recentering. The retained 512
exact baseline peaks at .81448 mm, ending at .11476 mm. The shared definition
reduces that motion but does not eliminate placement dependence.

Only already retained first-saddle entries are replayed—no further microscopic
near-release brackets or gain searches. Physical generalized curvatures use
vᵀM_phi v=1 on the admissible critical space:

| Mesh | First discrete saddle (s) | lambda_min (Pa) | L2 cosine with −phi_y |
|---:|---:|---:|---:|
| 511 | .5498046875 | −.477864 | .999973 |
| 512 | .5625 | −.474366 | .999972 |
| 1024 | .5458984375 | −.152366 | .999994 |

These events are at different times, so the table is not a same-state convergence
rate. Independent derivative checks resolve curvatures within 2.5e-6 Pa.
At the **common** .59375/.625 s states, accepted reduced curvatures decrease
from 1.1709/1.3813 Pa on 512 to .2851/.3153 Pa on 1024. The next mode is about
340 Pa. The increasingly translation-like soft mode and smaller excursion are
consistent with diminishing discrete pinning; they do not resolve a non-neutral
asymmetric continuum branch.

In the continuum exact-bulk reduction, translation preserves I and fracture
energy for a localized profile that remains inside the domain. But the fixed
previous-phase obstacle excludes translation wherever a sloping contact would
be crossed; natural boundaries alone do not remove this restriction. The
reference has tiny nonzero tails, so an exactly translated *entire* profile is
not automatically admissible. The computed critical-space mode is effectively
translation inside the released band and vanishes at strict contacts. Neutrality
is conditional, not a blanket symmetry claim. A unique positional continuum
limit remains unqualified at these resolutions; imposing centering would hide
the issue rather than verify it.

## Complete retained 512 FE spatial qualification

Independent 24-point sampling, checked with 48 points, uses actual Q2 strain
and Q1 phase away from the six assembly points. The 24→48 changes are <1.7e-10 Pa in stress
metrics and <6.3e-14 J/m in storage. Fixed historical scales remain
1332.521975 Pa and .887807407 J/m; no near-zero startup denominator is used.
This is the full **FE prototype** case, not a new native coupled trajectory.

| Quantity, maximum over 0–.625 s | Measured | Retained target | Result |
|---|---:|---:|---|
| Full phase difference | .002424 | .01 | Pass |
| Cohesive traction difference | .02717 Pa | 10 Pa | Pass |
| Slip / final reference slip | .0000603% | .5% | Pass |
| Whole-section stress RMS / fixed scale | .7863% | 5% | Pass |
| Localization-weighted stress RMS / scale | 1.8888% | 5% | Pass |
| Bulk storage difference / fixed scale | .01012% | 5% | Pass |

Mean stress error is only .02943 Pa; oscillatory RMS is 10.4779 Pa. Final
independent extrema are 1097.623 / 1234.069 Pa; largest interface jump is
10.4856 Pa. Final bulk storage .7067213 J/m decomposes into .7066664 mean-stress
storage and 5.4893e-5 oscillatory storage. These sampled oscillations pass the
RMS target but have not vanished. Centroid, release timing, and unaligned
localization remain selection-sensitive. The stricter temporal phase target
of .001 is not a substitute spatial criterion and is not qualified here.

### Separate energy contributions

Full-interval signed residual works in J/m, same documented linear-phase /
exact-Maxwell diagnostic path and audited reaction work:

| Case | Mechanical imbalance work | Friction residual work | Phase residual work | External energy remainder |
|---|---:|---:|---:|---:|
| Retained 512 FE | .0129524 | −.00448019 | −.00840375 | −6.848e-5 |
| Shared 512 exact | .0129571 | −.00448232 | −.00840729 | −6.750e-5 |
| Shared 1024 exact | .0129448 | −.00448938 | −.00838739 | −6.801e-5 |

For retained FE, accumulated absolute path integrals are .0129525 / .00460085 /
.00840922 J/m; the largest absolute step contributions are 2.881e-4 / 8.753e-5 /
2.006e-4 J/m. These remain individually finite at this dt and nearly unchanged
by spatial refinement; cancellation is not energetic qualification. External
remainder sum of absolute steps is 7.899e-5 J/m. Storage changes are
bulk .7067213, cohesive 4.5987951, fracture 4.4051163 J/m. Bulk/cohesive viscous
losses are .00725189/.02093060 J/m; friction loss 48.8047300 J/m.
Bulk-profile/normalization/fracture path works are −.00836639/−4.40515363/
+4.40511627 J/m, retained separately. The accumulated **signed incoming-state
profile cost** is +4.2927592 J/m; it is neither phase dissipation nor an extra
compensating loss. No D_relax term is invented for this diagnostic path.

Exact-control chain-rule identities close within 2.6e-13 J/m; selected 16→32
path quadrature changes are below 1e-10 J/m. This verifies bookkeeping, not that
intermediate equations are equilibrated. Spatial storage error, residual work,
and profile transition interpretation remain different questions.

## Verification, cost and source changes

All controls pass bounds, no measured healing, endpoint KKT/contact/force,
immutable-anchor/single-publication checks and exact Maxwell recurrence.
Maximum KKT is 8.351e-12; maximum force is 1.792e-10 Pa. The original failed
states and **1315 preserved source/evidence files** retain their SHA256 hashes.
The whole-repository `git diff --check` still reports inherited trailing whitespace
in pf_rsf.tex; that incoming mathematical document was left untouched.
Runtimes for 511/512/1024 are 69.5/67.9/117.2 s, with peak RSS approximately
208/207/352 MiB. Timings cover evolution and diagnostics; reference generation
and constructor initialization precede the timer. Peak RSS covers the process. The algebraic
Hessian agrees with the original Schur construction to 3.06e-13; independent
gradient differences check it without changing the sampled operator.

| Changed/new file and functions | Purpose |
|---|---|
| matched_initialization.py: Seed, reference/check_reference, SharedReference.initialize, Exact.hessian, run | Common physical initialization, independent refinement checks, unchanged efficient exact-bulk tangent, fresh one-direction trajectories |
| assess_matched_initialization.py: baseline, compare, independent_stress, baseline_energy, assess | Reuse original fields, separate position/shape, full-interval off-quadrature stress and energy ledgers |
| check_matched_modes.py: main | Mass-normalized curvature/derivative replay of retained entries only |
| snapshot_visualization/snapshot_visualization.cc: AcceptedSnapshot::execute | Output-only registered ASPECT postprocessor, scratch scalar FE fields, compressed VTU/PVD on native mesh |
| snapshot_visualization/CMakeLists.txt; render_matched_snapshot.py: main | Separate test-plugin build and fresh no-solve native rendering fixture; no existing build reconfiguration |
| check_matched_visualization.py: decode/main | Decode compressed payloads, verify accepted values, size and metadata |
| Report/equations, STATUS, PLAN, CURRENT_STATUS, current_design, README, AGENTS | Record scope, evidence, qualification and next-task recommendation |

**Production source and governing specification equations are unchanged.**
No material sweep, new phase law, history conversion, kinetic term, generic
projector change, centering constraint or P3 work is included.

## ASPECT visualization output

Open [accepted.pvd](../../../benchmarks/reconstructed_fault/propagation/matched_initialization_evidence/aspect-visualization/accepted.pvd)
in ParaView and use the time controls. Five frames at 0/.53125/.5625/.59375/.625 s
contain phase, localization, shear stress, cohesive traction, cumulative slip,
velocity_x and velocity_y. Traction/slip are the uniform surface values extended
for display; localization and velocity use linear visualization patches.

These are **actual ASPECT postprocessor/deal.II DataOut outputs**, rendering the
accepted shared-512 exact-bulk prototype fields on the native 16×512 mesh. The
native runner uses `no Advection, no Stokes`, t=0 only. It imports no physical
history and does not claim a native coupled solve. Rendering takes 6.19 s / 323 MiB peak RSS;
compressed Float32 files are 152–157 kB each, **780605 bytes total**. Solver
fields remain Float64. Decoded phase differs by at most 2.98e-8, as expected
from display precision; other fields and metadata are independently checked.
Full build/input/binary hashes, timings, units and checks are in snapshot-native.json.
Two setup errors and an unnecessary mechanical setup interrupted before output
are retained separately; the final rendering fixture solves no physical equations.

## Reproduction and stopping point

Evidence is in benchmarks/reconstructed_fault/propagation/matched_initialization_evidence.
Scripts refuse to overwrite completed trajectories/VTU outputs. Initial generation:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/matched_initialization.py reference
# Run `run --cells 512`, `run --cells 511`, `run --cells 1024` into fresh evidence.
# Existing retained evidence is checked without trajectory advancement:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/matched_initialization.py check-reference
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/matched_initialization.py equivalence
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/p2-matched-mpl python3 benchmarks/reconstructed_fault/propagation/assess_matched_initialization.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/check_matched_visualization.py
```

Baseline remains worktree /home/ein/repository/aspect-fault-propagation,
fault-propagation, HEAD de7beaa65846c7372711f7922abf1c5d9a745ee6, with all incoming
changes preserved. Native integration is recommended, not started. Its first
scope should be operator/accepted-state validation against the matching discrete
prototype, followed by one bounded fresh comparison. A full 640-step native
run needs a cost budget: naive scaling of the older 34.4-minute 16×256/40-step
baseline gives roughly 18 hours before additional coupled iterations, not a
measured native estimate for this option. Coupled branch selection and general
Stokes localization must remain explicit limitations. No P3 readiness claim.
