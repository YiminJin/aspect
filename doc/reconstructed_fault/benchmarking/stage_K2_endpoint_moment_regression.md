# K2 endpoint-moment regression and correction proposal

2026-09-09. This follows the accepted dominant-contribution diagnosis in
`stage_K2_endpoint_diagnosis.md`. It is a manufactured quadrature regression,
not a corrected mechanical trajectory or completion of K2.2.

**Outcome:** the current point-volume rule reproduces constants but has
displacement-dependent endpoint errors for smooth, nonconstant transverse
stress. Changing domain areas alone is not a reliable correction. Integrating
the surface Q1 test/trial moments over the unchanged wall-clipped domains
recovers the independent mass matrix and gives decreasing traction errors in
both manufactured fields. This supports a **surface quadrature revision for
review**, not a particle-domain construction or history-transfer correction.
No production rule, parameters, histories, support, I_h, endpoint topology or
solver tolerances were changed. No mechanical replay was run in this step.

## 1. Controlled experiment and independent reference

Use the physical K2 influence strip

    Omega = [0,L] x [-w,w], L = .25 m, w = .3088215939070757 m.

For bulk-resolution label nx, particle spacing is a_x=L/(3 nx), and fault
spacing is d_s=2 L/nx=6 a_x, as in the saved runs. Choose
ny=round(2w/a_x), a_y=2w/ny and initially centered rectangular particle rows.
Apply the controlled shear displacement

    x_p = x_p,0 + rho a_x y_p/w,    y_p = y_p,0.

All tested rho are below .5, so no particle crosses a tangential wall. The
manufactured particle-domain partition ends exactly at the strip's normal
edges. This intentionally removes normal-edge particle-admission/tail error:
the production bulk box extends farther, to y=+/-.5 m. Thus this experiment
isolates tangential endpoint moments on the same integration strip; it does
not validate a new normal-support clipping or association policy.

The prescribed stresses are

    q_even(y) = 1000 + 5 [(y/w)^2 - 1/3] Pa,
    q_odd(y)  = 1000 + 5 y/w Pa.

Both have exact transverse mean 1000 Pa at every x. Values are sampled at
the displaced points (y is unchanged). There is no constitutive evolution,
FE history transfer, initialization solve or stress smoothing in this test.

The independent reference integrates the prescribed polynomial over Omega,
not a finer version of the particle projection. For each surface segment,

    M_ref,local = (2w d_s/6) [[2,1],[1,2]],
    b_ref = 1000 M_ref 1,        q_Gamma,ref = 1000 Pa.

The centered weak load for q-1000 is exactly zero, including endpoint and
interior test functions. Loads have units Pa m^2 and mass entries m^2 (unit
out-of-plane thickness). We report centered loads as well as full loads;
otherwise the large constant component can mask the nonconstant error.
Traction RMS uses the reference mass norm divided by strip area, not a
Euclidean nodal norm. Endpoints remain independent finite-fault DoFs.

Saved first-column particle data in the admitted strip give these maximum
tangential displacements from the initial particle positions:

| nx | max displacement/a_x at .5 s | at 1 s |
|---:|---:|---:|
| 32 | .0227768 | .0962324 |
| 64 | .0456200 | .1928593 |
| 128 | .0912760 | .3859266 |

The controlled ratios 0, .05, .1, .2 and .4 bracket this range. The cross
(nx,rho)=(32,.1),(64,.2),(128,.4) holds physical maximum displacement fixed
at .000260416667 m. It approximates the saved shear magnitude, not the exact
saved displacement field; it is not fitted to a solver stopping residual.

## 2. Rules compared and geometry verification

1. **Current point-wall rule:** actual wall-clipped Voronoi area times
   N_i(x_p) and N_i(x_p)N_j(x_p), exactly the authoritative surface formula.
2. **Interior-area counterfactual:** same point coordinates, but use a_x a_y
   for every area. This isolates the contribution of boundary-modified
   weights; it is not a proposal to freeze production volumes.
3. **Domain-integrated rule:** keep the same wall-clipped polygons and q_p;
   split polygons at fault-node planes and integrate Q1 first/second moments
   exactly. q_p remains piecewise constant on its own domain. The remaining
   nonconstant-field error therefore includes transverse stress sampling.
4. **Boundary/crossing-only integration:** integrate domains touching walls
   or crossing fault-node planes; retain the current point rule elsewhere.
   This tests whether a smaller exceptional treatment suffices.
5. **Periodic-area diagnostic:** independently build actual tangentially
   tiled Voronoi domains, but keep the finite-fault point-coordinate rule.
   Wrap polygon pieces onto the same physical strip to verify its coverage
   and moments. This is offline only: no production periodic flag is enabled.

The geometry calculation uses reflected seeds to construct finite-wall
Voronoi polygons with SciPy/Qhull. Reflections enforce walls, not periodicity.
A separate executable cross-checks all 22,752 areas of the nx32, rho=.4
cloud against the Voro++ library used by ASPECT, with all periodic flags
false. Maximum relative area difference is **3.75255e-14**; total area is
.15441079695345655 m^2, within 8.2e-14 m^2 of the analytic strip area.
This verifies the finite-wall geometry independently, not the entire
distributed ParticleDomainHandler path. The point assembly also matches the
existing saved-data projection kernel in a direct test.

## 3. Results: constants are necessary but not decisive

All compared matrices recover a 1000 Pa constant from b=1000 M1 to
approximately 1e-13 Pa. The regression also independently assembles constant
loads through the point and full-domain routines and verifies this identity
and their represented constants. **Constant reproduction does not certify
the weak mass or nonconstant weak loads.**

For the even stress at fixed physical displacement:

| nx | rho | Current maximum traction error (Pa) | Domain-integrated maximum error (Pa) | Current interior error magnitude (Pa) |
|---:|---:|---:|---:|---:|
| 32 | .1 | 5.83047e-4 | 3.03426e-5 | 2.97545e-5 |
| 64 | .2 | 2.21960e-3 | 8.08725e-6 | 7.41810e-6 |
| 128 | .4 | 8.85504e-3 | 2.51110e-6 | 1.85062e-6 |

The point-rule endpoint error grows while the interior error decreases.
With nx32 fixed, its maximum errors at rho=0,.05,.1,.2,.4 are
2.96724e-5, 1.68010e-4, 5.83047e-4, 2.24353e-3 and 8.89108e-3 Pa.
After accounting for the zero-displacement midpoint sampling error, this is
consistent with the near-quadratic displacement-ratio effect inferred from
the saved runs. It is not a general convergence law.

At nx32, rho=.4, even stress, the left endpoint and interior control show:

| Quantity | Current left endpoint | Domain-integrated left endpoint | Current midpoint |
|---|---:|---:|---:|
| Traction error (Pa) | -8.89108e-3 | -4.01408e-5 | -3.08518e-5 |
| Centered weak load (Pa m^2) | -1.92311e-5 | -1.72544e-7 | -2.86358e-7 |
| Full weak-load error (Pa m^2) | -1.43162e-2 | -1.72544e-7 | -2.86358e-7 |

The current mass relative Frobenius error is 3.61944e-3; full domain moments
reduce it to 1.32301e-15. Boundary/crossing-only integration gives a small
traction error (4.02503e-5 Pa) but leaves a 6.58649e-3 mass error from the
point evaluation of quadratic products in the remaining domains. It does
not recover the independent weak mass, despite improving this traction test.

### Separating boundary areas from point-coordinate quadrature

At these same fixed coordinates, replacing only wall-modified areas changes
M by 4.41949e-5 m^2 in Frobenius norm and the centered load by
3.81458e-5 Pa m^2 in vector norm. Keeping the actual wall domains and replacing
only point quadrature by domain moments changes M by 9.74151e-5 m^2 and the
centered load by 3.81172e-5 Pa m^2.

These are **different controlled comparisons, not additive contributions**.
The exact additive split against the reference is

    point - reference = (point - domain moments) + (domain moments - reference).

The first term is the point-coordinate quadrature defect on the same domain
partition; the second retains the error of representing smooth q by q_p on
each domain. The area-only comparison tests a candidate boundary mechanism
without conflating it with that split.

For the even field, periodic/interior areas appear successful: the maximum
error drops to 2.96724e-5 Pa. But the **odd field decisively rejects an
area-only correction**:

| nx, rho=.4 | Current wall rule (Pa) | Actual periodic-area point rule (Pa) | Domain-integrated rule (Pa) |
|---|---:|---:|---:|
| 32 | 4.14971e-2 | 3.86832e-1 | 8.74710e-6 |
| 64 | 4.14968e-2 | 3.86838e-1 (interior-area diagnostic) | 2.19346e-6 |

The explicit periodic-domain construction was checked at nx32; its areas
agree with interior areas to 2.96e-14 relative and its wrapped mass agrees
with the reference to 1.31e-15 relative. Thus the failure is not an uncovered
strip or an assumed periodic volume. The finite-endpoint point coordinates
do not represent those domain weak moments. The odd-field current error
hardly decreases at fixed rho, while domain-moment error decreases by about
four on doubling resolution. No one manufactured field alone establishes
correctness for the coupled nonlinear system.

## 4. Smallest evidence-supported correction proposal — approval required

Leave particle-domain construction unchanged. Revise the **surface quadrature
rule**, replacing the point approximation to Q1 moments by integration over
the existing particle-domain measure. In this straight-strip experiment:

    M_ij = sum_p integral_Dp N_i(s(x)) N_j(s(x)) dA,
    b_i  = sum_p q_p integral_Dp N_i(s(x)) dA.

The domains are split where the surface basis changes segment. This is not
bulk CPDI shape-function substitution, stress smoothing, a new history
interpolator, periodic endpoint identification or frozen-volume weighting.
It changes the discrete projection objective from a sum of pointwise errors
to domain-integrated errors and therefore **requires an explicit revision of
the authoritative point-quadrature specification**, not merely a bug fix
claimed to preserve the existing rule. The current source implements that
existing rule; no undocumented source/specification conflict is alleged.

The smallest supported scope is a consistent surface quadrature correction,
not a global domain or bulk-transfer redesign. Full moments are better
supported here than a boundary-only exception because both weak loads and
mass must agree, not just a particular represented traction. This is not
proof that no cheaper accurate quadrature could exist.

A production implementation must use compatible test/trial quadrature for
surface projections, R_Gamma, K_V and G, and their diagnostics. In particular,
inserting a new mass matrix while retaining the old single-coordinate
nonlinear residual evaluation is not justified. Surface fields and their
derivatives must be evaluated consistently at the selected integration
coordinates, with the existing pointwise material ownership and frozen
history semantics. The current one-segment association/cache is insufficient
for a domain crossing a surface node; this must be addressed explicitly in
the reviewed quadrature implementation, not hidden in an area scalar.

Keep the admitted particle set, normal support width and full I_h unchanged.
The experiment does not settle domain portions outside that admitted strip,
curved-fault closest coordinates, internal free tips, overlap or 3-D. Before
implementing, specify how the existing admitted-domain measure is retained
in those cases or restrict the initial implementation explicitly. Do not
silently clip normal domains or admit extra particles to match this reference.
No correction has been implemented or authoritative equation edited here.

## 5. Separate bulk-history transfer consistency item

The saved-data diagnosis still shows cell-average assignment into continuous
Q2 history with shared-node last-writer dependence. It is not corrected by
surface moments and is not proposed for modification in this patch.

A focused production transfer regression should prescribe constant and
smooth nonconstant tensor histories at stable particle IDs; compare actual
cell averages, shared-node writes and their weak bulk history load against
independent cellwise references. Reverse/permutate cell traversal and compare
one/two ranks. Use periodic-compatible smooth data as well as an affine
nonperiodic control. Constants must reproduce, but nonconstant shared-node
and weak-load differences must be measured, not averaged away.

Record two distinct fields: (1) the **published FE history** after particle
transfer and (2) the **physically constrained working history** used by
mechanics. Validate the latter against the physical constraint application
to the former; do not demand their unconstrained periodic traces coincide.
Compare particle-old and transferred-old contributions with the same beta,
geometry and test functions. Verify that preparing the working iterate does
not publish or evolve histories. Existing diagnostic tests cover constant
reproduction, stable-ID joins and a nonconstant last-cell example; they are
not a substitute for this production MPI/constraint regression.

## 6. Verification, artifacts and review boundary

New files for this step:

- `nonuniform/endpoint_moments.py`: controlled geometry, independent weak
  reference, point/domain/area-only comparisons and raw matrix/load export.
- `nonuniform/test_endpoint_moments.py`: five focused regression cases.
- `nonuniform/endpoint/verify_voronoi_areas.cc` and its `CMakeLists.txt`:
  independent finite-wall Voro++ geometry cross-check.
- This report and progress/README links. Existing working-tree changes,
  including the earlier diagnostic plugin edit, were preserved.

Paths below are relative to `benchmarks/reconstructed_fault/uniform_shear`:

```sh
B=benchmarks/reconstructed_fault/uniform_shear
python3 -m unittest discover -s "$B/nonuniform" -p 'test_*.py' -v
python3 "$B/nonuniform/endpoint_moments.py" --nx 32 --ratio .4 --check-periodic
python3 "$B/nonuniform/endpoint_moments.py" --nx 32 --ratio .4 --stress linear --check-periodic
python3 "$B/nonuniform/endpoint_moments.py" --nx 32 --ratio .1
python3 "$B/nonuniform/endpoint_moments.py" --nx 64 --ratio .2
python3 "$B/nonuniform/endpoint_moments.py" --nx 128 --ratio .4
cmake -S "$B/nonuniform/endpoint" -B "$B/nonuniform/endpoint/geometry-build" \
  -DVORO_ROOT=/home/ein/local/voro++/0.4.6
cmake --build "$B/nonuniform/endpoint/geometry-build" -j4
"$B/nonuniform/endpoint/geometry-build/verify_voronoi_areas" \
  "$B/nonuniform/endpoint/moments/voro_check_points.txt"
```

**Results:** 10/10 Python tests passed (five new, five existing), 2.383 s;
Voro++ executable exited 0. The nine main manufactured cases completed in
about 38.7 s of accumulated evaluation time; the finest cost 19.43 s.
Peak RSS of the sequence process was 725,288 KiB (not a per-case memory
measurement). No ASPECT rebuild, MPI mechanics test or full suite was run.

`endpoint/moments/` contains per-case JSON summaries and compressed NPZ
files with full mass matrices, centered weak loads, represented tractions,
points and areas. `sequence.log`, `nx32-r04.log`, `regression-tests.log`,
`voro-comparison.json` and geometry build logs preserve execution evidence.
The Voro input lists ID, x, y and independently calculated area from the
nx32/rho=.4 quadratic NPZ. The earlier endpoint report remains the source
for saved-run displacement and actual mechanical history/traction evidence.

**Next gate:** obtain correction approval first; then implement and verify
the agreed quadrature consistently, including constant/nonconstant moments,
MPI ownership, action/Jacobian consistency and lifecycle/rollback coverage.
Only after those pass, replay the unchanged 64/128 physical cases through
1 s and compare actual particle-based endpoint traction, weak loads/mass/
balances, initial projections separately from subsequent changes, support,
actual slip normalization and published/constrained history. Any changed
initial projection caused by the revised quadrature must be reported, not
hidden by resetting retained histories. Temporal refinement, true-normal-
stress tests and a claim of Gate K2 completion remain held for review.
