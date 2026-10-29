# Weak-admission qualification and native fixed-surface seeded growth

**Weak intact admission, activation lifecycle and the bounded native growth test
pass within the stated fixed-surface scope.** The specimen first develops a broad
cohesive zone, then fully decoheres; it does not demonstrate isolated free-path
crack propagation. Early spatial onset remains underresolved. No physical
parameter, force/energy tolerance, production/BP3 path or old evidence was changed.

[Weak formulation](COHESIVE_INTERFACE_FRONT_FORMULATION.md) ·
[selection](COHESIVE_INTERFACE_FRONT_SELECTION.md) ·
[phi/a/Γ architecture proposal](COHESIVE_INTERFACE_ARCHITECTURE_PROPOSAL.md) ·
[figures, GIF and data](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_front/INDEX.md).

## What was qualified

An intact Gauss record enforces zero jump, while bounded multipliers must
**collectively** balance the weak mechanical load:
`Q_T δ=b_T`, `s−R_D r_D=R_T λ`, `|λ|≤3000 Pa`.
The native solver uses a positive first-formation predictor under this positive
loading, followed by an unrestricted weak box certificate before publication.
Established damaged material remains free to reverse. This is not a nodal stress
cutoff or a growth mask. The multiplier certificate covers negative/alternating
virtual directions too; a failure stops acceptance.

Independent linear programming shows why the distinction matters. Alternating
nodal loads of only 0.9 times the peak violate the weak admission condition.
Conversely, a feasible bounded multiplier field projects to 1.331 times the peak
at some Gauss points. Neither projected nor nodal pointwise cutoffs implement the
correct discrete test. Tied multipliers can be nonunique; reconstructed physical
stress and the work-conjugate weak traction are reported separately.

| Check | Result |
|---|---|
| Tied physical jump / multiplier excess | exactly zero / zero |
| Native weak equilibrium | ≤1e−8 Pa; independent reevaluation includes explicit arithmetic roundoff |
| Independent full-history reference through activation | maximum normalized jump/damage difference 1.63e−16 |
| Uniform identity | retained native/scalar comparison and independent weak-predictor/scalar comparison pass 1e−7 Pa |
| Complete admissible tangent, current pressure included | refined FD errors <1e−6 at activation; actual values in CSV |
| Activation rejection/retry | rejected trial leaves accepted owner unchanged; retry bitwise identical |
| Publication and real ASPECT restart across activation | exactly one commit per accepted step; all histories, rates, energies and tractions bitwise identical; timing column excluded |
| Admissible relaxed curvature during growth | minimum 6.77e6 Pa/m |
| One-sided formation-release curvature | minimum 4.50e6 Pa/m |
| Closed-contact side pressure | minimum 9123.97 Pa |
| Exact backward-Euler ledger | maximum defect 2.11e−14 J/m |

The reduced stability space is the nullspace of the tied Gauss equalities.
For this monotone Q1 front every tied element has all three Gauss constraints,
which force both endpoint variations to vanish. Artificial tied-row diagonals
are excluded. A partially tied sign-changing element is explicitly unsupported;
it would require a general constraint-nullspace solve. The unrestricted
one-sided release curvature is reported separately. The negative cohesive slope
−τp/δf=−4.5e6 Pa/m is retained. **No negative incremental curvature occurred**, so
there was no reason to alter the law or add damping. This is a local incremental
check, not the old phase-potential certificate or a global branch theorem.

Two initial independent semismooth solver drafts stalled at admission; their logs
and retained draft are preserved. They were algorithmic failures, not evidence
of physical instability. The certified predictor resolves the tested branch.

## Fresh specimen and observed mechanism

The same homogeneous exact-jump block is periodic over 1 m, with clamped walls
.05 m below and .075 m above Γ. The initial mature interval is .375<x<.625 m;
elsewhere a=0 exactly. Bulk/jump memories start at zero, Θ=1. The endpoints align
on both meshes, so the prescribed initial Gauss state and its **0.25 J/m fracture
inventory** agree exactly; no imported coarse field or relaxation is involved.
No bulk phi or second fracture energy is enabled.

Keep G=1e6 Pa, eta=1e8 Pa s, Gc=1 J/m², τp=3000 Pa, Dc=.001 m and the retained
RSF/radiation calibration. Load at Udot=.001 m/s through .84 s, then hold through
.86 s. This is a finite periodic cell, not a distant-boundary rupture specimen.

| Milestone | 32 segments, Δt=.002 s | 64 segments, Δt=.002 s | 64 segments, Δt=.001 s |
|---|---|---|---|
| First new admission bracket (s) | [.222,.224] | [.204,.206] | [.205,.206] |
| Entire candidate line admitted (s) | [.366,.368] | [.366,.368] | [.367,.368] |
| First new full decohesion, a=1 (s) | [.822,.824] | [.822,.824] | [.823,.824] |
| Entire candidate line decohered (s) | [.824,.826] | [.824,.826] | [.825,.826] |
| Maximum a change per step | .004460 | .004460 | .002230 |

In the fine half-step run the cohesive length is .1094 m at .30 s (about seven
fine segments), .4531 m at .36 s (29 segments), and .75 m at .40 s (48 segments).
The original mature region has **not advanced** at those times. The a=.05 right
contour reaches x=.62968 m at .30 s and .68678 m at .36 s, then spans the cell;
these are process-zone contours, not a fully broken crack tip.

Full decohesion starts adjacent to the left seed flank in [.823,.824] s; the two
flanks are not artificially aligned or averaged. By .826 s the whole line is
mature. No disconnected a≥.01 growth is detected at the saved Gauss resolution.
The rapid traversal of the a=1 contour occurs through material already near
complete damage: the maximum phase increment remains small and halves with Δt.
Do **not** interpret that contour speed as a validated rupture velocity or an
isolated traveling process zone. At the hold, the already mature interface keeps
sliding under stored bulk stress; stopping wall motion does not remove stress.

## Spatial, temporal and physical-energy accuracy

Δx=.03125/.015625 m; Δy=.00078125 m on both meshes. Field comparisons use the
same physical **1/32 m box average**, evaluated with Gauss weights, and normalize
by the actual new damage, not the large initial seed. No interpolation crosses
the pre-existing damage discontinuities. Native peaks and contour positions are
reported separately. Thus initial representation and diagnostic interpolation
are not misidentified as evolution error.

| Time | Spatial new-damage L2 difference | Spatial new-fracture difference | Temporal new-damage L2 difference |
|---|---:|---:|---:|
| .24 s | 6.80% | 3.56% | .0102% |
| .30 s | 8.24% | 4.36% | .0329% |
| .36 s | 2.04% | .907% | .0212% |
| .40 s | .687% | .157% | .00278% |
| .824 s | .000354% | .000129% | .00127% |

Early onset shifts by 18 ms between meshes, and the peak-growth difference at
.24 s is 45.5%. That tiny initial zone is **not spatially qualified**. By .36 s
the zone is resolved by many elements and the measured integrated criteria are
small. The a=.05 spatial contour difference at .30 s is .00196 m, below one
coarse segment; the a=1 difference at .824 s is .00176 m. Temporal a=1 positions
at common sampled times agree; the finer timestep narrows the event bracket.
Final reaction differs by .139% under halving. This is a bounded comparison,
not a convergence-order claim or general propagation qualification.

New fracture inventory ends at **0.75 J/m**, separately from the initial .25.
The absolute sum of trapezoidal physical-work remainders is .0013913 J/m at
Δt=.002 s and .00069668 J/m at Δt=.001 s: **.246%/.123% of sliding loss**, and
**.1855%/.0929% of added fracture energy**. All are below the selected 1% screen.
Endpoint work, bulk/bond storage, Maxwell/friction/radiation losses, signed bond
endpoint remainder and bulk BE remainder are retained individually. No final
cancellation or algebraic ledger closure substitutes for that physical screen.

## Architecture and next boundary

The proposal assigns bulk Maxwell history to the bulk; physical geometry and
normals to Γ; actual discontinuity/velocity to δ; irreversible fracture compliance
and progressive contact activation to a; and, initially, only geometric/path
representation to phi. New candidate planes use **maximum-shear intact release**,
then a declared preference for exact ties. New tied records cost no fracture
energy and preserve bulk history; their subsequent damage pays GcΔa once.
Friction activates as a and reaches geometric-interface RSF at maturity.

A thermodynamic bulk phi alternative would require deriving a regularization of
this **same** fracture measure and its transfer to Γ/a. Adding the old bulk phi
energy to Gc∫Γ a would double-count fracture. General path generation, secondary
fracture and junctions are still scientific/design work, not inferred from this
straight candidate-line test. Recommend a bounded tied-record insertion/extension
and state/energy-transfer test before enabling free-path growth. A larger physical
specimen would also be needed to study an isolated moving process zone.

## Files, runtime and reproduction

Modified the existing optional `cohesive_interface_native.cc`; added its focused
`cohesive_interface_front.inc`, the independent weak reference, run/check/analysis/
plot scripts, and this report/formulation/selection/architecture proposal. Updated
STATUS, SOURCE_AUDIT and CURRENT_STATUS. Production source/header files, BP3,
old packages and failed evidence are unchanged; preservation hashes are indexed.
No new production parameter, CMake target or checkpoint conversion was introduced.

Build the existing `cohesive_interface_native` target under
`build-particle-free/cohesive-interface-front/plugin`, using `Aspect_DIR` from
`build-particle-free/native`; retain the current host
`build-particle-free/coulomb-native/aspect-release`. Then run, from the root:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_front.py qualification
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_front.py identity
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/check_cohesive_interface_front.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_front.py growth
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/analyze_cohesive_interface_front.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/plot_cohesive_interface_front.py
```

The growth driver requires a passing qualification JSON. All native runs are
serial, sequential and bounded to 180 s each. Measured growth wall times were
**24.6 / 112.5 / 166.2 s** for coarse/fine/half-step cases. Qualification and
restart runs took 17.6–20.1 s each. Full raw solver/checkpoint archives remain
under the task build directory; compact source, figures, CSV/JSON and provenance
are packaged. Rejection is exercised at the experimental history owner; the production outer
timestep/cutback driver is not exercised. This is native FE/manager/checkpoint
component integration, not replacement of the production Stokes/phase driver, MPI or free geometry evolution.
