# Native cohesive–RSF propagation pilot

Baseline `f1f9f29cf40c32d1ddcea000b724375b64853532` plus preserved local work;
branch `pf-rsf-particle-free`. This task implements the default-off native route,
without changing production/BP3, the cohesive–RSF law, confinement, radiation,
physical R, numerical support or acceptance tolerances. No server job was run.

**Result:** ASPECT itself advances the curved interface, admits new material,
fully decoheres two newly admitted cells and continues RSF sliding there during
a displacement hold. General tensor memory, actual outer rejection/cutback,
once-only publication and same-rank native restart are exercised. This is a
bounded finite-wall continuation, not free-topology or quantitative propagation
qualification.

[Animation](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_native/native-propagation.gif)
· [geometry](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_native/native-geometry.png)
· [new-material response](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_native/native-mature-material.png)
· [indexed data](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_native/INDEX.md)
· [direct build/run instructions](../../../benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/README.md)
· [equations and ownership](COHESIVE_RSF_NATIVE_FORMULATION.md).

## Implemented problem and native ownership

The explicit switch is `Formulation/Enable cohesive RSF propagation pilot = true`
(default false). The actual ASPECT constrained, distributed Q2/P1-discontinuous
mixed inverse supplies both the smooth displacement and current-pressure
coupling. Condensation retains
`f0 + beta K dj + Q^T W r = 0`, including `r_p beta N` in its nonsymmetric
tangent. No Python evolution, external kernel process or trajectory replay is
used by the native solver. The retained signed point/admission solver is shared
with the reference rather than duplicated.

The bulk owner keeps all 27 DGQ2 coefficients per cell plus the complete native
FE gradient remainder, so physical stress is not restricted to the prototype's
affine/lifting response span. The manager owns three physical Gauss histories
per formed cell: signed jump, irreversible damage, normalized age, cohesive
reference, relaxed jump and accumulated absolute slip. The system owns the exact
curve, Q1 jump/memory/rate, immutable initial Gauss offsets, identities, admission
clocks and temporary candidate view. Derived traction diagnostics are not history
feedback. Native bulk ownership/ghosts are distributed; small surface operators
and records are replicated. Geometry/operator caches rebuild on restart.

Trials anchor the same accepted owner. The actual outer repeat decision discards
the complete trial; successful acceptance publishes once. Checkpoints reject a
live trial and store physical CellId records and every surface history. The
default-off route adds no checkpoint payload. Candidates with zero damage,
slip and memory have no formed identity; admission never rewrites formed material.
Metadata adds zero fracture/work; `Gc integral(a ds)` is the only fracture charge.
There is no additional bulk-phase fracture inventory.

## Reproducible specimen and measurements

| Quantity | Value |
|---|---:|
| Domain | x=0–12 m, y=−6–6 m; prescribed affine displacement on all walls |
| Bulk mesh / DoFs | 64×64 cells; 33,282 velocity + 12,288 pressure + 16,641 frozen temperature = 62,211 |
| Mechanics / source / surface quadrature | cut/support-partitioned Gauss6 / Gauss3 / Gauss3 |
| G / eta / Gc / intact peak | 1e6 Pa / 1e8 Pa s / 1 J m−2 / 3000 Pa |
| p0 / mu0 / a_RSF / b_RSF | 10,000 Pa / .1 / .005 / .01 |
| Dc / V0 / radiation | .001 m / 5e−6 m s−1 / 25,836.0213655 Pa s m−1 |
| Physical R / numerical lifting half-width | .75 m / .75 m, distinct quantities |
| Surface spacing / reserve / refresh | .09375 m / .1875 m / .0025 s |
| Physical interval / timestep / ranks | .440–.462375 s / .000125 s / 2 |

Loading is gamma=.0003+.004 min(t,.455). Its prior smooth direction turn from
−20 degrees ends at .440 s; subsequent direction is fixed and displacement is
held after .455 s. The pilot starts from the retained loaded, already-curved
.440 s snapshot, through an explicit checked conversion. It does not reconstruct
that entire preparation history natively. Conversion differences are
7.822e−11 Pa in physical tensor stress, 2.802e−10 Pa in pressure and
4.206e−10 J/m in bulk storage; native K/N differences are 9.040e−15 /
1.001e−13 relative. A supplied **native** .441 s anchor also permits running
without the prototype files; that path was tested with the prototype absent.

The uninterrupted final-source run has 179 positive steps and no superseded
output rows. Added damage integral is **2.10012434 m**. The seed-connected a=1
front advances **1.44259415 m** in arc and ends near x=7.91405345 m. Fronts are
bracketed by neighboring Gauss points, not exact point locations.

| Native-admitted cell | Birth (s) | Whole-cell a=1 bracket (s) | Subsequent slip / Dc | Normalized age change | Final V (m/s) | Final pn (Pa) |
|---|---:|---:|---:|---:|---:|---:|
| 50 | .440500 | (.461125, .461250] | .0474933 | −.0235748 | .0419205 | 10590.93 |
| 51 | .441500 | (.461875, .462000] | .0160611 | −.00806693 | .0427512 | 10521.00 |

Both have exactly zero cohesive traction after full decohesion. These are weighted
three-Gauss-point cell measurements with persistent material IDs, not a newly
drawn tied tail or a claim of steady-state RSF. The geometry figure distinguishes
temporary tail, process front, developed front and newly admitted records. The
animation uses physical reconstructed shear stress and signed slip-rate vectors.

## Focused verification

| Check | Measured result |
|---|---|
| Distorted-cell non-proportional source, 1/2/4 ranks | discrepancy <=3.32e−13 Pa; non-DG remainder .00529771 Pa retained; all nine modes/component including mixed x²y² |
| Affine identity | remainder <=2.52e−13 Pa; source <=3.26e−13 Pa |
| General native history, 1/2/4 ranks | independent incompatible Airy tensor; physical-record difference <=1.64e−11; owned/ghost coverage retained |
| Actual rejection/cutback and same-rank restart | matched accepted sequences preserve every bulk record and six Gauss histories exactly; IDs and once-only publication agree |
| Rejection at first geometric admission | unchanged accepted geometry/history; deterministic retry matches the preselected cutback control bitwise |
| Restart with formed curved material | physical bulk records, surface histories and identities reproduce exactly |
| Final-source restart, steps 175–179 | all surface columns, geometry, bulk physical records and reconstructed fields bitwise identical; 61.66 s plus 4.02 s completed-checkpoint readback |
| Retained trajectory, .440–.460 s | growth <=2.20e−13 m; slip <=6.16e−17 m²; bulk storage <=4.56e−10 J/m; front x identical |
| Legacy unit controls | 18 test cases / 1042 assertions pass; not a new full BP3 trajectory |

The general-history control adds the divergence-free tensor
`(20(x−6)^2/36, 20y^2/36, −40(x−6)y/36)` Pa. Its elastic incompatibility is
80/(36G) m−2: it cannot be represented merely by compatible affine/lifting
response coefficients. It enters native mechanics, pressure, storage, relaxation
and checkpoints. Cross-rank differences are reduction/linear-solve roundoff;
same-rank lifecycle comparisons are exact.

A focused native half-step replay uses the same .460 s incoming state and
dt=.0000625 s through .462375 s. Added damage over this window changes by
−.000416465 m (**−.127322%**) and integrated absolute slip by
−2.13167e−7 m² (**−.043615%**). Developed front brackets agree at all 20 common
times. First full-cell upper times are .461250 / .4619375 s; the second differs
by one half-step. This bounds the final maturation response, not the complete
spatial error. The tiny 5.813e−6 m arc difference from the old analysis is due
to the independent Gauss20 arc calculation; physical front x is unchanged.

## Energy and numerical acceptance

Maximum scaled independent source error is 1.923e−13 (bound 1e−12), mechanical
weak error 1.129e−10 Pa and surface weak error 9.977e−9 Pa (bound 1e−8 Pa).
Maximum BE ledger error is 1.574e−11 J/m, against the retained
1e−10 max(1 J/m, |endpoint work|) allowance. The full plane-strain metric is
used for bulk storage and Maxwell dissipation. Surface storage, Gc da, friction,
radiation and endpoint numerical remainders remain separate.

| Window (s) | Sum absolute physical-work remainder (J/m) | / sliding | / added fracture |
|---|---:|---:|---:|
| .440–.455 loading | .00545411 | .17189% | .48466% |
| .455–.460 hold | .00363411 | .23689% | .56109% |
| .460–.462375 hold | .00186965 | .22333% | .57159% |

These use actual trapezoidal boundary work, not algebraic closure or cancellation
of signed errors. The final half-step remainder is .000933763 J/m, approximately
halved (.11167% of sliding, .28584% of fracture). Changing cut partitions at
admission produces a separately reported same-field integration defect:
max 2.502e−12 J/m, sum absolute 4.849e−11 J/m. It is not physical fracture work
or hidden history transfer. Bulk storage changes 508.932110→531.157576 J/m.

## Corrections and preserved failures

1. Owned DG updates now add only locally owned entries before ghost refresh;
   the initial ghost-add MPI failure remains saved.
2. Early experimental dispatch initializes native pressure scaling through the
   existing routine. The original uninitialized-assembly failure remains saved.
3. The manager's unused legacy nonnegative-rate container is initialized; signed
   physical rates remain in the new owner. The original first-append failure is
   preserved.
4. An independent interpolation fixture finds nonfinite unmasked P1 pressure;
   initialization now interpolates only velocity and solves pressure natively.
   Finite-field guards use owned vectors. The corrected prefix preserves all
   physical histories bitwise; original NaN diagnostics/failed guard remain.
5. A fixed-eight-ulp endpoint clock allowance failed after 160 additions. It is
   replaced by an operation-count roundoff bound including actual outer retries,
   persisted in the checkpoint; physical tolerances are unchanged. Original tiny
   closure-step/ramp outputs remain. The final uninterrupted run closes in 179
   meaningful steps.
6. An attempted longer extension reached the **existing 7.5 m formed-length
   budget** at trial .462597690 s (last accepted .462472690 s). It stopped before
   publication; trial/error/last checkpoint are retained. The bound was not
   increased. The qualified endpoint .462375 s precedes it.

## Cost, reproducibility and remaining scope

The final uninterrupted run takes **700.50 s wall time** on two local ranks.
Accepted-output sampling observes a maximum rank high-water RSS of
**385,036,288 bytes** (367.2 MiB); this is not a measured scheduler/node peak.
Native timers: 392 Stokes calls 241 s, coupled route 681 s, 23 checkpoints
5.68 s, one preconditioner .0916 s. Scopes overlap. Final maturation half-step
replay takes 173.67 s. The tiny general-history 1/2/4-rank controls take
4.38/10.61/17.21 s: no MPI speedup is claimed. Setup/column solves dominate;
surface replication/checkpoint gathering remain bounded-pilot implementations.

The [README](../../../benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/README.md)
provides direct CMake/mpiexec commands, both initialization routes and checkpoint
branching. `qualify.py` records exact input, command, binary hash, timeout and
wall time; `analyze.py` reads accepted CSVs only. The evidence index describes
units, weights, thresholds, source/environment identities and raw-run locations.
All nine native parameter files pass ASPECT syntax validation; actual execution
and lifecycle checks above remain separate from syntax validation.

The retained fitted-control normal differences (2.545% bend / 3.255% late turn),
raw normal-jump RMS 422.8 versus 18.0 Pa, ahead-region/point-traction failures,
earlier spatial/branch failures and finite-wall boundary sensitivity are **not
resolved**. Their measured bounded response sensitivity remains inherited
evidence, not point-stress accuracy. The causal maximum-shear/persistence rule
is still a propagation hypothesis; no R calibration or unique general path
claim is made. There is no inertia/rupture-speed validation, opening, crossing,
3D, AMR, transport, arbitrary mesh or changed-rank restart qualification.

The next usable-model work is review/integration of this explicit pilot API,
scalable surface/column/checkpoint storage, broader geometry/pressure accuracy
and scientific path/contact qualification. It should preserve the general
tensor owner and atomic lifecycle now demonstrated. No further campaign or
physical-model change is started automatically.
