# P3 prescribed straight-extension diagnostic

Worktree `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`. Entry
`612b75d3570d81763407875f3a12c12999b9fa8e`; curated reusable P2/hanging-node
baseline committed as `2d0d40183`. This is original propagation PLAN P3
(synthetic geometry/state transfer), not historical particle-free PF3.

## Transfer contract fixed before measurements

No physical time, Maxwell interval, phase evolution, bulk material identity,
mesh or material coefficient changes during the append. All bulk material,
including the diffuse cap ahead of the old physical tip, already exists. Keep
its physical Q1 phase and lower obstacle exactly. Old surface vertex indices,
Z=G s_elas, accumulated slip and accepted rate retain their values. New surface
records are new constitutive entities with **prescribed fixture birth values
Z=s_elas=accumulated slip=rate=0**. The old endpoint values are exactly zero,
allowing this C0 Q1 join. A loaded nonzero endpoint would need a separately
specified compatible join; this diagnostic does not select that birth law.
There is no RSF theta in this cohesive fixture.

Rebuild consistent normalization I and set the stress-coordinate target to

\[
 T_*^+=T^-+2\{Z^+h(\phi)/I^+-Z^-h(\phi)/I^-\}S,
 \quad T_h^+=T_h^-+\Pi_{Q2,h}(T_*^+-T_h^-).
\]

Outside the corresponding finite normal strip its localization term is zero;
the independent bulk tensor is retained. This preserves physical tau exactly
before projection, not the old coefficients. All nine Q2 modes are retained;
the full metric is xx²+yy²+(xx+yy)²+2xy². G and eta retain 1e6 Pa and 1e8 Pa s.
This is a representation operation, not an extra constitutive update.

At fixed physical stress, bulk storage is invariant in the exact map. Its DG
defect is measured. Cohesive storage is integral Z²/(2GI) ds: a changed
consistent I representation changes storage on the old overlap even with
unchanged Z. That known imposed-geometry change is reported separately; it is
neither numerical dissipation nor evidence for a physical birth energy law.
The new zero-Z surface adds no cohesive storage. Fracture storage stays fixed.
Weak work uses the existing cohesive functional integral N chi tau_xy dOmega;
the old-overlap measure change, new-surface contribution and DG projection
defect are separated. Prescribed mu sigma_n=1000 Pa adds 250 Pa m over the
0.25-m new surface; no contact solve or stress unloading is implied.

## Fixture and implementation boundaries

The fixed uniform 32×64 native distributed Cartesian bulk mesh carries DGQ2
and ordinary Q1 phase. The finite seed is (0,.5)–(.5,.5); one append adds sixteen
collinear cells through (.75,.5), retaining surface spacing 1/64 m. The imposed
phase is sampled directly from native compact `PhaseFieldProfile` at distance
to the original finite segment (core .6, ell .15625, p=1, m=128, 5000 profile
points). The initialized rounded cap is not detected as extension. The append
is explicit new surface representation within already initialized bulk damage;
**this does not demonstrate activation of virgin bulk material**. No tail,
normalization floor, artificial phase creation or phase cap is introduced.
The computational boundary contact at x=0 is distinct from the interior tip;
its normal-completion influence length is zero. No continuation is assigned
to either interior tip.
The continuum cap ends at x=.808821594; native Q1 interpolation has zero
support beyond x=.8125. Both are already present before the append and are
reported separately from its endpoint x=.75.

Loaded input has nonproportional complete-Q2 T, including x²y², and nonzero
Z=1000 sin²(2pi x), accumulated slip=.002 sin²(2pi x), rate=1e-5 sin²(2pi x)
on the old interval, with exact zero histories at its tip. This is a prescribed
loaded snapshot, not a claimed mechanical equilibrium or P2 trajectory.

`ReconstructedFaultManager::append_reconstructed_fault` stages geometry,
generic property rows, support widths and all rate representations before
nonthrowing swaps; birth data must be explicit and finite. Old generic rows
and material indices survive. It invalidates particle projection systems,
Stokes-QP associations and the indexed segment search. The existing archive
schema is unchanged. Configured automatic/legacy boundary completion is
rejected until a joint endpoint-support rebuild is qualified; stale completion
frames are not retained. No production caller selects this operation.

`unit_tests/reconstructed_fault_extension.cc` stages the **synthetic** joint
candidate with native manager/archive, native FEValues/DGQ2 mass projection
and distributed DG owned/ghost vectors. Rebuild normalization, point layout,
quadrature and empty fixed-grid constraints in the candidate. Accept the whole
candidate by one pointer swap after validation; reject by discarding it. Its
geometry-publication counter is distinct from the simulator's physical-step
counter. This test does **not** extend native finite-tip mechanical eligibility,
modify `FEStrainState` ownership, or claim a simulator outer-loop geometry
transaction/checkpoint. The existing finite-tip evolution guard stays intact;
no Stokes/preconditioner cache exists in this isolated fixture. Integrating
this map with those owners/caches remains a subsequent native task.

Projection uses fixed Gauss8 per normal cell and per half-cell tangential
panel, split at every surface knot; Gauss12 checks it. Existing physical-step
Gauss/source rules, limiters and defaults are untouched. Independent replay
uses Gauss16, a dense consistent surface solve, the analytic profile ODE and
orthogonal Legendre projection instead of the native nodal mass inverse.

Predeclared `evidence/p3_extension/selection.json` gates: independent
normalization/projection/weak-load agreement 1e-11 relative; reconstructed
full-tensor L2, bulk storage and weak-load transfer defects <0.5%; exact
persistent identity/rollback/retry/restart and old histories/obstacle. I must
be positive without repair. Cell mean defect screen is 1e-11 Pa m².

## Measured results

[Compact evidence and portable replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/p3_extension/README.md)
retain the pre-run selection, logs, independent evaluations and all 31 native
files in a 416,572-byte archive. Replay verifies every member and reruns the
mathematical checks without an ASPECT build. Tested native binary SHA256:
`a1ee3080a14ef02778d2631c5e114720de6c5789cfbec803b424d140c3a88019`.

| Quantity | Measured result |
|---|---:|
| Bulk cells / tensor DoFs / physical phase vertices | 2048 / 55296 / 2145 |
| Initial continuum-to-Q1 phase relative L2 error | 0.339721% (same frozen field on both sides) |
| Compact cap radius / old and appended physical tip | .308821594 m / .5 and .75 m |
| Minimum new I / maximum old-node I change | .813411129 m / .050608458 m |
| Independent I / complete-Q2 reproduction error | 2.10e-15 / 7.82e-16 relative |
| Independent nodal-versus-Legendre correction error | 1.60e-12 scaled |
| Gauss8/12 coefficient difference | 5.74e-15 relative |
| Full-tensor stress transfer L2 / localization-weighted L2 | 2.11e-7 / 3.10e-7 relative |
| Maximum sampled full-tensor stress change | .0154396 Pa |
| Unchanged-T negative control | 3.73e-7 relative L2 |
| Maximum physical cell mean defect | 1.06e-16 Pa m² |
| Bulk storage / transfer defect | .637166456 J/m / +3.12e-11 J/m |
| Cohesive storage / imposed-geometry change | .000865178 J/m / +4.96e-14 J/m |
| New-surface cohesive storage / fracture change | 0 / 0 |
| Independent weak-load agreement / transfer defect | 8.26e-15 / 3.11e-8 relative |
| Integrated old / new shear weak load | -706.478612 / -502.745148 Pa m |
| Intended overlap-measure / new-surface shear contribution | +.00172255 / +203.731741 Pa m |
| Integrated shear transfer defect | -7.08e-10 Pa m |
| Prescribed new-surface friction contribution | +250 Pa m |

All predeclared screens pass. Old surface rows, all physical phase/obstacle
values and material IDs are exact. The normal tensor components retain their
nine coefficients exactly. The nonzero projection defect remains measured,
not called an exact stress transfer or propagation accuracy result.

The identity control leaves the complete archive unchanged. Fully prepared
candidates are rejected without changing accepted geometry, histories or cache
ownership; deterministic retry reproduces the entire candidate archive. One
geometry publication succeeds; null/duplicate/stale publications reject.
Pre-append and post-append archives restore exactly. A **separate two-rank
process** reads the serial pre-append archive, rebuilds native DG ownership,
performs the append and matches the uninterrupted accepted archive exactly.
Post-append restart recomputes I exactly and checks all DG owned/ghost values.
One/two-rank persistent fields and both archives are byte-identical; native
scalar reductions differ only at roundoff (bulk-energy delta about 1e-15 J/m).

Serial: **292 assertions /2 cases**; two ranks: **295 assertions /2 cases on
each rank**; separate process restart: **12 assertions per rank**; full native
suite: **165436 assertions /131 cases**. Bounded command times are 5.43 s,
2.72 s, 1.72 s and 5.97 s respectively, including fixture setup/output; no
scalability or event-time prediction is made. All runs use 300-s deadlines.
The independent replays take about 1.6 s each with a 180-s deadline.

All **34 protected BP3 identities** pass. Of 858 baseline source/header files,
only the selected manager declaration/implementation change; the original P2
binary is preserved separately. No server work, physical evolution or subsequent
stage is launched. The initial compile failure (ambiguous unity-build namespace,
corrected by explicit qualification) and all prior scientific failures remain
preserved. No numerical acceptance tolerance was relaxed.

## Remaining qualification

This isolates a geometry-transfer coordinate change. It does not qualify an
evolving finite-tip native solve, physical propagation or a general birth law.
The physical-zero-phase region cannot receive a normalized cohesive segment
without phase activation: that scientific rule is deliberately unselected.
General nonzero-tip joins, tip phase driving/configurational work, true new
damage, coupled geometry acceptance, solver rebuilds in that route, RSF,
runtime AMR, transport and moving geometry are outside this test. P2's
`strict_reference_match_passed=false`, all original branch/error findings and
protected BP3 evidence remain. No P4 task begins automatically.
