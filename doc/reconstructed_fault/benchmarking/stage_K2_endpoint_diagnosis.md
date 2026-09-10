# K2.2 endpoint traction: bounded diagnosis

2026-09-09. The first deterioration is predominantly associated with the
**moving particle-volume / point-coordinate quadrature at the finite box
endpoints**, acting on nonconstant transverse stress. It is not explained by
using the wrong timestep's particle stress. The bulk history transfer has a
separate, measurable discretization error, but its endpoint signature does not
match the dominant growing term.

No production correction has been made. No support, full I_h, initial state,
committed history, tolerance, loading or endpoint topology was changed.
Temporal refinement and true-normal-stress testing remain held for review.
The earlier spatial review remains the record of the three accepted runs;
this report diagnoses its 0.5-to-1 s deterioration.

## Evidence and the one targeted replay

Saved 32/64/128 runs supply stable particle IDs, positions, volumes, fault/
segment/xi, accepted particle stresses, Q1 surface states, and native bulk
quadrature/cell data. Previous accepted stresses were joined by ID, **not by
container position or by nearest particle**. All joins are complete.

One missing distinction required a replay. The earlier bulk CSV exported the
published solution's FE stress history. The coupled solve lifts a separate
working vector with physical constraints, then publishes only its velocity
and pressure blocks. Consequently the published FE history is not necessarily
the constrained history actually used by bulk assembly. The accepted
linearization retains the latter.

The benchmark-only `history_transfer_*.csv` now exports both full history
tensors at the same native quadrature points, using existing public accessors.
The approved 64x256 pilot was replayed through 1 s, with no change except the
diagnostic and the stopping time/output directory:

- Expected before running: 200--350 s, about 1 GiB RSS, plus plugin build.
- Measured: **283.290 s, 916352 KiB RSS**, one rank, exit 0.
- Plugin built with `-j4`; no production rebuild or second replay was needed.
- The existing replay comparison passes at 0, 0.5 and 1 s; all compared
  bulk/surface/history fields are **bit-identical** to the accepted pilot.
- The replay retains the family allowance and actual normalization checks:
  maximum normalization error 6.00643e-5; particle weak-balance RMS agrees with
  production within 4.89e-14 Pa. This check covers the replay's three states,
  not an additional trajectory through 2 s.

The executed binary remains SHA256
`27a2defaf8fa44492d33c22235927cdf2a60cb4ed1c35e722c72f9345d9b61b4`.
The measured replay plugin SHA256 is
`c632970437931153b24d80711655b7a737a85d12e227b14d811a65f0de71ace2`.
These are the executed artifacts recorded in `replay64.resources.json`, not
a claim that any subsequent rebuild has identical bytes.

## 1. Old history and actual FE transfer

For this horizontal homogeneous fixed-profile fixture, the saved data close
the identity

    q_reconstructed_with_published_FE_history(x_p) - q_particle,k
      = beta * (tau_FE,published,k-1(x_p) - tau_particle,k-1,p).

The maximum closure error is 1.03e-10 Pa across the three meshes at the two
times. The rate term uses the saved FE strain, current chi V and frozen
history-slip correction, not a fitted constitutive expression. At 0.5 s,
the correct old particle stress is the retained **1500 Pa** initial stress,
not the evaluated t=0 response. At 1 s it is the accepted 0.5 s particle stress.
The surface assembly reads that particle property directly. No wrong-index,
initial-history reset or premature history update was found.

The bulk transfer is **`cell average` into continuous Q2 composition**, not
CPDI. CPDI participates in the phase-field infrastructure, but the initial
phase field is frozen here. The relevant source path is:

1. `source/particle/interpolator/cell_average.cc`: unweighted arithmetic mean
   of the nine particles in each current cell.
2. `source/simulator/initial_conditions.cc`, `interpolate_particle_properties`:
   assign that mean to every local composition support point. Shared DoFs take
   the value written by the last visited adjacent cell.
3. `source/simulator/solver.cc`: distribute physical constraints on the private
   working vector before residual evaluation. In this replay, the right
   periodic trace is set to the left trace.

Reconstructing the cell means from the correct old particles, current cell
locations and native exported traversal reproduces the published Q2 xy nodal
values within 1.37e-12 Pa on the 64 mesh. The replay's xx/yy reconstruction
agrees within 1.78e-15 Pa. Thus the transfer is identified, rather than guessed
from stress extrema. All cells retain nine particles; no empty-cell fallback
or insertion/removal is involved. A small diagnostic test also demonstrates
the traversal dependence of a nonconstant field at a shared node. This is
the existing transfer's behavior, not a new approved stress-transfer rule.

At 1 s, the published right-minus-left trace mismatches and the actual
constraint corrections are:

| Component | Published trace jump max (Pa) | Working-minus-published QP max (Pa) | Actual working trace mismatch max (Pa) |
|---|---:|---:|---:|
| xx | .0485782 | .0333979 | 2.78e-17 |
| yy | .0485718 | .0333935 | 2.78e-17 |
| xy | 6.90736e-5 | 4.75159e-5 | 1.37e-12 |

The physical constraint is therefore being applied; the unconstrained
published trace must not be reported as the one used in mechanics. In the
direct surface-traction history diagnostic, using the actual working xy
instead of the published xy changes the right endpoint result by only
6.53e-8 Pa. Normal history components affect bulk mechanics, not the direct
horizontal shear contraction. Their induced bulk response is not estimated
by that small direct-xy number.

This also qualifies earlier raw bulk-stress reconstruction: its published
history is not exactly the assembly history near the periodic seam. At replay
1 s, the missing xy contribution is bounded by beta*4.75159e-5 Pa at native
QPs. The actual particle/Q1 traction plateau is unaffected by this diagnostic
distinction. No updated full-tensor fine-grid convergence claim is made here.

## 2. Associations and endpoint domain measures

Between 0.5 and 1 s, all three meshes have **zero changed active flags and
zero changed associated segment indices**. Interpolated fault coordinates
match actual x to roundoff (1.49e-16 m on the replay). The geometry/support
is unchanged. The coordinate xi nevertheless moves continuously with particle
advection, as intended.

Use fixed physical control regions: left x<0.015625 m, interior
0.109375<x<0.140625 m, and right x>0.234375 m, restricted to associated
particles. At 64 resolution, the left/interior/right sample counts are
5688/11376/5688. The endpoint Q1 row has 2844 samples and mass 0.00240745 m2;
the center row has 5688 samples and mass 0.00482178 m2. The domains remain
positive and well supported; the observed issue is not an empty endpoint.

| nx | Largest outer-column volume change, .5 to 1 s | Interior maximum relative change | RMS beta*(published FE old - particle old), left (Pa) |
|---:|---:|---:|---:|
| 32 | 7.5233% | 6.0509e-7 | .415759 |
| 64 | 15.4396% | 3.1261e-6 | .104580 |
| 128 | 32.4436% | 1.7322e-5 | .0268584 |

The raw transfer error decreases with refinement and is similar in the
interior (.108354 and .0308187 Pa on 64/128). It is not uniquely localized to
the endpoints. The domain-measure change has the opposite refinement trend
and a strong endpoint/interior contrast.

`ParticleDomainHandler::generate_particle_domains` builds containers from
ordinary vertex-neighbor cells, inserts unshifted particle coordinates, and
sets all three Voro++ periodic flags to false. At the outer boundary these
are **fixed-box-wall-clipped domains**, not periodic-image domains. This is
confirmed quantitatively from the saved first/last particle columns.

For row spacing a=h/3 and first-column displacement delta_x, their leading
area change is

    delta_m_left = a * delta_x,
    delta_m_right = -a * delta_x.

The measured .5-to-1 s changes match this wall-motion model with relative RMS
errors 4.30e-5 / 2.19e-5 / 1.26e-5 on the left 32/64/128 columns (right similar).
Correlations exceed 0.999999999. This is evidence for wall clipping, not an
assumption inferred from absence of particle wrapping. It does **not** show
that the positive computed areas are corrupt or nonconservative. Their
appropriateness as periodic-seam surface quadrature is the distinct issue.

## 3. Weak traction: exact diagnostic split

Let P_k be the production-form Q1 projection using the accepted particle
volumes and coordinates at step k. For the same current stress samples q_k,
the following is an algebraic identity:

    P_k q_k = P_previous q_k + (P_k - P_previous) q_k.

Both projections use the same stable particle IDs; no stress sample is
changed. P_previous here is only a **diagnostic previous-measure projection**,
not a corrected solve, approved reference, smoothing operation or history
update. The weak loads and row-mass-normalized moments are retained along
with the mass-inverse representation in `weak_terms_*.csv`.

At 1 s, the 64-minus-128 **nonuniform** traction differences split as follows
(each difference has its arclength mean removed):

| Quantity | Left endpoint (Pa) | Right endpoint (Pa) | Along-fault RMS (Pa) |
|---|---:|---:|---:|
| Actual difference | .00367339 | .00366451 | 4.92231e-4 |
| Difference of (P_k-P_previous) q_k | .00351934 | .00352305 | 4.69991e-4 |
| Algebraic remainder | .00015405 | .00014146 | 4.27374e-5 |

The measure/coordinate term accounts for about **96% of the endpoint
difference**, and removing it algebraically reduces the RMS discrepancy by
about 11.5 times. This is the leading measured signature, not a claim that
re-solving mechanics with different weights would give the remainder.
At 0.5 s the corresponding actual RMS is only 2.71228e-5 Pa; its previous-
measure remainder is 2.47554e-5 Pa.

The nonconstant **old particle history alone** contributes .00217139 Pa to
the left endpoint difference through the measure/coordinate change at 1 s;
this contribution is zero to accumulation roundoff at .5 s, when old stress
is still constant. Initial retained C0 offsets and initial-state projection
differences remain as documented in the spatial review; this identity does
not reset or remove them. Within each run, it changes neither initialization
nor the stress sample being analyzed.

A second check applies only the previous volumes, keeping current xi and
the current accepted residual samples F_p. At the left endpoint at 1 s:

| nx | Current-volume projected F (Pa) | Previous-volume projected F (Pa) |
|---:|---:|---:|
| 64 | 9.8491e-7 | 9.3620e-4 |
| 128 | 9.9516e-7 | 3.7913e-3 |

The interior changes are only about 2e-8 Pa. The volume-only shift of projected
q is -.0009352 / -.0037903 Pa at these endpoints; the raw transfer error's
direct endpoint projection is much smaller (-3.14e-5 / +7.78e-5 Pa).
Direct transfer-error modes elsewhere on the fault remain measurable and
must not be dismissed as nonexistent. The two mechanisms are spatially
distinct, and the actual surface solve balances the **current** weak measure.

An explanatory inference is consistent with the observed near-fourfold
growth: for shear-like odd transverse displacement and approximately even
transverse old shear stress, first-order weighted moment changes largely
cancel. Products of the changing clipped volume and changing endpoint Q1
shape function leave second-order terms involving displacement/fault-spacing.
Halving spacing at fixed advection does not hold that ratio fixed. This is
not a proved asymptotic convergence law or a new numerical parameter.

## 4. Diagnosis and smallest next action

Established:

- The correct old particle history is used and the accepted stress
  decomposition closes. Constant initial history transfers accurately.
- The actual bulk FE history is the cell-average/shared-node transfer followed
  by the physical periodic constraint; the replay removes ambiguity about
  that constraint, not just its documented intention.
- Moving wall-clipped endpoint domains and coordinates strongly change the
  weak sampling of nonconstant transverse stress. Their signed projection
  contribution explains the dominant observed endpoint discrepancy.
- No evidence identifies a new domain-ownership error, lost particle, negative
  volume, missing support or incorrect association in this interval.

Remaining uncertainty: the algebraic split does not separately compute the
bulk-mediated response to a different transfer or quadrature policy. It does
not prove that a particular periodic-domain change would fix the coupled
trajectory. Bulk transfer quality and the periodic/finite endpoint quadrature
semantics must not be conflated.

**The smallest justified next action is a focused endpoint-moment regression,
not a Maxwell-history or Jacobian patch.** Use a smooth prescribed transverse
stress and controlled shear displacement at a periodic seam, measuring the
current point-volume Q1 rule against independent weak moments, with an
interior control. The test should isolate the finite-wall/domain-boundary
contribution and the point-coordinate contribution separately. Then review
the required seam/domain-boundary treatment before a narrowly scoped
correction. Merely turning on periodic Voro++ flags, freezing old volumes,
identifying fault endpoint DoFs, or substituting a new history interpolator is
not justified by this diagnosis and has not been done. Any boundary-quadrature
change must respect, or explicitly revise, the authoritative m_p N_i(xi_p)
surface rule; it is not an incidental debugging change.

Keep a possible bulk-history transfer correction as a separate review item:
continuous shared-node assignment and its traversal dependence warrant a
dedicated transfer-consistency regression. They are not a demonstrated reason
to change the Maxwell equations, commit timing or Jacobian. No Jacobian
documentation/source modification is part of this endpoint-only follow-up.

## Reproduction and files

From the repository root:

```sh
B=benchmarks/reconstructed_fault/uniform_shear
cmake --build "$B/diagnostics/pilot-build" -j4
python3 "$B/convergence/run_case.py" "$B/nonuniform/endpoint/replay64.prm" --configuration Release --timeout 600
python3 "$B/compare_replays.py" "$B/nonuniform/output" "$B/nonuniform/endpoint/replay64" --last-step 2
python3 "$B/nonuniform/diagnose_endpoint.py" "$B/nonuniform/endpoint/replay64" "$B/nonuniform/endpoint/replay64-analysis"
python3 "$B/nonuniform/diagnose_endpoint.py" "$B/nonuniform/refinement/space32" "$B/nonuniform/endpoint/saved32"
python3 "$B/nonuniform/diagnose_endpoint.py" "$B/nonuniform/refinement/space128" "$B/nonuniform/endpoint/saved128"
python3 "$B/nonuniform/summarize_endpoint.py"
python3 "$B/nonuniform/measure_case.py" "$B/nonuniform/endpoint/replay64" "$B/nonuniform/endpoint/replay64.log"
python3 -m unittest discover -s "$B/nonuniform" -p 'test_*.py' -v
python3 -m unittest discover -s "$B" -p 'test_*.py' -v
```

Results: replay exit 0; replay agreement and allowance/balance checks pass;
**5 diagnostic tests pass** (including 3 new ID/transfer tests), **15 existing
reference/analysis tests pass**. No full ASPECT suite, MPI campaign or further
refinement trajectory was run. This diagnostic export has been checked on
one rank only.

New files: `nonuniform/diagnose_endpoint.py`, `summarize_endpoint.py`,
`test_endpoint.py`, `endpoint/replay64.prm`, this report and endpoint README.
The only C++ change is the additional benchmark history export in
`uniform_shear.cc`; there is no production source/API change. The current
progress and refinement README link this diagnosis. Earlier uncommitted work
and all raw data are preserved.

Local artifacts under `nonuniform/endpoint/`:

- `replay64.resources.json`, `replay64.log`, `replay-comparison.json`:
  executed command/hashes/resources and unchanged accepted-state checks.
- `replay64/history_transfer_*.csv`: actual working versus published FE tensor;
  `replay64/bulk_cell_ids_*.csv` maps each row to native cell/QP provenance.
- `replay64-analysis/report.json`, `saved32/report.json`, `saved128/report.json`:
  ID/association, volume, transfer and weak-residual measurements.
- Each analysis directory's `weak_terms_*.csv` and `transfer_nodes_*.csv`:
  full fault weak-moment profiles and reconstructed last-writing cell indices.
- `summary.json`, `adjacent_split_at_1s.csv`, `endpoint_diagnosis.png`:
  wall-model check and the signed endpoint split.
- `replay64-measurements/report.json`, `diagnostic-tests.log`,
  `reference-tests.log`: allowance, balance and focused test results.

The replay also retains normal ParaView bulk/fault outputs. No corrected
trajectory, time-converged K2 reference or Gate K2 completion is claimed.
