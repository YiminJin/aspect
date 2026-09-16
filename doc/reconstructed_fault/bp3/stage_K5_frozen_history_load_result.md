# K5: frozen step-12 stress-history/load audit

## Decision summary

The 40-km concentration is already present in retained particle stress before
step-12 mechanics. The FE transfer attenuates its normal-stress RMS by 2.9%,
rather than generating or globally amplifying it. Physical constraints make
no change to the history at the sampled junction/interior parents. The
production frozen weak load agrees with independent integration to roundoff.

There is nevertheless a significant **local particle-versus-FE mismatch**:
at the tensile junction sample the old normal contraction is 5.216 MPa in
the particle but 0.440 MPa in its FE transfer. This mismatch matters to the
sign of a near-zero pointwise normal traction. It does not explain away the
much larger pressure dipole, nor establish an implementation defect. The
bottom tip has a still larger mismatch and remains a separate concern.

One four-rank extraction took 26.4 s and stopped before Newton. No new
mechanical solve, accepted state, history update, physical-model change,
or long-run continuation occurred. Neither checkpoint was changed.

## 1. What was actually sampled

Command, from the repository root:

```sh
cmake --build benchmarks/reconstructed_fault/bp3/build -j4
python3 benchmarks/reconstructed_fault/bp3/run_history_load_diagnostic.py
python3 benchmarks/reconstructed_fault/bp3/analyze_history_load.py
```

The runner copied the compatible **local** step-11 `restart/03`, never the
failed in-memory step 12 or the incompatible server checkpoint. It retained
the normal 40-km prescribed mask and performed ordinary step-12 particle
advection, transfer and constitutive preparation. At the solver's physical
constraint-creation callback, before its first residual/Newton evaluation,
the benchmark reconstructed the exact private working-vector lift used by
the solver and invoked the existing bulk assembler noncommittingly.

The callback is armed only after ordinary temperature/advection preparation
at step 12. A private closed copy of the pending physical constraints is
applied to a private owned solution copy; the published solution is not
modified. Temporary replacement of the current-linearization vector is
protected by an unwind restore guard. Homogeneous constraints are used for
weak-load distribution. Every bulk cell is assembled only on its owner,
followed by MPI ADD. The callback throws an explicitly labeled pre-Newton
stop after closing its output files. Normal coupled failure cleanup then
closes the nonlinear lifecycle. The process status is intentionally 1, not
a convergence claim.

This is a source-matched reconstruction of the first working state and a
direct call to the real cell assembler, **not** an observation of a new
Newton iteration. In particular, `bulk_11` was not substituted for the
particle tau11 newly transferred for step 12.

Data:

- time = 2232176379.2516127 s; dt = 376359254.49064183 s;
- beta = 0.9999998794215765;
- all 36,106 owned bulk cells assembled across four ranks;
- 317,016 unique parent IDs and 317,004 bulk QPs exported in the near-fault
  diagnostic strip; no duplicated ownership in parent/cell/owned-DoF exports;
- parent rows contain the full retained tensor and published/constrained FE
  tensor evaluated **at that same current particle position**;
- QP rows contain actual working-history tensor and beta*tau:N;
- all cells have unassembled local load norms; all nonzero owned load DoFs
  have assembled global and region-decomposed loads.

The exported strip is a diagnostic selection, not a change to physical
support. The full bulk weak load includes cells outside this strip.

## 2. The concentration predates the transfer

Here N=n tensor n, n=(sin60,-cos60). Values below are MPa, measured at common
parent positions within 400 m of the sharp fault. RMS is **unweighted over
parents**, not a surface-domain quadrature norm or an analytic error.

| Down-dip window | particle tau11:N RMS | constrained FE RMS | FE-minus-particle RMS | FE RMS change |
|---|---:|---:|---:|---:|
| 24–26 km interior | 0.628013 | 0.621740 | 0.063276 | −1.00% |
| 34–36 km control | 1.107070 | 1.097639 | 0.088887 | −0.85% |
| 39–41 km junction | 6.418345 | 6.231753 | 0.709266 | −2.91% |
| 13–20 km transition | 1.749316 | 1.736299 | 0.035885 | −0.74% |
| 114–116 km bottom | 4.588670 | 2.663962 | 3.053236 | −41.94% |

At 39–41 km, particle tau11:N spans −15.5985 to +16.2912 MPa. The FE field
at those parents spans −14.1386 to +14.6907 MPa. The 40-km structure is visible
in both representations, with nearly coincident broad profiles. The local
transfer discrepancy reaches −4.948 to +5.851 MPa: modest RMS attenuation does
not imply pointwise equivalence.

Constraint application changes neither the sampled tensors nor their normal
contractions in any of these near-fault windows. Across the larger exported
strip, the maximum component constraint correction is 0.178038 MPa and the
maximum normal-contraction correction is 0.052642 MPa. Thus constrained and
published histories are distinguished explicitly, but constraints are not
the source of this junction anomaly.

`history_profiles.csv/png` use separate signed normal strips 100–300 m on
either side and 100-m down-dip bins. They do not cancel the dipole by taking
a single normal-column average. The binned curves are descriptive diagnostics;
the raw parent/QP CSVs remain available without binning or smoothing.

## 3. Weak-load verification

The exact production frozen-history RHS is

\[
 f_i^{hist}=-\int_\Omega \beta\tau_{11,h}:\epsilon(w_i)\,d\Omega.
\]

The audit calls `ReconstructedFaultStokes::execute` on the actual working
vector, extracts `local_frozen_fault_rhs`, and independently integrates the
formula using the same material `evaluate_frozen_maxwell_stress` operation.
Its pointwise coefficient is evaluated from working bulk composition and
temperature. The fixed-profile history-localization term is zero: agreement
with the full production frozen accumulator checks this rather than adding
or removing it by assumption. The V-dependent accumulator is not counted as
history. No pressure, viscous-current, or boundary-traction contribution is
included in these isolated loads.

- Maximum per-entry production/independent local difference: 9.53674e-7
  in 2-D weak-load units (Pa m), roundoff relative to O(1e9) cell loads.
- Full constrained global norm: **1.60659914537e10**.
- Isolated normal-tensor contribution norm: **7.54636894870e9**. This tensor
  component alone is not an independently equilibrated stress.
- Independent QGauss(4) versus production quadrature global difference:
  2.86494e-5, or **1.78323e-15 relative**.
- Region-vector sum versus full load: **2.42348e-16 relative**.

| Region of bulk integration | constrained regional load norm |
|---|---:|
| 34–36 km | 7.776033236e9 |
| 39–41 km | 9.630521478e9 |
| 13–20 km | 9.055995817e9 |
| bottom y<1 km (after preceding disjoint windows) | 6.070943066e9 |
| remaining bulk | 1.885984717e10 |

These are a **vector decomposition**, not additive scalar percentages.
Cutting the integration domain introduces artificial subdomain interfaces;
the regional load norm is not a local force imbalance. The full frozen load
must be balanced by other terms in mechanics. Its nonzero norm alone is not
evidence of an erroneous prestress.

For comparable fine cells with centers within 400 m of the fault, local
unassembled load RMS is 1.45749e9 near 40 km (168 cells), versus 3.10714e8 near
35 km (168 cells), and 1.45279e8 near 25 km (167 cells). Normal-tensor-only
local RMS is respectively 6.20750e8, 1.09590e8 and 6.13566e7. This confirms
an inherited localized loading concentration; these element norms do not
measure equilibrium or prove a transfer-induced force error.

## 4. Decomposition at the already saved surface extrema

The stable-ID join verifies matching parent coordinates to 1e-8 m. With the
fixed profile and S:N=0,

\[
 \sigma_n=50\ {\rm MPa}+p-
  \underbrace{\beta\tau_{11,p}:N}_{\text{retained parent history}}-
  \underbrace{(\Delta\tau:N-\beta\tau_{11,p}:N)}_{\text{current strain contribution}}.
\]

All table entries except location are MPa. Only **saved** converged mechanics
are used; the frozen audit solves nothing.

| Sample | p | retained normal history | current strain contraction | sigma_n | FE old normal history at same parent |
|---|---:|---:|---:|---:|---:|
| baseline 39.961334 km, ID 217673 | −43.096021 | 5.216355 | 2.347476 | −0.659852 | 0.439753 |
| saved 35-km probe at same old hotspot | −34.243044 | 5.216355 | 0.867685 | 9.672915 | 0.439753 |
| saved probe new hotspot 34.960291 km, ID 230195 | −10.499438 | −0.906002 | −0.531527 | 40.938090 | −0.886542 |
| baseline bottom, ID 11936 | −21.065329 | 40.855151 | 8.485873 | −20.406352 | 19.098902 |

At the old junction, the baseline total normal contraction is 7.563831 MPa,
with 5.216355 MPa inherited. The pressure decrement is larger: 43.096021 MPa.
Both old particle stress and its weak FE load remain concentrated near 40 km
when the *current* diagnostic cutoff is moved. This is consistent with stress
accumulated under the old loading, not an immediate new-junction-only effect.

For diagnostic accounting only, replacing the parent old history by FE old
history **while holding the saved u,p fixed** changes baseline sigma_n from
−0.659852 to +4.116751 MPa at ID 217673, and from −20.406352 to +1.349896 MPa
at the bottom minimum. These are **not corrected solutions** and are not
permission to change surface sampling. Such a replacement would violate the
current parent-history specification; rerunning mechanics could also change
p and u. It shows why the local tensile sign needs representation scrutiny,
not that either representation is the exact solution.

The earlier 35-km probe still carries its documented incomplete in-memory
rollback-check status. This audit uses its converged exports as such; it does
not retroactively qualify that rollback checker.

## 5. Smallest justified next decision

No production correction is supported by this audit. It rules out creation
of the broad anomaly by this one FE transfer, amplification of its overall
normal-stress RMS, and a missing/wrong-sign frozen weak load in this assembly.
It does **not** prove spatial convergence of inherited stress, exactness of
the particle history, or absence of accumulated transfer error over time.

Before changing history representation, the discriminating follow-up would
be a frozen-data **common-test-function weak-moment comparison** between the
retained particle-domain stress and its FE representation, separately near
40 km and the bottom, with a smooth interior control. It must account for
the actual particle-domain measure rather than replace it by point weights.
That would test mechanical significance of the representation mismatch;
the present pointwise differences and FE-only load norms cannot do so alone.
No such new integration-rule comparison, transfer correction, changed-cutoff
solve, or first-event continuation was launched.

## 6. Files and checks

New benchmark-local files:

- `history_load_diagnostic.h`: opt-in private-history sampling and actual
  assembler/independent weak-load audit, terminating before Newton;
- `run_history_load_diagnostic.py`: one-run copied-checkpoint provenance and
  preservation checks;
- `analyze_history_load.py`: stable-ID joins, raw/FE comparisons, independent
  quadrature/region checks and plots.

`bp3.cc` only adds the include, opt-in arming and constraint callback.
No production source, equations, parameters or normal output are modified
in this task. The header reuses the earlier small history-transfer audit's
direct assembler/private-vector method rather than adding a core API.

Release/debug plugin builds (`-j4`) passed; pre-existing test warnings remain.
The one four-rank audit passed its numerical and checkpoint-preservation
checks. Offline checks passed; Python compilation and `git diff --check`
passed. No broad suite, fresh trajectory, or new one-/two-rank invariance
claim is made. The unwind restore is implemented but was not independently
instrumented with a new in-memory full-state equality test; checkpoint
hashes and absence of accepted output were verified.

Artifacts are under
`benchmarks/reconstructed_fault/bp3/normal-stress-history-load-local4/`:
`run.log`, `provenance.json`, `execution.json`, `build.log`,
`history_load_{parents,qp,cells,loads}_rank*.csv`, `analysis.json`,
`hotspot_decomposition.csv`, and `history_profiles.csv/png`.
