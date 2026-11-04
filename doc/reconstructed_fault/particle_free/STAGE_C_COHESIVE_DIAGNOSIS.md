# Stage C cohesive energy and localized-tip diagnosis

This records the narrower continuation selected on 3 November 2026. The
cohesive-interface calculation is distinct from production BP3 and the mature
moving-fault experiment. Its endpoint scheme, physical loads and acceptance
tolerances are retained. The energy reduction below uses existing outputs;
the tip investigation uses separately labelled diagnostic repetitions.

## Energy decomposition from retained outputs

Reproduce with
`python3 benchmarks/reconstructed_fault/particle_free/transported_objective/diagnose_stage_c_energy.py`.
The [JSON with input hashes and per-step terms](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_c_coupled/cohesive-energy-decomposition.json),
[CSV](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_c_coupled/cohesive-energy-decomposition.csv)
and [figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_c_coupled/cohesive-energy-decomposition.png)
retain signed sums and sums of absolute per-step contributions separately.

The independent terms already emitted by `cohesive_advance.cc` obey

```
physical corrected remainder = BE bulk numerical dissipation
                            + surface endpoint remainder
                            - (endpoint work - trapezoidal work)
                            + endpoint ledger error
                            - quadrature repartition defect
                            - objective projection defect.
```

This is an accounting identity, not a new acceptance criterion or a definition
of a missing work term by closure. Each term comes from the existing stress,
work or energy diagnostic. The reduction checks the identity per step with
an absolute tolerance of 1e-11 J/m.

| Steps over 0.001 s | dt (s) | Signed physical remainder (J/m) | Sum of absolute remainders (J/m) | Fraction of new fracture work | Wall time (s) | Bulk solves |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 1.25e-4 | 3.42486134e-4 | 3.42486134e-4 | 149.63% | 7.936 | 46 |
| 16 | 6.25e-5 | 1.74111685e-4 | 1.74111685e-4 | 74.78% | 13.545 | 54 |
| 32 | 3.125e-5 | 8.78601723e-5 | 8.78601723e-5 | 37.43% | 18.762 | 70 |

All physical step remainders are positive: cancellation does not hide error.
Measured cumulative orders on the two halvings are 0.976/0.987 for the
physical remainder, 0.984/0.991 for bulk BE dissipation, 0.991/0.995 for
the surface endpoint remainder, and 1.000/1.000 for endpoint minus
trapezoidal mechanical work. At eight steps these three principal signed
contributions are respectively 3.99480646e-4, 1.67702387e-4 and
2.24696898e-4 J/m. Both positive numerical contributions matter; substituting
trapezoidal work alone cannot remove the endpoint integration error.

Bulk BE dissipation is the measured full-plane-strain increment norm
`|tau_plus-tau_anchor|_3^2/(4G)`, distinct from physical Maxwell loss.
Surface endpoint work minus bond/fracture/friction/radiation increments is
the separately recorded surface remainder. For a frozen damage state, its
elastic part is `k(a_plus) (Delta j)^2/2`; changing damage also contributes.
The present outputs resolve the total surface term, not that additional split.

Quadrature repartition and objective projection are around roundoff in these
fixed-geometry cases (eight-step absolute sums 5.68e-13 and 6.89e-13 J/m).
They cannot explain a 3.42e-4 J/m discrepancy. The endpoint ledger closes
at the same scale. The zero transfer-stress diagnostic is measured in Pa,
not J/m; these runs perform no moving-mesh or geometry transfer and establish
no energy bound for that operation.

The 1% fracture-relative screen still fails at every resolution. A first-order
extrapolation from 32 steps suggests about 1,198 steps over this same interval
to reach 1%; this is not a verified accuracy result or an authorized replay.
The observed solve counts are `38 + number of steps`; setup and other fixed
costs prevent treating the three wall times as a precise large-run cost model.
The next energy experiment should be one targeted halving only if a tighter
accuracy estimate is needed, with identical endpoint physics and independent
work diagnostics. No broad matrix or approximately 1,200-step run was launched.

## Localized-tip investigation

The three fresh two-rank traces retain the original Airy amplitude 3500 Pa,
width 0.4 m and location (6,0), with unchanged law, loading, line search and
acceptance criteria. Reproduce the reduction with
`python3 benchmarks/reconstructed_fault/particle_free/transported_objective/diagnose_stage_c_tip.py`.
[Trace identities and reduction](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_c_coupled/tip-diagnosis.json)
link the exact binaries, parameters and failed runs. Per-backtrack CSVs are
retained beside that JSON. The added diagnostics do not change the Newton
direction, residual, accepted state or line-search decisions.

| Case | dt (s) | Failing Newton iteration (zero based) | Force residual (Pa) | Complementarity Linf | Minimum contact (Pa) | Relative linear-system defect |
|---|---:|---:|---:|---:|---:|---:|
| Cut objective | 1.25e-4 | 35 | 8.25717e-8 | 2.78e-17 | 9747.5446 | 2.32e-14 |
| Legacy | 1.25e-4 | 35 | 8.25717e-8 | 2.78e-17 | 9747.5446 | 2.32e-14 |
| Cut objective, half dt | 6.25e-5 | 33 | 8.05005e-6 | 9.71e-17 | 9747.8627 | 3.56e-15 |

**The first failing condition is the force-residual merit decrease**, after
35 backtracks. The unchanged force threshold is 2e-9 Pa; complementarity
passes its 2e-13 threshold and contact stays strictly compressive. At the
original dt the entry residual L2 is 2.77641231e-11, the full-step value is
4.46281639e-11, and the smallest-step value equals the entry value. The
linearized merit derivative is negative (-7.70847e-22) and the Newton linear
system is solved accurately. Thus neither contact failure nor an inaccurate
linear solve explains this stall. Tip admission is not reached.

Objective and legacy backtrack traces are byte-identical. In this first
increment their objective anchors have no accumulated rotation difference;
the experiment rejects a rotation-specific explanation of this failure.
Halving dt makes the force stall larger, so dt reduction alone does not solve
it. The trace is consistent with a residual-evaluation or active-branch
precision problem but does not prove which arithmetic operation causes it.
In particular, the law computes `jump + increment - reference` before a
potentially large small-damage stiffness multiplies it. A higher-precision
replay of the stalled residual and its branches is the next justified tip
diagnostic. No tolerance, loading or solver behavior was changed, and no
propagation or successful admission is claimed.
