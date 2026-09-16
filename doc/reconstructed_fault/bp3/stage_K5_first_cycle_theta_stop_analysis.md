# BP3 first-cycle stop: Theta audit analysis

Date: 2026-09-14. Analysis only: no production changes, parameter changes,
restart, or ASPECT execution.

## Decision summary

The supplied run reached mechanical convergence at step 12, then stopped in
postprocessing. Saved nodal histories through step 11 obey the approved exact
split aging law to double-precision accuracy. The benchmark's independent
Theta check uses a cancellation-prone algebraic expression: a small offline
reproducer shows that it can reject a correct production update at admissible
near-bound rates, even with extended-precision arithmetic.

This is a demonstrated defect in the checker, and the leading explanation
for the reported stop, **not a demonstrated extra history lag**. The exact
step-12 violation cannot be reconstructed: its nodal V/Theta data and exception
stderr were not supplied. Keep this distinction explicit before restarting.

Separately, the global sampled normal-stress minimum becomes tensile at
step 11. The official station values remain compressive. This spatially
unlocalized issue deserves its own check; it is not evidence of a Theta-update
failure or permission to change pressure treatment.

## Artifacts and provenance

The requested `doc/reconstructed_fault/bp3/first_cycle_coarse` directory is
absent. The supplied outputs are actually in
`benchmarks/reconstructed_fault/bp3/first_cycle_coarse/`:

- `log.txt`: Release, 64 MPI ranks, deal.II 9.6.0; 36,106 active cells and
  1,282,726 DoFs after initial refinement.
- `parameters.prm`, `original.prm`: resolved/run configuration; Dc=0.008 m,
  Vmin=1e-20 m/s, artificial initial interval 4e6 s.
- `accepted_steps.csv`, `stations.csv`: rows through step 11 only.
- `fault_0.csv`, `fault_2.csv` through `fault_11.csv`: nodal histories;
  `fault_1.csv` is absent, consistent with sparse profile output.
- `history_*.csv`: existing checker results; step 11 reports
  9.103828801926284e-15 relative error.
- `first_event.csv`: started=0, complete=0; no seismic event detected.
- `restart/last_good_checkpoint.txt`: 3; log identifies `restart/03` as the
  completed step-11 checkpoint.

The log ends at step-12 `Postprocessing:` without the exception text. The
specific exception is supplied by the user. No server plugin source/binary
fingerprint accompanies these outputs; code interpretation below uses the
current local source. In particular, that source writes some step records
before the Theta assertion, but no step-12 rows were supplied. Incomplete
artifact capture, early MPI abort, or source-version differences cannot be
distinguished from these files alone.

## Where execution stopped

| Quantity | Last fully reported state, step 11 | Mechanically converged step 12 |
|---|---:|---:|
| Physical time (s) | 1,855,817,124.7613616 | 2,232,176,379.2382126 |
| Physical time (Julian yr) | 58.8072960 | 70.7334011 |
| dt (s) | 884,306,322.0546608 | 376,359,254.47685081 |
| Final bulk residual | 0.001521170663 | 0.000636069694 |
| Final relative bulk residual | 1.800082922e-13 | 1.671794219e-13 |
| Final surface residual (Pa, RMS) | 9.636445384e-8 | 2.598859093e-8 |
| Final relative surface residual | 1.464656751e-14 | 8.267451776e-15 |

At step 12 the bulk target was 38.0471284438 and the fixed surface scale
3,143,482.61566 Pa; the configured relative target remains 1e-8. The final
returned linear direction had estimated residual 5.548325711e-13, fresh
residual 5.548325616e-13, and requested target 6.400703413e-13. All 105 logged
fresh checks over the supplied run passed; the largest fresh/target ratio
was 0.986132639. Step 12 took five accepted Newton updates; its first alpha
was 0.5442997200 and subsequent alphas were 1, with no logged Armijo rejections.

These are positive convergence measurements, not an inference from exit status.
They do not certify postprocessing or checkpoint completion at step 12.

## Production law versus the check

The specification requires mechanics at frozen preceding Theta, followed by
the exact aging-law update using accepted V. At timestep zero supplied Theta
is retained, not aged over the artificial initialization interval.

`source/material_model/phase_field_fault.cc` constructs the candidate with
the accepted nodal rate, preceding nodal state, current dt and surface mixture.
`source/material_model/rheology/fault_friction.cc::update_state` evaluates

\[
 x=V_k\Delta t_k/D_c,\qquad
 \Theta_k=\Theta_{k-1}e^{-x}-(D_c/V_k)\operatorname{expm1}(-x).
\]

This remains stable as x tends to zero. In contrast,
`benchmarks/reconstructed_fault/bp3/bp3.cc` checks, in long double,

\[
 s=D_c/V_k,\qquad \Theta_{\rm check}=s+(\Theta_{k-1}-s)e^{-\Delta t_k/s}.
\]

The two terms being added have magnitude near s, while the result can be
orders of magnitude smaller. Long double reduces but does not remove this
cancellation. The assertion compares the rounded result with a 1e-12 relative
threshold and gives neither the failing node nor the numerical discrepancy.

### Independent verification of saved updates

Offline Python/mpmath evaluation with 70 decimal digits, using saved nodal
V, preceding Theta and actual dt, gives the following maximum relative errors
against published Theta. Dc uses the binary double value of the configured
0.008, and CSV inputs retain their exported precision.

| Step | Maximum relative error, stable high-precision reference |
|---|---:|
| 3 | 2.0843e-16 |
| 4 | 2.1526e-16 |
| 5 | 2.2296e-16 |
| 6 | 2.3369e-16 |
| 7 | 2.6079e-16 |
| 8 | 2.4732e-16 |
| 9 | 2.9436e-16 |
| 10 | 2.4078e-16 |
| 11 | 2.5736e-16 |

Thus all available consecutive full-profile pairs (3–11) consume the
immediately preceding published state correctly. Missing full step-1 data
prevent this same all-node test for steps 1–2; no history was inferred or filled
in to create that evidence. Step-0 diagnostic error is 6.8834e-15 and retained
particle Maxwell perturbation is exactly zero in its history report.

### Bounded cancellation reproducer, not a reconstructed failed state

Saved step-11 node 1154, xd approximately 100 m, has
Theta_old=1,855,415,838.0411193 s and V=3.0611065791756763e-17 m/s.
Use this actual old state and the actual step-12 dt, but prescribe the trial
rates below. A standalone C++ program compiled with `c++ -O2` evaluates the
exact expressions used by production and the checker; it performs no solve.

| Hypothetical accepted V (m/s) | Correct stable Theta (s) | Checker relative disagreement |
|---|---:|---:|
| 3.061106579e-17 | 2,231,772,149.5489492 | 7.21645e-15 |
| 1e-18 | 2,231,774,996.377214 | 1.16795e-13 |
| 1e-19 | 2,231,775,082.9038944 | **1.05549e-12** |
| 1e-20 | 2,231,775,091.5565624 | **2.66043e-12** |

The last two admissible inputs falsely fail the unchanged assertion. The
stable expression agrees with the 70-digit answer within 1.01e-16 for these
four inputs. Local long double has a 64-bit significand; the server compiler
arithmetic has not been independently identified. **These rates are test
inputs, not measured step-12 rates.** The failing rate, node and error must
still be captured to close attribution of this particular abort.

## Other trajectory observations

Through step 11 all 400 non-prescribed nodes were reported free (zero lower
active). The minimum free rate fell from 2.1444e-13 at step 2 to 3.0611e-17
at step 11. Global maximum V remained at the deep prescribed 1e-9 m/s from
step 2 onward, far below the 1e-3 event threshold. Step-11 maximum free V was
6.5110871e-10 m/s. Do not interpret the global max as an accelerating rupture.

The production surface-sample normal-stress range widened from
[49.8526, 50.0754] MPa initially to [17.4589, 69.5494] MPa at step 10 and
**[-12.2348, 90.8439] MPa at step 11**. Meanwhile the twelve station values
at step 11 span [50.0805, 52.5636] MPa. These stations represent the mass-solved
surface traction field, not extrema over every domain quadrature sample.
Consequently the negative sample cannot be located, assigned to deep/free
support, or declared a physical tensile segment from the present summaries.
Its location and weight need separate diagnostic capture. No causal link to
the Theta assertion is established by this observation.

## Smallest recommended next action

1. Correct only the benchmark reference arithmetic to a cancellation-safe
   extended/high-precision aging-law evaluation, retaining the 1e-12 check
   and the independently specified law. Add a regression with the captured
   old state/dt and near-bound rates above, plus an intentionally wrong-history
   input that must fail. No production state law or tolerance change is needed.
2. Make failure reporting identify step, fault/node/xd, accepted V, old Theta,
   committed Theta, reference and relative error, with MPI-safe reporting.
   Keep the original artifacts. Confirm the server uses the matching source.
3. Only after that change is reviewed, replay from completed step-11
   `restart/03` to inspect step 12. It is the last saved coherent state; an
   in-memory postprocessing failure does not create a valid step-12 restart.
   Capture the normal-stress extrema's sample locations separately during
   the same bounded replay rather than launching the first-event continuation.

No replay or fix has been performed for this analysis. This report establishes
an unreliable diagnostic and accurate preceding histories, not permission to
ignore the exception or claim first-event verification complete.
