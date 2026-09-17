# K5: origin of the first resolved 40-km dip

## Conclusion

At the first clearly growing dip, releasing the free-side endpoint removes
**94.88% of the newly generated neighboring-chord defect**. The new endpoint
is 0.999674194 Vp; the deep side stays exactly Vp. Incoming local Theta and
old stress have very little notch-shaped structure. This is substantially
stronger evidence than the late-state comparison: **the shared prescribed
velocity generates most of the first strong dip in this discrete model**,
rather than merely responding to an already developed local Theta peak.

A small initialization seed and a small residual early dip remain. The
experiment does not prove that all later notch growth has this one cause,
or qualify a discontinuous committing formulation. Nor does it remove the
raw stress variation: normal traction changes only slightly. No further
mechanical experiment or trajectory continuation was performed.

## 1. When the notch starts

Define the velocity chord defect at the last free node by

\[
 d_k={\tfrac12[V_k(39.90)+V_k(40^-)]-V_k(39.95)\over V_p}.
\]

In the original control, 40-minus is the shared prescribed vertex. The
following values use the saved revised-work 50-m trajectory. Theta peaks
are **incoming** nodal values at 39.95 km minus their neighboring chord.

| Accepted step | Time (yr) | Velocity chord defect | Incoming Theta peak (s) |
|---|---:|---:|---:|
| Initialization | 0 | 5.23662e-6 | 0 |
| 1 | 0.0788151 | 3.22731e-6 | 0 |
| 2 | 0.1574943 | 2.80841e-4 | 6.89908 |
| 3 | 0.3077717 | 8.64352e-4 | 604.580 |
| 4 | 0.5948016 | 2.15721e-3 | 3428.256 |

The nonzero initialization seed is not concealed. It decreases at step 1;
step 2 is the first clear growth, about 87 times the preceding defect.
It is therefore the selected test, not a late state with a large inherited
notch. The growing Theta peak follows the rate depression in the split
history cycle: step 2 consumes the 6.90-s peak and leaves the 604.58-s peak
for mechanics 3. These updates were observed, not recomputed as inputs.

There is also a broad physical forcing change at step 2. At 15 km the supplied
initial Theta is 8000 s, and its first accepted aging update gives
2.143589e6 s. V(15 km)/Vp falls from 0.999956 at step 1 to 0.000362469 at
step 2. At 39 km it falls to 0.999625851. Thus the free region develops a
nonuniform mechanical response while the deep region remains fixed at Vp.
This timing is consistent with the junction exciting the previously measured
alternating discrete response. It does not by itself prove a unique remote
loading mechanism, and no additional altered-history solve was used.

## 2. How much local structure is already incoming?

At mechanics 2 the last three retained Theta values are:

- 39.90 km: 8,000,008.059231 s;
- 39.95 km: 8,000,010.928694 s;
- 40 km: 8,000,000 s.

The 6.899079-s peak is only 8.62e-7 of the local state. Offline substitution
of its neighboring chord into the **diagnostic friction calculation only**,
with the same V, normal traction and work quadrature, changes the last free
row's mass-normalized friction by **0.430667 Pa**. No such substitution is
made in either mechanics solve.

The retained working stress is recovered independently from saved current
constitutive fields as

\[
 \beta\tau_{\rm old}^{\rm FE}
 =\tau_k-2\kappa_k(\dot\epsilon_k-\chi V_k S),\qquad
 \kappa_k=-\eta\operatorname{expm1}(-G\Delta t_k/\eta).
\]

Here the resolved fixture has G=32038120320 Pa and eta=1e26 Pa s. The
initialization interval is still 4e6 s. No published post-commit stress array
is substituted for this old working field. During the actual B preflight,
the production material evaluator exports that same frozen FE tensor
directly; it agrees with the control reconstruction to **8.64e-10 Pa**.

At row 39.95 km, the work-weighted incoming contributions are:

- old shear: +86.918286 Pa;
- old contribution to minus tau:N: -2.571820 Pa;
- total old contribution to R, old shear minus mu times old minus-normal:
  +88.283700 Pa;
- departure of that last quantity from the neighboring weak-row chord:
  **-0.038513 Pa**.

This old stress is not pointwise zero or perfectly smooth. Raw sampling
shows similar transverse/discretization variation at the junction and an
interior control, rather than a strong unique incoming junction hotspot:

| Window (km) | Old shear range (Pa) | Weighted shear RMS about mean (Pa) | Old minus-normal range (Pa) |
|---|---:|---:|---:|
| 39.8--40.2 | [-3.751, 190.657] | 55.175 | [-39.471, 15.288] |
| 39.0--39.4 | [-3.391, 189.131] | 54.642 | [-39.921, 16.052] |

The small weak-row chord departure does not claim exact pointwise history
homogeneity. Both raw and weak evidence are retained. Initialization and
mechanics 1 consume zero retained perturbation stress to reconstruction
roundoff, consistent with timestep-zero retention of supplied zero history.

## 3. The bounded early A/B comparison

A is the saved shared-trace step 2. B starts from a verified reconstruction
of initialization and accepted step 1, using the exact same physical time
and dt=2482929.0612529977 s. The existing independent-trace basis is unchanged:
the free endpoint is an unknown; the deep segment stays prescribed Vp;
geometric Theta interpolation is untouched. Nothing is smoothed, projected
again, transferred between fault spaces or updated within Newton.

| Quantity, normalized by Vp | Incoming step 1 | A: shared | B: independent |
|---|---:|---:|---:|
| V(39.90 km) | 0.999996230 | 0.999680893 | 0.999657919 |
| V(39.95 km) | 0.999994888 | 0.999559605 | 0.999648602 |
| V(40-minus) | 1 | 1 | 0.999674194 |
| Chord defect | 3.22731e-6 | 2.80841e-4 | 1.74545e-5 |
| Depth below both free-side neighbors | 1.34230e-6 | 1.21288e-4 | 9.31696e-6 |
| Separate deep-minus-free endpoint jump | 0 | 0 | 3.25806e-4 |

The **newly generated** chord defect is 2.77614e-4 in A and 1.42272e-5 in B:
a 94.88% reduction. The free-side dip is therefore substantially removed,
not merely hidden by comparing V(39.95) with Vp. A small 9.32e-6 local minimum
remains; the separate 3.25806e-4 endpoint jump is not counted as that minimum.
All 441 B free nodes remain above the lower bound.

The two incident-element residuals at node 39.95 km change from
**-1785.496 / +1785.496 Pa m** in A to **-154.385 / +154.385 Pa m** in B.
The new endpoint equation has only its free-side contribution, with residual
6.60e-9 Pa m and zero deep-side contribution. The detailed force budgets
include driving traction, friction and damping separately.

## 4. Raw stress and interpretation

At the same 2387 physical QPs in 39.8--40.2 km:

| Current constitutive quantity | A | B |
|---|---:|---:|
| Perturbation pressure (Pa) | [34.963, 150.846] | [35.326, 151.432] |
| tau_xx (Pa) | [-173.470, 1074.479] | [-178.161, 1072.667] |
| tau_yy (Pa) | [-1471.909, 846.893] | [-1471.296, 846.838] |
| tau_xy (Pa) | [-458.593, 707.588] | [-466.511, 707.110] |
| sigma_n minus 50 MPa (Pa) | [-213.780, 113.700] | [-225.481, 103.615] |

Maximum matched-point pressure, normal-traction and shear-traction changes
are 7.04, 17.18 and 36.33 Pa. There is no tensile normal stress. The early
raw normal-stress variation is almost unchanged, not cured by the much
smaller rate dip. The broader 37--43-km normal range is likewise essentially
unchanged (379.472 vs 379.471 Pa peak-to-peak). This discriminates creation
of the velocity notch from a claim to have removed all stress errors.

Together with the prior negative adjacent-rate impulse response, the result
supports this sequence: broader free-region slowdown meets a hard shared
Vp vertex; the continuous discrete coupling produces an exaggerated adjacent
minimum; the accepted aging update then imprints and amplifies that minimum
in Theta. The early test removes nearly all new growth, whereas releasing
the same trace late removes only about half the local defect because it
does not erase the already evolved histories. The last statement is a
mechanistic interpretation consistent with both tests, not a new committing
experiment or proof of the exact division among later history effects.

## 5. Verification, attempts and cost

The original early checkpoint no longer existed. The accepted short prefix
was reconstructed and checked against original fault, particle and QP data.
Maximum relative differences: V 3.10e-15, Theta 4.66e-16, I_h zero, slip
2.61e-15, pressure 3.20e-11 and tau_xx 7.51e-12. Particle IDs and geometry
match; all history comparisons pass. The directly exported incoming FE
tensor check above additionally verifies the actual mechanics input after
ordinary advection and transfer.

An initial harness mistake is preserved under `early-free-trace/`. Truncating
the clock after step 1 produced the right accepted fields but checkpointed
an uncapped next dt approximately 2.66627e6 s instead of the original
2482929.061 s.
The resumed probe's exact-clock assertion stopped it **before any linear
direction or mechanical update**. Its files and checkpoint remain intact.
The harness was corrected to retain the step-2 clock entry while still
stopping after accepted step 1. No checkpoint bytes, physical state or
controller semantics were patched. The corrected prefix is in
`early-free-trace-matched/`; its log records the exact next dt selection.

Only one alternative mechanical solve actually ran. It converged in
186.239 s on four ranks, with 241 Krylov iterations and **14 passing fresh
linear checks**. Final normalized bulk/surface residuals are 2.06832e-12
and 6.90711e-10, respectively; final absolute bulk norm is 4.18990e-6 and
surface RMS 0.00182173 Pa. K column finite differences contract as expected
(new endpoint error 1.80e-7 to 1.80e-9). B and G velocity derivative errors
are 1.88e-15 and 3.19e-15; pressure derivative 6.89e-11; virtual work error
1.88e-15. Sparse/reference actions and surface-inverse checks pass.

The benchmark-only early guard selects step 2 instead of step 10 and exports
the actual incoming FE history. The unrelated candidate-aging derivative
test is not run in this lagged-Theta probe: its tiny-rate finite difference
is cancellation-limited at the shorter dt, and no state derivative enters
these equations. No K/G accuracy check or solver tolerance was weakened.

Rollback restores bulk, current/committed V, all surface/particle histories
and geometry. Original and copied checkpoint hashes are unchanged and no
accepted step-2 record is written. The status-1 exception is the intentional
post-convergence diagnostic termination. The prefix commits only the ordinary
reconstructed initial state and step 1 in its disposable output directory.

Successful prefix + probe cost 80.837 + 186.239 = 267.076 s. Including the
preserved first prefix (81.084 s) and pre-solve clock rejection (29.721 s),
simulation execution totaled 377.881 s. Each MPI launch had a 600-s cap. Maximum
recorded child RSS for the successful probe was 1,488,796 KiB (1.420 GiB),
not aggregate MPI memory. Plugin Release build used -j4. The offline checks,
Python syntax checks and `git diff --check` pass. No complete test suite,
late replay, particle-density change or additional spatial level was run.

## 6. Artifacts and changed files

- [Onset and incoming-history table](../../../benchmarks/reconstructed_fault/bp3/early-free-trace-matched/analysis/onset.csv)
- [Comparison summary](../../../benchmarks/reconstructed_fault/bp3/early-free-trace-matched/analysis/comparison.json)
- [Velocity/raw-stress plot](../../../benchmarks/reconstructed_fault/bp3/early-free-trace-matched/analysis/early_comparison.png)
- [Incident-element force budgets](../../../benchmarks/reconstructed_fault/bp3/early-free-trace-matched/analysis/early_element_budgets.csv)
- [Matched full stress tensors](../../../benchmarks/reconstructed_fault/bp3/early-free-trace-matched/analysis/early_matched_stress.csv)
- [Incoming raw-history/control statistics](../../../benchmarks/reconstructed_fault/bp3/early-free-trace-matched/analysis/incoming_raw_history.csv)
- [Prefix agreement](../../../benchmarks/reconstructed_fault/bp3/early-free-trace-matched/prefix/comparison.json)
- [Probe log](../../../benchmarks/reconstructed_fault/bp3/early-free-trace-matched/probe/run.log)
- [Execution and checkpoint hashes](../../../benchmarks/reconstructed_fault/bp3/early-free-trace-matched/probe/execution.json)

New maintained scripts: `run_early_trace.py`, `analyze_early_trace.py`.
The existing benchmark `bp3.cc`, `junction_diagnostic.h` and
`within_step_diagnostic.h` receive only selected-clock, input-export and
diagnostic-check changes. The existing addendum and README record the new
bounded task. No additional `source/` or production-header changes were made
in this task; prior working-tree changes are preserved. Build log, plugin
and executable hashes and a benchmark source diff are saved with the probe.

The executed commands were:

```sh
python3 benchmarks/reconstructed_fault/bp3/run_early_trace.py prefix
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/bp3/analyze_early_trace.py prefix
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
python3 benchmarks/reconstructed_fault/bp3/run_early_trace.py probe
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/bp3/analyze_early_trace.py probe
```

The driver refuses to overwrite existing runs. Stop here: the first-dip
mechanism is substantially identified for this frozen state; choosing a
physically and numerically justified production representation remains a
separate decision.
