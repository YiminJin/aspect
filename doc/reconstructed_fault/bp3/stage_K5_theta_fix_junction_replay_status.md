# Theta audit fix and junction diagnosis: replay blocked by build compatibility

Date: 2026-09-14. This is a partial implementation/analysis report, not a
completed step-12 or shifted-junction verification.

## Completed checker correction

Only the BP3 benchmark's independent reference calculation has changed.
`BP3::aging_state_reference()` in `benchmarks/reconstructed_fault/bp3/bp3_model.h`
evaluates in long double

\[
 x=V\Delta t/D_c,\qquad
 \Theta^{ref}=\Theta_{old}e^{-x}-(D_c/V)\operatorname{expm1}(-x).
\]

The benchmark postprocessor uses this helper at real timesteps and continues
to check retained supplied Theta at timestep zero. Its 1e-12 relative
assertion is unchanged. On failure it selects a worst node collectively and
prints step, fault/node, xd, accepted V, old Theta, committed Theta, reference,
absolute error and relative error before all ranks throw. Nonfinite audit
errors are failures rather than being lost by a maximum operation.

Production `FaultFriction::update_state`, history publication, physical
parameters, pressure treatment, timestep controller, output visualization,
and solver criteria were not changed. No constitutive diagnostic was generated
by treating committed stress as an old Maxwell input.

### Focused verification

```
c++ -std=c++17 -O2 \
  -isystem /opt/dealii/9.6-local/include/deal.II/bundled \
  benchmarks/reconstructed_fault/bp3/test_theta_audit.cc \
  -o /tmp/bp3_test_theta_audit
/tmp/bp3_test_theta_audit
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
```

- PASS: five rates (3.0611065791756763e-17, 1e-18, 1e-19, 1e-20, 1e-9 m/s)
  against an independent 50-decimal-digit calculation. Published-double versus
  rounded-reference relative errors are all zero for these cases; the
  long-double reference also passes the stricter high-precision comparison.
- PASS: a deliberately wrong committed Theta (relative perturbation 1e-8)
  is rejected at each rate with the existing 1e-12 criterion.
- PASS: zero interval retains the input state.
- PASS: Release benchmark plugin build with -j4; `git diff --check`.
- The first standalone compile lacked Boost's include path and failed before
  testing. The command above supplies the existing deal.II bundled headers;
  no dependency or numerical change was made.
- Not yet tested: actual step-12 audit, MPI failure-message execution, raw
  sample output, or noncommitting shifted-junction solve.

Full unit output: `benchmarks/reconstructed_fault/bp3/theta_junction_audit/theta_unit.log`.

## Why no checkpoint replay was launched

The supplied run used deal.II 9.6.0 **with 64-bit indices**, Trilinos 15.0.0,
p4est 2.8.5 and 64 MPI ranks (`first_cycle_coarse/log.txt`). The existing local
Release build uses deal.II 9.6-local **without 64-bit indices**, Trilinos 14.2
and p4est 2.8.7. The other installed deal.II configurations inspected
(9.5-local and master) also disable 64-bit indices.

This is not just an MPI rank-count difference: deal.II's
`particles/property_pool.h` selects `types::particle_index` as uint64_t or
unsigned int using `DEAL_II_WITH_64BIT_INDICES`. The checkpoint contains
serialized particle data and binary archives. Cross-width restoration has not
been established; it must not be assumed to reconstruct identical particle
identities/history. No archive conversion or unchecked local replay was tried.
A compatible server build/environment is needed to perform the requested
baseline and diagnostic safely, or a separately arranged compatible local build.

`restart/03` remains untouched. SHA256 hashes of its mesh, fixed/variable data
and resume archive are recorded in
`benchmarks/reconstructed_fault/bp3/theta_junction_audit/checkpoint03.sha256`.

## Available junction evidence, without substituting lifecycle quantities

`analyze_saved_junction.py` extracts only the saved surface Q1/nodal fields:

```
python3 benchmarks/reconstructed_fault/bp3/analyze_saved_junction.py \
  benchmarks/reconstructed_fault/bp3/first_cycle_coarse \
  benchmarks/reconstructed_fault/bp3/theta_junction_audit
```

This extraction has completed. The script uses exclusive-create output files
so repeating it cannot overwrite evidence. `saved_junction_history.csv`
contains all 11 available full-profile states (0, 2–11), including shear extrema
and their locations. `saved_nodal_11.csv` contains 10–45 km, explicitly marking
15, 18 and 40 km. It reports **consistent projected shear q and committed
Theta**, not raw constitutive samples or mechanics-input Theta. It deliberately
does not invent pressure/normal-stress columns that the input lacks.

| Step | V at last free node / Vp | Relative rate difference | q peak-to-peak, 13–20 km (MPa) | q peak-to-peak, 37–43 km (MPa) |
|---|---:|---:|---:|---:|
| 0 | 1.000124 | +0.000124 | 0.401653 | 0.018985 |
| 5 | 0.898689 | -0.101311 | 0.341605 | 0.031791 |
| 8 | 0.483349 | -0.516651 | 1.409482 | 0.721899 |
| 10 | 0.166805 | -0.833195 | 2.747990 | 3.933680 |
| 11 | 0.068583 | -0.931417 | 3.770770 | 8.172248 |

The last free node is at 39.9 km; the first prescribed node is at 40.0 km,
where Vp=1e-9 m/s. These are **neighboring-node differences**, not a
discontinuous value jump in the Q1 field: the mathematical left limit at the
40-km node equals Vp. At step 11 the 100-m segment accommodates a
9.31417199e-10 m/s rate change (slope magnitude 9.31417199e-12 s^-1).
This increasing unresolved gradient is relevant to the junction hypothesis.

The q variation over 37–43 km grows to 2.167 times that over 13–20 km by
step 11. This is a comparison of **total shear** over fixed windows, not a
pressure or normal-stress dipole measurement, and does not establish causality.

## Missing evidence and required continuation

1. **Corrected step-12 audit:** pending; no ASPECT replay was launched.
2. **Exact raw minimum/classification/decomposition:** pending. Existing
   summary minimum -12.234839 MPa cannot establish which parent/sample is
   tensile, whether its Q1 support crosses Wf, or its integrated weight.
   Existing station normal values are projected and cannot substitute.
3. **Step-11 raw constitutive samples:** not archived. A step-11 checkpoint
   contains committed stress/Theta/C, not automatically the preceding inputs
   consumed by that solve. Reconstruction must recover those inputs using
   the retained earlier state, or explicitly obtain approval for a step-11
   replay from the preceding checkpoint. Simply evaluating a new Maxwell
   update at restart/03 would be the wrong quantity.
4. **Step-12 saved graphics:** only 53 of 64 `bulk_12_*.vtu` pieces are present,
   and `bulk_12.pvtu` is absent. They are not a complete accepted state. The
   new audit must use a genuine replay from restart/03, not these partial files.
5. **Raw and consistent normal/pressure profiles, region widths and tensile
   weights:** pending capture of pre-publication samples. Existing graphical
   `tau_xx/tau_yy/tau_xy` are transferred FE history fields; they must not be
   silently relabeled as the actual constitutive stress.
6. **35-km causal diagnostic:** not run or implemented. It remains conditional
   on the fully characterized baseline and must use frozen preceding history,
   no state commit, the existing additional prescribed nodes and no other
   parameter changes. No claim that the hotspot moves has been made.

No smoothing, normal-stress clamp, altered Vp, pressure adjustment, history
transfer change, standard graphical-output modification or long-run
continuation has been performed. The junction explanation is supported by
correlation in saved nodal data, not yet a completed causal diagnosis.
