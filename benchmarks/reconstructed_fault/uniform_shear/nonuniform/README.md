# Bounded K2.1 pilot (prescribed normal stress)

Completed through 2 s in 414.14 s / 933372 KiB peak RSS. See
`doc/reconstructed_fault/benchmarking/stage_K2_1_pilot.md` and
`measurements/report.json`. Actual normalization and lifecycle checks pass;
K2's original 1e-6 containment requirement remains unmet (5.914e-5 omitted).
No support change, inherited K1 allowance or K2.2 run has been made.

This is a full coupled ASPECT solve, not independent scalar K1 solves along
the fault. It reuses the accepted 64x256 K1 bulk mesh, 1/128 m fault spacing,
3x3 particles per cell, ell=0.15625 m, dt=0.5 s and all physical coefficients,
loading, initial stress/H and numerical tolerances. Only Theta's physical
initial field changes. End time is 2 s (initialization plus four real steps).

For a=0.0625 m and xc=0.125 m,

    Theta0(x) = 200 s * (1 + 0.05 f(x)),
    f(x) = [(1+cos(pi*(x-xc)/a))/2]^2   if |x-xc|<a,
           0                           otherwise.

This positive compact C3 bump is constant near the periodic x boundaries.
The full width is 0.125 m (32 bulk cells / 16 fault intervals). At fixed V,
the peak friction change is approximately 1000*0.013*log(1.05)=0.6343 Pa.
A local slip-rate response of roughly 1e-7--1e-6 m/s is expected, accompanied
by a coupled response outside the initially perturbed interval. Its sign
and magnitude must be measured, not imposed. ell/a=2.5: this pilot verifies
finite-width coupling, not a sharp-fault asymptotic approximation.

The benchmark plugin constrains the *converged initial Q1 phi* only in the
phase-field solve, not mechanics. Its existing fixed-H mode retains H.
Theta0, particle stress0=1500 Pa and initial C0 are retained through the
artificial Maxwell initialization interval; the solved initial V is committed.
Real timesteps update Theta/C/stress. The generic theta_initial field does
not enter chemical material fractions. Surface temperature remains 293 K.

Production postprocessors write `output/solution.pvd` and
`output/reconstructed_faults.pvd`. Bulk composition tau_xy is a history field,
not the evaluated mechanical stress at t=0; use the diagnostic raw-QP stress
CSV for the complete evaluated Maxwell decomposition. No stress smoothing
or filtering is applied. The homogeneous comparison is the fully coupled
K1 space64_dt05 trajectory with identical physical times and initialization,
not a local scalar reference.

Estimated cost before execution: 400--600 s, about 1 GiB peak resident memory
on one rank, based on the validated seven-/thirteen-state K1 runs. The runner
has a 1200 s wall cap. This is not the K2.2 convergence campaign. The original
1e-6 support-containment target remains unapproved for relaxation in K2;
measure and report it even if the mechanical pilot succeeds.

Run from the repository root:

```sh
python3 benchmarks/reconstructed_fault/uniform_shear/convergence/run_case.py \
  benchmarks/reconstructed_fault/uniform_shear/nonuniform/pilot.prm \
  --configuration Release --timeout 1200
```

## Proposed K2.2 sequence (review only; not run)

1. At fixed ell, perturbation and dt=.5 s, compare 32x128, the current
   64x256 pilot, and 128x512 bulk meshes, with fault spacing 1/64, 1/128,
   1/256 m respectively. Keep the physical H/stress/Theta initializer fixed.
   Record changes in converged initial phi, projected C0 and realized Theta0
   separately from the later response differences.
2. On the spatially resolved configuration compare dt=.5, .25, .125 s at
   common times through 2 s. Do not run the Cartesian product. If bulk/fault
   errors cannot be separated, one fault-only spacing refinement at fixed
   bulk mesh is preferable to expanding the entire matrix.
3. The finest **full coupled nonuniform solve** is the numerical reference.
   Compare V, Theta, cumulative slip and traction on common arclength points
   using mass/arclength-weighted norms; compare bulk u/p/raw stress on common
   physical sections. A homogeneous paired run can isolate perturbation
   response from initialization changes, but is not a K2 reference itself.
   Do not use corresponding node indices or per-vertex scalar K1 solutions.
4. Retain the analytic compact window and record outside-window response at
   each level. A nonzero far-field signal alone is not convergence evidence.
   Native raw stress, normal averages and actual surface-traction samples
   must remain distinct diagnostics. The present pilot exports the first
   two; a direct surface-traction diagnostic is needed for the K2.2 table.
5. Review K2's measured containment budget before this campaign. No K1-only
   relaxation transfers automatically. Keep support/full I_h unchanged unless
   a separate decision approves a correction. The true-normal-stress branch,
   orientation study and K3 are outside this pilot.

The 128x512 case has four times the cells/particles of this pilot and may
need several GiB and tens of minutes; estimate from the realized pilot cost
and obtain review before running it locally or requesting server resources.
