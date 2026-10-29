# One scalar rate/state candidate: failed mechanical prerequisite

Read `doc/reconstructed_fault/particle_free/PRANGER_FEASIBILITY_REPORT.md` and
the companion FORMULATION/SELECTION files. `decision.json` is a STOP decision.
No cross-fault trajectory, native update, mesh/timestep comparison or phase
propagation is represented by these plots.

The ZIP uses repository-relative paths. From this worktree's root, reproduce
the independent audit and figures with installed NumPy/SciPy/mpmath/Matplotlib:

```
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py pranger-replay 90 python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/check_pranger_feasibility.py
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py pranger-replay-plots 60 python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/plot_pranger_feasibility.py
```

Thread environment and measured commands are recorded in `commands.json`.
No ASPECT build or native shared library is required for this audit. The
compact archived single-system input is bundled, so its old build directory
is not required either. Its provenance JSON identifies the original file and
the intentional rate rescaling. Two-system states are synthetic, explicitly
admissible instantaneous controls; the steady one is constructed analytically.
None is relabeled an accepted candidate checkpoint. Exact maturity is a
separate Q=0 specialization, not a finite-h native phi=1 API call.

CSV definitions:

| File | Meaning/units |
|---|---|
| `mixed-spectrum.csv` | Full 3-D plane-strain, incompressible mixed symbol; wave normal in degrees from physical +x; k ell_r dimensionless; growth in 1/s. `determined=False` means singular evolution, not stable/unstable numeric growth. k ell_r=0 is the k→0 limit of nonzero-wave equilibrium. |
| `oblique-pole.csv` | Frozen reversed-memory angular approach; offset in degrees, growth in 1/s. State-dependent and b=0 laws share the directional pole. Values closest to the pole are floating-point sensitive; use the independent analytic denominator/root for its location. |
| `angular-denominator.csv` | Analytic dimensionless 1/h+kappa cos²(2alpha) versus mixed-matrix action, with absolute error. |
| `mature-null-mode.csv` | Assigned exponents [1/s] for a verified null direction, not predicted growth rates; delta T [Pa], A/K [1], W [1/s], p [Pa]. |
| `mature-null-field.csv` | Analytic illustration on x/y [m], W [1/s] and velocity [m/s]; wavelength .02 m. No numerical grid/phase mask is used. |
| `operator-multipliers.csv` | Selected explicit, local identity and UNSELECTED implicit Fourier multipliers. Plotting the implicit multiplier does not test it as a replacement. |
| `parameters.csv` | Retained inputs versus uncalibrated feasibility choices. |
| `verification.csv` | Measured checks and their unchanged allowances. |

In the full symbol comparison, `explicit_state` uses q+ell_r² Delta q;
`same_RSF_local` retains the identical rate/state law without spatial
interaction; `direct_effect_only` uses b=0, retaining positive a. The
single-system direct-effect limit is separate from the already singular
mature two-system mode. No Kelvin/radiation/slip viscosity or other damping
has been added.

`retained-native-guard.json` preserves a separate original constant-friction
return's three-root error. It is not validation of the proposed law.
`original-limit-arithmetic.json` preserves the auxiliary direct-sinh overflow;
the same asinh law now uses stable logarithmic arithmetic. Original and corrected
logs, including a plotting API error, are retained rather than hidden.
`preservation.json` verifies the 71 protected entry artifacts/sources and prior
status/audit text. `manifest.json` hashes package members except itself; the
outer ZIP has a separate SHA-256. Production/BP3/defaults are unchanged.
