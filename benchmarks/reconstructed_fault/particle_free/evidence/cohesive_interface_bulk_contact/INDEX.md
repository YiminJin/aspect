# Fixed-path bulk/contact evidence

[Concise report](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_BULK_CONTACT_REPORT.md) · [predeclared controls and amendments](../../../../../doc/reconstructed_fault/particle_free/COHESIVE_INTERFACE_BULK_CONTACT_SELECTION.md) · [commands](COMMANDS.md) · [cleanup and restore mapping](../../../maintenance/disk_cleanup_2026_10_27/README.md).

The numerical correction uses a resolved, fixed lifting support. The fitted
reference checks work-conjugate loads, storage and subsequent evolution.
Raw point-traction mismatch remains unqualified on the Cartesian cut mesh;
the tied-ahead 2% normal-perturbation target also retains a small failure.
Production/BP3, physical laws and the original failed evidence are unchanged.
This is serial isolated native mixed mechanics/C++ evolution, not a production
checkpoint conversion or Simulator integration.

Read the figures in this order:

1. [Lifting mechanism](lifting-mechanism.png): an arbitrary representation edge
   acts as an artificial spring. Independent pressure-free 1D prediction versus
   actual 2D load change.
2. [Frozen shear/contact](frozen-shear-contact.png): common physical geometry,
   Q1 trace/tests and memory; weak loads alongside retained raw Cauchy diagnostics.
3. [Storage/incompressibility](bulk-storage-incompressibility.png): perturbation
   energy, excluding the large background; strong versus weak incompressibility.
4. [Short imported-history controls](short-growth-fronts.png): useful operator
   diagnostics, with separately measured initialization disequilibrium.
5. [Fresh fixed-path confirmation](fresh-fixed-path.png): actual physical
   evolution from matching equilibrated initial states, without imported history.
6. [Fresh causal workflow](causal-fronts.png): successive insertions and real
   fully decohered/process fronts; old results remain visible.

| Compact file | Definition |
|---|---|
| `frozen-loads.csv`, `frozen-errors.csv` | Regional common-test loads and absolute/relative differences from fitted96; all failed targets retained |
| `frozen-profiles.csv` | Identical physical Gauss coordinates/weights, total physical jump, applied strain, relaxed affine/Maxwell memory, Q1 dual shear, normal work reaction, raw sided normal pressures and raw sided-mean shear |
| `frozen-assessment.json` | Common-test definitions, regions and reference scope |
| `incompressibility.csv` | Physical measures, total/perturbation storage, divergence norms, pressure norms and ∫p′div(r); includes hard-region measure uncertainty |
| `lifting-reference.csv`, `lifting-reference-profiles.csv` | Independent 1D FE/projection calculation; energy per fault area J/m², shear Pa, cross-fault coordinate m |
| `lifting-mechanism.json` | Analytic prediction, sign convention and measured 2D comparison |
| `short-comparison.csv`, `short-errors.csv`, `short-profiles.csv` | .42025–.445 s diagnostic controls and two timestep halvings; increments relative to common incoming histories |
| `fresh-comparison.csv`, `fresh-errors.csv`, `fresh-profiles.csv` | Fresh equilibrium initialization and fixed-path evolution to .445 s |
| `own-time-errors.csv`, `own-time-*-history.csv` | Late time halvings from each fresh run's own accepted checkpoint, same basis and complete histories, no reconstruction |
| `fresh-*-initial-checks.json` | Actual bounded initial tied reactions and true weak/deviator checks |
| `causal-comparison.csv` | Original and corrected fresh causal trajectories, actual fronts, transfer defects and publication counts |
| `*-history.csv`, `*-summary.json`, `short-*-evolution.json` | Per-step physical work/storage/dissipation, source checks, lifecycle evidence and measured costs |
| `legacy-identity.json` | Seven retained default operator/trace tables bitwise unchanged |
| `native-cost.csv`, `cost-summary.json` | Actual matrix sizes and measured native/controller times; avoid double-counting nested timers |
| `verification.json`, `preservation.json`, `changed-files.json` | Artifact checks, protected sources/BP3/checkpoints and changes relative to task entry |

All lengths are m, times s, stresses Pa (figure axes often kPa), slip m, rate
m/s, surface energy J/m² and 2D integrated energy J/m of out-of-plane thickness.
The 1D strip energy is per unit fault area, unlike whole 2D stored energy.
All plotted profiles are evaluations of actual saved fields; constitutive
histories are not interpolated into another state. Q1 dual shear is the common
mass projection of the tangential work functional, not a pointwise stress.
`raw_normal_same_bubble_Pa` is an additional matching-functional comparison;
the original raw sided values remain available separately.

Physical p0=10000 Pa and mean-zero solved p′ are distinguished. Numerical
lifting and normal-test vertical half-widths .75 m have different roles from physical
R=.75 m; their equality is not a constitutive identification. Test bumps are
the common Q1 trace of sin² at fixed physical nodes on regions old
[3.75,5.5], developed [6,6.5], zone [6.5,7.75], ahead [7.75,8.4] m.

Imported short controls independently reconstruct the old bulk response. Their
initial free residual and storage difference are explicit; they are not valid
accepted-state conversions. Missing new tied reactions are distinguished from
that free residual. An unconstrained least-squares multiplier peak is not a
bounded-admission certificate. Fresh controls instead start from the same
continuum prescription and pass a bounded weak-equilibrium certificate.

Large basis tables and all accepted states remain under
`build-particle-free/cohesive-interface-bulk-contact/`, indexed by hashes in
`raw-index.json`; complete new state archives are hashed in `state-index.json`.
Original work96/checkpoints remain unchanged. `physical-diagnostic.csv` in a
kernel directory evaluates its specified probe coefficients, not a trajectory;
accepted physical quantities come from states and history tables. The fitted
geometry is used only for independent/fixed-path checks, not arbitrary history
transfer on extension. Corrected causal runs use the fixed Cartesian mesh and
nested old basis; every zero-time event verifies physical stress/storage/history
preservation. There is no new ZIP: the useful compact outputs stay directly
accessible, with checksums, following the cleanup request.
