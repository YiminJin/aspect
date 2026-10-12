# Bounded P2 material study

Current result: [P2 direct irreversibility](../../../doc/reconstructed_fault/propagation/P2_irreversibility_report.md).
The physical admissible set now uses the accepted phase as lower bound; initial
relaxation remains box-only. Three fresh a=.256 paths all complete to t=1 with
zero healing and improving temporal agreement. Source normalization error and
profile work remain unresolved; material search and P3 stay paused.
[Run/replay commands and evidence](IRREVERSIBILITY.md). Earlier results below
remain historical, including their failed qualifications.



Current result: [P2 coupling repair](../../../doc/reconstructed_fault/propagation/P2_coupling_report.md).
Surface-owned history and conjugate mechanical work pass fixed-phase checks.
The reference material stays unchanged; one a=0.256 ramp at three dt exposes
resolved tail healing. Previous high-load, preloaded and material-study evidence
remains historical and failed. No material optimization or P3 is resumed.

Current result: [recoverable-energy P2 revision](../../../doc/reconstructed_fault/propagation/P2_recoverable_report.md).
The unloaded fixture and local state driver pass, but three coupled runs remain
unqualified because of traction-transfer and work-exchange discrepancies.

Current follow-up: [history semantics and timestep diagnosis](../../../doc/reconstructed_fault/propagation/P2_history_report.md).
Material tuning is paused. Native tests confirm the incremental-candidate/max-H
problem; no replacement law or stress-scaling qualification has been selected.

The question is whether increasing Gc, while retaining strength, elasticity,
viscosity and length scale, broadens the phase-versus-slip response. This is a
fresh initialization for each material; previous high-load failures and the
normalization report remain valid and preserved. No production algorithm change
is selected. Results are reported in
[the material-study report](../../../doc/reconstructed_fault/propagation/P2_material_report.md).

Domain [0,1]² m, sharp fault (0,0.5)–(1,0.5) m, fixed 32×32 square bulk mesh,
Q1 phase/Q2 velocity, fault spacing 0.1 m, 3×3 reference particles per cell
(9,216 total), full normal association width 0.50000000000001588 m. Particles,
geometry, mesh and material fractions (0.5/0.5) stay fixed. The existing frozen
RK2 diagnostic suppresses position transport only, not constitutive time.

Both constituents use G=1e6 Pa, c=1000 Pa, eta=1e8 Pa s (bounds 1e6–1e10 Pa s),
ell=0.15625 m, AT1 with c0=8/3, Lorentz curvature p=1. Density is 3300 kg/m³,
gravity zero, static temperature 293 K. Stateless rate-dependent friction has
reference=dynamic coefficient 0.1, characteristic rate 2e-6 m/s, minimum slip
rate 1e-20 m/s. Normal friction pressure is the native adiabatic 10000 Pa;
pressure normalization is surface. Radiation damping is zero. No RSF theta is
active. The only material parameter varied is Gc:

| Case | Gc (J/m²) | m=Gc/(c0 ell Hc) | Hc=c²/(2G) (Pa) | Gc/c (m) |
|---|---:|---:|---:|---:|
| Reference | 26.6666666667 | approximately 128 | 0.5 | 0.0266666666667 |
| Primary | 106.6666666667 | approximately 512 | 0.5 | 0.1066666666667 |
| Optional, only for abrupt primary evolution | 266.6666666667 | approximately 1280 | 0.5 | 0.2666666666667 |

Nominal initial core phase is 0.6. Native geometry-dependent profiles initialize
particle H, the rounded guess relaxes at fixed H, and native routines initialize
q and current/previous I_h from the accepted phase. The native energy coefficient
E=Gc/(c0 ell) is 64, 256 or 640 Pa (up to decimal input precision), not a fixed
64 Pa. Inside the prescribed active profile H0=E*0.6/[h(0.6)*g(phi_guess)²];
outside it the intact initializer retains Hc. Stress initially vanishes. No
histories or relaxed fields are imported from another material case.

All velocity boundaries impose vx=a*t*(y−0.5), vy=0. Phase has natural zero
normal flux. The interval is 0–1 s, initially dt=0.125 s; imposed top-minus-bottom
displacement is a*t²/2 m. Calibrate reference a through 0.004, 0.008 and 0.016
s^-2 only, stopping at meaningful accepted feedback. Use that same loading for
all Gc comparisons. Repeat only a preferred evolving case with dt=0.0625 s.
Consistent normalization is fixed throughout this study. No damping, limiter,
phase cap, H change or previous-phase inequality is introduced.

Phase/H/q/velocity/pressure/stress/slip and normalization evolve in the native
sequential cycle: incoming H -> phase -> mechanics -> committed H. Slip in the
observer is the accumulated accepted native V*dt, consistent with discrete
step-end velocities. This is not particle-transport, birth, coupled-restart,
rejected-step rollback or fault-propagation qualification.

Phase bounds are exact nodally, CPDI allowance 3e-14, KKT allowance 1e-11,
healing allowance 4.922e-12; native phase relative tolerance 1e-12 and linear
2e-7 are unchanged. Mechanical tolerance is 1e-8. Bad phase entries are saved
before mechanics and stop the run; positive normalization is required and
accuracy is measured separately against exact transverse Q1 integration,
including endpoints. No arbitrary projection-accuracy pass is inferred from the
native 1e-8 quadrature/tail tolerances.

Build and example command (labels refuse overwrite):

```sh
cmake --build build-propagation-integration/plugin --target material_feedback -j2
python3 benchmarks/reconstructed_fault/propagation/run_material.py material_reference_a004 --case reference --amplitude 0.004
```

The observer reuses `coupled_feedback.cc` for history-publication and native
candidate/max-H checks. Its core-profile diagnostic now reads the configured
fault core. `material_feedback.cc` adds runtime calibration, frozen nodal states,
direct normalization checks, local phase/q/V/slip and CPDI-versus-pointwise
normalized driving. Data go into fresh `build-propagation-integration/output-*`
directories; logs, input hashes and final compact plots are retained under
`material_evidence/`. No hidden parameter-selection environment switches exist.
