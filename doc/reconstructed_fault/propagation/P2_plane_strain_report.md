# P2 plane-strain audit and bounded spatial/stability follow-up

Updated 2026-10-09. **The optional accepted-FE history path now uses one 3D
plane-strain deviator consistently. Compatible shear is unchanged; the
nonzero-divergence operator is corrected. Both short loaded controls complete,
but spatial accuracy and full phase stability remain unqualified. No P3.**

The helpers themselves are correct and unchanged. The defect was using raw
in-plane symmetric gradients in the FE-owner stress/energy while the framework's
invariant helper assumes an in-plane block of a three-dimensional deviator.
The existing volumetric subtraction was also absent from the selected
incompressible Newton assembler registration. The correction reuses that
assembler and includes the missing out-of-plane stress in storage and loss.
It introduces no pressure reinterpretation, material change, phase kinetics,
new bound, history projection or normalization change.

Q3 velocity enrichment substantially improves the phase/traction comparison
with the rich 1D reference. However, near-constant stress at the native Gauss
points conceals a 409 Pa off-quadrature RMS error and a 10.52% increase in bulk
storage when the frozen field is accurately integrated. This is a concrete
quadrature/velocity-space limitation, not evidence of spatial convergence.
Positive sampled free-set curvature is useful local evidence, not proof of a
minimum on the complete constrained critical cone. The prior affine negative
modes, failures and reports remain preserved.

## Actual model and restrictions

| Quantity | Value / definition | Units |
|---|---|---|
| Domain / fault | [0,1] x [0,1]; through-going horizontal y=0.5 | m |
| Bulk mesh / phase | 16 x 16 uniform cells; Q1 phase; dx=dy=0.0625 | m |
| Velocity / pressure | Baseline Q2/Q1; enrichment Q3/Q2 | — |
| Native normal mechanical quadrature | 3 Gauss points per cell, unchanged in both runs | — |
| Fault / particles | 64 Q1 segments, spacing0.015625; 3x3 particles/cell,2304 total | m / — |
| Associations | Fixed entire normal cross-section, width0.5; fixed sharp endpoints | m |
| G / eta / lambda | 1e6 / 1e8 / 100 | Pa / Pa s / s |
| Viscosity bounds | 1e6–1e10, unchanged | Pa s |
| c / Gc / ell | 1000 / 26.6666666667 / 0.15625 | Pa / J m^-2 / m |
| Lorentz p / m / Hc | 1 / 128 / 0.5; c0=8/3 | — / — / Pa |
| Friction / normal pressure | mu=0.1 constant; prescribed adiabatic10000; manager background0 | — / Pa |
| Radiation damping / gravity | 0 / 0 | Pa s m^-1 / m s^-2 |
| Temperature | 293, fixed | K |
| Seed | Native nominal core0.6, native seeded H, box-relaxed maximum0.613176697049462 | — |
| Initial mechanics | Accepted FE d_R=0; surface Z=0; bulk stress=0 | m / Pa m / Pa |
| Loading | ux=0.256 max(0,t-0.0078125)(y-0.5), uy=0 on top/bottom | m s^-1 |
| Side conditions | uy=0, ux natural on left/right; pressure normalization=no | — |
| Time | dt=0.015625, T=0.625,40 accepted steps after initialization | s |
| Normalization / phase | Consistent I_h; immutable accepted-phase lower bound, upper1 | m / — |

G=1e6 Pa remains a **dimensional synthetic verification value**, not an inferred
rock modulus. The boundary velocity is an interval-mean rate; endpoint mechanics
uses that rate as its constant-rate approximation over the interval. Reported
V_interval is the solved interval rate, not an independently sampled exact
instantaneous velocity. Phase trials freeze accepted d_R and Z; H+ remains a
diagnostic under total-energy driving. Mesh, sharp geometry, particles and
associations are fixed; transport/birth are explicitly disabled. This fresh,
serial FE option still rejects checkpoint conversion/resume and varying
coefficients. These controls do not qualify propagation or general Stokes
localization. The Q3 run changes only the existing Stokes degree setting;
pressure enrichment follows ASPECT's existing element choice.

## Tensor ownership and consumer audit

For a raw zero-zz plane-strain block A, define

\[
\widehat A=\begin{pmatrix}A&0\\0&0\end{pmatrix},\quad
D_3(A)=\widehat A-\tfrac13\operatorname{tr}_2(A)I_3,\quad
D_2(A)=A-\tfrac13\operatorname{tr}_2(A)I_2.
\]

An already deviatoric block T instead has Tzz=-tr2(T), so
`<T,U>3=T:U+tr2(T)tr2(U)`. A genuinely 2D deviator subtracts trace/2;
that is a different operator and is not selected here.

| Consumer / variable | Input meaning | Audit result |
|---|---|---|
| consistent_deviator(A) | Raw zero-zz plane-strain symmetric block | Correct trace/3; output is D2(A); **not idempotent** in2D |
| consistent_second_invariant_of_deviatoric_tensor(T) | In-plane block of 3D deviator | Correct `-0.5*(T:T+tr2(T)^2)`; does not deviate its input |
| Standard strain-rate rheologies / visualizations | Raw FE symmetric rate, then D2 once | Valid helper composition |
| Viscoelastic effective rate / dislocation rate | Linear combinations or scalar multiples of deviatoric blocks | Invariant directly; requires the documented deviatoric old-stress ownership |
| Selected FE d_R gradient / particle velocity gradient | Raw symgrad before correction | Now D2 once in stress, constitutive rate and candidate publication |
| Selected retained stress | Already reconstructed deviatoric block | Never deviated again; legacy particle branch unchanged |
| Stress norms used for publication/error checks | Stored in-plane error block | Norm is a numerical component-error scale, not physical energy |
| Incompressible shear-heating / stress-residual diagnostics | Raw rate in incompressible branch | Valid as a deviator only at pointwise zero divergence; discrete-divergence caveat |
| stress_second_invariant visualization | Total in-plane stress includes p, unknown sigma_zz | Pre-existing inconsistent raw-zero-zz embedding / possible re-deviation; outside selected fixture |
| strain_dependent finite-strain L | Positive left stretch, not a deviator | Pre-existing invariant-helper precondition violation; outside selected rheology |

The complete source-site inventory, including the four new selected calls, is
[tensor-consumers.csv](../../../benchmarks/reconstructed_fault/propagation/plane_strain_evidence/tensor-consumers.csv):35 qualified call sites. No tensor utility,
generic rheology, visualization or projector is changed. The unrelated two
consumer issues are reported rather than bundled into this correction. In
particular, a hydrostatic in-plane total stress cannot be embedded with zz=0
when diagnosing its physical deviator.

## Reduced equations and assembly audit

The single accepted owner remains the velocity-basis coefficient vector d_R.
Set R=D3(symgrad d_R), epsilon=D3(symgrad u), and use the trace-free slip tensor
S, with Sxy=Syx=1/2 and Szz=0. Then

\[
\tau_2=2G D_2(\operatorname{symgrad}d_R)-2Z\chi S_2,
\qquad\tau_{zz}=-\operatorname{tr}_2\tau_2,
\]
\[
E_b=\int (\tau_2:\tau_2+(\operatorname{tr}_2\tau_2)^2)/(4G),\quad
\dot D_b=\int (\tau_2:\tau_2+(\operatorname{tr}_2\tau_2)^2)/(2\eta).
\]

Bulk power is tau2:symgrad u, which equals the full 3D deviatoric contraction.
The momentum residual is `integral(tau2:symgrad w-p div w)-external work`;
its velocity tangent is
`2*kappa*integral(symgrad w:symgrad du-div w*div du/3)`.
Slip cross blocks require no trace correction because tr(S)=0. At fixed d_R,Z,
the additional trace storage is phase independent: the existing normalization
adjoint and direct phase gradient/Hessian are unchanged by it. Mechanics and
future accepted anchors can still change with nonzero discrete divergence.

The exact coefficient recurrence remains

\[
d_R^{new}=\beta d_R^a+(\kappa/G)u^{new},\quad
Z^{new}=\beta Z^a+\kappa V^{new},\quad
\beta=e^{-G\Delta t/\eta},\quad\kappa=\eta(1-\beta).
\]

Applying the linear D3 symgrad gives the embedded Maxwell endpoint with the
same general beta, including zz. There is no strain/stress projection or
small-dt expansion. Normal and exceptional trial restoration and terminal-only
publication checks remain enabled.

The generic Stokes compressible-strain-rate assembler **already contains**
`-2*kappa/3 div w div u`. Its coefficient is the material output kappa, not
creep eta. Previously it was registered only for compressible material or
prescribed dilation, neither selected here. The Newton variant also supplies
the matching residual. Both registration entry points now include the optional
FE owner, once only, along with their existing simplified preconditioner term
when applicable. The incompressible mass equation remains unchanged.

The initial Picard-only registration attempt left the actual Newton matrix
unchanged; its run and build identity are retained in
[picard-only-attempt.json](../../../benchmarks/reconstructed_fault/propagation/plane_strain_evidence/picard-only-attempt.json).
The final Newton-path manufactured matrix check detects and verifies the fix.

Pressure remains the physical Stokes multiplier. A formal raw-stress pressure
shift by `2G tr(symgrad d_R)/3` is not implemented and generally is not a field
in the selected pressure space. Friction is audited separately:
- dynamic mode: `sigma_n=background+p-n.tau2.n`;
- selected prescribed adiabatic mode: `sigma_n=background+p_ad`, **without**
  dynamic p or deviatoric normal-stress correction.

The supplied derivation and earlier localization prose claimed a retained
stress correction in that prescribed mode. That wording is corrected to the
existing authoritative specification and actual implementation; the friction
model itself is unchanged. Background0 and p_ad10000 produce1000 Pa friction.

## Numerical evidence

| Check | Result | Interpretation |
|---|---|---|
| Native helpers vs explicit3x3, shear/extension/nonzero trace | Block error0; invariant error<=5.55e-17 | Helpers correct |
| Applying deviator twice, nonzero-trace case | Error0.15713484 | Demonstrates invalid re-deviation |
| Energy / phase directional derivatives | Relative<=2.51e-11 / <=1.93e-9 | Embedded stress and normalization chain rule verified |
| Raw vs3D energy at trace1 | Difference333333.333333 Pa, constant across phase trials | Extra storage is phase independent at fixed anchor |
| Algebraic recurrence, dt/lambda=.00125,1,4 | Relative<=2.37e-16 | General beta retained |
| Exact finite-step bulk work/loss balance | Relative<=1.99e-16 | Out-of-plane term included |
| Native nonuniform FE fields, tracefree and nonzero trace | Maxwell relative<=3.24e-15; max raw trace2.97362e-5 s^-1 | Same accepted FE representation works for both |
| Manufactured matrix, before |33167.2208361 vs3D25796.7273170 | Missing7370.49351913 trace subtraction |
| Manufactured matrix, corrected Newton |25796.727316973; error1.27e-9, relative4.92e-14 | Within original256eps positive operand allowance1.89e-9 |
| Small compatible run before/final | Five phase and five surface files byte-identical; encoding difference3.91e-18 m | Expected shear invariance |
| Loaded Q2 before/final,41 common states | Phase1.92e-15; q2.62e-12 Pa; slip1.32e-16 m | Invariance persists through onset |
| Mature default | Two diagnostic CSVs byte-identical | Existing path compatibility |

Nonzero divergence is an **operator diagnostic**, not a proposed compressible
physical benchmark. It uses a representable Q2 velocity vanishing on the
boundary, so pressure/constraint elimination does not conceal the stiffness
comparison. Pure shear and trace-free extension test zero-trace invariance.
The separate standalone algebra report is
[algebra.json](../../../benchmarks/reconstructed_fault/propagation/plane_strain_evidence/algebra.json).

## Bounded spatial and phase-stability result

[Transverse phase, chi and stress; phase, traction and slip rate histories](../../../benchmarks/reconstructed_fault/propagation/plane_strain_evidence/spatial-comparison.png).
Profiles are also retained as dense CSVs and an exportable PDF. Here widths are
full widths at half maximum; the sampling spacing is0.00005 m. The center and
both sharp side endpoints agree to arithmetic precision, with phase spread
along x below4.5e-15. The phase maximum remains on y=0.5; its x location is
degenerate in this tangentially uniform control.

| At T=.625s | Q2/Q1 | Q3/Q2 |
|---|---:|---:|
| First resolved step, max increment>1e-5 | .390625s | .515625s |
| Phase maximum | .65743615 | .68338396 |
| Largest phase increment | .01595771 | .01680195 |
| phi FWHM / chi FWHM | .27885 / .11915 m | .27795 / .10865 m |
| chi peak / 95% width |7.40519 m^-1 / .300012 m |7.82066 m^-1 / .297923 m |
| Cohesive q |301.68809 Pa |258.95282 Pa |
| Interval slip rate | .15823920 m/s | .15884560 m/s |
| Cumulative slip | .04851880 m | .04873486 m |
| Full phase vs256-cell rich1D, common times | .03642610 | .01311956 |
| q / slip common-time errors vs rich1D |46.63045 Pa / .00021966 m |3.89519 Pa / .000003603 m |
| Full phase vs matching16-cell exact-bulk1D | .02594781 |7.44e-9 |
| Native bulk storage |5.271185 J/m | .792491 J/m |
| D_relax accumulated | .369648 J/m | .181722 J/m |
| Sum absolute mechanical work discrepancies | .429188 J/m | .205155 J/m |
| Fracture expenditure |2.224313 J/m |3.060868 J/m |

These are accepted states, not failed-entry fields. Every original KKT target
passes; bounds and the existing4.922e-12 healing allowance pass, measured healing
is zero, and coefficient recurrence error is exactly zero at saved precision.
Maximum terminal bulk/fault relative residuals are below3.5e-14/8.5e-15.
The fine1D run has its independently relaxed fine seed (maximum about0.60006),
whereas native coarse initialization is0.61318. The fine comparison therefore
includes phase initialization/discretization error. The matching16-cell
reference separates this: Q3 phase agrees within7.44e-9 despite the remaining
between-point stress error. Agreement of sampled phase alone is insufficient.

Frozen integration uses the same continuous FE d_R, phase, I and Z, actual
Gauss-Lobatto support nodes, and3/6/12/24 normal Gauss rules. No bulk field or
normalization is changed. The Q12/Q24 bulk-energy difference is <=5.4e-15 J/m.

| Frozen final field | Q2 | Q3 |
|---|---:|---:|
| Accurate bulk storage,Q24 |5.417641 J/m | .875859 J/m |
| Relative native underestimate (denominatorQ24) |2.7033% |9.5184% |
| Accurate transverse stress RMS about mean |2943.07 Pa |409.27 Pa |
| Native-Q3 stress RMS |2892.78 Pa |1.08e-9 Pa |
| Dense stress range |[-13422,7977] Pa |[-1701,3051] Pa |
| Direct-reference normalization/source error |<=1.23e-15 |<=1.12e-15 |

Thus enrichment improves velocity representation, but using only three normal
points makes a Q3 derivative able to interpolate the rational chi profile at
those points. The resulting apparent constant stress is not constant between
them. The accurate Q3 storage exceeds native storage by10.52%; a projection
or normalization change would not address this. No coupled quadrature correction
is inserted in this task. A concrete next numerical experiment is one matched
native/prototype comparison with sufficient normal mechanical/energy quadrature,
before another velocity/time refinement or propagation claim.

At t=.5/.625, Q2 free Hessian minima are9.13612/2.45025, with0 negative
modes. Q3 has no free nodes at.5 and minimum2.24255 at.625. Feasible lowest-mode
perturbations of +/-1e-4 raise energy by approximately4.56805e-8,1.22513e-8
and1.12127e-8 J/m respectively, matching the quadratic predictions; separately
scaled32eps energy allowances are1.50e-13–2.03e-13 J/m. This checks only the
sampled free subspace above the healing allowance. It does not test all active
critical-cone directions, the entire path, or a global minimum. At an identical
accepted anchor the plane-strain energy correction is a phase-independent
offset, so it cannot remove the historical affine negative-curvature evidence.

Energy discrepancies remain separate from each other and from quadrature error.
Their reduced Q3 values above are values of the native sampled problem, not a
qualified continuum balance. No cancellation is used to certify accuracy.

## Changes, provenance and verification

The task-only [production diff](../../../benchmarks/reconstructed_fault/propagation/plane_strain_evidence/production-correction.diff)
is separate from derivation/specification amendments and benchmark diagnostics.
Five production files, six functions:

| File / function | Purpose / resulting behavior |
|---|---|
| simulator/assembly.cc :: Simulator::set_stokes_assemblers | Register existing kappa trace term for optional FE owner, preserve ordinary defaults |
| simulator/newton.cc :: NewtonHandler::set_assemblers | Register matching Newton residual/tangent and existing preconditioner variant |
| phase_field_fault/fe_strain.cc :: evaluate_fe_phase_energy | Deviate raw anchor once; include reconstructed zz storage; phase derivative unchanged |
| phase_field_fault/fe_strain.cc :: evaluate_fe_strain_stress | Same authoritative stress for mechanics, trials and publication |
| phase_field_fault/constitutive.cc :: evaluate_reconstructed_fault_point | Deviate selected raw FE rate once, never accepted deviatoric stress; legacy branch retained |
| phase_field_fault/history.cc :: compute_history_candidates | Consistent selected terminal Maxwell candidate before accepted-owner comparison |

No new production parameter/default/interface/checkpoint conversion is added.
Benchmark changes: native_fe_strain.cc gains helper/operator tests, a second
nonuniform rate field and sampled Hessian/mode probes; its energy diagnostics
now include zz. check_plane_strain_algebra.py verifies independent explicit3D
energy/work/recurrence. assess_plane_strain.py reconstructs actual FE fields,
checks trajectories, isolates frozen quadrature and makes plots.
check_plane_strain.py verifies immutable incoming/historical artifacts and
current evidence. Small fixture files select existing parameters only.
Documentation changes: new derivation and specification reduction/pressure
audit, current_design and recovery/status/plan/readme links, this report and
actual task-only TeX diffs. Incoming manual pf_rsf.tex edits are preserved exactly.

Baseline HEAD de7beaa65846c7372711f7922abf1c5d9a745ee6, branch fault-propagation;
18 incoming files/binaries are copied in build-propagation-integration/plane-strain-baseline.
Compiler GCC12.4, Release Unity; deal.II9.6.2, Trilinos14.2. Final executable
SHA6fbedb652a2b0af8eb0d8b202b14d12edd0c93b26b8d8ee60c8155f2a57e0c50;
plugin SHA936e9d0a55d14f501766f3f7a26c6328f80ff52503a54cc55f1d7b18170cddf5.
Commands/configuration identities and artifact hashes are in
plane_strain_evidence/native-build.json and manifest.json. Builds use the
existing core build and plugin build. Runs use one thread and the existing
explicit particle-advection diagnostic freeze. Fresh outputs have unique names.

Reproduction/verification commands:

```sh
cmake --build build-propagation-integration --target aspect.exe.release -j2
cmake --build build-propagation-integration/plugin --target native_fe_strain -j2
ASPECT_SOURCE_DIR="$PWD" ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-propagation-integration/aspect-release benchmarks/reconstructed_fault/propagation/plane_strain_evidence/compatible-q2-native.prm
ASPECT_SOURCE_DIR="$PWD" ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build-propagation-integration/aspect-release benchmarks/reconstructed_fault/propagation/plane_strain_evidence/compatible-q3-native.prm
python3 benchmarks/reconstructed_fault/propagation/check_plane_strain_algebra.py
MPLCONFIGDIR=/tmp/plane-strain-matplotlib python3 benchmarks/reconstructed_fault/propagation/assess_plane_strain.py
python3 benchmarks/reconstructed_fault/propagation/check_plane_strain.py
```

The derivation is compiled in two LaTeX passes. Actual
[derivation diff](../../../benchmarks/reconstructed_fault/propagation/plane_strain_evidence/derivation.diff) and
[specification diff](../../../benchmarks/reconstructed_fault/propagation/plane_strain_evidence/specification.diff)
are relative to the incoming task snapshot, not the much older git baseline.
The final read-only checker passes:18 incoming snapshots,2015 unchanged historical
artifacts and537 current evidence files;35 helper call sites;10 identical small
shear files and2 identical mature CSVs. These pass counts do not alter the
spatial/stability limitations above. No broad unit/regression matrix, MPI, restart, transport, rejected-physical-step
rollback, time-refinement qualification or P3 was run. Existing earlier evidence
is retained and hash checked. Conclusions concern the declared optional fixed
fixture; they do not extend to unrelated helper consumers or the default
particle-owned model.
