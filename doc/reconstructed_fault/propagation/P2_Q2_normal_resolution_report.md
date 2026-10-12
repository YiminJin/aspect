# P2 Q2 normal-resolution decision

**Working choice: Q2/Q1 on 16×256 cells, with six normal Gauss points.**
The inexpensive study selects this as the least tested resolution meeting all
five provisional phase/traction/slip/stress/storage criteria. At 128 normal
cells the whole-section stress error passes, but localization-weighted error is
8.41%, above the 5% target. The 256-cell errors are 1.09% and 2.10%, respectively;
storage error is 0.013% of the fixed reference peak. The native confirmation
finishes all 40 steps and reproduces the matching discrete 1D calculation.
Its measured cost is **34.4 minutes and 1.95 GiB peak RSS**, including diagnostics.
Stress and storage meet the practical spatial targets for the next P2 accuracy
assessment. The remaining uncertainty is time integration and finite-step
energy accounting, rather than bulk quadrature or these spatial RMS targets.

This is a spatial decision for the restricted fixed-mesh P2 benchmark. Temporal
accuracy, finite phase-relaxation release, endpoint-mechanics work discrepancies,
full constrained stability, general localization and P3 are not qualified.

## Fixed settings and error definitions

| Setting | Value / units |
|---|---|
| Domain and geometry | [0,1]×[0,1] m; one fixed horizontal through-going fault at y=0.5 m |
| Bulk discretization | Q2 velocity/Q1 pressure, Q1 phase; 16 tangential cells; 16/32/64/128/256 normal cells |
| Bulk quadrature | Six normal Gauss points, existing split tangential rule, shared by all coupled bulk operators |
| Surface and associations | 64 Q1 segments, spacing 0.015625 m; fixed full-normal-domain association |
| Particles | 3×3 per cell, frozen positions; no transport or replenishment |
| Material | G=10⁶ Pa, η=10⁸ Pa s, λ=100 s; c=1000 Pa, Gc=26.6666666667 J/m², ℓ=0.15625 m, p=1; m=128, Hc=0.5 Pa |
| Viscosity bounds | 10⁶–10¹⁰ Pa s; identical homogeneous constituents |
| Friction and pressure | Constant μ=0.1, prescribed adiabatic normal pressure 10000 Pa, background pressure 0; radiation damping 0 |
| Initialization | Native nominal-core-0.6 profile/H, fresh native relaxation at each mesh; unloaded accepted FE strain and surface elastic slip |
| Histories and phase | One authoritative accepted FE strain; immutable accepted-phase lower bound, upper bound 1; diagnostic H+ |
| Normalization/fracture | Retained consistent normalization and exact transverse integral/adjoint; native CPDI fracture |
| Boundaries/loading | Compatible sides: uy=0, ux natural; top/bottom ux=0.256 max(0,t−dt/2)(y−0.5) m/s, uy=0; pressure normalization=no |
| Time | dt=1/64 s, T=0.625 s, 40 accepted feedback cycles |
| Tolerances | Native phase relative 10⁻¹², phase linear 2×10⁻⁷; mechanical 10⁻⁸; unchanged arithmetic/healing allowances |

G remains a dimensional synthetic verification value. The imposed velocity is
the interval mean of the same smooth ramp. Reported slip rates are solved
interval rates, not instantaneous endpoint samples. Q3 evidence is preserved.

The reference is the retained 256-phase-cell, exact-bulk one-dimensional path
at identical times. Its endpoint shear stress τ_ref(t) is uniform. Use fixed
scales over the complete interval, **1332.521975 Pa** and **0.887807407 J/m**,
for maximum stress and storage errors. No denominator tends to zero at startup.
The traction scale remains cohesion; slip errors use reference final slip.
Targets are maximum full-phase error ≤0.01, q error ≤10 Pa, slip error ≤0.5%,
both stress RMS errors ≤5%, and bulk-storage error ≤5%.

For unit height and independently evaluated τ(y),

\[
 \bar\tau=\int_0^1\tau\,dy,\qquad
 \sigma_{\rm osc}^2=\int_0^1(\tau-\bar\tau)^2dy,\qquad
 e_{\rm all}^2=(\bar\tau-\tau_{\rm ref})^2+\sigma_{\rm osc}^2.
\]

The localized error uses the **current field's** positive χ(y):
\(e_\chi^2=\int\chi(\tau-\tau_{\rm ref})^2dy/\int\chi dy\).
Its localized mean and oscillation components are retained separately. This
weight is a practical localization diagnostic, not a bound on every term of
the phase residual. In this pure-shear control, embedded 3D bulk storage is
\(E_b=\bar\tau^2/(2G)+\sigma_{\rm osc}^2/(2G)\).
Normal stress and tangential uniformity are checked natively.

Stress oscillations matter directly for damage feedback in this transversely
varying, tangentially uniform control. At fixed accepted strain and Z,
\(\delta E_b=-(Z/G)\int\tau\,\delta\chi\,dy\). A uniform stress gives zero
because \(\int\delta\chi\,dy=0\); numerical stress oscillations can introduce
additional bulk phase driving even when phase maxima look close.

Stress moments and energy use 24 points/cell, checked with 48, on frozen
accepted FE coefficients. These are independent evaluations; the evolving
solves retain six points. Cell one-sided limits and interior stationary points
of the actual Q2-strain/Q1-phase stress give extrema; cell-interface jumps are
evaluated from both limits. Dense plotting samples do not define these metrics.
All common-time histories, locations and signed mean errors are saved.

## Spatial convergence and initialization

Maximum errors over common accepted times; all worst stress/storage errors
occur at 0.625 s. Percentages use the fixed scales above.

| Normal cells | Full phase error | q error (Pa) | Slip error (%) | Whole stress RMS (%) | Localized stress RMS (%) | Storage error (%) | All targets |
|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 0.03823 | 50.172 | 0.4650 | 219.76 | 426.63 | 514.92 | Fail |
| 32 | 0.006478 | 5.047 | 0.04520 | 68.14 | 130.89 | 49.55 | Fail |
| 64 | 0.001356 | 0.4643 | 0.003131 | 17.41 | 33.63 | 3.246 | Fail |
| 128 | 0.0003676 | 0.05418 | 0.0002428 | 4.358 | 8.409 | 0.2065 | Fail |
| 256 | 6.30×10⁻⁷ | 0.001051 | 0.00001107 | 1.090 | 2.103 | 0.01264 | Pass |

The first three trajectories are reused unchanged. Only 128 and the necessary
256-cell extension were added. Prototype runtimes are approximately 2.05 and
10.43 s. The 256-cell phase initialization matches the reference's phase grid;
its exceptionally small phase error is **not** an independent claim about the
continuum seed. The retained exact-bulk 128/256 repeat differs by up to
0.0383 Pa in stress, 4.86×10⁻⁵ J/m in storage and about 0.000357 in full phase
at these times. This is small against the practical targets, and limits what
may be inferred from tiny matching-grid errors.

At the final time, mean-stress error and oscillations are distinct:

| Normal cells | Mean error (Pa) | Whole oscillation RMS (Pa) | Stress min/max (Pa) | Largest interface jump (Pa) | Mean / oscillation storage (J/m) |
|---:|---:|---:|---:|---:|---:|
| 16 | 226.234 | 2919.59 | −13190.6 / 7951.67 | 6708.39 | 1.097112 / 4.261989 |
| 32 | 21.9963 | 907.77 | −5179.18 / 4257.89 | 4736.40 | 0.815433 / 0.412025 |
| 64 | 1.5238 | 231.98 | −338.44 / 2020.98 | 760.32 | 0.789498 / 0.026907 |
| 128 | 0.1178 | 58.07 | 857.58 / 1449.90 | 105.77 | 0.787733 / 0.001686 |
| 256 | 0.00539 | 14.52 | 1156.35 / 1303.92 | 13.49 | 0.787592 / 0.00010546 |

The reference final stress/storage are 1255.057639 Pa / 0.787584839 J/m.
The RMS targets do not assert 5% pointwise stress accuracy: at 256 cells the
minimum is still about 7.4% of the peak reference scale below the reference.
The stress extrema and jumps remain explicit residual spatial errors.

Different meshes have different native particle sampling, relaxed seed profiles
and immutable irreversible obstacles. Subtracting each initial phase removes
only the offset; it cannot undo the effect of these obstacles on evolution.
Initial maxima are 0.61318/0.60355/0.60090/0.60023/0.600057 for the five meshes.
The comparison retains this documented initialization rather than substituting
another mesh's history.

## Same-physical-profile control

A separate frozen equilibrium uses the **identical retained reference phase,
Z and mean total elastic strain** at t=0.5, 0.546875 and 0.625 s. No phase/history
is advanced. Solve the Q2 elastic-strain-space weak force balance at the same
endpoint difference, with τ=G d_R,y−Zχ. The exact continuum comparison is the
uniform reference stress. The prescribed reference Q1 profile is evaluated as
the same physical function even where it is not representable on a coarse phase
mesh; it is not interpolated into different obstacles. All spaces use a common
integration split at those physical phase knots, independently checked at
24/48 points. This diagnostic isolates FE strain/velocity-space error, rather
than claiming a new coupled discrete model at a coarse phase resolution.

Final localized RMS errors with the same physical profile are
6982.9/2990.1/774.5/191.1/28.02 Pa. Mean-stress errors are at arithmetic precision;
excess bulk storage is entirely the oscillation component. Thus velocity-space
error itself decreases strongly with refinement. The evolving 128-cell error
(112.1 Pa) is smaller than its frozen common-profile error (191.1 Pa): this
confirms that altered profiles/obstacles affect apparent convergence.

The frozen control fixes Z, so its coarse discrete surface-force defect is
reported rather than corrected by changing Z or slip. At the final time this
falls from −211.7 Pa to −0.00434 Pa over the mesh family. It is a controlled
bulk approximation test, not another accepted coupled trajectory.

## Native confirmation, cost and selected curvature

The single selected mesh completes to 0.625 s. The fixed tangential mesh and
64-segment fault are unchanged. Native common-time maxima agree with the cheap
selection: whole/localized stress error **1.08987% / 2.10268%**, storage error
**0.0126403%**. Final mean stress error is 0.005391 Pa, oscillation RMS
14.52279 Pa, and localized RMS 28.01867 Pa. Native extrema are
1156.35181 / 1303.91854 Pa; largest jump is 13.48997 Pa.

| Native check | Measured result |
|---|---|
| Matching discrete 1D maximum full phase / q / slip differences | 1.293×10⁻⁸ / 1.690×10⁻⁵ Pa / 1.693×10⁻¹¹ m |
| Exact-bulk reference maximum phase / q / slip errors | 6.351×10⁻⁷ / 0.001055 Pa / 5.399×10⁻⁹ m |
| Phase bounds and healing | All fields in [0,1), no decrease; accepted lower bound unchanged |
| Actual KKT target range / final defect | 5.763×10⁻¹²–3.399×10⁻¹¹ / 1.421×10⁻¹³ |
| Largest terminal mechanical residual, bulk / fault | 5.781×10⁻¹⁴ / 5.919×10⁻¹⁴ |
| FE coefficient Maxwell recurrence error | 1.735×10⁻¹⁸ |
| Largest stress-reconstruction / energy-reconstruction errors | 1.690×10⁻⁸ Pa / 3.442×10⁻¹⁵ J/m |
| Maximum tangential phase spread / normal stress | 4.830×10⁻¹⁴ / 1.487×10⁻⁹ Pa |
| Direct-normalization source error at initial/0.5/final states | ≤6.662×10⁻¹⁶, including both endpoints; all I_h positive |

Damage begins at 0.546875 s, at y=0.5 throughout the fault. The core rises from
0.60005713 to 0.67310927; the largest accepted full-field increment is 0.0170487.
Final q is 255.05869 Pa, solved interval slip rate 0.15886030 m/s, cumulative
slip 0.048738459 m. Final phase/chi FWHM are 0.27628 / 0.11795 m; the 95% chi
width is 0.29051 m. This is widening of a prescribed through-going band.

Native energy derivatives and free-Hessian checks use the **proper incoming
accepted FE strain/Z from steps 31/39** and the immutable previous-phase bound.
At 0.5 s no phase node is free: a positive free curvature cannot be claimed.
At 0.625 s, 2839 nodes exceed the healing allowance; the minimum free-Hessian
eigenvalue is 0.139611 with no negative eigenvalues. Both signs of a 10⁻⁴
minimum-mode perturbation increase energy by about 6.985×10⁻¹⁰ J/m, against
an arithmetic estimate 1.711×10⁻¹³ J/m. This is a selected free-subspace check,
not full constrained or coupled stability. Best finite-difference energy/Hessian
direction errors are ≤2.10×10⁻⁹ / 9.83×10⁻¹⁰. At those incoming states,
six-versus-24-point phase gradient and relative Hessian differences are
≤2.99×10⁻¹⁴ and 1.43×10⁻¹⁵; independent 24/48 bulk storage differences
are ≤7.2×10⁻¹⁵ J/m.

| Cost / size (one serial rank, one thread) | Measured value |
|---|---:|
| Cells / velocity DoFs / particles | 4096 / 33858 / 36864 |
| Wall / user / system time | 2063.04 / 2033.24 / 15.58 s |
| Peak RSS (per-child wait4) | 2041916 KiB = 1.947 GiB |
| Stokes matrix / RHS assembly | 119 / 209 s |
| Stokes preconditioner / condensed linear solves | 54.4 / 699 s |
| Phase bounded / phase linear solves | 467 / 184 s (nested timings) |
| FE phase energy / postprocessing | 175 / 173 s (nested timings) |
| Phase Newton iterations / mechanical linear iterations | 78 / 2425, including initialization |
| Mechanical nonlinear evaluations | 123 |

Timers overlap and must not be added. Cost includes the selected curvature,
derivative and publication diagnostics, and excludes the interrupted attempts
below and offline assessment. Earlier Q2/Q3 costs use the pre-cache executable;
they are preserved but are not a controlled cost-scaling comparison with this
run. The recommendation is the least tested resolution passing all targets.

The first initialization attempt was interrupted after 1070 s (1060 s user
CPU, 661 MiB peak RSS), before publishing an initialized or physical accepted
state. A read-only stack sample found repeated Epetra row extraction inside
`constrain_phase_newton_system`: constructing `SparseMatrix::end(row)` at every
inner-loop comparison repeatedly copies a complete row. The dense phase block
makes that otherwise unnecessary cost grow rapidly. Each row-end iterator is
now constructed once in the three affected loops. Matrix values, iteration
order, contact decisions, lifting, compression, bounds and tolerances are
unchanged. The retained interrupted files are computational diagnosis, not a
failed physical trajectory. The same numerical input is rerun from fresh
initialization with the rebuilt executable. Its cost excludes the earlier
attempt and debugger sampling. A subsequent initialization completed, then the
observer stopped at a manufactured dt/lambda=1 Maxwell probe: error
6.24e-10 Pa exceeded its net-gradient operand estimate by about 5%. The
production coefficient-times-basis-gradient uncertainty estimate is now exposed
through the test accessor and used by that observer. The 256-epsilon factor and
all solver tolerances remain fixed; net errors remain recorded separately. That
initial-only failed probe and its accepted unloaded state are preserved. The
final fresh rerun uses the same numerical input and corrected diagnostic.

## Energy and temporal limits

Spatial storage error, phase-relaxation release and mechanical work discrepancy
are different quantities. At 64/128/256 cells the cumulative phase release is
0.179629/0.179809/0.179911 J/m. Absolute internal-work discrepancies are
0.188386/0.174664/0.175633 J/m; boundary-lifting discrepancies are
0.203795/0.204195/0.204356 J/m. These do not disappear as spatial resolution
improves. Signed profile storage changes and fracture expenditure are retained
alongside these quantities. No cancellation, altered phase driver or time-step
change is used to obtain the spatial pass.

The native totals confirm this separation: spatial storage difference
1.1222×10⁻⁴ J/m; phase release 0.179911 J/m; absolute internal/boundary work
discrepancies 0.175633 / 0.204356 J/m; fracture expenditure 3.102945 J/m.
Signed bulk/cohesive profile storage changes are +0.002618 / −3.285474 J/m.
Ordinary bulk/cohesive work is 59.190461 / 9.510973 J/m, boundary work
59.219184 J/m, bulk/cohesive viscous loss 0.007361 / 0.022357 J/m, and
frictional loss 48.738459 J/m. These profile changes belong to the phase
redistribution accounting and are not extra mechanical work to add twice.

The present comparison holds dt fixed. It establishes practical spatial
accuracy of selected stress/storage/feedback outputs for this restricted
fixture; it does not establish native time convergence, vanishing phase-release
slack or exact finite-step boundary work closure. One-step phase/mechanical lag,
endpoint mechanics and possible rate-independent relaxation release remain
separate temporal/formulation questions. No P3 or new temporal family is run.

## Source changes and evidence

| File | Purpose |
|---|---|
| source/simulator/phase_field.cc :: constrain_phase_newton_system | Performance-only cache of the row-end iterator in contact-column elimination and contact-flag import; identical algebra and iteration order |
| native_fe_strain.cc :: MomentCycle::local_probe / tests/phase_field_fault_test_access.h :: PhaseFieldFaultTestAccess::prepared_fe_stress | Reuse existing production FE coefficient/basis uncertainty estimate in the observer; preserve the 256-epsilon factor and report net errors separately |
| normal_resolution.py | Independent common-time stress moments, exact cell extrema/jumps, storage decomposition, fixed-scale selection and same-physical-profile control |
| run_normal_resolution.py | Fresh 128/256 extensions and exactly one selected native mesh; input/build hashes and per-child runtime/RSS |
| assess_normal_resolution.py | Native history/numerical checks, matching-reference comparison, proper incoming curvature and compact plots |
| check_normal_resolution.py | Read-only preservation and numerical/selection evidence checks |
| This report, STATUS, PLAN, CURRENT_STATUS, benchmark README, current_design | Record the working Q2 choice, evidence and restricted qualification |

**No constitutive, mathematical, tolerance, projector or quadrature change.**
One production performance fix is compiled into the executable; the modified
observer plugin is rebuilt only for that diagnostic, with unchanged compiler/
dependency configuration.
Previous Q3 and three-point artifacts are preserved. Evidence and exact command
identities are in `benchmarks/reconstructed_fault/propagation/normal_resolution_evidence/`.

Build provenance remains GCC 12.4 Release Unity, deal.II 9.6.2, Trilinos 14.2,
at HEAD de7beaa65846c7372711f7922abf1c5d9a745ee6 on fault-propagation with
incoming work preserved. Native input SHA is
ceeef524a04775c12860c25ed72d8926be0fb56fdc4b42d50aa6fbeb586fb8f5;
core/plugin identities and before-fix sources are retained in the evidence.
No new MPI, restart, transport or generic-projector campaign was performed.

Commands (OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 for calculations):

```text
python3 benchmarks/reconstructed_fault/propagation/run_normal_resolution.py --cells 128
python3 benchmarks/reconstructed_fault/propagation/run_normal_resolution.py --cells 256
python3 benchmarks/reconstructed_fault/propagation/normal_resolution.py
cmake --build build-propagation-integration --target aspect -j2
cmake --build build-propagation-integration/plugin --target native_fe_strain -j2
python3 benchmarks/reconstructed_fault/propagation/run_normal_resolution.py --native
python3 benchmarks/reconstructed_fault/propagation/assess_normal_resolution.py
python3 benchmarks/reconstructed_fault/propagation/check_normal_resolution.py
```

[Convergence plots](../../../benchmarks/reconstructed_fault/propagation/normal_resolution_evidence/normal-convergence.png),
[same-profile and feedback comparisons](../../../benchmarks/reconstructed_fault/propagation/normal_resolution_evidence/frozen-feedback.png),
[selected native profiles and response](../../../benchmarks/reconstructed_fault/propagation/normal_resolution_evidence/selected-native.png).
