# Bulk energetic tip implementation and bounded qualification

**Selected closure, 7 November 2026.** The user-approved control prescribes one
fixed Mode-II direction and one age on each complete cross-section. Only fault
deformation enters `V = integral w dn`; ordinary elastic and viscous bulk strain
does not. Uniform section resistance is the discrete work adjoint of that
integral. This is an additional nonlocal constitutive assumption, not a result
of the bulk energy or a general turning/branching model.

The implementation now solves direct physical stress, physical cohesive tensor
history, authoritative DGQ2 bulk damage, signed regularized RSF, and positive
backward-Euler section age together. The loaded coarse control and its timestep
halving complete through 0.08 s. The finer mesh fails its nonlinear line search
at 0.05 s. Thus the closed law and bounded coupled implementation are demonstrated;
spatial convergence, stable branch selection and general propagation are not.
The single next proposed experiment is a constrained-stability/globalization
study of the existing refined 0.0475→0.05 s window with unchanged physical inputs.
No further run is launched by this report.

Read [TIP_TREATMENT.md](TIP_TREATMENT.md) for the current algorithm and its linked
model, discrete equations, ownership, configuration and limitations. The supplied
[fault_tip_treatment.tex](fault_tip_treatment.tex) remains a continuum proposal;
the historical interface episode remains evidence for a different model.

## Starting state and preservation

Actual root: `/home/ein/repository/aspect-pf-rsf-particle-free`; branch
`pf-rsf-particle-free`; HEAD `36de6bcf3633dce96f904d27b1256c37b6675ccc`, with
incoming manual/recovery changes preserved. Numerical baseline is the restored
`84f57503c` behavior, not the inactive shared-assembler review. No reset, checkout,
commit, server submission or production checkpoint conversion was performed.
The entry preservation record from propagation recovery and the new
`evidence/bulk_tip/baseline.json` distinguish prior edits from new source work.

All seven protected native-integration executables/DSOs rehash unchanged;
`evidence/bulk_tip/protected-binaries.json` records the check. In particular the
mature executable remains `6ea374c5b9b50a20af00fa6e2c365df61d7ac59ed33ad26c7b338d7bb7ba67d9`.
The candidate was built separately in `build-particle-free/bulk-tip/native`:

- Core SHA256: `340b6c37516a8f60792d46412e031023daa04ea366f467bb404a5f56ee03ebf2`.
- Checker SHA256: `5c5ba77cddf02b5cd5bf38cafef2cda16536b36406f0f62ed377dc9090743446`.
- Final point suite: 632 assertions, 10 cases, all passing.

Each run retains `input.prm`, `command.json` with executable/checker/input/source
hashes, complete command/environment, elapsed time and exit code, plus logs,
physical exports and available checkpoints. HEAD alone does not identify these
uncommitted numerical changes. The last core rebuild changed only a comment and
produced the same binary hash. A retained executable/checker copy is in
`build-particle-free/bulk-tip/qualified-340b6c37/`; final source hashes, numerical
patch and new support sources are retained with compact evidence. Build-log intervals total approximately 1,414 s
(23.6 min), separately from about 175.3 s of all new simulation launches, including
failed probes and diagnostic repeats. Unit/reduction overhead is seconds. The
2-hour simulation/check budget was not approached. At most four build workers,
two MPI ranks here, one numerical thread per rank and one heavy job at a time.

## Implemented equations and ownership

At fixed reference and zero spin, with matched constant G and eta,

\[
\tau^+=\beta[\tau^-+2G\Delta t(D_3^+-w^+S)],\qquad
A^+=\beta[A^-+\Delta t w^+S],\quad \beta=(1+\Delta tG/\eta)^{-1}.
\]

Full 3-D plane-strain tau and A are retained at native mechanical quadrature
points; no uncut DGQ2 stress projection is substituted. The control restricts
A=eS but does not eliminate its physical memory or energy. The phase field alone
is DGQ2. `bulk_tip.cc` supplies endpoint elimination, derivatives and an independent
BE ledger. Existing bulk solver/history owners, native SIPG fracture operators,
Bernstein obstacle coordinates and ordinary simulator transactions are reused.
No ordinary advection assembler or mature timestep path was replaced.

For h=m phi(1+phi)/(1-phi)^2, analytic endpoint limits handle phi=0 and phi=1
without floors or caps. Intact A=0 has reaction q=tau:S-f_section; a wholly intact
section has V=f=0. Hence Y(0)=m(tau:S)^2/(2G), not an imported Coulomb excess rule.
Intact points on a partly deforming section see that section's resistance.
This nonlocal influence is part of the selected physical assumption.

For aligned sections, ds is the tangential Gauss weight and dn=JxW/ds. The code
checks compatible section layouts and full normal coverage, and uses exactly
these measures in rate, residual, tangent and work:

\[
\sum_q JxW_q f_{s(q)}w_q=\sum_s ds_s f_s\sum_{q\in s}dn_qw_q.
\]

One age is initialized to 10 s everywhere, including ahead of the initial patch,
and advances with `(theta_old+dt)/(1+dt*abs(V)/Dc)`. There is no reconstructed
surface, segment admission, or age reset. The local deformation may reverse;
only the section sum has nonnegative physical RSF/radiation dissipation.

The signed existing RSF constants are reused. A scalar elimination solves
V+C R(V)=B, where B is free integrated fault rate and C is integrated mobility.
The mixed section equation uses f-f_star and its analytic nonlocal derivative.
This avoids evaluating steep cold-creep RSF on a cancellation-dominated rate.
The monotonicity guard is derived, not a tolerance adjustment:
eta_rad >= pN dt/Dc (max(0,sqrt(b)-sqrt(a)))^2. The chosen b>a still gives
steady velocity weakening. Unsupported multiple-root selections reject.

Actual integrated V, scalar-resolved V, their defect, raw RSF(V), signed/absolute
rates and physical power are exported separately. At cold-creep rates below
arithmetic resolution, raw RSF(V) can differ by pascals despite negligible slip
and power errors. Age/slip use the actual integral; scalar-law diagnostic power
versus actual bulk f*w is explicitly accounted as residual work.

Phase growth solves the full DG/SIPG fracture gradient minus Y with both bounds.
The accepted Bernstein lower bound guarantees polynomial irreversibility, a
sufficient constraint stronger than arbitrary pointwise Q2 admissibility.
Trial stress, A, section state and phase share one accepted parent and one
publication; retry rolls all of them back. New checkpoint law is
`bulk tip physical tau A DGQ2 sections BE v1`, with material, RSF, phase,
quadrature and geometry checks; old checkpoint meanings are preserved.

## Bounded evidence

Fixture: 2×1 m box, h=.25 m (8×4 cells), ell=.25 m, fixed Mode II, imposed
boundary velocity (0.02(y-.5),0) m/s, natural phase boundaries. Initially unloaded
compact damaged patch has support x=[.5,1.5], y=[.25,.75], maximum phi=.2;
all other material is intact. G=1 MPa, eta=100 MPa s, Gc=26.6666666667 J/m²,
calibration cohesion=1000 Pa (m=80), prescribed pN=1000 Pa, radiation=1000 Pa s/m,
mu0=.6, a=.015, b=.02, V0=.001 m/s, Dc=.01 m. Prescribed confinement is distinct
from the solved pressure gauge. Gauss6 mechanics and Gauss5 fracture are retained.

| Run | dt / mesh | Accepted endpoint | Result / wall seconds |
|---|---|---|---|
| `control04` | .005 / 8×4 | .08, 16 steps | Pass; rejection/retry and coupled tangent; 9.03 |
| `half03` | .0025 / 8×4 | .08, 32 steps | Pass; coupled tangent; 12.64 |
| `unloaded03` | .005 / 8×4 | .08, 16 steps | Pass; zero mechanics/slip; 4.72 |
| `intact01` | .005 / 8×4 | .02, 4 steps | Pass; exactly intact, zero fault slip; 3.92 |
| `fine01` | .0025 / 16×8 | .0475, 19 steps | Line-search failure at .05; 21.56 |
| `fine-diagnostic01` | same | same | Reproduces failure, derivative checked; 22.77 |
| `fine-unloaded01` | .0025 / 16×8 | .045 | Pass; identifies initial profile adjustment; 8.43 |
| `mpi02`, `restart01` | .005 / 8×4 | .02, then .03 | 2→1 rank cold-state recovery; pass |
| `mpi03`, `restart03` | .005 / 8×4 | .06, restart from .04 to .08 | Active-slip 2→1 rank recovery; pass |

Wall times include instrumentation and differ between runs; these are costs of
bounded checks, not an isolated performance comparison. Exact records govern
rounding. `mpi03` also takes a roundoff-sized terminal step at .06. Extending
that terminal snapshot (`restart02`) retains its tiny timestep and controller
history, producing a different temporal mesh. This result is preserved, not
claimed identical to the constant-dt trajectory. `restart03` instead selects
its regular .04 checkpoint (snapshot 02) in a fresh copy; originals are untouched.

Independent checks include finite-h/endpoints, one-sided derivatives, reversal,
matched relaxation, BE energy, signed RSF/age derivatives, section balance and
nonlocal derivative, native SIPG energy/Hessian perturbations, and physical
stress/storage/source equivalence to the retained constant-resistance tensor
return. The latter is a shared-component regression, not mature-production
qualification of the new closure. The material-point mixture explicitly tests
opposing rates and nonuniform transverse variance. No repeated rotation campaign.

Coupled finite-difference relative errors: 5.45e-9 (coarse), 6.45e-9 (half dt),
9.77e-8 (refined failing iterate). Accepted coarse/half mechanical residuals are
<=9.75e-11 / 7.80e-10 and phase KKT <=2.42e-11 / 1.59e-8 under unchanged gates.
Independent algebraic ledger errors <=5.95e-15 J/m; work-adjoint errors
<=1.12e-16 in the completed pair. Maximum section-law residual work is
6.86e-11 / 7.24e-10 J/m, exported separately from the adjoint identity. Final integral-rate defects <=4.0e-14 m/s.
Rollback checks compare full fields/histories, deterministic retry and a second
rollback. MPI/restart physical point differences are <=3.1e-12 (maximum among
mixed-unit exported components), section state differences <=3.5e-13 excluding
the deliberately sensitive raw RSF-of-integral diagnostic. No bitwise MPI claim.

## Physical comparison and accuracy limits

At .08 s, coarse versus half dt:

| Quantity | .005 s | .0025 s |
|---|---:|---:|
| Damage integral in initially intact material, m² | 8.22190e-4 | 7.98116e-4 |
| Maximum section rate, m/s | .0215353 | .0241574 |
| Maximum cumulative slip, m | .000692512 | .000690913 |
| Minimum age, s | 9.40921 | 9.40966 |
| Central mean shear stress, Pa | 979.662 | 980.210 |
| Endpoint-region mean shear stress, Pa | 1302.789 | 1303.705 |
| Full bulk cohesive energy, J/m | .0756309 | .0746066 |
| Transverse variance part, J/m | .00540588 | .00541371 |

Central means use x=[.75,1.25]; endpoint means x=[.25,.75) union (1.25,1.75].
Normal deviatoric stress averages cancel by symmetry; they are not a contact
or total-normal-traction test. Stress relative L2 difference is .1432%; phase
L2 difference 3.8198e-5 m; Axy tensor-component norm including the off-diagonal
metric is 2.3570e-6 m. Peak slip rate differs by about 11% of the half-dt peak,
so small bulk stress error alone does not qualify the slip transient.

The unloaded coarse profile also redistributes under its obstacle and fracture
functional: virgin damage integral 1.72300e-5 m², with zero slip and age=10.08 s.
Loaded excess is therefore not inferred from a damage contour alone. Loaded
phi>.001 reaches the sampled outer x boundaries, whereas phi>.01 remains
x≈[.5423,1.4577], and no phi>.99 exists. This is diffuse damage activation plus
slip on pre-existing damage, not a demonstrated fully developed advancing fault.
The subcritical entirely intact control stays exactly intact. The homogeneous
intact threshold for this calibration is |q|=1000 Pa (K=40 Pa). Under uniform
imposed shear its nominal attainment time is about .05 s, so this loading also
tests widespread intact activation, not only a localized tip process zone.
The largest endpoint-time section rates occur near the outer x boundaries.
Neither those peaks nor the weak halo should be labeled a localized propagating
tip. The coincidence of refined solver failure with this threshold scale is a
diagnostic clue, not proof of a physical bifurcation.

At common t=.045, fixed ell, coarse/half versus fine **coarse-cell-mean** stress
relative differences are .990%/.998%. No arbitrary-point interpolation of native
stress is used. Refined virgin damage is .00118573 m², but its unloaded counterpart
already has .00116146 m²: most is mesh-dependent initial fracture-profile
adjustment. Loaded-minus-unloaded increments are about 2.43e-5 m² fine versus
1.90e-6 coarse. This is not spatial convergence. The fine case's later failure
precludes a same-time .08 comparison; no extrapolated success is reported.

For both completed coarse cases, sampled active sections have one band and no
reverse rate above the explicit diagnostic threshold max(1e-10/s,1e-6 max|w|).
Raw negative local work remains visible: -5.27e-17 / -4.28e-18 J/m, versus
positive .387904/.386523 J/m. In the refined partial trajectory, two steps have
up to three sampled bands, reverse w reaches 7.62e-9/s, and negative local work
is -1.82e-11 J/m versus positive .113092 J/m. These tiny reversals are not
established physical bands; they remain resolution/tolerance-sensitive evidence.
No rate is clipped by this diagnostic. At the final coarse endpoint, variance
is 7.15% of cohesive energy: replacing the complete bulk history by one section
elastic moment would lose measurable stored energy.

For the same active interval (.04,.08], componentwise endpoint work remainders:

| Numerical term, J/m | dt=.005 | dt=.0025 |
|---|---:|---:|
| Stress BE loss | .0487986 | .0245710 |
| Cohesive conjugate BE loss | .00390566 | .00196121 |
| Compliance work minus Y dphi | -.000121859 | -.0000642985 |
| Fracture quadratic increment | .000295125 | .000174276 |
| Physical remainder, signed | .0528775 | .0266422 |
| Physical remainder, accumulated absolute | .0528775 | .0266422 |
| Genuinely added diffuse fracture | .0308367 | .0298617 |
| Absolute external work | 1.72271 | 1.69660 |

Phase/mechanical residual work and closure defects are separately retained in
CSV/JSON. The remainder halves approximately as expected for BE, but is 171.5%
/89.2% of added fracture and 3.07%/1.57% of active external work. Algebraic energy
closure is not physical-work accuracy. No remainder is charged to Gc. Fixed
reference has no advective flux, transfer or history projection; nonlinear
quadrature error remains unmeasured. The old interface four-step 21.72% remainder
was separately decomposed from existing data into bulk BE, surface endpoint,
endpoint/trapezoid difference and roundoff ledger error; no interface replay.
See `evidence/bulk_tip/interface_decomposition.json` and the guide energy page.

## Preserved failures and next experiment

The initial direct RSF-on-integral probe stalled at cold-creep rates due to
cancellation and very large R'; implicit section elimination fixed that arithmetic
problem without changing the constitutive equation. A pressure-gauge predictor
issue was fixed within the new selector. Coarse/half failures then exposed loss
of coupled force correction when phase increments were projected after solving.
The tip-only linearized phase-box solve now includes those bound displacements
before resolving the coupled direction. No loading, tolerance, phase cap,
friction constant or physical obstacle was changed.

The finer .05 failure remains: at Newton iteration zero, the bound loop fixes
all 1,152 phase coordinates (1,083 lower, 69 upper). All 30 trial step lengths
fail the residual merit test: normalized merit starts 1.30e7, reaches 4.50e25 at
full step and is still 8.36e16 at the smallest step. The coupled derivative
passes, but this constrained direction is not acceptable. No contact/admission
law rejects first: this control has prescribed compression and no reconstruction
admission. The result does not establish instability or uniqueness. The add-only
linearized active-set heuristic is not a complete bound-constrained Newton
method and is a specific suspect; stable branch selection is also unqualified.

**Next single experiment:** reuse the refined accepted state at .0475 and audit
the .05 linearized bound problem, including multiplier signs and release of
incorrect active constraints. Evaluate the constrained second variation of the
[restricted incremental stationarity functional](tip_treatment/energy_and_validation.md#restricted-incremental-stationarity-functional).
Keep physical inputs and tolerances fixed. Distinguish a wrong active set from a
loss of constrained stability before choosing a solver change or extending time.
Do not start a general refactor, old-interface continuation or production campaign.

Implementation is complete for the declared closed control; local energetic,
derivative and lifecycle checks pass; temporal physical accuracy remains limited;
spatial/stability qualification and production readiness remain **not established**.
General turning, branching, variable coefficients, current-pressure contact,
material advection/growth and cohesive spin remain unsupported. Mature production
and its four-run closure are separate. Historical shared-assembler evidence and
cleanup losses (39/45 native pilot frames and early temporary logs) remain explicit.

## Reproduction and compact artifacts

Build and one fresh run: [configuration](tip_treatment/configuration.md).
All labels above use `bulk_tip/run.py`; `--dt .0025`, `--refinement 1`, `--unloaded`,
`--intact`, `--ranks 2`, `--rejection` and `--tangent` specify the checks. Full
arguments and environment are retained per run. Active recovery uses
`--resume-from build-particle-free/bulk-tip/runs/mpi03/output --snapshot-id 2`.
The failing-iterate diagnosis sets `ASPECT_BULK_CONTINUATION_TIME=.05` and
`ASPECT_TIP_CHECK_ITERATION=0`, with `--tangent --dt .0025 --refinement 1`.

```sh
MPLCONFIGDIR=/tmp/bulk-tip-mpl /usr/bin/python3 benchmarks/reconstructed_fault/particle_free/bulk_tip/analyze.py control04 half03 unloaded03 fine01 intact01 --figures
MPLCONFIGDIR=/tmp/bulk-tip-mpl /usr/bin/python3 benchmarks/reconstructed_fault/particle_free/bulk_tip/analyze.py control04 half03 fine01 fine-unloaded01 --time .045
/usr/bin/python3 benchmarks/reconstructed_fault/particle_free/bulk_tip/lifecycle.py
/usr/bin/python3 benchmarks/reconstructed_fault/particle_free/bulk_tip/diagnose_interface.py
```

Reducers write compact evidence under `benchmarks/reconstructed_fault/particle_free/evidence/bulk_tip/`.
The figures show the first three same-time cases, with logarithmic damage scale
revealing the weak halo; they do not replace numerical norms:

- [Specimen and damage](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_tip/tip-specimen.png)
- [Resistance, slip, age, driving and shear stress](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_tip/tip-sections.png)
- [Signed/absolute work error and measured cost](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_tip/tip-energy-cost.png)
