# Frozen along-fault wavelength diagnostic

The user-selected continuation now includes targeted 512×128 actions, an
independent compatible short-wave reference, and a separate native pilot.
Read [the continuation report](../../../../doc/reconstructed_fault/particle_free/STAGE4_SHORTWAVE_CONTINUATION_REPORT.md)
for current progress. The sections below describe the original frozen-action
experiment; its original early-stop decision is historical.

This benchmark extends `../stage4_rsf/DERIVATION.md`. It is a read-only
derivative experiment, not a checkpoint conversion, physical restart, or
preparation of an unstable native trajectory. Production sources and the
protected first-free-turn executable are unchanged.

## Specimen and transfer

`prepare.py NX` reads the actual native Bernstein coefficients at step 15,
time .075 of `first-turn-positive-fourcore-end`. It subdivides each original
quadratic in x exactly, retaining all DG face jumps and the original normal
spacing .03125 m. It does not evolve or re-equilibrate phase. A random physical
point check compares parent and refined polynomials; `cross-section.csv`
records the common physical cross-section at x=4.01 m.

Angles come from the retained fixed-phase restoring diagnostic. Its formed
angles are checked against the original .075 parent; its virgin angles had
finished their mechanical selection during the fixed-phase .075–.0775
interval. They are frozen here too. Linear interpolation in physical x defines
the common angle field between the old material labels. New quadrature measures
are ds=dx/cos(angle) and dn=JxW/ds. Formed flags use nearest old label and do not
enter this frozen tangent: no orientation update is called.

The original six-mode candidate V is interpolated linearly in physical x;
theta is its unchanged uniform initial value. These are operating-point data
for the variational RSF analysis, **not** claims about the initial native
solution's rate, age or stress. All responsive ages, including those outside
the force patch, enter the spectrum without rate/phase thresholds.

## Native mechanical action

The benchmark imports phase before material initialization, freezes planes,
constructs the existing `BulkTipCoupling`, and uses ASPECT's ordinary assembled
Stokes preconditioner. At fixed phase, plane and constant G/eta the stress and
cohesive-strain derivatives are independent of the base tau/A histories.
Consequently zero histories suffice for this derivative measurement; the
original coarse responses are an independent reproduction check. There is no
physical interval, state publication, new growth or acceptance-tolerance change.

`prepare_checker.py` reuses the complete retained `stage4_rsf/stiffness.h`,
changing only its pattern count/formula in the generated build copy. The same
mixed equilibrium, point-update finite difference and energy/work checks run
for every action. The native stage duration is .00125 s, exactly the retained
diagnostic's stage duration. Elastic response is beta*dt*deltaV; it is not the
algorithmic friction Jacobian. Original quadrature and mechanical FE stay fixed.

## Patterns and stability

Force patterns are sin(n*pi*(x-2.5)/3) and cos(n*pi*(x-2.5)/3), zero outside
the established patch. They are normalized with actual ds. Six smooth sines
n=1…6 remain in every cumulative space. Extra pairs n=24,32,48,64 bracket
lambda=.125; n=96 is added at 256 columns, n=128,192 at 512 columns. Multiple
Gauss labels inside each cell distinguish sine and cosine even at two cells
per wavelength; the measured weighted Gram spectrum checks degeneracy rather
than assuming nodal Nyquist sampling.

For P the sampled force matrix and C the equilibrated responses, whiten the
positive weighted Gram matrix (coordinate change only). Then

    Cp=P^T W C, K=Cp^-1, Psi=C K, P^T W Psi=I.

Friction is projected on the actual slip responses Psi. State coordinates are
relative age perturbations, and elastic slip obeys zdot=deltaV-r*z, r=G/eta.
The continuous frozen variational matrix is the retained Stage 4 matrix.
The candidate, matched b=a and original Dc laws reuse identical mechanical
actions and the identical V/theta operating point. The original-Dc comparison
is therefore a same-operating-point control, not the old observed physical
trajectory. Cold passive ages can place the largest stable eigenvalue very
close to zero; no cutoff removes them to make stability appear stronger.

Reports distinguish individual force-pattern estimates, sine/cosine pair
spaces, six-smooth-plus-one-pair spaces and cumulative coupled spectra. Signed
complex rate profiles are phase-aligned and normalized to unit maximum modulus.
Their carrier wavelength is measured by a physical-x Fourier scan of the signed
rate (both real and imaginary parts), never by the broad amplitude envelope.
The scan is restricted to at most one cycle per mechanical cell, and at most
1.5 times the highest trial frequency. Fourier factors with many cycles per
cell are not integrated by six-point Gauss quadrature; using them would mistake
quadrature aliases for physical rate oscillations.

There is also a useful mesh-independent **upper bound**, not a measured plateau.
For a growing eigenvalue lambda with real part s>0, eliminate relative age from
the friction equation and test with the complex rate mode. With q=abs(V)/Dc,
c=ftheta*theta*sign(V)/Dc >= 0 and positive elastic K,

    0 = zeta* K zeta/(lambda+r)
        + sum ds |deltaV|^2 [fV - c/(lambda+q)].

Here zeta denotes the generalized rate coordinates. Taking real parts makes
the mechanical term positive, and Re(1/(lambda+q)) <= 1/(s+q). Consequently

    s <= max_sections(c/fV - q).

This rules out unbounded growth-rate divergence within this positive-energy,
frozen-phase/plane continuum law, but does not select a wavelength or prove
that a sampled rate has reached its finite short-wave plateau. The bound uses
the actual regularized friction derivatives, including radiation and the
unchanged operating point. It is not a time-discrete stability criterion.

## Reproduction

From repository root, run once for each fresh input/output label:

```sh
python3 benchmarks/reconstructed_fault/particle_free/stage4_wavelength/prepare.py 128
python3 benchmarks/reconstructed_fault/particle_free/stage4_wavelength/prepare.py 256
python3 benchmarks/reconstructed_fault/particle_free/stage4_wavelength/prepare_checker.py
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py stage4-wavelength-128 1900 python3 benchmarks/reconstructed_fault/particle_free/stage4_wavelength/run.py 128
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py stage4-wavelength-256 1900 python3 benchmarks/reconstructed_fault/particle_free/stage4_wavelength/run.py 256
python3 benchmarks/reconstructed_fault/particle_free/stage4_wavelength/analyze.py 128
python3 benchmarks/reconstructed_fault/particle_free/stage4_wavelength/analyze.py 256
python3 benchmarks/reconstructed_fault/particle_free/stage4_wavelength/compare.py
python3 benchmarks/reconstructed_fault/particle_free/stage4_wavelength/retain.py
python3 benchmarks/reconstructed_fault/particle_free/stage4_wavelength/plot.py
```

Run native commands sequentially: each binds four MPI ranks to four cores.
Use a new `--suffix` on reruns; retain failures. Build provenance and generated
sources are under `build-particle-free/stage4-wavelength`; exact physical-run
commands, runtime hashes and timings are in each run's `command.json`.
The executed outcome and next proposed comparison are documented in
[`STAGE4_RSF_WAVELENGTH_REPORT.md`](../../../../doc/reconstructed_fault/particle_free/STAGE4_RSF_WAVELENGTH_REPORT.md).
Compact native outputs and compressed exact input coefficients are retained in
`../evidence/stage4_wavelength/native-NX/`. The original build/run data remain
untouched. In the original task, the conditional `prepare.py 512` case was
deferred because the two-mesh migration settled that immediate negative
qualification. The subsequently selected continuation below supplies 512 data.

## Matched continuation and interruption recovery

`prepare.py 512 --family targeted` selects six smooth actions and pairs n=96,192.
The shutdown and daemon interruptions are recorded without relabelling either
as a successful run. `continue_analysis.py` selects complete actions from the
original `-512-mpi`, `-512-resume`, and `-512-pairs` runs and verifies duplicate
rows and common section data. `prepare_checker.py --build ...` can build an
independent checker with flushed completed-action CSVs. This changes output
lifetime only; the duplicated smooth result is bit-for-bit identical.

Run `shortwave.py`, then `check_shortwave.py` for the compatible Fourier and
asymptotic references; run `continue_analysis.py`, then `plot_shortwave.py` for
the native matched comparison. All analysis uses one numerical worker.

## Separate native slipping pilot

`prepare_pilot.py` builds only benchmark support in `build-particle-free/stage4-pilot`.
It leaves production sources and all protected executables/checkers unchanged.
`preload.h` imports the same frozen phase/planes, equilibrates physical stress
and cohesive strain under a prescribed affine elastic boundary displacement,
and checks equilibrium, work, and the owned signed section friction law. The
initial age/rate target is the same physical operating profile used above.
No existing checkpoint is resumed with this synthetic history.

`run_pilot.py NX LAW CASE` supports meshes 256/512, laws weakening/neutral, and
background/perturbed. It uses native midpoint at dt=8e-5 s with fixed phase;
`--steps`, `--seconds`, `--suffix` bound the run and preserve previous evidence.
The default four ranks may be reduced to one/two to share the four-worker budget
with independent analysis. Initial and endpoint ages/slips are distinguished
from subsequent midpoint stage rates/tractions in `pilot-summary.csv` and
`pilot-sections-NNNN.csv`. The initial rate is instantaneous at t=0. The age
perturbation is a common physical 1% sin² patch; each law needs its own background.

This preparation chooses gamma=2e-4 elastic shear plus .4195498156 gamma extension,
then applies the original .01/s shear plus .004195498156/s extension velocity.
The benchmark's fixed-state signed-law evaluation analytically inverts the BE
age map before calling the existing public section operation, admitting zero
rate without clipping or inventing an age-evolution interval. The postprocessor
verifies the requested age is recovered to roundoff. All subsequent physical
steps use the unchanged native midpoint solver and its original acceptance gates.

`neutral` means the stateful b=a control, with its reference coefficient chosen
to match the weakening law's initial traction. Each mesh/law needs its own
background to distinguish changing base loading from the age perturbation.
The comparison window is four intervals; the initial 256 weakening background
also retains its original eight-interval endpoint. Compare only common prefixes.
No interval is a release of the old .075 checkpoint under changed material data.

After all native ranks exit, reproduce the inexpensive verification and figures:

```sh
python3 benchmarks/reconstructed_fault/particle_free/stage4_wavelength/check_native_spectra.py
python3 benchmarks/reconstructed_fault/particle_free/stage4_wavelength/check_limit_quadrature.py
python3 benchmarks/reconstructed_fault/particle_free/stage4_wavelength/plot_shortwave.py
python3 benchmarks/reconstructed_fault/particle_free/stage4_wavelength/analyze_pilot.py
python3 benchmarks/reconstructed_fault/particle_free/stage4_wavelength/retain_pilot.py
```

The quadrature check splits the along-fault integration at actual interpolation
knots and phase-cell faces. The native-root check substitutes growing roots into
the small physical dispersion matrix in original force coordinates, independently
of the full-age matrix's Gram whitening. `analyze_pilot.py` separates midpoint
rates/resistance from endpoint age/slip, compares common histories, and records
both signed and accumulated absolute work errors and signed local cancellation.
Section-profile L2 comparisons interpolate coarse Gauss observations onto fine
ones; they are not exact FE-field L2 errors. `retain_pilot.py` retains exact input
coefficients, checker/source hashes and checkpoint identities without copying
checkpoint payloads into the compact evidence directory. It does not delete
original outputs or change checkpoint meanings.
