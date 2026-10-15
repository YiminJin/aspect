# Prescribed moving-particle transport checks — 2026-10-14

**Seven controls pass their transport/constitutive/refinement gates. The full
selected gate remains open:** one warranted denser shear control fails the
strict domain-coverage check. It is preserved and independently diagnosed;
no tolerance, geometry algorithm or physical law was changed to pass it.

Baseline: `fault-propagation` at `c33edaaeb`, with all incoming P2/BP3 edits.
Native ASPECT executable remains `1ba2f7f43b37b75b1462d9cd5e48998fc8cc58b891e943829be8240f70776d12`.
Build uses GCC 12.4/deal.II 9.6.2/OpenMPI 5.0.6 and the retained native runtime.
All tests are serial; independent launches were batched, not MPI scaling tests.
Eleven launches, including one parameter failure and three failing-domain
attempts, total **76.76 s**, excluding compilation. No BP3 first-event run starts.

## Tests and measured behavior

The 1×1 m Box is periodic in x, fixed in y, with no mesh changes or particle
birth/deletion. Native regular seeding, RK2, cell sorting, limited LLS,
continuous-Q2 mapped stress fields and Voronoi/CPDI domain rebuilds are used.
Uniform translation U=(.4,0) runs to T=1 s with dt=.0625 s: each particle moves
3.2 coarse-cell widths or 6.4 refined-cell widths, including actual cell and
periodic-boundary crossings. Step zero is separate and preserves histories.

Initial stress is nonuniform and trace-free:

\[
 T_{xx}=1+.2\sin(2\pi x)\cos(2\pi y),\quad T_{yy}=-T_{xx},\quad
 T_{xy}=.3+.15\cos(2\pi x)\sin(2\pi y).
\]

G=2 Pa, eta=4 Pa s and lambda=2 s are named manufactured controls. The
translation reference is exp(-t/lambda) T0(x-Ut). These synthetic choices do
not change BP3 material or loading inputs. All subsequent stress generation
uses gradients of the prescribed **bulk FE velocity**, not particle gradients.

| Translation control | Particles | Minimum cell crossings per particle | Final particle stress maximum error, Pa | FE stress RMS error, Pa | Direct LLS stress RMS error, Pa |
|---|---:|---:|---:|---:|---:|
| 8×8 cells, 4×4/cell | 1024 | 3 | 2.33e-15 | .005419 | .008550 |
| 8×8, 8×8/cell | 4096 | 3 | 3.53e-15 | .004648 | .007740 |
| 8×8, 16×16/cell | 16384 | 3 | 5.71e-15 | .004281 | .007319 |
| 16×16, 8×8/cell | 16384 | 6 | 8.03e-15 | .001199 | .002057 |

Maximum translation position error is 4.45e-16 m; stress error over all times
is at most 8.15e-15 Pa. Those are roundoff floors, not convergence estimates.
At fixed 8×8/cell density, bulk refinement reduces FE and direct-LLS RMS errors
by factors 3.87 and 3.76. Increasing particle density alone leaves a bulk-field
resolution floor. The error measurement samples the actual pre-publication
mapped fields at native 3×3 integration points and evaluates the current
Maxwell response; direct LLS at those same points is recorded separately.
FE-minus-LLS RMS is also saved. In translation, particle error is negligible,
so these field errors measure reconstruction, not constitutive drift.

## Narrow passive history and its actual weak vector

The scalar `crack_driving_force` stores a passive Gaussian with center .213 m,
sigma=.0125 m, independent of y. Its FWHM=.02944 m is below a cell width in
both tested bulk meshes. It is not refreshed from position, reprojected back
to particles, or used to evolve damage. Every surviving particle retains its
initial pulse value **exactly**, and the initialized particle peak stays
unchanged through all crossings. Initial peak undersampling is reported rather
than confused with transport loss.

Using the **native CPDI domain volumes and weighting-function values**, the
observer assembles every entry of F_i=sum_p H_p volume_p Nbar_ip. It compares
the whole periodic Q1 weak vector to integration of the analytical transported
pulse against those same Q1 tests. This is the H-loading basis in the phase
weak form, isolated without a phase solve; it is not a visualization norm.
The Gauss-32 reference additionally agrees with independent Gaussian
antiderivatives to 1.89e-15 in relative L1.

| Fixed 8×8 bulk mesh | Final weak-vector relative L1 error | Relative mass error | Retained particle peak |
|---|---:|---:|---:|
| 4×4 particles/cell | 11.5768% | 3.42507% | .731945 |
| 8×8/cell | 1.51648% | 4.40621e-6 | .986480 |
| 16×16/cell | .359093% | 2.22e-15 | .989181 |

Weak-vector errors decrease under particle refinement even while the pulse
remains subcell. Sampled mass is constant in time to the 2e-13 check; coarse
mass error is initial sampling error. Final `pulse_particles.csv` contains
actual values and material-label references, and `weak_pulse.csv` contains
every assembled/reference weak entry. They preserve evidence beyond smoothed
fields or global mass alone.

## Transport plus constitutive generation

The second flow is u=(.4+.2 t(y-.5),0), a trace-free shear-rate ramp. Its exact
characteristics are x=x0+.4t+.1t²(y0-.5), y=y0, and the non-rotational Maxwell
reference is

\[
 T(t)=e^{-t/\lambda}T_0(X_0),\qquad
 T_{xy}(t)\mathrel{+}=\eta(.2)[t-\lambda(1-e^{-t/\lambda})].
\]

Each particle crosses at least two cells. On the same 8×8 mesh and 8×8/cell
particle set:

| dt, s | Positive steps | Final particle stress error, Pa | Position error, m | FE total RMS error, Pa |
|---:|---:|---:|---:|---:|
| .125 | 8 | .0275327 | .000769043 | .0278981 |
| .0625 | 16 | .0138388 | .000192261 | .0145482 |
| .03125 | 32 | .00693751 | .0000480652 | .00826160 |

Stress-error ratios are 1.9895/1.9948; position ratios are 4.000/4.000. Stress
errors decrease at every common positive physical time. The source is frozen
at each interval endpoint, explaining its observed first-order trend against
the exact ramp convolution. Before the first positive solve, RK2 uses the
available velocity predictor: its first shear displacement is missing
.1 dt²(y0-.5). This accounts for the measured position error; it is not a
claim of second-order accuracy for coupled mechanics/history. Pulse weak error
has a spatial floor and is not monotone in temporal maxima; timestep halving
does not qualify its spatial representation.

The prescribed no-Stokes scheme has no native terminal mechanical-history call.
The named diagnostic driver therefore calls the **production Maxwell coefficient
and stress kernel** once, using each moved particle's own old tensor and direct
FE strain. A step-number guard prevents duplicate updates; the analytical
trajectory detects missing/doubled relaxation or generation. Maximum one-step
decomposition defect is 2.33e-16 Pa. Reconstructed FE fields are never the
authoritative old particle tensor. This tests native transport/transfer and the
native constitutive kernel, **not** the complete coupled terminal publication
method, surface RSF, B/G tangents or complete-step rejection.

The written law excludes objective rotation, so no Jaumann/corotational law or
rigid-rotation qualification is introduced. Trace-free deformation makes the
legacy full-gradient and intended 3D-deviator conventions identical here; their
general mismatch and the previously identified G trace gap remain unresolved.

## Preserved dense-shear failure

One targeted 16×16/cell shear control at dt=.03125 was added to resolve the
weak-pulse spatial floor. It aborts at **t=.0625 s, step 2**. Last complete
diagnostic row is t=.03125 s. Two unchanged-input diagnostic replays reproduce
the failure; no continuation bypasses the guard.

- Double/extended sums of domain area: .99999999375459347 /
  .99999999375455651. The 6.24544e-9 deficit is not accumulation roundoff.
- Maximum CPDI partition-of-unity error: 4.44e-16. Basis normalization is working.
- One saved polygon at (.653799819946289,.13671875) has native area
  6.103515873679456e-5. Independent half-plane clipping of its actual neighboring
  sites gives 6.103515625e-5, the lattice area: relative discrepancy 4.07436e-8.
  A native vertex violates a nearest-site half-plane by 2.53144e-12.

The discrepancy is already in the constructed Voronoi polygon, before CPDI
weight normalization or weak assembly; it is unrelated to Maxwell recurrence
or FE stress transfer. Installed Voro++ has fixed tolerance 1e-11, a relevant
conditioning lead, not a proved complete diagnosis. The native Debug global
volume guard is 1e-6; this test deliberately checks 1e-10. The deficit is below
that existing guard and far below the measured pulse discretization errors,
but **the stricter test failed**. Neither threshold was changed. Saved polygon,
neighbors and independent reconstruction are under the diagnostic output and
`failed-domain-independent.json`. A bounded frozen-entry geometry/precision
investigation should resolve this before claiming an unconditional moving gate.

## Files, reproduction and limits

New fixtures/plugin/analysis are in [transport/](transport/). The only existing
code edit is the narrow test-access wrapper in `tests/phase_field_fault_test_access.h`
calling the existing private production kernel; no production algorithm or
default changes. Core and original stationary server archive hashes remain
unchanged. Incoming manual/P2/BP3 work and all successful/failed evidence remain.
ASPECT astyle 2.04 and `git diff --check` pass. Plugin build passes; the initial
compile used an unavailable tensor `norm_square()` API and was corrected, and
the first input needed an explicit unused radiation-damping value. Both failed
logs and the old input are retained. No physical tuning followed a numerical
comparison failure.

Build manually with CMake and `Aspect_DIR=build-bp3-native`, as recorded in
`build-bp3-native/transport-checks/`. Runs use the existing
`refactoring_boundary/run_logged.py`, 300-second external limits and T=1 s.
Reproduction must select fresh output paths. Run analysis with
`python3 benchmarks/reconstructed_fault/bp3_native/transport/analyze.py --output NEW.json`;
its final exit code is **1** because the dense shear is incomplete. Passing
completed-family refinements and the failed gate are both explicit in
`assessment-final-with-diagnosis.json`. Provenance and preservation manifests
retain exact commands, artifact identities and input/source hashes.

No MPI migration, births/deletion, objective rotation, restart, automatic
cutback, full coupled moving BP3 or first-event qualification follows. The
remaining next steps are the localized domain-geometry failure, the scoped
common-convention/G correction, and a subsequent actual coupled moving-history
publication check. Earlier stationary results and their limits are unchanged.
