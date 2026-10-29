## P4 continuation/stability diagnostic — 21 October 2026 (bounded completion)

The default bulk initial guess is dt times accepted velocity, accepted pressure
and accepted phase. Current physical boundary constraints override its boundary
values. Thus its first hold trial can retain a nonzero *interior* displacement
predictor even though boundary motion is zero. Trial constitutive variables are
always re-evaluated from the same immutable bulk owner and phase obstacle.

For constant matched parameters, let Tplus=beta(Tn+2G devsymgrad(du)) and
abar=beta an. The exact local return minimizes

    ||Tplus-2GaS||_3D^2/(4G)+G a^2/(2h)+R |a-abar|.

Its envelope derivatives are beta times physical mechanical stress and -Y.
Including AT1 gives the discrete incremental potential
Eb+Ec+Ef+beta*dt*Dfriction, subject to incompressibility and the physical phase
obstacle. This is a diagnostic of the existing BE equations, not a new viscosity
or jump law. The smooth reduced phase second variation must eliminate mixed
mechanics/pressure and local return response. Existing row scales G/Kf explain
the apparent nonsymmetric cross blocks: multiply mechanical/pressure rows by
beta*G and phase rows by Kf to obtain the symmetric physical tangent.

A test-only ASPECT_BULK_CONTINUATION_TIME selects diagnostic export at one
physical time. ASPECT_BULK_RELAXATION_GUESS then selects zero incremental
velocity, beta-scaled pressure and accepted phase before imposing current
boundary constraints. It changes neither the accepted history nor the default.
The exported tangent excludes constrained/artificial rows after native
condensation. Offline mechanically relaxed critical-space checks must not
condense it twice. Strict positive lower-obstacle multipliers remove phase
critical directions; pressure saddle eigenvalues are not phase instabilities.
Independent point-potential checks cover both stick and slip; native derivative
and stability measurements are recorded in the continuation report/evidence.

The propagation observer connects the already existing post_resume_time_step
signal, reducing only a saved pending interval to the requested maximum dt.
Accepted histories and old_time_step are untouched. Optional compact output
suppresses only dense diagnostic QP CSV, retaining every accepted phase/force/
cell-mean/reaction/history guard and checkpoint. Existing output defaults and
protected mature BP3 paths remain unchanged.

### Critical-space verification correction

The constant absolute phase FD probe can cross Coulomb branches near tiny
positive h; its failures remain evidence. A separate stability check uses
interior/near-critical phase masters and relative-to-obstacle phase directions,
with unchanged epsilon1e-7/5e-8 and both the original 1e-6 scaled bound and an
additional 1e-6 relative bound. The complete operator is P^T J P: restricting
only input directions leaves irrelevant strictly active phase residual rows in
the comparison. That initial incomplete-output test failed (1.9791e-5 relative)
and is preserved. Correct output restriction passes at 8.6009e-7, while reporting
the unprojected error separately. This is a verification-space correction;
Newton, constitutive derivatives, source projection and physical acceptance are
unchanged. Native C^T reduction is already complete and is never applied twice.
Exported master gradients and mass use C^T g/C^T m, with constrained/artificial
rows excluded. Current continuation experiments are uniform Cartesian, four-rank
fixed-mesh controls; no new hanging-node/MPI scope is implied.

The homogeneously weighted BE incremental potential gives positive relaxed
curvature .139907 at the loaded state and negative curvature -.208558 at the
predictor's hold-growth root. Its frozen-mechanics phase block is nevertheless
positive (6.37136). Thus a phase-only observer would miss the coupled instability.
The actual continuation initial guess solves with phase available throughout
Omega; its no-growth result follows strict native obstacle gaps, not a freeze.

### Enlarged-boundary diagnostic correction (P4)

The first 1.75 m control passed the native step-41 mechanics (2.49207e-18),
source and energy checks, then its postprocessor failed the unchanged natural
side residual gate. `MomentCycle::export_state()` still classified x=1.5 as a
free boundary. On the enlarged box that selects interior DoFs at y=.0078125
and .9921875, integrating only their boundary-cell contributions and omitting
cancelling interior cells. The observed false force norm .165726080264519 N/m
matches sqrt(2)*tau*Delta x/3 to 4.49e-13 relative, using the independent
homogeneous first-step Maxwell stress. It also misses the actual x=1.75 side.

The benchmark now selects supported velocity DoFs from native boundary faces
and retains the existing constraint classification and 1e-9 scaled residual
bound. No solver, equation, material or acceptance change is involved. The
failed output and old diagnostic DSO/source are preserved separately; the
corrected observer is built in `bulk-continuation/plugin-boundary`. A matching
one-step 1.5 m replay and fresh same-input 1.75 m run verify the correction.
This is a post-publication diagnostic failure, not an outer-loop rejection;
no checkpoint containing step 41 was published by the failed observer run.


### Completed P4 controls and limits

All three shared-parent timestep branches and the two fresh controls complete
through .17875 s. Predictor hold roots retain negative relaxed curvature under
halving; stable continuation is a test-only native initial guess, not a new
production acceptance policy. The refined/wider loaded curvatures are +.120646/
+2.34054; their holds have strictly positive obstacle multipliers. Exact BE
closure and physical force-displacement work are separate from stability.

The full exported-field scan finds refined onset (.12750,.12875], correcting
the preliminary log estimate. Maximum growth L2 difference on the refined
control is 7.442%; the wider-domain final difference is 94.51% of its growth
norm. Domain growth integral falls 57.26%. No reflection, history interpolation,
new phase cap, limiter, material change or tolerance relaxation is used.
The three edited implementation files are the bulk solver's opt-in diagnostics/
initial guess and the existing parallel/propagation benchmark observers. Point
law and accepted-history-owner files retain their original hashes. Source and
publication checks retain 1e-12 coverage. Frozen ID/coordinate fingerprints,
positive-step pressure gates and once-only ownership/publication are checked;
step-zero ASPECT pressure retains its existing pre-bulk-solve exclusion.

Native full-space FD failures and the incomplete critical-output test remain
recorded. Final Q1 peak errors use union-grid vertices; earlier Gauss-point peak
calculations remain labeled development evidence. Zero interval growth at the
parent is explicitly marked, without a roundoff-normalized error claim. The
[P5 specification](P5_DIRECTION_SPECIFICATION.md) is a conditional tensor-memory
constitutive proposal, not an implementation or a change to this scalar model.
Current quadrature and earlier-prefix uncertainty, branch selection, boundary
sensitivity and prior strict-reference failures are not erased by P4 closeout.
