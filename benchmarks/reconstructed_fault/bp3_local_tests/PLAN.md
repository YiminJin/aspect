# Section 5 bounded local comparison (declared before runs)

Reference: accepted Section 4 commit `1678949c2`, core executable
`build-refactor-r6b/aspect-filter-derivative-qualified` (SHA256
`5ded9532b898a8b5f569ef9c568644c4e929c11db550515a74373344c0d2fa83`).
The Section-3 unbuffered graded-completion admission failure remains open.
This test does not fix core completion or change the maintained production default.

Use one/two ranks, AMG, at most 1200 seconds total simulation wall time including
pilots/failures. Preserve logs and partial outputs. Build time is separate. Reuse
Section-3 (1888 comparisons) / Section-4 (211 checks) geometry, lifecycle, retry,
RNG and restart evidence rather than repeating that matrix.

Two local meshes only, fixed after native initial-global refinement: 2x1 km,
60 degrees, ell=20 m, finest edge 3.90625 m, coarse bound 250 m. Both keep the
accepted fine band max(2 ell,R+2 h_fine). A has exterior slope 1/4. B recovers
slope 1/2 and cap 12500 m from `tests/bp3_length_scale_mesh.cc`, applying the
same protected band rather than that historical generator's smaller 1.977 ell
band. The cap is inactive locally. Both protect boundary cells intersecting
an endpoint interval of half width (R+h_coarse*(|n_x|+|n_y|))/sin(dip), plus
one fine edge, in a two-fine-edge boundary strip. This conservatively covers
current core admission even before realized h_max is known. It is a diagnostic
buffer, not qualification of the unbuffered A/production policy. Native balancing
and conservative cell distances remain active.

The isolated build compiles the maintained plugin sources with
BP3_LOCAL_OSCILLATION_TEST=ON; the default build remains unchanged. Functional
uniform strengthening: a=.025,b=.015,Dc=.008,Vinit=Vp=.001 m/s, steady Theta=8 s.
Use the live friction law for initial shear with the retained BP3 damping and
50 MPa background. Frozen H initializer, native Maxwell transfer, 4x4 seeding,
12/24 bounds, LLS/Q2/limiter and filter20 remain. No prescribed fault slip.

Pilot dt=.05 s (S=.00625 at target V) with all safety limits retained. Choose
20--40 common fixed steps only after timing and safety inspection; no mismatched
adaptive schedules are called a controlled comparison. Save an intermediate A
checkpoint after nonzero history, then branch with dt/4 for two base intervals.

Predeclared comparison scale: 1e-3 m/s for both V and u differences. Provisional
engineering target is 1% RMS active-fault V difference relative to this common
scale; maxima/localized differences reported separately. Active subset V>1e-5
m/s on both meshes; report absolute differences outside it. Existing positivity,
solver, H, work, and lifecycle assertions are not loosened. Roughness is a
centered physical-window second difference, with raw curves retained; it is not
an exact error. Analytic transport alone has a known exact reference.

The actual Maxwell law is beta=exp(-dt G/eta), eta_ve=-eta*expm1(-dt G/eta).
Constitutive step zero uses Initial time step; subsequent steps use simulator dt.
The fixture sets the former to pilot dt. No backward-Euler substitution is made.

Transport reuses prescribed velocity/native particle management and LLS/Q2
sampling infrastructure. Use one curved deviatoric tensor [q,-q,q/2], amplitude
1e8 Pa, with the precise field/flow and sampling convention documented before
transport runs. Exact-position shadow samples must not modify coupled history.
Report realized interface crossings and motion, not a presumed excitation.

Stop after this local comparison. No production policy replacement, solver change,
server submission, long-event run or further refactoring stage is selected.

Transport declaration (before runs): U=(1,.6) m/s, dt=.4 s, end=1.6 s;
q(x,y,t)=1e8*[1+.1*sin(pi*(x-t)/64)*cos(pi*(y-.6*t)/64)] Pa,
tensor [q,-q,q/2], wavelength128 m in both coordinates. Four moves displace
1.866 m (0.478 fine-cell edges). Native spatially refreshed initial-composition
properties provide an independent exact-position shadow tensor on the same
particles; stored Maxwell components are never overwritten. Both sets pass
through native limited LLS and simulator continuous-Q2 shared-DoF averaging and
constraints. At actual Gauss3 coordinates compare both mapped fields with q.
Uniform/interface and boundary/interior cells are reported separately. The
analytic field is defined at every inflow/birth location. Retained stable IDs
carry path sums through all-gathered observation maps; Cp uses the incoming cell
edge, Ap sums each accepted fractional displacement. Births begin at zero and
removed IDs are discarded, so reported maxima belong to surviving populations.

The realized periodic checkpoint is after accepted step9 at t=.45 s (the native
checkpoint cadence saves after advancing its pending clock). The two continuations
both start there. The existing post_resume_time_step hook reduces only the
pending .05 s interval to .0125 s, preserving accepted time and old dt. The
constant-dt replay takes two steps to .55 s and reproduces the uninterrupted
trajectory bitwise. The changed-dt branch takes eight steps to that same time.
No new production clock interface or checkpoint rewrite is used.

During transport, native max-ID-based allocation reuses an ID after its prior
particle exits. This invalidates ID-only path tracking. Final diagnostics require
both the ID and the known constant-flow trajectory to match within a geometric
floating-point bound (256 epsilon times coordinate scale); a reintroduced ID is
counted as a birth and its path begins at zero. Active IDs remain globally unique.
This is a diagnostic identification rule, not a change to native allocation or
BP3 audit/population counting. The serial diagnostic checks the same issue.
