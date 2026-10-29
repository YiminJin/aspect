# P4 continuation controls selected before execution

Entry f1f9f29cf plus retained local work, 21 October 2026. Preserve the bulk
cohesive law, all physical data of the Gc=1/C=2000/ell=.046875 specimen,
96x128 mesh, Gauss8/source3, native FE complement, unlimited DG, obstacles,
acceptance tolerances and the released-side loading. No direction freedom.

Branch from the retained accepted step 127 at .15875 s, not the checkpoint's
pending clock .16. Use the existing native post_resume_time_step signal to
reduce only the pending interval. Compare dt=.00125 and .000625 to .17875;
loading ends exactly at .175. Split at .175 to preserve the identical incoming
hold checkpoint for both native initial guesses. If growth-relative phase L2
or peak differences exceed 5%, a .1 contour differs by more than dx=.015625 m,
or the hold selects materially different branches, add .0003125. No threshold
or physical parameter search. Compare contours .001/.01/.05/.1/.3/.5 against
time and cumulative imposed relative grip displacement. Record all step maxima,
reaction, slip activity, storage/dissipation/work and physical energy remainder.

Diagnostic initial guesses change only the Newton starting point: (a) existing
accepted-velocity extrapolation; (b) zero displacement increment with Maxwell-
scaled pressure, accepted phase, and the actual current boundary constraints.
Every constitutive evaluation uses the same immutable incoming history. Both
must satisfy the unchanged native acceptance tests. Retain the existing default.
Analyze the physical critical cone and mechanically reduced consistent tangent,
including pressure/constraints and constitutive return response. On smooth return
branches, multiplying physical mechanical/pressure rows by beta symmetrizes the
cross blocks. Thus Hred=Hphiphi-beta Huphi^T Kmech^-1 Huphi, with incompressible
mechanics eliminated. A phase block at frozen mechanics is insufficient. Strict
positive obstacle multipliers exclude corresponding phase directions from the
critical cone. Report switches/numerical eigenvalue bounds explicitly; no new
physical jump or hold-growth prohibition is selected.

After identifying the branch, use one targeted fixed-mesh spatial control and
one enlarged-domain control of the straight-growth interval. Initialize changed
meshes freshly from the same continuum seed and zero histories, never remap
accepted quadrature records. Preserve local cell sizes when moving boundaries
and preserve affine shear-strain history. Declare actual meshes/cost before
launch. Practical response targets remain 5% growth-relative field/integral
changes and one original cell in front position; report failures, not a two-mesh
convergence order. New-regime errors are separate from earlier specimen results.

Prepare P5 as a specification only. Preserve BP3, earlier failures and
strict_reference_match_passed=false. Compact evidence is PNG, one GIF, CSV/JSON
and report in one archive; full solver/checkpoint records remain separate.

## Focused controls selected after the native branch test

The loaded .175 state has positive relaxed curvature (minimum .139907); the
predictor's hold-growth root has negative curvature (-.208558), while the
accepted-state continuation has strictly positive obstacle gaps and no phase
critical directions. Select local stable continuation from the accepted state
for the remaining controls, without a hold-growth constraint or default change.
The half-step loaded field differs by 5.07e-5 relative to interval growth.
Proceed with spatial/domain controls only after confirming the same selected
branch at the half timestep through .17875; further halve if the declared
physical-field/front/branch criteria fail.

One fresh tangential refinement: 192x128 on 1.5x1 m, Delta x=Delta y=.0078125 m.
This doubles resolution along straight growth and retains the old normal spacing;
it is not an isotropic spatial convergence study. One fresh right-boundary control:
112x128 on 1.75x1 m, retaining Delta x=.015625/Delta y=.0078125 m and the seed at
(1,.5). Top/bottom velocities, y boundaries, p_N, law, Gauss8/source3 and physical
shear-strain history remain identical. This tests the right boundary, not full
domain independence. Compare fields without reflecting or aligning branches.

Use the same .00125 s timestep if the selected half-step control passes; fresh
zero histories and the same continuum seed on each mesh. Stop at .17875, split
at .175 with native end-step termination to retain the incoming hold checkpoint.
The physical end-time cap stays .185. Export loaded and first-hold tangents and
check native guards. Costs estimated from the retained 1906 s full trajectory:
roughly 45--90 minutes for the doubled-cell spatial case and 30--50 minutes for
the enlarged domain; bounded native timeouts 7200/5400 s, four ranks, sequential.
No added mesh or domain is selected if a target fails.

Report raw reactions/storage and also reaction per width and excess relative
to the homogeneous intact Maxwell solution: enlarging the grips and bulk volume
adds a background response even without a seed. Use common original-domain
field/functionals plus the added region separately. Compare initial represented
phase before growth, integrated growth, several contours and energy/work terms.
Keep the 5% growth-relative field/integral and one original-cell front targets;
report failure and residual normal/quadrature limitations explicitly.

## Further-halving trigger

At common accepted times the original/half-step default trajectories differ by
only .21365% in interval-growth L2 at .17875. However, the coarse linear temporal
representation misses the actual saved half-step field at .174375 by .0192199
in phase, or 8.68245% of interval growth. This exceeds the declared 5% peak
criterion for the rapid interval. Run dt=.0003125 from the same accepted .15875
checkpoint, retaining a clean .175 incoming hold state. Report temporal sampling
uncertainty separately from differences at native common times; do not evaluate
constitutive histories on interpolated states. This is the single requested
further halving, not a parameter search. Existing point/operator derivative
checks have passed; additional trajectories keep all physical guards and export
stability matrices without redundantly enabling the exploratory FD probes.
