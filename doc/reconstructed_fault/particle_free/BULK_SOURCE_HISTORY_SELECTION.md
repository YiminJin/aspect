# Non-affine native source-history qualification

Entry f1f9f29cf plus retained local work, 20 October 2026. No law, seed, loading,
time integration, quadrature, tolerance, limiter or BP3 path is changed.

For the fixed mesh and homogeneous constant coefficients let P3 be the retained
cellwise Gauss3 DGQ2 projection. Store native FE velocity memory A in the same
mixed FE vector representation as the retained FE-history path, with all other
blocks zero. The full FEStrainState surface owner is deliberately not reused:
its normalization/Z/geometry contracts describe another constitutive route.
Reuse the native FE shapes, gradient evaluation and Maxwell memory recurrence:

    A_next = beta (A_n + du), beta=(1+dt G/eta)^(-1),
    T_DG_next = beta (T_DG_n + 2G P3 dev3 symgrad du),
    T_phys = T_DG + 2G dev3 symgrad A - P3(2G dev3 symgrad A).

The last projection is a derived cache, rebuilt from A after restart. This
retains arbitrary independent DG modes; it does not replace T_DG by a sampled
gradient. The remainder obeys beta[R_n+2G(Id-P3) dev3 symgrad du]. It is point-
evaluable on every matching cell, including non-Gauss diagnostic points. The
cohesive a/k/L/q/rate histories remain at their original mechanical QPs; they
are never interpolated into another constitutive state.

Every trial anchors the same accepted T_DG/A/local histories/physical obstacle.
Mechanics, tangent, local return and energy use the same full incoming tensor
and native incremental gradient. Bulk energy is the full physical stress norm,
including cross terms, zz and both xy contributions; there is no extra energy
assigned independently to the remainder. Both tensor and FE owners publish
once in the existing endpoint callback. Rejection discards both. The existing
checkpoint mechanism gains the FE velocity-memory values and an explicit v2
bulk signature; old v1 bulk checkpoints are rejected without conversion. BP3
checkpoint bytes and behavior are outside this default-off branch and unchanged.

Before orientation runs, qualify one 24x32 small affine/distorted pair on the
same 1.5x1 m rectangle. The distortion is prescribed once at mesh construction,
with boundary fixed: dx=.006 sin(2pi x/1.5)sin(2pi y),
dy=.004 sin(2pi x/1.5)sin(2pi y). Start unloaded from the unchanged continuum
seed; stop at .12 s. Use native Gauss8/source3, dt=.0025 s and current tolerances:
source reconstruction <=1e-12, mechanics <=1e-9, phase KKT <=1e-7, native
tangent FD <=1e-6 and existing energy ledger/positive-dissipation checks.
Independent coefficient/Jacobian point reconstructions use the established
1e-10 verification screen. Check nonzero source remainder, all nine modes,
native FE memory advancement, stress/storage consistency, deliberate history
inconsistency rejection with actual error/location, true outer rejection/retry,
single accepted publication and prefix/restart versus uninterrupted histories.
Do not retune an acceptance tolerance to pass a control.

Only after those checks pass, repeat the original aligned wide and single 45
degree core/annulus control from fresh histories through .145 s, with all
physical inputs and local core resolution unchanged. Use the same physical
radial kernel of radius .0625 m alongside each native nodal KKT condition.
Compare at common physical times and actual first-growing nodes. Distinguish
initial representation, the different far-field discretization and remaining
quadrature uncertainty; neither mesh is an accuracy reference.

The existing physical external-energy failure is separate from exact local
accounting and source consistency. Preserve it, strict_reference_match_passed=
false, prior native failures and BP3 artifacts. No AMR, MPI expansion, transport,
surface birth, moving geometry, cap reshaping or increased drive is selected.

A retained failure of the first complement implementation at step 47 is
reproduced from its checkpoint with failure-only operand exports. Its 70-digit
reconstruction isolates cancellation in the accumulated native memory gradient.
Before the fresh final orientation pair, remove one constant cell translation
per velocity component and accumulate that gradient in long double. This uses
partition of unity; no source, projection, increment gradient, constitutive
formula or 1e-12 check changes. Repeat small independent/restart controls with
the final arithmetic before that orientation pair. Preserve both failed paths.
