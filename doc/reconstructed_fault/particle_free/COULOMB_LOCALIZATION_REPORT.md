# Coulomb mechanical localization audit — 23 October 2026

**The mechanical prerequisite fails. Do not advance this zero-dilation law
to phase-active or larger specimens.** The actual mixed displacement–pressure
tangent admits short-wavelength mechanical localization at developed damage.
The phase gradient and current Maxwell relaxation do not control that mode.
The bounded native runs show deteriorating mechanical conditioning, but do
not demonstrate a collapsing slip band in their short observation window.
No constitutive remedy or acceptance relaxation has been applied.

Baseline: `pf-rsf-particle-free`, HEAD
`f1f9f29cf40c32d1ddcea000b724375b64853532` plus preserved entry work.
[Preselected scope](COULOMB_LOCALIZATION_SELECTION.md),
[equations and remedy derivations](COULOMB_LOCALIZATION_FORMULATION.md).
All native commands have ended. Full runs/checkpoints remain separately in
`build-particle-free/coulomb-localization`; the
[compact review archive](../../../benchmarks/reconstructed_fault/particle_free/evidence/coulomb_localization.zip)
contains PNG/CSV/JSON, sources, input/command records and checksums.

## Mixed mechanical conclusion

Physical stress is `sigma=tau-(10000+p')Id`, not T. Yield resistance uses
`pn=10000+p'-nu.tau.nu`; flow is tangential, with both conjugate systems and
changing directions. The native derivative includes all those changes.
For incompressible wave normal n and transverse amplitude a, analyze

```
M(n)=[ Q(n)  b(n) ; n^T  0 ],
Q(n)a=[(partial tau/partial epsilon):D3 sym(a*n)]n,
b(n)=[partial tau/partial p-Id]n,
Lambda=a.Qa-(a.b)(n.Qa)/(n.b).
```

This pressure elimination is essential. At a selected phi=.8 single-system
state after the loading-direction change, the actual dt=.0125 s tangent has
`min Lambda/G=-.00195452998`, whereas its fixed-pressure minimum is
`+.00467886991`. There are two real full-symbol rank-loss normals, 2.89038449
and 2.93703921 radians. Independently reconstructed 3x3 null vectors have
defects below 3e-16. The pressure-normal block remains nonsingular; this is
not a pressure-reference artifact or just a negative eigenvalue of the
symmetric part. Wave-normal and band-axis angles are distinct from the
constitutive slip planes and from phase-front geometry.

An independent continuous-rate single-system derivation gives loss at
`h>100.49813740695`, or **phi>.76761127929** for m=4, mu=.1. At phi=.8,
`h=144`, its minimum is `-.003014999912 G`, while fixed-pressure stiffness
remains positive. The closed formula agrees with independent tensor
contractions to 5.57e-16 over the sampled single-system states. The audit
covers phi=0,.2,.6,.75,.8,.9,.95,.99, sticking, both conjugate single systems,
and simultaneous slip. The double-system sampled minimum is positive at
.8, but negative at .9, .95 and .99 (respectively -.001020, -.002140 and
-.002463 G). These are sampled state-dependent results, not a universal
double-slip threshold. Phi=1 is the infinite-compliance limit. The intact
mechanical tangent is elastic. All 96 pressure-independent constant-resistance
controls remain positive; their smallest algorithmic coefficient is 1.27e-5 G.

![Mixed symbol, phase range and timestep diagnostics](../../../benchmarks/reconstructed_fault/particle_free/evidence/coulomb_localization/mechanical-symbol.png)

The top-right panel compares the same actual derivative with and without
pressure response. Crosses are rank-loss normals. Its step advances the point
path from .1 to .1125 s. The lower-left probes start from that step's returned
state and use one immutable incoming state, rather than different evolved trajectories:
at dt=.0125/.00625/1e-8 s its mixed minimum is
-.0007167/-.0018464/-.0030150 G. Decreasing dt approaches the nonelliptic
continuous-rate limit. The top-left two-system curve is an envelope of
sampled states. Lower-right free-phase elimination is a diagnostic, not an
admissible phase wave or a new stability certificate.

## What the retained regularizing terms control

At the selected state the physical obstacle force is `Kf-Y=7.95047 Pa>0`.
An admissible critical perturbation can keep phase fixed. The AT1 gradient
term controls phase gradients, not gradients of A/K or mechanical slip with
delta phi=0. Even formally releasing phase gives feedback O((k ell)^-2);
the coupled diagnostic approaches the negative fixed-phase symbol to
3.93e-14 G at k ell=1e5. Thus the phase length cannot repair this mechanical
short-wave defect. No old symmetric reduced-Hessian certificate is reused.

Maxwell time eta/G=100 s supplies local relaxation, with positive
`beta=(1+dt/100)^-1`. Its continuous relaxation terms cancel in a jump of
rates. It is neither a slip-gradient penalty nor direct slip-rate viscosity.
The loss occurs in compressive, strictly active states with admissible small
mechanical perturbations and fixed phase. This is a continuum lack of
ellipticity, separate from a Newton initialization problem or timestep error.
The quasistatic symbol does not establish a dynamic growth speed.

## Bounded native refinement and timestep check

Fresh .25 m square blocks use the same nonuniform, divergence-free boundary
loading: engineering shear rate .02/s through .1 s, then deviatoric normal
rate .02/s; the retained sinusoidal perturbation has amplitude .006/s.
Stop at .125 s. G=1e6 Pa, eta=1e8 Pa s, Gc=1 J/m², ell=.046875 m,
Kf=8 Pa, m=4, strength-drop input=2000 Pa, mu=.1, c_r=0, p0=10000 Pa.
Uniform initial phase .8 (and .2 controls) is represented exactly on every
mesh; all histories start at zero. Unlimited DGQ2, native Q2/Q1 mechanics,
Gauss6/source3, four ranks and stationary diagnostic particles are unchanged.
There is no geometry change, transport, remapping or phase-evolution run.

At common t=.125 s and **dt=.00625 s from fresh initialization**:

| Mesh | Cell size (m) | Mean / peak equivalent slip rate (1/s) | Participation width (m) | Scaled mixed condition number | Negative-symbol area |
|---|---:|---:|---:|---:|---:|
| 4×4 | .0625 | .028187 / .081071 | .148887 | 13.77 | 60.47% |
| 8×8 | .03125 | .028071 / .079388 | .144964 | 31.04 | 58.58% |
| 16×16 | .015625 | .028070 / .087906 | .141767 | 73.17 | 59.28% |

Equivalent slip rate is `sqrt(2 W:W)` using the full tensor metric: a bulk
inelastic shear-rate measure (1/s), not surface slip velocity (m/s).
Participation width is `(integral w)^2/[.25 m integral w²]`, with physical
Gauss weights; no threshold or smoothing enters this width. A fluctuation
width based on `sqrt(2(W-mean W):(W-mean W))` is .19579/.19570/.19639 m.
The broad pattern is retained: **the small width change is not evidence
of a one-cell band shrinking to zero**, and these three short runs do not
qualify nonlinear localization convergence. They do sample a negative actual
algorithmic symbol over substantial physical area. Area fractions use JxW,
not counts of quadrature points. Native angular scans use 721 normals;
the point rank-loss check independently refines roots.

Conditioning uses the actual nonsymmetric constrained mixed matrix, with
elastic H1 velocity/pressure L2 Riesz scaling on the same mesh, including
the physical mean-pressure constraint once. Minimum singular values fall
.103549 -> .047967 -> .020495; maximum values stay 1.43–1.50. Independent
FE integration reproduces the native matrix to 1.06e-15 in the reported
scaled Frobenius measure (predeclared bound 1e-10). Low-phase controls have
positive local mixed symbols and much milder scaled condition numbers
7.38 coarse / 8.87 fine; their timestep histories differ after .1 s as
explicitly noted below. No artificial constrained-row diagonals are included.

![Actual native slip and mechanical conditioning](../../../benchmarks/reconstructed_fault/particle_free/evidence/coulomb_localization/native-refinement.png)

The maps show actual quadrature values with common physical axes and color
scale. Profiles average these values with the same Gaussian in physical
wave-normal coordinate, sigma=ell/2, divided by its integrated mass.
Blank tails with kernel mass below 1e-12 m² are omitted. This averaging is
only a plotting diagnostic; histories/tangents are never interpolated into
a different constitutive state. Widths are unsmoothed. Phase increments are
exactly zero. High-phase minimum weak obstacle forces are 7.237/6.743/5.583 Pa,
so that part of the phase critical cone allows only zero phase direction.
The .2 controls develop negative phase force while held: no free-phase KKT
pass is claimed for them.

The focused dt=.0125/.00625 comparison gives full-tensor W relative L²
differences of 9.60% on 4×4 and 11.92% on 8×8 at .125 s. This is not temporal
convergence. Both timestep choices retain nonelliptic states; smaller dt
does not supply missing constitutive regularization.

Both 16×16 **full-step** cases stop before step 9 publication, at the .1 s
accepted checkpoint. Their zero-interior initial Newton lift at the changed
boundary rate violates closed physical elastic-trial compression. At phi=.8:
CellId `0_0:`, q=18, reference=(.03376524,.61930959),
physical=(.000527582,.009676712) m, total pressure=8940.968 Pa.
At phi=.2: same cell, q=12, reference=(.03376524,.38069041),
physical=(.000527582,.005948288) m, pressure=9462.398 Pa.
This arises before a Newton update, including in the elliptic low-phase
control; boundary-lift gradients scale with dt/cell size. It is **not
classified as a localization failure**. Original errors and checkpoints
remain. Two explicit restart/cutback controls from those unchanged same-parent
checkpoints, clamping only pending dt to .00625 s, both reach .125 s and
publication 12. No automatic native cutback or improved contact initialization
is claimed, and no contact guard/tolerance is changed. These restarted paths
are distinct from the fresh half-step spatial comparison above.

Across 114 new accepted intervals, source error <=8.50e-15 (bound 1e-12),
mechanical residual <=9.524e-10 (bound 1e-9), absolute BE ledger defect
<=5.32e-17 J/m. Minimum actual trial/returned normal pressures are
6130.52/6683.86 Pa. Dissipation, unique ownership and single publication
checks remain active. Particle IDs/coordinate fingerprints are fixed with
effective freeze=1. No phase Hessian is run on the new law. A retained
low-phase trajectory reproduces 33 old native state/mean/phase exports
bitwise. All 60 protected source/artifact hashes, including BP3 and previous
archives, remain unchanged. The BE ledger is not physical-energy convergence.

Sequential four-rank command wall times for high-phase half-step runs are
3.12/6.08/22.56 s. The accepted full-step coarse/intermediate runs take
2.52/4.17 s. Failed fine full-step attempts take 10.84 s high phase and
9.23 s low phase; explicit cutbacks take 6.73/7.93 s. These include dense
diagnostics, checkpointing and startup, not production performance estimates.
`execution-summary.csv` separates existing assembly/linear/history/output
timers and rank peak memory. Independent full-matrix conditioning takes
61.4 s total; native history/symbol postprocessing 88.1 s. Native launch
timeouts are 240 s. A plotting script initially tried to parse deliberately
blank low-mass profile tails; it now omits them. Original failed log remains.

## Remedy and next decision

Recommend reviewing a **physical spatial slip regularization** before new
native phase-active work. One explicit option adds
`G ell_s² |grad K|²` to storage, microstress `2G ell_s² grad K`, and backstress
`X=-2G ell_s² Laplacian K` in `tau-Q-X`. It preserves homogeneous point
response and zero dilation while controlling high-k slip modes. Its physical
length, microboundary conditions, extra storage/work and transfer rules need
scientific selection; ell_s must not be set to the mesh spacing. It is not
yet implemented or qualified. Local slip viscosity would be another new
rate law, not the existing Maxwell term or a guaranteed quasistatic band width.

A consistently derived dilatant alternative requires normal plastic flow,
new deviatoric/volumetric memory, modified continuity/pressure tangent and
compatible boundary volume change. Its plastic dissipation is
`sum[c_r+(mu-d)pn]w`. Associated d=mu removes the specific non-associated
route with positive hardening but, at c_r=0, changes frictional heating to
zero and does not alone prescribe a finite band width as hardening vanishes.
Partial dilation still needs a mixed-symbol audit. Adding dilation only to
an orientation selector or tangent would be inconsistent. The present
zero-net-volume all-displacement block cannot accept net positive dilation.
[The derivation](COULOMB_LOCALIZATION_FORMULATION.md) separates these choices.

**Completed:** mixed continuum/algorithmic audit, bounded native refinement,
timestep and same-parent cutback controls, compact evidence and preservation
checks. **Not begun:** phase-active Coulomb qualification, new regularization,
larger specimens or production changes. A reduced phase operator cannot
certify a mechanically nonelliptic law. Retain the fixed-phase guard, earlier
P4/P5c spatial/temporal/boundary and branch failures, outstanding physical
energy limits, and `strict_reference_match_passed=false`.

## Files changed for this task

- `source/simulator/solver/bulk_cohesive_phase.cc`: opt-in read-only exports of
  actual converged/failing mixed matrices and native constitutive derivatives.
- `benchmarks/.../bulk_cohesive/parallel.cc`: test-only phase initialization,
  point layout export and same-mode pending-dt clamp on restart.
- `benchmarks/.../bulk_cohesive/CMakeLists.txt`: standalone audit target;
  `coulomb_localization.cc` and `coulomb_localization_{analysis,native,verify,gates,plots}.py`:
  bounded point audit, independent checks, native analysis and figures.
- Selection/formulation/this report; `STATUS.md`, `SOURCE_AUDIT.md` and
  `CURRENT_STATUS.md` prepend this decision while preserving prior text.

Physical constitutive bodies, native checkpoint layout, production defaults,
all tolerances and BP3 are unchanged. `provenance.json`, `changed-files.json`,
`entry-delta.patch`, logged commands and `manifest.json` in the evidence
package identify exact input/source hashes and changed files relative to entry,
not the much older git HEAD alone.
