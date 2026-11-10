> **Subsequent free-phase pair completed:** [response and decision](SYNTHETIC_FREE_EVOLUTION.md). The preflight and selection below remain original evidence. The next priority is time integration; no further growth campaign is selected.

# Selected synthetic propagation benchmark — 9 November 2026

This is a **synthetic verification specimen**, not a calibration to a natural
fault. The selected input is
[`bulk_tip/synthetic_propagation.prm`](../../../benchmarks/reconstructed_fault/particle_free/bulk_tip/synthetic_propagation.prm).
It uses the existing native DG Bernstein bulk-tip formulation, direct physical
stress/cohesive history, BE reference update and section RSF closure. The mature
fault production model and historical surface-cohesive formulation are separate.
No compact perturbation is added.

Entry is `5f1fc6f3c` on `pf-rsf-particle-free`, in the current worktree. The prior
specimen assessment and its nine modified documentation files are preserved.
The core executable remains `5d8f9ab1…c18e3e`; only benchmark observation/checker
code is rebuilt. No physical solver, quadrature, acceptance tolerance or restart
meaning changes.

## Joint selection

| Quantity | Selected value |
|---|---:|
| Shear modulus G; viscosity eta | 1e6 Pa; 1e8 Pa s |
| Strength calibration c; toughness Gc | 1000 Pa; 2/3 J/m² |
| Phase length ell; curvature parameter | .125 m; 1 |
| Fracture potential K=Gc/[(8/3)ell]; compliance parameter m=2GK/c² | 2 Pa; 4 |
| Box; center | 8×4 m; (4,2) m |
| Seed half-length; central plateau half-length | 2 m; 1.5 m |
| Seed peak; input normal half-support | .95; .25 m |
| Prescribed frictional confinement pN; reference friction mu0 | 200 Pa; .6 |
| RSF a, b; V0; Dc; initial age | .015, .02; .001 m/s; .0005 m; .5 s |
| Radiation coefficient | 1000 Pa s/m |
| Initial mesh | 64×32 cells, h=.125 m; native DGQ2 Bernstein phase |
| Fixed-phase preflight | BE dt=.01 s, eight intervals to .08 s, two MPI ranks |

The initial phase is

\[
 \phi_0=.95\left[1-\left(\frac{\max(0,|x-4|-1.5)}{.5}\right)^2\right]_+^2
                \left[1-\frac{|y-2|}{.25}\right]_+^2.
\]

The existing zero-clock constrained fracture minimization equilibrates this
obstacle; that discrete seed is then held fixed in the preflight. Initial tau
and A are zero; age .5 s and slip zero exist on every section, including ahead
of both ends. No age is reset when a damaged region is later observed.

All four walls prescribe v=(.01(y−2),0) m/s. Side velocities are the same affine
field as top/bottom; this is compatible incompressible uniform shear, not the
old pure compact return flow. Nominal intact Maxwell shear is
1e6[1−exp(−t/100)] Pa: about 200–800 Pa over .02–.08 s. The nominal strain is
at most .0008. This is the selected loading range, not a promised growth onset.
No increase beyond .08 s is selected by this task.

### Fracture and cohesive scales

The unchanged response is h=m phi(1+phi)/(1−phi)² and
Y=h' q²/(2G). The intact strength condition in a wholly intact section is
Y/K=(tau_xy/c)². The global damage criterion still includes SIPG fracture forces
and the Bernstein obstacle; c is not an extra Coulomb yield resistance.

The parser also retains g''(0)=2m(m−3)>0 for curvature one. Choosing m=4 leaves
g''(0)=8, rather than weakening this guard. Its importance was missed in the
first proposed Gc=.12, ell=.25 combination; see the preserved startup failures
below. The final configuration is selected with this requirement included.

| Dimensionless indicator | Value |
|---|---:|
| a/ell; side ligament/ell; center-to-normal-wall/ell | 16; 16; 16 |
| G Gc/c² | .6667 m; a divided by this length = 3 |
| Reference positive resistance at V0 and initial age | 121 Pa, including 1 Pa radiation; f/c=.121 |
| G Gc/(c−121)² | .8628 m; a divided by this illustrative length = 2.318 |
| c/G; Gc/(c Dc); Dc/(c a/G) | .001; 1.333; .25 |
| Maxwell duration tG/eta; dt/initial age | .0008; .02 |

The central AT1 transverse-profile half-width is about .244 m. The 2 m normal
wall distance and side ligaments provide a larger separation than the rejected
specimen, while retaining a modest finite box. This is not an infinite-body
crack approximation. A fully developed extension by one .125 m cell at each
end costs .1667 J/m; the homogeneous whole-box elastic reference at tau=c is
16 J/m. These scales offer energy headroom without increasing loading through
remote intact activation; they are not a propagation theorem.

### RSF and cohesive stiffness together

The positive-rate spring-slider RSF marker is
k_RSF=pN(b−a)/Dc=2000 Pa/m. Bulk markers G/a and G/H are 500000 and
250000 Pa/m. Include the cohesive response using the central input profile,
I_h=integral h(phi) dn=33.06056 m. In the instantaneous elastic, uniform-section,
fixed-phase approximation, let delta_f=integral e dn and delta be imposed total
shear displacement. Then

\[
 q=G\delta_f/I_h,\qquad
 \tau=G(\delta-\delta_f)/H,\qquad
 f=\tau-q=G\delta/H-(G/H+G/I_h)\delta_f.
\]

Thus the restoring feedback against fault slip is G/H+G/I_h=280248 Pa/m,
about **140 k_RSF**. In contrast, G/(H+I_h)=26982.9 Pa/m is the external
traction/displacement stiffness at fixed friction; it is **not** the spring-slider
feedback and must not be substituted for it. These are section scale estimates,
not eigenvalues or a stability proof for the nonlocal, evolving 2-D phase problem.
The chosen Dc deliberately permits measurable state evolution while keeping this
initial control away from targeted RSF instability. Theta0=Dc/V0=.5 s preserves
reference initial friction. At the preflight endpoint, max slip is already 1.805 Dc.

The existing BE endpoint monotonicity bound requires only 1.436 Pa s/m radiation
at this dt; the retained 1000 Pa s/m satisfies it. No extra damping is introduced.
The smaller prescribed confinement is an explicit synthetic physical input,
not a replacement for solved pressure or a numerical stabilization. Dynamic
normal stresses are reported separately; opening/contact and current-pressure
friction are not qualified by this fixed-direction control.

## Mesh, cost and reproduction

The preflight has 2048 cells, 18432 phase coefficients and 37347 coupled
velocity/pressure/phase unknowns (101028 total simulator DoFs including static
histories and temperature). h=ell is an economical DGQ2 **screen**, not a spatial
accuracy claim. A subsequent qualification needs h=ell/2 (8192 cells, 73728
phase coefficients), retaining the same physical seed and ell. That mesh is not
run here. At fixed mesh, reducing dt from .01 to .0025 multiplies interval count
by four; active phase growth and finer meshes can cost more per interval.

Predeclared cost was 2–6 minutes for the one eight-interval preflight, with a
600 s simulation cap. No long propagation run or additional loading case is
selected. The preflight's coarse BE timestep is for mechanical screening;
fracture-work accuracy must be checked before interpreting a growth trajectory.

```sh
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py synthetic-preflight03 620 \
  python3 benchmarks/reconstructed_fault/particle_free/bulk_tip/run.py synthetic-preflight03 \
  --fixture benchmarks/reconstructed_fault/particle_free/bulk_tip/synthetic_propagation.prm \
  --checker build-particle-free/bulk-tip/synthetic-checks/libbulk_tip_checks.release.so \
  --fixed-phase --phase-balance --time-update 'backward Euler' \
  --dt .01 --end .08 --ranks 2 --seconds 600
```

Use a new label to reproduce: the runner refuses to overwrite old runs. The
input file retains `Evolve phase field = true` to use the integrated bulk-tip
path; **the preflight command's `--fixed-phase` hook sets upper=accepted lower**.
Running the input without this hook would be a growth run and is not this test.
The new `--phase-balance` flag exports the actual fracture/driving rows without
activating the older fixture's hard-coded region stops. It does not alter equations.

## Preflight findings

**Retain this specified synthetic specimen for the next bounded numerical growth
verification.** It passes the fixed-phase mechanical preflight with substantially
larger tip driving than at the walls. This selects a benchmark, not a claim of
propagation accuracy or an invitation to run a long trajectory.

`synthetic-preflight03` completes all eight intervals: **79.99 s** inside the
runner, 80.13 s including its wrapper (76.5 s in the simulator timer). The two
startup failures add 8.19 s; only this run advances physical time. The original
80-iteration and mechanics/KKT/ledger gates remain. Maximum mechanical residual
is 8.02e-10, ledger defect 8.98e-15 J/m. Phase coefficients and exported fields
are exactly unchanged; the frozen phase KKT is zero by construction and is
**not** a test of the unfrozen phase equilibrium.

| Time, s | Tip / boundary max abs(shear), Pa | Tip / boundary max Y/K | Tip / ahead / boundary max native release |
|---|---:|---:|---:|
| .04 | 708.08 / 465.80 | .5014 / .2170 | −.04085 / −.50741 / −.78536 |
| .05 | 925.60 / 590.98 | .8567 / .3493 | +.04318 / −.20910 / −.65484 |
| .06 | 1143.14 / 716.15 | 1.3495 / .5129 | +.41480 / +.16740 / −.49346 |
| .08 | 1578.09 / 966.44 | 2.7511 / .9340 | +1.42114 / +1.15454 / −.07823 |

Native release means (assembled driving load − full SIPG fracture gradient)/
(K times the coefficient's diagonal mass), evaluated on coefficients below one.
A positive value permits an infinitesimal increasing-coefficient direction when
the artificial fixed-phase upper bound is removed. It includes the actual face
terms; it is not inferred from Y/K. Tip masks are within .5 m of x=2 or6 and
|y−2|≤.5; ahead is |x−4|≥2, |y−2|≤.5, using cell centers for native rows.
Remote is |x−4|>3; boundary is distance≤.25 to any wall. Masks overlap, and the
remote/boundary maxima coincide in this run.

Tip coefficients first have positive release between .04 and .05 s, even though
max local tip Y/K is then below one. Rows in cells ahead of the nominal seed
first become favorable between .05 and .06 s. At .06 s, the maximum ahead row
is at cell center (6.0625,1.9375), with accepted coefficient zero; all coefficients of that cell are zero. This
is a favorable direction in an intact cell, but still does not establish actual
contour advance because phase was held fixed. Central profile broadening is also favorable, from .02–.03 s onward;
the specimen is not a pure translating crack-tip solution. Remote/boundary rows
remain unfavorable through .08 s. The next bounded growth check should therefore
focus on **the .04–.06 s window**, retaining a meaningful remote margin, rather
than extending this screen beyond .08 s.

The endpoint tip stress concentration is 1.97 times nominal homogeneous shear,
or 1.63 times the boundary maximum. Left/right maxima agree to about 3e-10 Pa.
Maximum dynamic normal stress is about 2148 Pa at the tips and 422 Pa at the
boundary, in the solved pressure gauge. These are distinct from the prescribed
200 Pa frictional confinement. The chosen closure does not test tensile contact.

Signed deformation remains predominantly coherent but is **not everywhere
same-sign**. Endpoint spatial integrals of positive and negative fault-only w
are +.04368795 and −.000232825 m²/s. Integral(ds abs(V)) is .04345513 m²/s,
versus integral(abs(w))=.04392078 m²/s: **1.0602% cancellation**. Both tip masks
have positive w only; reversals occur mainly along the seeded core. The sampled
sign test finds 134 of 384 sections with both signs, but no separated active
bands at the documented threshold max(1e-10,1e-6 max(abs(w))). Opposite signs
within one connected band are still reported. No clipping or relocalization.
Maximum section rate is .013674 m/s, accumulated slip .00090235 m=1.805 Dc,
and endpoint age ranges from .13267 to .58 s. Maximum resistance is 136.21 Pa.

Accumulated positive/negative local friction-plus-radiation work is
+.39279978 / −.00211165 J/m; net .39068813 J/m. Local/section power agrees within
1.3e-16 J/m per step. The full physical work remainder is **1.13034370 J/m**
(signed and accumulated absolute agree), split into BE stress loss 1.10271433
and traction loss .02762821 J/m, up to solve roundoff. Fracture remains
2.15735162 J/m, with a −4.9e-15 J/m summation difference. Thus the coarse
preflight is suitable for mechanics/selection but **not fracture-work accuracy**.
A subsequent growth comparison still needs timestep and spatial error measured
against its actual added fracture; dt=.0025 is a candidate, not a qualified value.

![Fixed-phase preflight](../../../benchmarks/reconstructed_fault/particle_free/evidence/synthetic_propagation/preflight.png)

The measured fixed-phase interval cost is roughly 10 s including setup and
exports. At the same mesh, a 24-interval run to .06 at dt=.0025 has a roughly
four-minute fixed-phase cost estimate; an active phase solve can cost substantially
more. Doubling mesh resolution gives four times as many cells in 2-D, with no
claim that runtime scales linearly. Neither additional run is launched here.

[`analyze_synthetic_preflight.py`](../../../benchmarks/reconstructed_fault/particle_free/bulk_tip/analyze_synthetic_preflight.py)
reproduces the JSON reduction and PNG/PDF from the retained CSV exports. It
checks fixed phase, the native row identity and the actual section power adjoint.
The [analysis](../../../benchmarks/reconstructed_fault/particle_free/evidence/synthetic_propagation/synthetic-preflight03-analysis.json)
and [closeout](../../../benchmarks/reconstructed_fault/particle_free/evidence/synthetic_propagation/closeout.json)
retain commands, identities, per-region driving/stress/rate data and cost.
The 37k-unknown coupled preflight and the independently reported fracture rows
are the enabled configuration; no new general propagation claim follows.

## Preserved startup failures and boundaries

`synthetic-preflight01` stops before initialization because m=.36 violates the
retained degradation-curvature guard. Its input and proposed selection remain
preserved; the final calibration above replaces it without changing that guard.
`synthetic-preflight02` initializes the selected specimen but stops at time zero
in the benchmark's signed-RSF derivative check. The checker had fixed age10,
Dc=.01 in its independent aging reference, and a fixed zero-rate finite-
difference probe. The selected smaller Dc exposes that probe's nonlinear range.
The checker now uses configured age, dt, pressure, radiation and Dc, and scales
the zero-rate probe to a 1e-6 Pa traction increment. Original derivative error
tolerances are unchanged. These two startup attempts contain no physical step;
`synthetic-preflight03` is the one physical fixed-phase preflight.

Prior failed evidence, protected binaries and checkpoints remain. Inherited
losses are still 39/45 pilot frames, early temporary logs and the unavailable old
`bulk_continuation.zip`. The original large-toughness specimen remains rejected;
its historical assessment is not rewritten as evidence for this new calibration.
