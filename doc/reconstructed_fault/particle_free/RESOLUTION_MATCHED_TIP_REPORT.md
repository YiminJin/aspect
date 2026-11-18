# Resolution-matched peak-0.60 tip screen

This selected experiment follows `CODEX_RESOLUTION_MATCHED_TIP_TEST.txt`.
**Closed: retain this pair as a mechanical screen, but reject it as a tip-led
onset specimen under the screened uniform shear. No growth run was launched.**
Both meshes favor damage in the existing central band before the tips. Slip
agrees closely; native phase-row driving is still sensitive to its test space.
The peak-.95 trajectories and accepted step-89 checkpoint remain separate and
unchanged; their proposed seven-step continuation is deferred. This is the
synthetic fixed-direction bulk section-RSF control, not mature-fault production
or a general turning/branching model.

## Seed, mechanical width and cost selection

The physical input is

\[
 \phi_0(x,y)=.60\left[1-\left(\frac{\max(0,|x-4|-1.5)}{.5}\right)^2\right]_+^2
                 \left[1-\frac{|y-2|}{.25}\right]_+^2.
\]

The existing zero-clock constrained fracture initialization equilibrates this
obstacle independently on each mesh. Both start from zero stress/A/slip and age
.5 s on every section. The 8×4 m body, ell=.125 m, G=1e6 Pa, eta=1e8 Pa s,
Gc=2/3 J/m², strength 1000 Pa, Dc=.0005 m, prescribed frictional confinement
200 Pa and all RSF parameters remain those of the selected synthetic specimen.
Every wall prescribes v=(.01(y−2),0). Both screens use midpoint dt=.01 s from
t=0, with the same loading and age history, and the existing fixed-phase hook.
Phase bounds and physical solver/ledger gates are unchanged.

For central input peak p and normal half-support b=.25 m, define the **excess**
compliance c(phi)=4phi(1+phi)/(1−phi)². At p=.60, c(p)=24; the half-height
crossing is phi=.5. Thus W=2b[1−sqrt(.5/.6)]=.0435645354 m and integral c dn
=1.6722283975 m. The intact inverse-stiffness background is excluded. This
physical peak change alters compliance as well as resolution requirements;
only the same-peak mesh pair measures resolution sensitivity.

| Quantity | Less expensive mesh | More resolved mesh |
|---|---:|---:|
| Box repetitions | 128×128 | 128×256 |
| Along-fault hx | .0625 m | .0625 m |
| Normal hy | .03125 m | .015625 m |
| Cells | 16,384 | 32,768 |
| Simulator DoFs | 804,612 | 1,608,324 |
| Coupled velocity/pressure/phase unknowns | 296,195 | 591,747 |
| Input W/hy | 1.3941 | 2.7881 |

Both generated meshes match these counts. The coarse screen completes eight
intervals to .08 s in 1163.44 s process time (1169.71 s including its wrapper),
using two ranks. Its first two intervals cost 67.35 and 160.78 s. On that measured
basis, the fine screen selects the identical history through .08 s, four MPI
ranks and a 2400 s process cap. Compression and numerical reduction wait until
all four ranks exit; aggregate numerical workers never exceed four. A 4.1 GB raw
output/checkpoint allowance was checked against 5.50 GB available before launch.
The fine run accepts six intervals through .06 s and reaches its cap during
trial seven: process 2400.37 s, wrapper including compression 2438.13 s, exit124.
This is a time limit, not failed physical acceptance. Its first interval costs
170.70 s and subsequent accepted intervals 376.80–400.21 s, versus coarse
153.30–175.29 s at the same steps. Total process time is **59.40 minutes**.
Fine accepted .07/.08 outputs and a fine checkpoint are absent. The coarse
checkpoints remain; no restart check was needed or claimed for this unchanged
solver. Runtime does not scale simply with cells/ranks; no tuning was applied.

## Observation contract

The existing native phase-balance export supplies stage fracture gradient F_i,
mechanical driving load D_i and Bernstein mass m_i. The reported signed surplus
is S_i=(D_i−F_i)/(K m_i), K=2 Pa, on coefficients B_i<1. **Positive S_i** means
an admissible positive coefficient variation lowers the unfrozen restricted
functional. It includes SIPG faces, not just local Y/K. The artificial fixed-phase
upper bound is removed only in this diagnostic interpretation; a held-phase
reaction is not a demonstrated growing solution. Midpoint stage times are shown
separately from endpoint age/slip and accumulated work.

Regions use actual physical cell centers for rows and quadrature positions for
point observations: old central band |x−4|<1.5, |y−2|<.25; old tip taper
1.5≤|x−4|<2, |y−2|<.25; immediately ahead 2≤|x−4|<2.5, |y−2|<.25; remote
|x−4|>3 or |y−2|>1. Additional central flanks (.25≤|y−2|<.5), outer taper
(1.875≤|x−4|<2), and boundary masks distinguish central broadening and inner
taper shoulders from the physical seed ends. Extra masks overlap explicitly.
Region borders align with cells on both meshes; no square-cell indexing is used.

The reducer evaluates the **native Bernstein polynomials** with separate hx,hy,
checks them against native phase samples, and measures central widths on both
x=4 traces. Compliance integrals use per-cell 24- and 48-point quadrature as an
independent diagnostic, without changing solver quadrature. Mechanical Q6 tensor
quadrature, JxW, Bernstein row masses, section ds and integrated normal measure
are checked against the actual generated geometry. The existing SIPG operator
uses mapped face/cell Jacobians (normal spacing) and retains its Q5 rules.
The power adjoint and integral of fault-only signed w are checked against section
V; ordinary elastic and viscous bulk deformation are not included in slip.

The fine forces are also compared on **identical admissible coarse variations**.
Exact normal Q2 Bernstein subdivision T maps one coarse polynomial to its two
fine cells; T has nonnegative entries, so increasing a coarse coefficient gives
an admissible increasing fine variation. The comparison uses T^T(D_f−F_f) and
coarse K m, preserving directional work. A mixed-sign-force/positive-variation
adjoint check has defect 2.22e-15. Thus native row maxima with different supports
are distinguished from mechanical changes measured on common variations.

On **both independently equilibrated seeds**, W=.0435645354148 m and integral
c dn=1.6722283974803 m on both central traces. W/hy is 1.39406513327 / 2.78813026654.
Exact normal subdivision gives maximum seed coefficient difference 2.22e-16
and relative polynomial L2 difference 4.67e-17. Diagnostic quadrature convergence
is ≤2.3e-16 m; native reconstruction defect ≤1.17e-15. Q6 coordinate error is
1.39e-17 m, JxW error ≤4.07e-20 m², section normal-measure error ≤1.14e-13 m,
and the largest power-adjoint defect is 3.82e-14 J/(m s). Held coefficients and
widths stay exactly unchanged. These checks establish the rectangular measures,
not growth convergence.

## Results and growth decision

At matching midpoint stages, regional **native** maximum S values are below.
Each entry is coarse / fine; endpoint time is stage time + .005 s. For orientation,
the homogeneous no-fault Maxwell shear is
tau_bg=1e6[1−exp(−t_stage/100)] Pa, approximately 350, 450 and 550 Pa in these rows;
it is a loading reference, not the measured fault stress.

| Stage (s) | Central band | Old taper | Outer taper | Intact axial ahead | Remote |
|---|---:|---:|---:|---:|---:|
| .035 | −.0881 / −.0098 | −.3701 / −.3416 | −.3756 / −.3416 | −.7223 / −.7206 | −.8527 / −.8527 |
| .045 | .2466 / .4087 | −.1977 / −.1577 | −.1977 / −.1577 | −.5269 / −.5244 | −.7514 / −.7514 |
| .055 | .7374 / 1.0227 | .0652 / .1380 | .0652 / .1246 | −.2689 / −.2652 | −.6225 / −.6225 |

Both grids bracket first central activation between stages .035 and .045 s;
outer-taper activation follows between .045 and .055 s. These are sampled
held-phase brackets, not continuous-time growth onset. At .055, representative
fine maxima lie in cells centered at (4.03125,1.9921875) m (central),
(2.15625,1.9921875) m (old taper), (2.09375,2.0078125) m (outer taper), and
(1.96875,1.9921875) m (ahead); mirror-symmetric maxima need not choose the same
side after MPI roundoff. These are cell centers, not pointwise maxima.

On **identical coarse admissible variations** at stage .055:

| Region | Coarse S maximum | Fine S maximum on coarse variations | Largest absolute row difference |
|---|---:|---:|---:|
| Central band | .737422 | .702465 | .038323 |
| Old taper | .065218 | .066368 | .014909 |
| Outer taper | .065218 | .066368 | .001952 |
| Axial ahead | −.268851 | −.268816 | .000993 |
| Remote | −.622543 | −.622513 | .0000294 |

The central maximum differs by 4.98% relative to the fine common-variation
value; native central maxima differ much more (.7374 versus 1.0227), because
finer admissible variations resolve a more concentrated driving field. Small
slip error does not qualify the free phase solution. Both comparisons preserve
central-first ordering, with central surplus about ten times the outer-taper
surplus on common variations. Remote and central-flank material remain below
activation. The coarse-only stages .065/.075 show ahead surplus .0419/.4136,
but central surplus is already 1.3376/2.0855; these later observations have **no
fine partner** and do not establish a resolved extension window.

| Endpoint (s) | Stage integral V ds, coarse / fine (m²/s) | Cumulative integral slip ds, coarse / fine (m²) |
|---|---:|---:|
| .02 | .01236190 / .01236569 | .000123619 / .000123657 |
| .04 | .02093843 / .02094792 | .000696676 / .000696948 |
| .06 | .02442868 / .02443954 | .001251078 / .001251569 |

At .06, integrated slip differs .03921%, section-slip weighted L2 .04085%,
section-rate L2 .04624%, age L2 .01089%, and section-force L2 .00258% (fine
denominators). Over non-negligible shared stages the largest integrated-rate
difference is .04534% and largest rate L2 .04717%. All appreciable fault
deformation has the same sign and a single active band in each section. The
first, nearly motionless stage has positive/negative integrated w of about
±3.8e-11 m²/s, net 2.9e-13 m²/s: its cancellation ratio is .996, and the
diagnostic detects opposite signs in 78/71 and multiple bands in 107/104
sections (coarse/fine). Those small residual rates are retained, not clipped.
The band detector uses |w|>max(1e-10 s⁻¹,1e-6 max_section|w|); from step two
onward both counts and signed cancellation are zero.

**Decision:** do not start the conditional growth run. The screen favors further
damage within the pre-existing central band over localized tip-led onset. It
does not demonstrate geometric broadening: intact central flanks remain
subcritical and no phase was released. Under uniform shear the compliant central
band accumulates cohesive deformation and driving; its fracture derivative is
already only F/(K mass)=.4 on the winning interior rows of the quadratic seed.
The outer taper develops stress concentration, but not enough net driving to
overtake the center. At the last shared stage the fine pointwise max Y/K is
1.774 in the center, .819 in the outer taper and .733 ahead; the actual decision
uses the full discrete derivative above, including SIPG. This is consistent
with Y=G h' A:A/h² and the initialized fracture gradient, rather than a contour
threshold or an integrated-slip proxy for growth.

Lowering the peak materially changes the specimen: c(.60)=24 versus c(.95)=2964,
and the central input compliance integral is 1.6722 m versus 33.0606 m. The new
same-peak pair supports the narrow-compliance explanation for poor mechanical
resolution, but does not establish that it alone caused the earlier evolving
phase sensitivity. There is still phase-space sensitivity even at peak .60.

Physical-work error Wext−deltaE−Db−Dc−Df through .06 is 3.506e-7 / 3.514e-7 J/m;
signed sums equal accumulated absolute sums here (all increments positive).
Coarse through .08 gives 1.952e-6 J/m. Added fracture and phase increments are
exactly zero because phase is held; division by added fracture is undefined.
The small ledger error and zero constrained KKT do not establish temporal
accuracy. Stage slip-rate oscillations remain visible at dt=.01; no timestep
refinement or free-growth accuracy claim is made.

The next justified work, if tip-led propagation remains the objective, is a
bounded constitutive/specimen scaling assessment aimed at reducing central
cohesive driving relative to the tip driving, followed by a held-phase screen
only if a concrete change is justified. No loading, strength or toughness change
is selected here. Before quantitative onset claims, a short timestep comparison
must also resolve the rate oscillation. The old peak-.95 89→96 continuation and
all longer trajectories remain deferred.

[Comparison figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution_matched_tip/screen-comparison.png)
and [machine-readable paired results](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution_matched_tip/comparison.json)
include the full common history; late coarse-only data are visibly separate.

## Reproducibility and preservation

Entry: `pf-rsf-particle-free`, `b32a2d4f6e21c219f9af786d1b6e3725ea78a5a4`,
with preceding uncommitted work preserved in an entry patch. No core or checker
rebuild is required: the protected step-89 pair is used (core `034d2a84…3c27`,
checker `b800ef43…9d84`). The input is `bulk_tip/resolution_matched_seed.prm`;
`run_resolution_matched.py` creates fresh run directories and retains actual
commands, identities and inputs. `analyze_resolution_matched.py` reads native
rectangular-cell observations without changing physical data.

New closed point/phase CSV files alone are compressed losslessly, with SHA256
verification before replacing the new raw copy. Checkpoints and pre-existing
outputs are untouched. Compression manifests retain names, content hashes and
sizes. This avoids exhausting the approximately 5.8 GB free at entry. Inherited
cleanup losses remain unchanged. Evidence is under
`benchmarks/reconstructed_fault/particle_free/evidence/resolution_matched_tip/`.
The runner appends explicit midpoint and fresh output overrides. The reusable
fixture now also states midpoint and its own output path; archived generated
inputs preserve exactly what was launched. No native solver, material law,
quadrature, acceptance gate or protected executable changed in this task.

Final checks: the protected executable/checker and all recorded old step-89
checkpoint hashes are unchanged; incoming core/checker/helper edits match entry
hashes. Both native reductions, the common-variation comparison, Python syntax
and `git diff --check` pass. No numerical processes remain. New exports occupy
1.56 GB compressed instead of 4.52 GB raw; the runner verified all uncompressed
content hashes. See `final-verification.json` for identities and checkpoint inventory.
