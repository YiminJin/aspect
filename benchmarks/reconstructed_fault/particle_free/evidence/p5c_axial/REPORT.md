# P5c targeted axial refinement

Completed 22 October 2026 on f1f9f29cf40c32d1ddcea000b724375b64853532 plus
preserved local work. **Radial tensor flow still reduces pre-growth cap driving
and integrated/peak growth. The reduction is smaller after axial refinement;
the growth field is not spatially converged.** Neither case activates initially
intact material. This completes the selected comparison, not propagation
qualification.

## Matched inputs and initialization

Each 192×128 case starts freshly from the same continuum seed and zero histories,
using four MPI ranks. The 1.5×1 m finite body, peak-.6 seed ending at (1,.5) m,
released lateral horizontal/retained vertical restraints and top/bottom loading
are unchanged. G=1e6 Pa, eta=1e8 Pa s, cohesion=2000 Pa, Gc=1 J/m²,
ell=.046875 m, m=4, residual resistance=1000 Pa. Retain unlimited DGQ2/native FE
source complement, Gauss8/source3 and all acceptance/stability settings.
[Explicit parameters](parameters.csv)
and resolved-input-comparison.json verify that only axial resolution, output
paths and the prefix stop change relative to the corresponding P5c inputs.

Δx changes from .015625 to .0078125 m; Δy stays .0078125 m. Both new
trajectories use **Δt=.00125 s through .11875 s, then .000625 s through
.145 s**. A native same-mode checkpoint at publication95 changes the pending
interval through the existing restart hook; accepted clocks/histories are not
edited. Shear rate .018/s from .05 through .140 s, followed by displacement
hold. The reference is the complete P5c half-step trajectory with this same
schedule, including its prefix.

Relative initial L² representation error decreases from **.591954% to .500005%**;
nodal prescription error is 1.11e-16. Each mode has identical initialization on
its own mesh, phi in [0,.6], zero physical stress/cohesive histories and exactly
zero unloaded phase change/KKT. Initial fracture storage changes from
.569413700 to .569586716 J/m (+0.030385%). The two initial physical fields differ
by .2466% L² relative to the refined field, with peak difference .01790.
All increments below subtract **each mesh's own initial Q1 field**.

## Measured response

Onset means accepted nodal increment>1e-8; brackets reflect saved timesteps.
Integrals use the Q1 field over the whole body, where all growth remains near
the seed. Peak and integral values are at .145 s; the hold adds no phase growth.

| Mesh | Constitutive flow | Onset bracket [s] | Integrated Δphi [m²] | Peak Δphi | Added fracture [J/m] |
|---|---|---|---:|---:|---:|
| 96×128 | Fixed direction | (.133750,.134375] | 1.868816e-6 | .00335143 | 5.253210e-6 |
| 96×128 | Radial tensor, direction free | (.135625,.136250] | 1.202099e-6 | .00198701 | 3.166616e-6 |
| 192×128 | Fixed direction | (.128125,.128750] | 2.468600e-6 | .00337147 | 6.601450e-6 |
| 192×128 | Radial tensor, direction free | (.129375,.130000] | 1.693120e-6 | .00225052 | 4.151832e-6 |

Radial reduction changes from **35.68% to 31.41% integrated** (−4.26 percentage
points) and **40.71% to 33.25% peak** (−7.46 points). Refinement increases fixed/
radial integrated growth by 32.09%/40.85%; peak changes are+.598%/+13.26%.

Both refined modes first grow at (1.0859375,.5) m, inside the pre-existing cap
(initial phi=.00460670). The coarse first-growing column is x=1.078125 m.
Fixed refined growth occupies 23 nodes: x∈[1.0703125,1.0859375],
y∈[.46875,.53125]m. Radial growth occupies 31 nodes, mostly in that cap;
four weak flank nodes at x=.9921875 m extend its total y range to [.4140625,.5859375] m.
Those outer flank increments contribute only **.0976% of radial integrated
growth**, with peak 1.32e-5. Thus radial flow does not suppress growth everywhere.
These nodes remain within pre-existing damage; they are not a new branch or
intact-material nucleation.

Centerline phi=.01 advances relative to initialization are .977/.613 mm coarse
and 1.483/1.023 mm refined (fixed/radial). Its initial position itself changes
from 1.086496 to 1.083449 m, so absolute front positions must not be mistaken for
growth. Phi≥.1 fronts do not advance; initially intact increments and
disconnected superlevel components remain zero.

At the common pre-growth time **.128125 s**, compare the same physical Q1 hat
centered at (1.078125,.5) m, radii .015625/.0078125m. Define driving
−Σw_i g_mech,i/Σw_i M_i and resistance Σw_i g_frac,i/Σw_i M_i, using complete
native weak forces. The refined mesh represents this test function exactly.

| Mesh | Fixed driving [Pa] | Radial driving [Pa] | Fracture resistance [Pa], both modes |
|---|---:|---:|---:|
| 96×128 | 1.545074 | 1.370276 | 2.417528 |
| 192×128 | 1.536873 | 1.362290 | 2.365027 |

The radial driving reduction is 11.31%/11.36%, while fixed driving changes only
−.531% across meshes. Independent AT1 integration of the same weak fracture
functional agrees within 4.44e-16 relative to Kf=8 Pa. Native refined nodal
tests are retained separately: at the new first-growing node, resistance is
1.518335 Pa and pre-growth fixed/radial driving 1.483891/1.334271 Pa. These have
different supports from coarse nodal tests. Earlier onset is associated with
the richer phase representation/test space; it is not evidence of increased
mechanical driving.

Endpoint growth-relative consistent-Q1 L² mesh differences are **28.68% fixed /
37.97% radial**, evaluated at common physical points. Two meshes establish no
order. Transverse resolution and refined-mesh temporal uncertainty are not
independently qualified here; the earlier half-step result applies to the
coarse mesh. Representation and subsequent spatial effects remain distinct.

## Verification, figures and next task

Both new trajectories pass unchanged mechanics≤1e-9, phase KKT≤1e-7,
source≤1e-12 and scaled BE ledger≤1e-10 gates. Measured maxima are
4.14e-12/6.56e-8/1.12e-14/1.13e-13 J/m respectively. Mechanically relaxed
curvature is ≥48.1289 on at most 31 critical masters, below the unchanged 1024 cap;
no stability retry occurs. Every accepted step publishes once. Actual endpoint
archives contain publication 95/.11875 s and 137/.145 s. All 221184 diagnostic
particles stay fixed with exact ID/coordinate fingerprints. Previous rejection
qualification is reused unchanged; no new rejection test is claimed.

The refined mesh has 24576 cells, 742409 native DoFs (247684 selected mechanical/
phase unknowns), plus 663552 independent DG stress DoFs. Four-rank wall costs are
2694.13/2891.79 s fixed/radial; maximum rank RSS 4.77 GiB. Complete runs/checkpoints
remain in build-particle-free/p5c-axial/runs. One omitted-freeze-flag launch was
stopped during rest after 102.8 s, preserved and excluded, then restarted freshly.
No native source/plugin, equation, tolerance or production default changes.
All 41 protected files, the prior P5c archive and qualified executable/plugin
hashes remain unchanged.

Refined endpoint/trapezoidal reaction work is 1.912609/1.890447 J/m fixed,
1.911742/1.889604 radial. Physical endpoint remainders are .022128/.022104 J/m;
trapezoidal remainders−3.376e-5/−3.412e-5 J/m. Full time histories retain storage,
dissipation and separate residual-work terms. BE closure is not a new physical
energy qualification. Preserve P4's failed 8.302% temporal/7.442% spatial
comparisons, 57.26% boundary sensitivity and **strict_reference_match_passed=false**.

![Common-scale cap comparison](cap-0129.png)

![Pre-growth physical-kernel balance](pregrowth-balance.png)

[Initialization PNG](initial-representation.png),
[growth/reaction PNG](growth-reaction.png),
[whole-body PNG](whole-domain.png),
[one GIF](axial-comparison.gif)
and compact CSV/JSON accompany the report. Dashed cyan contours are initial;
white are current (.01/.05/.1/.2/.4); dots are actual bulk vertices, green rings
growing phase nodes. Stress is physical tau, not auxiliary T. Y and W are
averages of actual Gauss8 physical values. The angle atan2(mean Wxx,mean Wxy)
is a constitutive tensor-component angle, **not a crack angle**; blank means no
defined active mean flow. The seed/loading were imposed; mechanics, histories
and phase increments were calculated.

**Recommend one bounded loading extension, not another mesh campaign.** Reuse
each mode's accepted step127/.13875s checkpoint before the original hold;
continue the same .018/s shear rate through .155s, then hold to .160s. Retain
Δt=.000625s and the current mesh/law/stability limits, with focused temporal
verification if intact activation occurs. At the next initially intact node
(1.09375,.5) m, current fixed/radial weak growth margins are still positive
.4191/1.2938 Pa. The extension aims to cross those conditions; it does not
guarantee activation. Measure genuinely initially-zero material, seed
connectivity and any remote nucleation without a growth mask, and stop at the
declared endpoint or existing acceptance/capacity limit. This remains a finite
physical specimen. The extension has **not** been run. BP3, RSF, transport,
geometry changes and AMR remain outside this task.
