# Enlarged horizontal-domain control

**Completed through loading and hold to 2.47 s. Horizontal-period independence
fails the selected growth, stress and normal-pressure targets.** Doubling the
period reduces new fracture by 11.9% and integrated slip by 6.11%. The fully
decohered region still advances about .73 m; this remains a finite-strip,
prescribed-path result, with substantial domain sensitivity.

[Predeclared selection](COHESIVE_INTERFACE_DOMAIN_SELECTION.md) ·
[figures, GIF, indexed CSVs and commands](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_domain/INDEX.md).
Baseline `pf-rsf-particle-free`, `f1f9f29cf40c32d1ddcea000b724375b64853532`
plus preserved local work. All 1,051 protected source/include/BP3 files and four
previous admission/energy/evolution archives retain their entry hashes.

## Matched experiment

Compare fresh 8×1 and 16×1 m periodic strips, physical x=[0,8] and [−4,12],
y=[0,1]. The large native coordinate is translated by −4 m in analysis only.
Both use the **same physical mature seed [2.5,5.5] m at y=.4 m**, zero initial
jump/Maxwell memory, Θ=1, a=1 on the seed and a=0 elsewhere. Initialization is
exactly represented on each mesh; no coarse history or phase is interpolated.
The pre-existing fracture inventory is 3 J/m, excluded from new fracture.

Keep Δx=.0625 m, Δy=.0125 m, Q2/Q1 native mixed mechanics, independent seam
pressure, Q1 jump, Gauss3 mechanics/source/surface histories, independent Gauss6
physical bulk reconstruction, current-pressure coupling, signed RSF and all
existing tolerances. Retain G=1e6 Pa, eta=1e8 Pa s, Gc=1 J/m², peak=3000 Pa,
p0=10000 Pa, RSF (.005,.01,.1), V0=5e−6 m/s, Dc=.001 m and the original
radiation coefficient. Relative wall speed is .001 m/s to 2.45 s, followed by
hold. Native timestep controls use Δt=.0025 s through 2.30 s and .00125 s afterward.
**All 1,057 accepted times and intervals match bitwise between domains.**

The longer candidate Γ=[−3.5,11.5] replaces [.5,7.5]. Every added Gauss record
has **exactly zero damage, jump, reference, relaxed jump and accumulated slip
at every accepted state**; new nodal jump is zero and Θ follows aging exactly.
There is no added fracture/sliding work. Gauss3 ties enforce a zero Q1 field on
these cells, so eliminating those variations leaves the old-Γ mechanical problem.
This isolates period/bulk-domain sensitivity from activation on added surface.
Bulk φ remains provisional geometry; a is surface damage, with Gc∫Γa the single
fracture reservoir. Height and wall distance are unchanged.

## Physical result

Differences below use the 8 m result as a scale, not as an accuracy reference.

| At 2.47 s | 8 m period | 16 m period | Difference |
|---|---:|---:|---:|
| New fracture energy (J/m) | 1.52123 | 1.33994 | −11.92% |
| Integrated absolute jump (m²) | .00416812 | .00391358 | −6.11% |
| Peak jump (mm) | 1.09852 | 1.07361 | −2.27% |
| Left fully decohered advance (m) | .752326 | .730207 | −.022119 m |
| Complete new full cells | 12 | 11 | — |
| Left cohesive-zone width (fine cells) | 14.23 | 14.00 | — |
| Fault-induced reaction deficit (N/m) | −4151.26 | −3897.15 | 6.12% smaller magnitude |
| Bulk storage in common 8 m box (J/m) | 15.61136 | 16.09647 | +.48511 |

The continuous left full front uses the actual Q1 jump threshold 2Gc/peak,
with monotone new-material increments verified; it is not inferred from one
activated point. The a=1 Gauss brackets are [1.74296,1.75704] and
[1.75704,1.78125] m. Neither right side develops new full decohesion. Left
integrated growth changes 1.14505→1.11312 m (−2.79%); right growth changes
.376176→.226827 m (−39.70%). No detached a≥.01 region occurs. First activation
has the same accepted bracket (1.5175,1.5200] s; precise onset remains unqualified.
At 1.8 s the absolute growth difference is .00025373 m (2.99%); by 2.0 s it is
.00244909 m (7.48%). The earlier **65.4% early spatial failure** remains.

On actual common Gauss6 points, the full 3D plane-strain tensor difference
uses xx²+yy²+zz²+2xy². At 2.47 s its relative L2 is **5.50%** over the common
8 m box and **2.53%** in [1,3]×[.2,.6]. Normal-pressure RMS differences are
**668.8 Pa** and **461.5 Pa**, respectively; common-Γ current normal pressure
differs by **583.4 Pa RMS**. The frozen, identical reference memory produces
31.1 Pa RMS shear-map change but 635.4 Pa normal-map change on common movable
nodes. Thus the normal coupling is particularly period-sensitive, before
accumulated evolution differences are included. Both global p′ means remain
below 1.1e−13 Pa; the pressure difference has spatial structure and is not merely
an arbitrary constant gauge shift. This does not isolate every nonlinear feedback.

The selected <2% growth/slip, <1% tip tensor and <10 Pa pressure targets **fail**.
The <.05 m full-front target passes (maximum sampled change .02472 m).
Retained 8 m timestep uncertainty at 2.47 s is .330% integrated growth,
3.08 mm full front, .240% tip tensor and 1.35 Pa tip pressure. These are much
smaller than the measured domain differences. No new timestep or mesh sweep ran.

![Matched accepted surface histories](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_domain/profiles.png)

![Actual near-tip physical fields](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_domain/tip-fields.png)

*Imposed: the period, fresh mature seed and common wall motion. Calculated:
mixed equilibrium, current-pressure RSF, irreversible surface damage, physical
stress and work. The added tied surface never activates. Neither a free path
nor distant-wall behavior is established.*

## Work, checks and the preserved diagnostic failure

Raw trapezoidal work remainders are retained per step, with no subtraction for
acceptance. For 8/16 m, sum|remainder| is .00689405/.00650879 J/m, respectively:
**.4532%/.4858% of new fracture** and **.1541%/.1548% of sliding loss**. Hold-only
ratios are .6332%/.6917% and .5366%/.5966%; both pass 1%. Signed sums are
−.00687504/−.00650879 J/m. The earlier coarse-timestep hold-screen failure remains.
The larger far field adds extensive storage/work: report raw totals separately
from excess relative to an independently integrated homogeneous Maxwell strip.
That recurrence reproduces wall memory exactly and never corrects the raw error.

Maximum weak residual is 1.00e−8/9.97e−9 Pa; BE ledger discrepancy is
4.33e−13/8.28e−13 J/m. Minimum tested mechanical symmetric curvature remains
positive, 4.93e6/4.94e6 Pa/m; this retains the component certificate's restricted
scope. Gauss6 storage reconstruction differs by at most 4.69e−13 J/m; Gauss3/6
surface integration differs by less than 9.69e−8 Pa/m at operator level.
The enlarged sampled bulk normal pressure stays ≥8081 Pa and old-Γ-exterior
sampled |τxy|≤2568 Pa, below the 3000 Pa intact strength.
Maximum accepted derivative-gate values are 4.10e−7/1.35e−7, below 1e−6.

The **first enlarged attempt failed** the original diagnostic derivative gate
at trial 1.9425 s, accepted 1.9400 s. Perturbations 2.02e−14→5.04e−15 m gave
errors 2.75e−6→1.09e−5. The smallest responsive rate scale was 2.02e−11 m near
physical x=2.31954 m. Subtraction of O(10³ Pa) residual operands makes the FD
roundoff term scale as uΣ|operands|/(epsilon‖J‖); further halving increased it.

An **opt-in diagnostic-only** extended-precision residual reference retains
the same perturbations, full active columns and **unchanged 1e−6 gate**. Error
becomes 1.33e−9→5.19e−9. Independent 50-digit reevaluation of saved worst-entry
and smallest-rate active-column operands has maximum relative FD discrepancy
4.99e−12. A deliberately 1% biased tangent fails at approximately .01. Constrained
raw coordinate perturbations across a tied cusp are excluded from this classical
derivative check. The C++ physical point return, Newton solve and timestep are
unchanged. Original failed errors, logs and checkpoint remain intact.

Restart from accepted 1.8975 s reproduces **all replayed six Gauss histories,
nodal histories and physical history columns through 1.9400 s bitwise**. The
initial and activation rejection/retry checks pass, and publication counters
advance exactly once per accepted state. Both pre-event checkpoints at accepted
2.2975 s are preserved separately from rolling output. This is same-rank component
restart evidence, not new MPI or production-coupled qualification.

| Serial cost | Cells | Mixed masters / jump unknowns | Wall time | Observed peak RSS |
|---|---:|---:|---:|---:|
| Fresh 8 m control | 10,240 | 91,903 / 128 | 270.20 s | 919.8 MiB |
| Enlarged first attempt, diagnostic stop | 20,480 | 183,807 / 256 | 1438.65 s | 2120.8 MiB |
| Enlarged restart to 2.47 s | same | same | 860.37 s | 2121.1 MiB |

Setup costs are 86.47, 454.19 and 474.62 s. The enlarged campaign totals
2299.02 s including the failed attempt, repeated setup and overlap. The host
has 134,580/268,596 DoFs including unused temperature. Runs were sequential,
single rank/thread, bounded at 900/1800/1200 s and 6 GiB. No server job ran.

## Conclusion and next action

The finite-strip propagation milestone survives enlargement qualitatively, but
its quantitative growth and normal loads are domain-sensitive. **Neither strip
is qualified as a distant-boundary reference.** Keep the 65.4% early spatial
failure, onset/branch limitations and fixed H=1 wall influence visible.

Before a longer quantitative trajectory, use the inexpensive fixed-strip
mixed/pressure reference to separate the period-dependent normal response from
the fixed-wall constraint, then select one focused distant-boundary or explicitly
finite-specimen control. Increasing the tied buffer alone cannot answer that
question. Do not change p0, lag pressure, tune the law, or extend this run to
conceal the sensitivity. Production solver integration, free paths, bends,
MPI/AMR/transport and a longer event remain separate tasks; none begins here.
