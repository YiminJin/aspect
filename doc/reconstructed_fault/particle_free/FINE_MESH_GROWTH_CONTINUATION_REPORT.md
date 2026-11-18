> Follow-up: the [selected 88→89 comparison](TIP_STEP89_SEARCH_REPORT.md) now
> supplies an accepted .055625 s endpoint with an opt-in search order; the
> default replay caps without acceptance. This report retains the preceding
> continuation and spatial comparison through .055 s as historical evidence.

# Fine-mesh midpoint growth continuation — 10 November 2026

**Fourteen new intervals accepted; the .06 s target is incomplete.** The fine
trajectory now reaches **step 88 / .055 s**, with unchanged physical equations and
acceptance criteria. Step 89 was deliberately stopped for affordability after a
second expensive constrained search. This was not a timeout or a failed physical
acceptance. The continuation used 95.61 minutes, including the discarded reference
trial. No simulation remains running.

Weak axial extension and band broadening survive refinement qualitatively, but
their onset, location and amount are not spatially resolved. At .055 s the
phase-increment difference is 189.14% in relative L2, while physical-work error
is only about .01% of added fracture energy. The late .06 comparison remains
missing; this report closes the selected bounded attempt with a quantified
numerical obstruction, not a completed trajectory.

The restart is
`build-particle-free/bulk-tip/runs/fine-growth-warm01/output/restart/02/`
(marker 2). Its read-only restore reproduces all compared coefficients, stress/A,
section histories and observations **exactly**, with accepted clock 88/.055 and
pending clock 89/.055625. Original parents and the other rolling slots remain.
Inputs, executable identities, logs and native exports are retained. No commit
or server run was made.

## Preserved state and reproducibility

Entry source is `b32a2d4f6e21c219f9af786d1b6e3725ea78a5a4` on
`pf-rsf-particle-free`, with no incoming tracked changes. No checkout/reset,
checkpoint conversion, fresh seed, parameter recalibration or API rollback was
used. This remains the synthetic fixed-direction, fixed-reference section-RSF
control, not an unstable RSF event or a free-direction/branching qualification.

The accepted trajectory consists of original `material-midpoint-S1` steps 1–73,
repaired `material-obstacle-window02` step 74, and the new continuation. The repair's
**snapshot 03** is the verified parent: accepted clock .04625 s/74, pending clock
.046875 s/75, native Bernstein phase and `midpoint exact stage aging v1`.
The current core's read-only restore reproduces retained native coefficients,
physical stress, cohesive A, section age/slip/rate/resistance exactly; the native
solution and history coefficients agree. Reading the checkpoint changes none of
these histories. The first diagnostic observer failed by calling stage-dependent
postprocessing before initialization; its log is retained. The corrected observer
reads FE/material data directly at the existing post-resume signal and passes.

The retained repaired input supplies all physical/numerical parameters. Only
output/checker paths, termination limits and checkpoint frequency change.
The specimen stays 8×4 m, h=.0625 m, dt=.000625 s, with midpoint, two MPI ranks,
one numerical thread per rank, and a checkpoint after each accepted interval.
Acceptance gates, bounds, seed, quadrature, constitutive laws and section closure
are unchanged. The coarse comparison is completed M3, h=.125 m at the same dt. A comparison
of actual parsed parameter values finds only mesh, restart, checkpoint-frequency,
output and termination differences; physical and solver parameters match.

[Evidence and exact commands](../../../benchmarks/reconstructed_fault/particle_free/evidence/fine_growth_continuation/README.md)
record source/binary hashes, input differences, restore comparison and budgets.
The entry core SHA256 is `5e9bd0623611e0f7aa65929bc52cde522bc8cbf3435149abc84b5849e2a440da`;
its executable and checker remain under `build-particle-free/fine-growth/reference/`.
The candidate core is `b4625dc837089de5afb4a7ca97e0fb1e0075f07557fcc7d42341587e7f111a38`.
The cached executable git banner is not used as source identity.

## One focused performance correction

The initial continuation remained expensive in step 75. Its two numerical ranks
were busy, not hung in MPI. It was deliberately stopped after 820.27 s, with no
new accepted state, to try the single permitted correction. This is discarded
trial cost, not a physical acceptance failure or part of the joined history.

In `source/simulator/solver/bulk_tip_linear.cc`, successive active-set linear
problems now start FGMRES from the previous direction minus the newly prescribed
bound displacement, with active free rows set exactly to zero. The first solve
still starts from zero. This reuses only a linear initial guess **within one
search**; it does not cache a classification or direction across Newton searches
or physical steps. The operator, right-hand side, preconditioner, linear tolerance,
600-iteration/24-pass limits, complete linearized VI and nonlinear acceptance
remain unchanged. Compact successful-solve timing/residual lines expose cost;
unfinished/failed solves are absent from these per-pass lines, so partial sums
are lower bounds. Complete accepted counters remain authoritative.

Two short one-rank midpoint controls use the existing direct/manufactured linear
oracle and tangent checks. Both pass. Krylov iterations fall 696→529 and Stokes
calls 714→547; nonlinear iterations remain 17, active passes 47, releases 84, shifts 0.
The summed physical-interval time is 17.657→15.983 s, but concurrent reference
work limits interpretation of wall-time speedup. Maximum endpoint differences
are phase 1.25e-14, stress component 2.82e-6 Pa, Axy 1.39e-11, age 6.32e-13 s,
slip 6.49e-16 m, rate 4.78e-15 m/s and resistance 1.15e-11 Pa. Candidate direct
relative defect is 2.37e-9, manufactured defect 1.53e-5 (existing 2e-5 gate), and
tangent action defect 2.45e-9. These controls establish numerical agreement;
the unfinished reference fine interval cannot establish a fine-grid speed ratio.

## Measured obstruction and cost

The warm-start change is retained as a locally verified optimization, not a
solution to the remaining constrained-search cost. It reduces work in the short
matched control; there is no completed original-core fine endpoint with which to
claim a fine-grid speedup or branch-uniqueness result.

| Interval | Outcome | Measured cost on two ranks |
|---|---|---:|
| Repaired 74 (prior evidence) | Accepted | 1657.66 s; 6391 Krylov iterations |
| New 75 | Accepted | 1032.10 s; 4240 Krylov; 45 active passes; 2 shifted searches |
| New 76–88 | All accepted | 183.40–257.03 s each; median 209.19 s; no shifted searches |
| Attempt 89 | Stopped, not accepted | 1021.78 s since the last checkpoint; at least 3499 Krylov iterations |

In step 89, the first exact search used all 24 permitted active-set passes without
producing a certified direction. The existing positive-metric search accepted
shift 1, alpha 1: phase KKT .0464328→.00307805. The next exact correction increased
phase KKT to .064212 while reducing mechanics to 2.4214e-9. A second exact search
completed eight linear solves before the stop. In total, 36 successful linear
subsolves alone consumed 952.76 s; unfinished/shifted solves are not included in
that lower bound. Small reported FGMRES residual estimates do not certify the
complete linearized VI. No scalar-obstacle accuracy exception was reported.

At the stop decision, about 1480 s remained. The **seven intervals after 89** alone
would require 1464 s at the recent median, excluding the still-unconverged current
interval. The process group was stopped explicitly, preserving step 88. Exit 1 is
the recorded intentional termination, not a claimed solver-failure certificate.
The candidate ran 4916.58 s, the discarded reference 820.27 s, total 5736.85 s.
Accepted new intervals account for 3809.34 s of recorded interval time. The
remaining difference includes the unfinished trial and setup/output overhead.

At the last accepted output, exact search accounts for 3420.11 s and phase-AMG
setup only 5.72 s. Stokes inverse time 1495.00 s is **nested** in search time, not an
additional fraction. `tip-step-timers.csv` contains cumulative snapshots within
each process segment; use differences for individual intervals. The joined
`tip-profile.csv` and cost figure give actual per-interval costs. The dominant
obstruction is repeated constrained mechanics/active-set work, not avoidable
setup or a recurrence of the repaired scalar stopping-floor defect.

## Comparison method

The reducer joins contiguous accepted step IDs with no restart duplicates, checks
mesh/method/dt and original prefix provenance, and keeps each segment's command
record. It reuses validated steps 1–73 reductions; new rows use the existing native
SIPG chain-rule regional partition and physical-work reducer. Raw tables, selected
fields, checkpoints and earlier failure evidence remain. Archived fields are
restored selectively with checksum verification. Earlier cleanup loss/availability
statements are unchanged; this task does not reconstruct lost campaigns.

Each phase increment is relative to its **own mesh's equilibrated initial seed**.
Relative L2 differences use the fine field norm as denominator.
Phase comparisons use exact cell-interior Q2 polynomials at common fine Gauss
points; gradients exclude DG jumps, which remain in full SIPG fracture energy.
Physical deviatoric stress tau and cohesive A use full 3-D plane-strain tensor norms and the
existing Q5 diagnostic interpolation of native histories, not a new stress space
or history transfer. Section age/slip are endpoint values; V/resistance are at
the matching midpoint stage. Native contours retain both DG traces and the
established thresholds; sampled plots alone do not define an onset. Regions are
unchanged: original support |x−4|<2, |y−2|<.25; central band |x−4|<1.5;
old tip band 1.5≤|x−4|<2 inside support; axial intact |x−4|≥2; the remaining
outside-support region is transverse intact. The onset thresholds are increments
1e-6, 1e-4 and .001. Contours phi=.001, .01, .1 and .5 are observation levels,
not a mesh-independent crack-front definition.

## Common-time findings through .055 s

[Phase increments](../../../benchmarks/reconstructed_fault/particle_free/evidence/fine_growth_continuation/phase-increment.png),
[onset, energy and cost](../../../benchmarks/reconstructed_fault/particle_free/evidence/fine_growth_continuation/onset-energy-cost.png),
and [both native DG traces](../../../benchmarks/reconstructed_fault/particle_free/evidence/fine_growth_continuation/native-traces.png)
are supplied as PNG and PDF. The cost plot marks unfinished 89 separately.
[Full numerical tables](../../../benchmarks/reconstructed_fault/particle_free/evidence/fine_growth_continuation/comparison-tables.md)
include every onset threshold, contour, section norm and signed/absolute work
channel. Common-time comparisons include .04, .045625, .04625, .05 and .055 s.

Initial seed phase differs by max .00229835 and L2 .000368320 (relative .06643%);
its gradient differs .32994% in relative L2. Initial SIPG fracture energies are
2.157351616 and 2.157382603 J/m. These are separated from subsequent evolution.
The unchanged step 73 prefix already showed a 257.31% phase-increment difference.

| Quantity at .055 s | Coarse h=.125 m | Fine h=.0625 m |
|---|---:|---:|
| Maximum phase increment | .0280422 | .0259632 |
| Maximum axial-intact increment | .000442599 | .0170496 |
| Added SIPG fracture, J/m | .004823528 | .002735625 |
| Old central band contribution, J/m | .004368552 | .001303750 |
| Old tip band contribution, J/m | .000449196 | .001286125 |
| Axial-intact contribution, J/m | .00000578034 | .000145750 |
| Transverse-intact contribution | 0 | 0 |
| Central increment first exceeds 1e-4, s | .030 | .0275 |
| Old-tip increment first exceeds 1e-4, s | .050 | .040625 |
| Axial increment first exceeds 1e-4, s | .055 | .0475 |

Fine axial onset is 12 intervals earlier. The fine old central and old tip bands
receive 47.66% and 47.01% of added fracture energy, compared with 90.57% and 9.31%
on the coarse mesh. Boundary and remote phase increments remain zero. Regional
SIPG partition defect at the fine endpoint is 9.31e-15 J/m; volume-only fracture
changes are retained separately and do not replace the face contributions.

At .055 s, phase-increment max difference is .0223661 and L2 difference .0131759
(relative 189.14%). Full deviatoric stress differs 157.462 in volume L2 (3.8735%);
cohesive A differs .000880826 (41.910%). Section age, cumulative slip, V and
resistance differ 1.375%, 4.302%, 2.877% and 13.616% in ds-weighted L2. Age/slip are
at endpoint .055; V/resistance, including radiation, are at stage .0546875 s.
Fine maximum slip is .000601647 m and maximum stage rate .0150898 m/s.

Both DG traces give fine outward axial shifts from their own seed of .058519 m
(.936h) at phi=.001 and .043797 m (.701h) at phi=.01. The latter ends at
x=6.017642 m: only .017642 m (.282h) beyond the original support x=6.
Coarse phi=.01 moves .004101 m (.0328h) and remains inside that support.
The phi=.5 axial contour does not move on either mesh. Normal phi=.5 broadening
is .005145 m (.0412h) coarse and .001487 m (.0238h) fine. This is weak process-zone
extension plus evolution/broadening inside the old band, not developed-front
advance. The .06 fine response cannot be inferred from this prefix.

Signed local deformation is retained. Fine/coarse section cancellation is
.3361%/.8863%; opposite-sign rates occur in 184/134 sections (of 768/384).
Two fine sections, s=2.002110 and 5.997890 m, trigger the existing multiple-band
sample diagnostic. Each has one dominant positive band, integrated rate
1.879082e-3 m/s, and two tiny negative wings; total negative rate is −1.82936e-8
m/s and the signed section rate 1.879064e-3 m/s. Their cancellation is .001947%.
These sampled clusters do not demonstrate separate physical faults. At the last
fine interval, positive/negative local work is .004160443/−.00000708992 J/m;
the signed sum equals section dissipation with defect 6.94e-18. No clipping or
additional localization rule was used. The [signed-section detail](../../../benchmarks/reconstructed_fault/particle_free/evidence/fine_growth_continuation/signed-sections.json)
records thresholds, coordinates and each contribution.

Physical work error is `Wext − ΔE − Db − Dc − Df`, with full recoverable storage
and physical dissipations. It is independent of the search functional.

| Window | Coarse signed / accumulated absolute error, J/m | Fine signed / accumulated absolute error, J/m |
|---|---:|---:|
| 0–.055 s | 4.75719e-7 /4.75719e-7 | 3.23053e-7 /3.23053e-7 |
| .04–.055 s | 2.90646e-7 /2.90646e-7 | 1.82406e-7 /1.82406e-7 |
| Full error / added fracture | .009862% | .011809% |
| Late error / late added fracture | .006703% | .007041% |

The equal signed/absolute physical errors show this small remainder is not hidden
by cancellation. The fine full/late phase remainder is 2.12628e-7/1.82406e-7 J/m;
full mechanical-work residual contribution is −1.10418e-7 J/m (absolute 1.11225e-7).
Stress, traction and fracture numerical-loss channels remain zero in midpoint.
Physical friction and radiation contributions are separate: fine full values
.239060 and .0238155 J/m, with bulk/cohesive viscous dissipation .00154563 and
.0000144878 J/m. The full tables retain both meshes and both windows. The fine
accumulated absolute ledger defect is 1.41e-13 J/m. Maximum accepted mechanics,
phase KKT and ledger residuals over the joined history are 5.51e-10, 9.35e-8 and
7.06e-15. These are acceptance evidence, not spatial-convergence certificates.

## A bounded diagnostic of spatial sensitivity

An existing-output diagnostic of interval 73→74 separates seed amplitude from
mechanical localization. Adding the matched Maxwell stress and cohesive equations
gives the midpoint strain rate

\[
 D_{xy}=\frac{(\tau_{xy}^{n+1}-\tau_{xy}^{n})/(2G)
                    +(A_{xy}^{n+1}-A_{xy}^{n})}{\Delta t}
       +\frac{\tau_{xy}^{m}}{2\eta}+\frac{G A_{xy}^{m}}{\eta}.
\]

Here total bulk Dxy and fault-only deformation rate w remain distinct. The
reconstructed Dxy is Q2 to 2.4e-15; stage averages match the retained endpoints.
At the same central material point, both meshes have phi=.95 initially and at
step 74, but Dxy rises from .063392 to .117818 s^-1 under refinement. The peak is
concentrated in the central mechanical cells; w has the same narrowing and
signed flanks. The central integrated fault/total engineering shear ratio changes
.36465→.37717. Thus essentially equal central seed amplitude does not imply a
resolved mechanical layer. The local fixed-phase/fixed-resistance shear stiffness
ratio is
`(1-phi)^2/[m*phi*(1+phi)+(1-phi)^2]`; its sampled minima are 9.10e-4 and 5.89e-4.
At the common central trace the ratio is 3.373e-4 on both meshes. The full
normal width over which its inverse exceeds half the central inverse is
5.470 mm coarse and 5.415 mm fine, computed from each retained midpoint phase
polynomial. The cell widths are respectively 22.85 and 11.54 times this width.
This is a fixed-phase/fixed-section-resistance point-tangent diagnostic, not the
complete condensed compliance. It identifies a mechanically narrow scale even
when the phase is well represented. It is **not** a measured condensed-Hessian
eigenvalue or proof of negative curvature. The point kernel is finite at intact phi=0; this
diagnostic does not resurrect the repaired scalar roundoff-floor diagnosis.

[Central layer comparison](../../../benchmarks/reconstructed_fault/particle_free/evidence/fine_growth_continuation/central-layer-parent74.png)
and its JSON/script preserve the reconstruction and limits. The evidence supports
unresolved mechanical/cohesive localization as a cause of spatial sensitivity;
two meshes do not establish its converged limit or uniquely separate seed-gradient
and coupled-response effects. No third mesh or physical modification was launched.

## Decision and next experiment

Retain the accepted .055 prefix, the small warm-start optimization and all failed
or stopped evidence. The selected source, midpoint endpoint convention, full
cohesive inventory, Bernstein representation and checkpoint meanings remain.
The .06 qualification is **not complete**, and the current growth response is
**not quantitatively mesh-converged**. Its qualitative weak-extension response
survives; onset differs .0075 s, growth-field L2 differs 189%, and the allocation
of added fracture changes markedly. A small physical-work error does not remove
that uncertainty. No production, unstable-RSF, free-direction or branching claim
is supported. The narrow mechanical scale identified above remains unresolved.

The next concrete experiment should replay **only the saved step 88→89 interval**:
compare the current exact-first search with the existing positive-metric family
selected first, retaining exact polishing and every physical/VI gate. Record
true per-pass constrained residuals, accepted fields, work and cost. This targets
the measured 24-pass search exhaustion and subsequent KKT increase, and can decide
whether the remaining eight intervals are affordable without another solver
framework or tolerance change. It is a proposed numerical experiment, not an
automatic continuation or a demonstrated cure for spatial sensitivity. Longer
growth, .08, a third mesh, loading changes and instability calibration remain
deferred.
