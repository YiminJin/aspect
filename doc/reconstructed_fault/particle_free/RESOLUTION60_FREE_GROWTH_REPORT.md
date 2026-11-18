> Follow-up: the selected four dt=.0025 intervals from .060 through .070 are
> complete. See [matched-parent comparison and phi0/phi/delta-phi figures](RESOLUTION60_HALFSTEP_REPORT.md).
> The original run and its evidence below are retained unchanged.

# Peak-.60 freely evolving fixed-direction response

The user selects one fresh 128×128 case from its independently equilibrated
peak-.60 seed at t=0, midpoint dt=.005 s to .08 s. This supersedes the held-phase
screen's recommendation to defer free evolution. The purpose is the sequence
of damage, not a precise onset time or a general turning/branching qualification.

## Configuration and preservation

Entry is `pf-rsf-particle-free`, `b32a2d4f6e21`, with incoming dirty work preserved.
The same protected step-89 core (`034d2a84…3c27`) and checker (`b800ef43…9d84`)
are used without rebuilding. Two MPI ranks, one thread each; at most one analysis
and one compression worker run alongside them. No physical model or solver
acceptance tolerance changes. The retained opt-in positive-metric-first search
with exact polishing is selected, as in the successful step-89 control.

The 8×4 m box has hx=.0625 m, hy=.03125 m; peak .60, normal half-support .25 m,
longitudinal plateau |x−4|≤1.5 m and taper to |x−4|=2 m. Ell=.125 m, G=1e6 Pa,
eta=1e8 Pa s, Gc=2/3 J/m², strength 1000 Pa, Dc=.0005 m, initial section age .5 s,
pN=200 Pa and all RSF parameters remain unchanged. All walls impose
v=(.01(y−2),0). Phase evolves from the fresh zero-clock equilibrated seed;
no frozen checkpoint or old peak-.95 trajectory is resumed.

Checkpoints are requested every accepted step. The existing integrity stop
remains: actual remote/boundary phase >1e-6 terminates after acceptance; a high
trial driving value alone does not. A 2.9-hour graceful wall limit and 3-hour
process cap protect against an unresolved solver stall. These are resource
limits, not a change to the scientific target. New closed diagnostic CSVs are
compressed losslessly, with hashes verified before removing their raw copies.

## Observation definitions

Every accepted step retains cumulative and interval phase increments and
fracture-energy increments in central band, old taper, initially intact axial
ahead and remote regions, using the previous screen's physical masks. A fifth
`other` region completes a disjoint partition of the body. Pointwise maxima use
native Q6 samples; coefficient maxima and endpoint profiles are also retained.

The fracture functional is E_f=.5 B^T H B+l^T B, l_i=K mass_i. With the exported
native derivative F=H B+l, each row receives .5 B_i(F_i+l_i). Differences of
these row energies define a telescoping regional partition, including SIPG.
They sum to the solver's fracture increment. Regional allocations of face
interactions are conventional and may be signed; they are not independent
positive fracture densities. The older directional-work partition
.5 (B−B0)_i(F+F0)_i is retained separately, not silently substituted for energy.

The endpoint profiles are delta phi=phi−phi0 at y=2 from both native traces,
and at y=2±hy/2, x=5.25–6.75 m, with the original endpoint x=6 marked. The
off-axis cuts catch damage beside the central cusp. No smoothing or clipping.
Connected extensions at both x=2 and x=6 use exact piecewise-Q2 superlevel
intervals at delta phi=1e-8,1e-6,1e-4; crossing a cell face requires both traces
to stay above the threshold. These diagnostics are not physical admission rules.

Central excess compliance c(phi)=4phi(1+phi)/(1−phi)^2 excludes intact background.
The native polynomial maximum and exact half-height crossings on each x=4
trace define W; disjoint half-height components are reported. A saturated peak
makes the compliance unbounded and the width diagnostic undefined, never capped.
Both W/hy=1 and 2 are displayed as screening guides, not accuracy guarantees.
The physical-work remainder is Wext−deltaE−Db−Dc−Df; signed and accumulated
absolute errors are compared with added fracture only when the latter is meaningful.

## Results and next decision

**Completed all 16 intervals through .080 s. Retain this specimen for a focused
comparison: the free response develops connected axial damage while remote
material remains quiet.** The held-phase central-first ordering was not sufficient
to reject the specimen for subsequent extension. No physical input changed and
no second run was launched.

The sequence is central damage at the .045 endpoint, old-taper damage at .060,
then axial-ahead damage at .065. Previous accepted observations bound these
sampled events to (.040,.045], (.055,.060] and (.060,.065] s respectively. They
are sequence brackets, **not precise onset times**. At .065, off-axis native
profiles connect the evolving central band through the taper and into initially
intact material beyond both original ends. At .075, taper and axial increments
have overtaken the central increment. Phase increments below are maxima at native
Q6 points, always relative to this run's equilibrated seed.

| Endpoint (s) | Central max delta phi | Old taper | Axial ahead | Remote | W/hy |
|---|---:|---:|---:|---:|---:|
| 0.040 | 0 | 0 | 0 | 0 | 1.3941 |
| 0.045 | 0.00113318 | 0 | 0 | 0 | 1.4087 |
| 0.050 | 0.00320508 | 0 | 0 | 0 | 1.4417 |
| 0.055 | 0.00622309 | 0 | 0 | 0 | 1.4869 |
| 0.060 | 0.0113039 | 0.00115455 | 0 | 0 | 1.5586 |
| 0.065 | 0.0169749 | 0.00546402 | 0.000975608 | 0 | 1.6438 |
| 0.070 | 0.0252378 | 0.0209359 | 0.0146323 | 0 | 1.7604 |
| 0.075 | 0.0360935 | 0.0536227 | 0.0402297 | 0 | 1.9005 |
| 0.080 | 0.0486787 | 0.121898 | 0.111234 | 0 | 2.0936 |

At every accepted interval, `history.csv` retains both interval and cumulative
phase integrals/maxima and fracture-energy increments for all four regions and
the complementary `other` region. The table below gives cumulative fracture
increments in J/m; it includes the actual SIPG row-energy partition.

| Endpoint (s) | Central added Ef | Old taper | Axial ahead | Remote | Total added Ef |
|---|---:|---:|---:|---:|---:|
| 0.040 | 0 | 0 | 0 | 0 | 0 |
| 0.045 | 8.54491e-05 | 0 | 0 | 0 | 8.54491e-05 |
| 0.050 | 0.000399629 | 0 | 0 | 0 | 0.000399629 |
| 0.055 | 0.00124079 | -7.42546e-05 | 0 | 0 | 0.00116654 |
| 0.060 | 0.00309798 | -0.000149603 | 0 | 0 | 0.00294838 |
| 0.065 | 0.00612504 | 6.58466e-05 | 2.60924e-06 | 0 | 0.0061935 |
| 0.070 | 0.0113516 | 0.0011134 | 0.000121255 | 0 | 0.0125862 |
| 0.075 | 0.0195336 | 0.00545718 | 0.000929519 | 0 | 0.0259203 |
| 0.080 | 0.032112 | 0.0179152 | 0.00733007 | 0 | 0.0573573 |

Negative taper entries before its phase evolves are shared-face energy
redistribution, not negative damage or discarded data. The independently stored
`fracture_directional_work` attributes work to changing coefficients and differs
regionally; the two partitions agree globally. `other` contributes
0 J/m at .080. No overlapping extra masks
are added to these totals.

### Endpoint profiles and pulse resolution

The figures show the native centerline traces and the cuts y=2±hy/2 across x=6.
The off-axis cuts are necessary because early central damage grows beside the
seed's cusp even while its peak stays fixed. Connected superlevel support crosses
the original endpoint; this is spatial connectivity, **not C0 continuity** of a
DG field. Native face jumps are retained in the CSVs and plots. Both original
ends and both normal sides are included in the machine-readable diagnostics.

| Endpoint (s) | Phase-increment threshold | Axis extension / hx | Half-cell-cut extension / hx |
|---|---:|---:|---:|
| 0.065 | 1e-08 | 0.996939 | 0.996052 |
| 0.065 | 1e-06 | 0.969385 | 0.960516 |
| 0.065 | 0.0001 | 0.693850 | 0.605156 |
| 0.080 | 1e-08 | 4.000000 | 4.000000 |
| 0.080 | 1e-06 | 4.000000 | 4.000000 |
| 0.080 | 0.0001 | 3.932491 | 3.914289 |

With hx=.0625 m, the .065 extension at threshold 1e-4 is .04337 m on the axis
and .03782 m on the half-cell cut. It is subcell at first detection and is strongly
threshold-dependent. This establishes connected damage, not a converged crack
length, speed, or high-phase-contour tip. At .080 the axial max increment is
0.111234; remote max remains
0.

Central W/hy grows from 1.394065 to 2.093576; its minimum
on the accepted trajectory is 1.394065. The final W is
0.065424246 m and native central peak is 0.6.
The pulse broadens in this run; it does not sharpen below its initial screening
resolution. This does **not** establish spatial convergence of the phase/front.
No finer normal spacing is selected on the basis of a width collapse.

### Physical work, acceptance and cost

Cumulative physical-work error is 7.26337237e-05 J/m; accumulated absolute error is
7.26337239e-05 J/m. Relative to 0.0573572793 J/m added fracture, these are
0.12663% signed and 0.12663% absolute. From .040 to .080,
the signed remainder is 7.21783674e-05 J/m. No numerical ledger term is subtracted
to improve this physical-work measure. At first axial detection (.065), total
cumulative work error (2.01524e-05 J/m)
exceeds the small axial regional fracture increment
(2.60924e-06 J/m). Global work agreement
therefore does not provide a local energetic accuracy certificate for onset.

All accepted bounds/history checks and the original mechanical/KKT gates pass.
Maximum accepted mechanical residual is 3.58494e-10;
maximum accepted phase KKT is 9.41614e-08.
The final KKT is 6.16566e-08. The largest reconstructed
section-power adjoint defect is 1.04e-13 J/(m s). Signed local rates, cancellation
and multiple-band counts are retained without clipping. Slip-rate transients
persist at dt=.005; halving the screening timestep is not a convergence study.

At the final stage, 14 of 768 sections contain opposite-sign local fault rates
above max(1e-10 s⁻¹,1e-6 max_section|w|); the sampled detector finds no multiple
active bands. All reverse deformation lies in the old taper. Integrated positive
and negative w are .0472801566 and −4.61828e-8 m²/s, giving net integral V ds
.0472801105 m²/s and cancellation fraction 1.95358e-6. The negative local
friction/radiation power is −6.12993e-6 J/(m s), or −3.06496e-8 J/m over this
interval, while total section dissipation is positive (.0320671 J/m). This small
local negative contribution is retained and is a consequence of applying the
common section force to signed local deformation; local positivity is not
asserted. Final integrated cumulative slip is .00200424234 m². See
`final-signed-deformation.json` for the regional signed-power check.

Physical process time is 5736.04 s (95.60 minutes), on two MPI ranks.
The last two intervals cost 790.84 and
2583.90 s. Their constrained searches dominate the late
cost: the last interval records 322 line trials,
12 repeated exact bound classifications and
12 failed search-direction attempts. These failed
linear/search attempts remain in the log; they are not failed physical steps.
No acceptance tolerance, timestep, physical input or solver algorithm was changed.

A plateau budget check was announced after iteration 23 of the final interval.
The third additional iteration reduced KKT by about 37%, so the stop condition
was not met and the original solve continued. The run exits normally; neither
the resource cap nor a manual termination was used. No 0.080 trial field was
counted before acceptance.

### Next justified experiment

Retain the specimen and the current physical parameters. Use the saved accepted
step-12/.060 parent for **one focused matched-parent midpoint comparison**:
dt=.0025 from .060 to .070 (four intervals), compared with this run's two retained
dt=.005 intervals at .065 and .070. Compare connected-profile increments at all
three thresholds, regional fracture increments, slip/state, W/hy, and signed
and absolute physical-work errors. This targets the first axial-growth window
and the still-visible time-integration error without repeating a long trajectory.
It inherits the same pre-.060 history and therefore qualifies only the local
window, not full-from-zero temporal convergence. Axial growth initially occupies
one cell; an along-fault mesh comparison remains necessary before quantitative
front-length claims. No redesign or longer propagation campaign is selected.

The parent is preserved at
`build-particle-free/resolution60-free-growth/parent-step12/output/`, with hashes
in `preaxial-checkpoint.json`. Its copy was checked before and after copying;
no additional restart simulation was run in this task.

[Growth and work history](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_free_growth/growth-history.png)
· [Endpoint detail](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_free_growth/endpoint-growth.png)
· [Slip history](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_free_growth/slip-history.png)
· [Every accepted interval](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_free_growth/history.csv)

## Reproduction

```sh
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py resolution60-coarse-growth 11100 python3 benchmarks/reconstructed_fault/particle_free/bulk_tip/run_resolution_matched.py coarse --growth --dt .005 --end .08 --seconds 10800 --checkpoint-steps 1 --wall-hours 2.9
python3 benchmarks/reconstructed_fault/particle_free/bulk_tip/analyze_resolution60_growth.py
```

`--live` reduces only accepted, closed, compressed observations and caches them.
Evidence: `benchmarks/reconstructed_fault/particle_free/evidence/resolution60_free_growth/`;
actual input/log/checkpoints: `build-particle-free/bulk-tip/runs/resolution60-coarse-growth/`.
Previous screen input/compression manifests stay in `evidence/resolution_matched_tip/`;
that runner also records this new run's generated input and compression manifest there.
Inherited cleanup losses and old failed evidence remain unchanged.

Final preservation checks pass: protected executable/checker and old step-89
hashes are unchanged, incoming core/checker/helper source is preserved, and the
copied .060 checkpoint matches its manifest. All 120 accepted-history/checker
checks pass. The 68 new large exports occupy 1.232 GB compressed rather than
3.447 GB raw; each uncompressed content hash was verified by the runner. Current
checkpoint hashes, exact run status and source identities are in
`final-verification.json` and `preservation-checks.json`. No numerical job remains;
Python syntax and `git diff --check` pass. No commit was requested or made.
