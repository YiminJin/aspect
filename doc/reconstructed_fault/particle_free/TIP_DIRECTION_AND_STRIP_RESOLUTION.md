# Prospective fault direction and material-strip resolution

**Retain the localized onset search and the default mesh-scale strip width for exploratory sliding in the single-forward-tip specimen.** The search itself took at most 0.62% of ordinary accepted-step solver time. During the first appreciable sliding interval, halving the width changed integrated slip by 0.16% and added fracture energy by 0.63%. The physical-work residual was about 71% of added fracture energy at both widths. A subsequent two-step replay increased the physical-work residual by 71.3% and the added fracture energy by 171%. Fracture work and slip in this window remain sensitive to time integration. No new plane formed during that sliding interval.

## Direction law, section geometry and cost

The material sections span the specimen perpendicular to the fixed incoming tangent. Each strip owns an age, signed slip, resistance and structural angle β relative to that tangent. With quadrature area Jq, fault-deformation rate wq and reference arclength Δsₖ = Δξₖ/cos βₖ, the section rate is Vₖ = Σq Jq wq / Δsₖ. Returning the same resistance fₖ to all its points gives Σₖ Δsₖ fₖ Vₖ = Σq Jq fₖ₍q₎ wq. This complete-section relation also supplies the candidate resistance for every trial angle. Only the **direction score** weights the squared onset-driving traction by the compact kernel ρ = [1 − |X − Xtip|²/R²]₊². The disk is fixed through each step and its retries. Centered stress moments preserve the score under near cancellation. [Formulation and code owners](tip_treatment/reference_sections.md).

The first interval uses the declared seed tip. Later intervals use the connected accepted front on the stored graph, with a phase tracking level of 0.05. That level locates the disk center. A prospective section needs positive phase and eligibility within the disk. Formed planes retain their histories. A loaded detached phase tail that developed physical fault history outside the disk caused rejection before publication. In a manufactured test the center followed a connected band and stayed at the seed when a gap separated it from a distant patch. In the short coupled experiments the center stayed at the seed. The historical 80/40 μs propagation endpoints had no eligible virgin section within 2ℓ under their original registration history. They remain a separate baseline. [Frozen archive analysis](../../../benchmarks/reconstructed_fault/particle_free/reference_sections/pilot_frozen.py).

The frozen radius test compared ℓ, 2ℓ and 4ℓ on the same state. All three selected −8.5° at the most strongly driven prospective station. The 2ℓ disk covered all 1,066 samples used to identify the diffuse tip region, while the ℓ disk missed 232. Direct remote contributions vanished outside the disk, although deformation elsewhere on the same section could still change its resistance. This supports **R = 2ℓ as a trial length for this specimen**. [Radius measurements](../../../build-particle-free/localized-sections/frozen-final.json).

For oblique faults the mesh-scale discretization uses one state per equal projected strip, with native integration split at each cell’s own corners and intersecting strip boundaries. On the unchanged 8 × 4 mesh at 30°, the independent polynomial integral agreed to 2.45 × 10⁻¹⁵. At 15 ms, integrated slip differed by 0.47% and added fracture energy by 0.91% from the more expensive exact-line result. The first-step tangent-action error was 7.53 × 10⁻⁹ relative. Rotated-box tests at 90° and 30° preserved formed flags and gave maximum back-rotated stress differences of 2.59 × 10⁻¹² and 7.36 × 10⁻⁹ Pa. The oblique control passed forced rejection, rollback, retry and restart. A two-ended exact-line control on the 16 × 8 mesh genuinely reached the angular boundary in a weakly driven section. That failed evidence limits extrapolation to narrower strips. [Fixed-frame baseline](TIP_REFERENCE_SECTIONS_REPORT.md) · [Independent edge scan](../../../benchmarks/reconstructed_fault/particle_free/reference_sections/edge_score.py).

| Geometry at 30° | Local-strip points | Exact-line points |
|---|---:|---:|
| 8 × 4 coupled control | 4,968 | 8,640 |
| 256 × 128 geometry only | 5,149,944 | 217,314,432 |

The larger local-strip geometry took 0.098 s on two ranks. One native history at that size requires 1.48 GB in aggregate, before candidate states, FE operators and checkpoint gathering. The exact-line figure is a count. Coupled server cost and checkpoint size remain unmeasured. [Geometry counter](../../../benchmarks/reconstructed_fault/particle_free/reference_sections/geometry_cost.cc).

Across seven loaded single-front steps, median direct search cost was 0.26%, with a maximum of 0.62%. Matched fixed-angle replays reached equivalent physical endpoints. Median effective overhead was −0.94%. The most negative comparison (−66.9%) had the same Stokes solve count and endpoint, with a longer time per iteration in the replay. Three forced-rejection comparisons gave +0.79%, +10.48% and +42.1%. Direct cost and effective overhead overlap. The measurements support retaining the integrated law, with substantial wall-time variability on retries. The tested localized Coulomb alternative used μdir = μ₀ = 0.6 and cost +2.25% and +2.70% in two matched replays. Adopting it would change the orientation hypothesis. [Per-step timings, matched endpoints and retries](../../../build-particle-free/localized-sections/single-front.json).

## Two widths on the same bulk mesh

Both experiments used a 2 × 1 m box with 16 × 8 Cartesian cells, a 30° single-ended seed and R = 2ℓ = 0.5 m. The seed reaches the upstream boundary and its forward tip is at ξ = 1.616025 m. The tests used the same DG Bernstein seed coefficients, mixed shear and normal loading, 200 Pa prescribed confinement, free phase evolution, six-point Gauss rules, 5 ms midpoint steps and two bound ranks. Each coarse interval contains two equal fine strips with independently evolving states. Halving the width increased native material points from 19,944 to 25,992.

| Interval | Observable | 18 strips, Δξ = 0.1240 m | 36 strips, Δξ = 0.0620 m |
|---|---|---:|---:|
| 0–5 ms | Eligible / newly registered | 5 / 3 | 9 / 5 |
| | Last newly registered angle | −8.47183° | −8.47153° |
| | Minimum raw score-maximizer clearance from ±45° | 13.01° | 5.63° |
| | Signed integrated slip increment (m²) | 7.79244 × 10⁻¹⁶ | 7.31289 × 10⁻¹⁶ |
| | Integrated absolute local deformation (m²) | 8.56491 × 10⁻¹² | 8.80096 × 10⁻¹² |
| | Added fracture energy (J/m) | −2.17976 × 10⁻¹⁰ | −2.22826 × 10⁻¹⁰ |
| 5–10 ms | Eligible / newly registered | 2 / 0 | 4 / 0 |
| | Minimum raw score-maximizer clearance from ±45° | 36.8242° | 36.8236° |
| | Signed integrated slip increment (m²) | 3.74890 × 10⁻⁵ | 3.74285 × 10⁻⁵ |
| | Peak stage slip rate (mm/s) | 5.3420 | 5.3992 |
| | Added fracture energy (J/m) | 5.55047 × 10⁻⁶ | 5.51560 × 10⁻⁶ |
| | Physical-work residual (J/m) | +3.96009 × 10⁻⁶ | +3.93815 × 10⁻⁶ |
| | Ledger error after numerical terms (J/m) | −4.35 × 10⁻¹⁵ | −5.34 × 10⁻¹⁶ |

**First registration, 0–5 ms.** A 257-angle independent scan found the fine strip nearest the angular boundary at ξ = 1.76704 m. Its best score exceeded the best edge by only 1.07 × 10⁻²¹ J/m. The existing tie rule retained the incoming 0° plane. This explains why the smallest clearance of the *stored* planes was about 36.53° at both widths even though the raw margins differed. The incoming seed already supplied positive DG phase in parts of the first two registering coarse strips. In the last registering coarse strip, initially intact at all points, peak endpoint phase was 3.70 × 10⁻⁶ at both widths. Only its upstream fine child registered. That split accounted for 99.99% of the 6.15% difference in signed creep slip. The two phase fields differed by just 1.31 × 10⁻¹² m in spatial L², against an actual increment norm of 0.01794 m. Registered-strip counts alone overstate the spatial difference in damage. [Angular profiles](../../../build-particle-free/strip-width/angles.png) · [Station measurements](../../../build-particle-free/strip-width/comparison.json).

**Appreciable sliding, 5–10 ms.** The fine run continued from its own 5 ms checkpoint and was compared with the retained coarse endpoint. Summing fine-child slip over each common coarse interval gave an absolute difference summed over the specimen of 0.2642% of coarse slip. Both peak rates occurred in the same coarse strip behind the initial tip. The endpoint phase difference was 5.23 × 10⁻⁹ m in L², against a coarse increment norm of 0.01149 m. Every eligible virgin strip remained intact, with zero slip and resistance. Independently maximizing its localized squared shear traction reproduced all prospective angles within 0.001°. These are measurements of plane selection during sliding. [Slip by position](../../../build-particle-free/strip-width/sliding10/slip.png) · [Common-interval and work measurements](../../../build-particle-free/strip-width/sliding10/comparison.json).

## Fracture-work accuracy during sliding

The physical-work residual subtracts stored-energy change and the three physical dissipations from external work. From 5 to 10 ms the midpoint phase remainder accounted for that residual to within 2 × 10⁻¹⁴ J/m. The residual was 71.35% and 71.40% of added fracture energy at the two widths, over 100 times the difference in added fracture energy between them. The small final ledger errors show that the numerical remainder is accounted for. Interpreting added fracture work requires measuring how that remainder changes with timestep.

## Bounded timestep comparison from the accepted 5 ms state

**Halving the timestep increased both the slip and the physical-work residual.** The 18-strip reference took one 5 ms step to 10 ms. The replay reached the same endpoint time in two accepted 2.5 ms steps from the reference run’s own parent, `localized-oneo1/output/restart/02`. It used the identical executable, checker, fixture, two bound ranks, checkpoint frequency and physical acceptance tests. The restart hook set the pending first interval to 2.5 ms. Both runs preserved formed planes and registered zero new sections.

| 5–10 ms observable | One 5 ms step | Two 2.5 ms steps |
|---|---:|---:|
| Signed integrated slip increment (m²) | 3.74890 × 10⁻⁵ | 5.80075 × 10⁻⁵ |
| Added fracture energy (μJ/m) | +5.55047 | +15.04138 |
| Spatial L² norm of total phase increment (m) | 0.0114891 | 0.0151855 |
| Maximum sampled total phase increment | 0.0798066 | 0.0798076 |
| Signed physical-work residual (μJ/m) | +3.960087 | +6.782957 |
| Accumulated absolute physical-work residual (μJ/m) | 3.960087 | 6.782957 |
| Signed midpoint compliance remainder (μJ/m) | +3.960087 | +6.782315 |
| Accumulated absolute midpoint remainder (μJ/m) | 3.960087 | 6.782315 |
| Absolute residual / added fracture energy | 71.35% | 45.10% |

The signed slip is the sum of Δsₖ times each section's accepted slip increment. Phase increments use the common 5 ms parent and the same native area quadrature on both paths. The L² norm of the difference between the endpoint phase fields is 0.00742665 m. Their nearly equal peak increments conceal a maximum pointwise difference of 0.0633307. The half-step solution at **7.5 ms** was already within 3.47 × 10⁻⁷ m in phase L² of the reference at **10 ms**. The second half step then produced a further phase increment of norm 0.00742678 m.

| Accepted interval (ms) | Peak stage rate (mm/s) | Integrated slip increment (m²) | Added fracture energy (μJ/m) | Physical-work residual (μJ/m) | Midpoint remainder (μJ/m) |
|---|---:|---:|---:|---:|---:|
| 5–7.5 | 1.3995 | 5.30779 × 10⁻⁶ | +0.134782 | +0.093895 | +0.093253 |
| 7.5–10 | 14.9161 | 5.26998 × 10⁻⁵ | +14.906594 | +6.689062 | +6.689062 |

For each accepted interval the physical-work residual is external work minus stored-energy change and bulk viscous, cohesive viscous and fault dissipation. We accumulated its signed value and its absolute value independently. Both half-step residuals were positive. The separately recorded midpoint remainder integrates compliance work minus midpoint phase driving times the endpoint phase increment. The 0.000642 μJ/m difference between the refined residual and the remainder is accounted for by the exported phase and mechanical residual work within ledger precision. The largest absolute ledger error was 9.84 × 10⁻¹⁵ J/m. [Per-step data and endpoint comparison](../../../build-particle-free/strip-width/timestep10/comparison.json) · [Reducer](../../../benchmarks/reconstructed_fault/particle_free/reference_sections/midpoint_timestep.py).

**Decision.** The work-residual fraction fell because the added fracture energy increased by a factor of 2.71. The absolute work residual increased by a factor of 1.71, so the condition for the four 1.25 ms steps was not met. The timestep comparison ends here. The subsequent audit found identical lower/free sets and verified the physical and stage clocks. Both first stages approach a fracture-energy minimum halfway along their phase increment, then extrapolate past it. The large second-half work remainder was independently recovered in already damaged material. [Phase constraints, clocks and spatial work audit](MIDPOINT_PHASE_AND_WORK_AUDIT.md).

## Reproduction and provenance

The measurements used branch `pf-rsf-particle-free` at `c407e612bf7c9aa1a9c0fc60fd623208dfd53b39` with local changes. The localized-selection executable and checker have SHA-256 prefixes `e640fa627141` and `977328e5d1a4`. The width-comparison pair has prefixes `6b758a7b857c` and `bffa8d70fdb3`. The default-width rebuild reproduced the earlier first-interval point, section and energy tables byte for byte. A cross-width restart rejected the section-count mismatch before solving. The retained physical propagation executable has prefix `5e6bdb65646c` and belongs to the historical full-section direction law.

First-registration outputs are in `localized-width1` and `localized-width2` under `build-particle-free/bulk-tip/runs/`, each with its generation 02 checkpoint. At 10 ms the reference is `localized-oneo2` and the fine continuation is `localized-width2-sliding10`, generation 03. The manifests record full hashes, inputs, logs and checkpoints. [Reproduction commands](../../../benchmarks/reconstructed_fault/particle_free/reference_sections/README.md) · [First-interval identities](../../../build-particle-free/strip-width/identities.json) · [Sliding-interval identities](../../../build-particle-free/strip-width/sliding10/identities.json).

The timestep replay is `build-particle-free/bulk-tip/runs/strip18-midpoint-half/` with its 7.5 and 10 ms exports and terminal restart generation 01. Both sides used the localized-selection executable and checker. [Matched parent, runtime and endpoint identities](../../../build-particle-free/strip-width/timestep10/identities.json).
