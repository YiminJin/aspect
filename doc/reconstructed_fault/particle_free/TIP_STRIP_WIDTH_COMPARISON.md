# First registration on nested oblique material strips

Historical experiment record. The [combined report](TIP_DIRECTION_AND_STRIP_RESOLUTION.md) gives the current decision across direction selection, first registration and appreciable sliding.

Both widths accepted the first loaded midpoint interval. Halving the strip width preserved the turning angle of the last newly registered plane to 0.00030° and the endpoint phase to a spatial L² difference of 1.31 × 10⁻¹² m. It changed the signed integrated slip by 6.15% during near-canceling creep. Keep the mesh-scale width for this first-registration control. The data do not establish a server discretization for rapid slip or propagation.

## Matched native experiment

The 2 × 1 m box has 16 × 8 fixed Cartesian cells, h = 0.125 m and an incoming fault at 30°. The single seed reaches the upstream boundary. Its forward tip is (1.433013, 0.750000) m, or ξ = 1.616025 m. Both runs start at zero stress and apply the same mixed shear and normal loading, prescribed 200 Pa confinement, RSF law and free DG Bernstein phase. The direction kernel stays centered at the seed tip with R = 2ℓ = 0.5 m. Both use six Gauss points per axis on each local integration piece and one 5 ms midpoint interval on two bound MPI ranks.

The integer input `Bulk tip strip subdivisions` multiplies ceil(projected span / minimum cell altitude). Subdivision 2 bisects every original strip. Each child owns its own age, resistance, plane and native quadrature histories. The fine run read the coarse run’s initial DG coefficients by cell and local shape before initialization. Their initial fracture energies are identical at 6.08819323040947 J/m. The original-width run reproduced the previous single-front step's section, direction, energy, power and all four point exports byte for byte. A restart from the coarse checkpoint at the fine width failed on both ranks at the section-count check, before a physical solve. [Section geometry and history ownership](tip_treatment/reference_sections.md).

| At 5 ms | 18 strips | 36 strips |
|---|---:|---:|
| Projected strip width (m) | 0.124003 | 0.0620014 |
| Native material points | 19,944 | 25,992 |
| Seed / eligible virgin / newly registered strips | 13 / 5 / 3 | 26 / 9 / 5 |
| Last newly registered β | −8.47183° | −8.47153° |
| Minimum clearance of selected β from ±45° | 36.52817° | 36.52847° |
| Minimum clearance of independent score maximum | 13.01° | 5.63° |
| Signed integrated slip, Σ Δs Δslip (m²) | 7.79244 × 10⁻¹⁶ | 7.31289 × 10⁻¹⁶ |
| Integrated absolute local deformation, Δt Σ J abs(w) (m²) | 8.56491 × 10⁻¹² | 8.80096 × 10⁻¹² |
| Added fracture energy (J/m) | −2.17976 × 10⁻¹⁰ | −2.22826 × 10⁻¹⁰ |
| External work (J/m) | 0.0108997276774 | 0.0108997276719 |
| Signed ledger error (J/m) | −1.42 × 10⁻¹⁵ | −2.71 × 10⁻¹⁵ |
| Absolute friction work-adjoint defect (J/m) | 5.32 × 10⁻²⁶ | 2.68 × 10⁻²⁶ |
| Mechanical residual / phase KKT | 1.50 × 10⁻¹¹ / 6.20 × 10⁻⁹ | 1.51 × 10⁻¹¹ / 6.18 × 10⁻⁹ |
| Accepted-step solver wall time, one sample (s) | 1.709 | 1.770 |
| Accepted checkpoint (bytes) | 1,345,565 | 1,683,941 |

**Angular margin.** The independent pointwise scan uses the accepted first-stage predictor reconstructed from endpoint stress and cohesive strain. At every trial angle it recomputes resistance from the complete section and sums the localized squared residual in physical area. The 257 trial angles are 0.352° apart. The smallest coarse clearance occurs at ξ = 1.67404 m. With half-width strips the closest raw maximum moves to ξ = 1.76704 m, β = −39.375°. Its peak score is 4.37 × 10⁻²⁰ J/m and its advantage over the better sector edge is 1.07 × 10⁻²¹ J/m. Both near-edge stations retain the incoming 0° plane under the existing roundoff tie rule. The native guard admitted both solves. The earlier exact-line edge conflict at ξ = 1.76838 m remains a reason to inspect raw maxima as well as stored angles; that earlier screen used a two-ended seed. Filled markers in the figure identify newly formed planes and open markers identify prospective planes. [Angular profiles and score magnitudes](../../../build-particle-free/strip-width/angles.png) · [Earlier edge conflict](LOCALIZED_TIP_DIRECTION.md).

**What registered.** The initial support rule gives the incoming direction to contiguous supported stations through the declared tip. The taper already has positive DG phase in parts of the first two newly registered coarse strips. In the last coarse strip, ξ ∈ [1.86004, 1.98405] m, the initial phase is exactly zero at all material points. Its endpoint peak is 3.70 × 10⁻⁶ at both widths. Only the upstream child of that strip develops positive phase and registers on the finer partition. The downstream child remains virgin and has zero slip. The width of the registered region therefore changes by a half-strip even though the two endpoint phase fields agree closely. As a check on the different quadrature locations, we recovered each native Q2 field cell by cell and integrated the squared polynomial difference exactly. The common-mesh L² difference was 1.18 × 10⁻¹⁶ m initially and 1.31 × 10⁻¹² m at the endpoint, against an actual coarse phase increment of 0.01794 m in the same norm.

**Slip and work.** On that last coarse strip the slip integral was 9.591 × 10⁻¹⁷ m². Its two children contributed 4.796 × 10⁻¹⁷ m² and zero. This accounts for 99.99% of the whole-specimen signed difference of 4.7955 × 10⁻¹⁷ m². Absolute local deformation was about four orders of magnitude larger than signed slip, and changed by 2.76%. The fracture functional decreased slightly for both evolving fields. Its difference between widths was −4.85 × 10⁻¹² J/m, against 0.01090 J/m of applied work. This interval demonstrates registration and phase redistribution. It supplies no positive net fracture-work signal for qualifying extension.

**Selected follow-up (completed below).** Compare the first interval with appreciable sliding, 5–10 ms. The retained 18-strip endpoint at 10 ms has a peak rate of 5.34 mm/s and integrated slip of 3.749 × 10⁻⁵ m². Only the 36-strip continuation from its own 5 ms checkpoint is needed. Compare signed slip over common coarse-strip intervals, fracture work, and raw angular margin where new planes register before choosing the longitudinal resolution for a physical run.

## Evidence and reproduction

The worktree was `pf-rsf-particle-free` at `c407e612bf7c9aa1a9c0fc60fd623208dfd53b39` with the preceding geometry and localized-selection work present. The 32 entries in the earlier preservation manifest verified. The previously qualified executable and checker retain SHA-256 `e640fa627141ea98343fee14526c532f005281d336da32aa2d2f3467c5589b10` and `977328e5d1a4b80c06e8ac8734caa8bf0b77a98232bf095463b154b64f6f44e8`.

The matched release executable is `6b758a7b857c09dfb3a6d4033e7ffde4f9c79e7d52150aac75d55da199ac8d32` and the checker is `bffa8d70fdb3400a2b601d3343c03846b2b201aa03d52c0da678fd6ee537eba2`. Both are retained in `build-particle-free/strip-width/`. The native run directories are `build-particle-free/bulk-tip/runs/localized-width1`, `localized-width2` and `localized-width-mismatch`. Each holds its input, invocation, identities and log. Both accepted runs retain restart generation 02. [Reproduction commands](../../../benchmarks/reconstructed_fault/particle_free/reference_sections/README.md) · [Per-strip measurements](../../../build-particle-free/strip-width/comparison.json).

## Appreciable sliding from 5 to 10 ms

**Retain the mesh-scale strip width for exploratory sliding with this fixture.** Halving the width changed the signed slip increment by −0.1615%, the peak stage rate by +1.070%, and the added fracture energy by −0.628%. The remaining accuracy question for fracture work is the 5 ms time update. Both widths have a physical-work residual about 71% as large as their added fracture energy.

Only the 36-strip run was advanced. It restarted from its own generation 02 at accepted step 1 and published step 2 at 10 ms into a fresh directory. The comparison uses the existing `localized-oneo2` run from `localized-oneo1`. Its coarse 5 ms physical outputs match the preceding 18-strip control. The bulk mesh, reference orientation, material law, localized selection, loading and physical timestep are those of the first interval. Both continuations used two bound local MPI ranks and their original acceptance tolerances.

| 5–10 ms observable | 18 strips, retained | 36 strips, new |
|---|---:|---:|
| Signed slip increment integrated along the fault (m²) | 3.748901 × 10⁻⁵ | 3.742847 × 10⁻⁵ |
| Peak stage slip rate (mm/s) | 5.3420 | 5.3992 |
| Added fracture energy (J/m) | 5.55047 × 10⁻⁶ | 5.51560 × 10⁻⁶ |
| External work (J/m) | 0.0308242 | 0.0308273 |
| Friction / radiation dissipation (J/m) | 0.00466845 / 0.000175501 | 0.00466164 / 0.000176013 |
| Physical-work residual (J/m) | +3.96009 × 10⁻⁶ | +3.93815 × 10⁻⁶ |
| Physical-work residual / added fracture energy | 71.35% | 71.40% |
| Signed ledger error after numerical terms (J/m) | −4.35 × 10⁻¹⁵ | −5.34 × 10⁻¹⁶ |
| Absolute work-adjoint defect (J/m) | 3.47 × 10⁻¹⁸ | 1.04 × 10⁻¹⁷ |
| Mechanical residual / phase KKT | 5.24 × 10⁻¹⁷ / 1.45 × 10⁻¹² | 9.59 × 10⁻¹⁷ / 1.49 × 10⁻¹² |
| Eligible prospective / newly formed strips | 2 / 0 | 4 / 0 |
| Minimum raw angular clearance from ±45° | 36.8242° | 36.8236° |
| Solver wall time, one observation (s) | 3.128 | 2.827 |

**Slip over common intervals.** We integrated Δsₖ times each section's accepted slip increment, then summed fine children over their coarse parent's ξ interval. The sum of absolute bin differences was 9.9043 × 10⁻⁸ m², or 0.2642% of coarse slip. The largest absolute difference was −3.390 × 10⁻⁸ m² in the first strip at the upstream boundary, 3.22% of that strip’s slip. The two peak-rate stations (ξ = 0.6820 and 0.7130 m) lie in the same coarse strip, behind the initial tip at 1.6160 m. Integrated absolute local deformation agreed with signed slip to the precision shown. The 6.15% relative width difference in the first interval's tiny, canceling creep did not persist into appreciable sliding. [Common-interval slip figure](../../../build-particle-free/strip-width/sliding10/slip.png) · [All bin increments and directional scores](../../../build-particle-free/strip-width/sliding10/comparison.json).

**Direction and phase.** No newly registered plane has an angle to compare in this interval. All previously formed planes kept their angles. The four fine and two coarse eligible virgin strips remained fully intact throughout the stage and had zero fault deformation. Their complete-section RSF solves give zero resistance. For this observed state the pointwise onset score is a weighted integral of squared resolved shear. We reconstructed the midpoint predictor from the accepted stresses and maximized this score independently over the angular sector. Its raw maxima agree with every native prospective angle within 0.001°. The minimum margin exceeds 36.8° at both widths. The common-mesh endpoint phase difference is 5.23 × 10⁻⁹ m in L², against a coarse phase increment norm of 0.01149 m. The absence of registration bounds what this interval establishes about a new tip direction.

**Fracture accounting and next experiment.** Added fracture energy is the change in the native fracture functional between 5 and 10 ms. Physical-work residual is external work minus stored-energy change and the three physical dissipations. The reported midpoint phase remainder accounts for that residual to within 2 × 10⁻¹⁴ J/m. This numerical remainder is over a hundred times the difference in added fracture energy between widths. Before using fracture work to interpret growth, take two 2.5 ms intervals from the same accepted **18-strip, 5 ms parent** and compare at 10 ms with this retained coarse endpoint. Track slip, added fracture energy and the signed and accumulated absolute physical-work residuals. The present comparison supports the default strip width for exploratory integrated sliding and leaves server-scale propagation unqualified.

**Run identity.** The worktree remained on the branch and revision recorded above. The 36-strip continuation used the retained width-comparison executable and checker. The 18-strip reference used the preceding localized-selection executable and checker. The earlier 26-file width manifest and 36-file preservation manifest verified. The new input SHA-256 is `2d1a78722d8accd0455ed4ac52b0d249879aa0c15cdc388658964e5cc56405c3`. Generation 03, both rank exports and the run record are in `build-particle-free/bulk-tip/runs/localized-width2-sliding10/`. [Reproduction commands](../../../benchmarks/reconstructed_fault/particle_free/reference_sections/README.md) · [Analysis code](../../../benchmarks/reconstructed_fault/particle_free/reference_sections/sliding_width.py).
