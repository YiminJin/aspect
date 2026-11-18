# Peak-.60 overnight time and axial-mesh comparison

**Close the fixed-direction synthetic control for a qualitative propagation demonstration.** Connected appreciable damage ahead, passage of the absolute phi=.05 contour, and positive contour displacement survive the completed time and axial-mesh comparisons. Quantitative onset, a converged displacement and rupture speed remain unqualified. At .080 s the tested advances from the seed contour span .1868–.1962 m.

All selected intervals complete: A has eight dt=.00125 intervals from .060 to .070; B starts fresh on 256×128 and reaches .080 at dt=.005; C continues the accepted half-step .070 state through four dt=.0025 intervals to .080. A required one bounded continuation after its initial cap. The first C launch failed before stepping on a clipped pending clock; a separate benchmark checker repairs that scheduling case. Failed evidence is retained.

## Configuration and identity

Entry remains `pf-rsf-particle-free`, `b32a2d4f6e21c219f9af786d1b6e3725ea78a5a4`, with incoming manual/dirty work preserved. No core or CMake change, core rebuild, constitutive change, tolerance relaxation or solver redesign. All runs use true stress, native DGQ2 Bernstein phase, midpoint, positive-metric-first search with exact polishing, Q6 material integration and Q5 SIPG. Phase KKT remains 1e-7 and mechanics 1e-9. Full bulk cohesive history/energy and the additional fixed-reference section-RSF closure are retained. This is a synthetic verification specimen, separate from mature-fault production and general turning/branching.

Box 8×4 m; G=1e6 Pa, eta=1e8 Pa s, Gc=2/3 J/m², ell=.125 m, strength=1000 Pa, m=4, Dc=.0005 m, pN=200 Pa, initial age=.5 s, mu0=.6, V0=.001 m/s, a=.015, b=.02, radiation coefficient=1000 Pa s/m. All walls impose v=(.01(y−2),0). Native physical bounds and irreversibility, the endpoint conventions and all acceptance checks are unchanged. See the [baseline](RESOLUTION60_FREE_GROWTH_REPORT.md), [half-step report](RESOLUTION60_HALFSTEP_REPORT.md) and [tip guide](TIP_TREATMENT.md) for equations.

Protected executable SHA256: `034d2a848a3a10c496fa097475f31497c00bc68c3456fb6d87c1ee700f453c27`. Protected checker: `b800ef431a2469760efd16befcba94e8cd743156629b55e4c921f769255d9d84`. C-only terminal-clock checker: `53f8c364cbe3b569206d1c4dbf19653184753147406424c9d9f94c68dee8fac4`. Binary identities are independent of HEAD and the cached executable banner.

| Run | Mesh; hx,hy (m) | Accepted history | Process minutes | Status |
|---|---|---|---:|---|
| A | 128×128; .0625,.03125 | Saved step12/.060 → step19/.06875 | 75.01 | Initial 75-minute cap; 7 accepted |
| Afinish | 128×128; .0625,.03125 | Copied accepted step19 → step20/.070 | 24.62 | Exit 0 |
| B | 256×128; .03125,.03125 | Own equilibrated t=0 seed → .080 | 159.60 | Exit 0 |
| C | 128×128; .0625,.03125 | Half-step step16/.070 → step20/.080 | 57.61 | Exit 0 |

One simulation at a time, two MPI ranks, one numerical thread each; at most one compression and one analysis worker accompany them. The checker build used one compiler worker alongside the two-rank continuation. A and its completion cost 99.63 process minutes including the discarded original .070 trial. B costs 159.60 minutes; C 57.61 minutes. These are process times, excluding final prose work. No simulation remains.

## Measurements and initial-field check

Each mesh uses its own actual equilibrated phi0. Phase norms use common physical Q6 points on the refined cell partition and exact native polynomial evaluation. A pair ratios share the finest change-since-.060 normalization at each time. B uses its fine field’s change since initialization. C uses its change since the half-step .070 parent; that parent already differs from the coarse .070 state, so the late comparison includes inherited earlier time sensitivity. CSV columns explicitly identify the normalization start and denominator. A short common-parent replay and two meshes do not establish a general convergence order.

Section slip/age are observed using the piecewise P5 interpolant through six native section quadrature values on a common axial partition. The observer reproduces native values and the retained half-step norm. This interpolant is an observation convention, not a transferred physical history. V is averaged over matching .005 s intervals. Refining x also refines section quadrature/history sampling (768→1536 sections); normal spacing is fixed.

Generated native coordinates verify B hx=hy=.03125 on both ranks; all-cell coordinates, cell counts, masses and domain measure are checked in reduction. Seed phase L2 difference is 2.92423e-5 m (.008351% of refined seed norm), maximum sampled difference .00019423. Initial fracture energies are 1.12091361669/1.12091424398 J/m. Thus the represented seeds are close but are not assumed identical.

Exact piecewise-Q2 superlevel intervals retain both DG traces at y=2 and cuts y=2±hy/2 across x=2 and x=6. A face joins only when both traces are strictly positive above threshold; exact touching/double-root tests pass. The legacy helper default is retained. The full 2D .05 contour is filtered through seed connectivity using overlapping native face superlevels and single-component cell checks. Subdivision 16/32 tests topology and extraction separately from physical refinement.

## Common-time comparisons

L2 is the two-dimensional area norm (phase L2 unit m). Percentages are differences divided by the indicated fine change, not errors against an exact solution. The common-time CSV also contains slip, age and matched-interval V norms.

| Comparison | Time | Normalization begins | Phase L2 difference (m) | Relative difference | Finer/coarser pair ratio |
|---|---:|---:|---:|---:|---:|
| base/half | 0.065 | 0.060 | 0.0005390675 | 15.0184% | 0.4668 |
| half/A | 0.065 | 0.060 | 0.0002516292 | 7.0104% | 0.4668 |
| base/half | 0.070 | 0.060 | 0.0006630421 | 7.4424% | 0.6003 |
| half/Afinish | 0.070 | 0.060 | 0.0003980355 | 4.4678% | 0.6003 |
| base/B | 0.065 | 0.000 | 0.0001140376 | 1.5809% | — |
| base/B | 0.070 | 0.000 | 0.0001852906 | 1.5429% | — |
| base/B | 0.075 | 0.000 | 0.0002211351 | 1.0780% | — |
| base/B | 0.080 | 0.000 | 0.0004075816 | 1.0665% | — |
| base/C | 0.075 | 0.070 | 0.001405925 | 13.9775% | — |
| base/C | 0.080 | 0.070 | 0.003012997 | 9.8164% | — |

A’s finer-pair differences shrink at both times: ratios .4668 and .6003. This supports improved state accuracy, while the finest-pair change at .070 remains 4.47%. At .080, the spatial difference is 1.07% of full growth; the late temporal difference is 9.82% of the change since .070 (absolute L2 .003013 m versus spatial .0004076 m). Time sensitivity dominates these comparisons.

| At .080 | Slip L2 difference | Age L2 difference | Matched-interval V L2 difference |
|---|---:|---:|---:|
| base/B | 0.0716% | 0.0465% | 0.7910% |
| base/C | 0.6869% | 0.5413% | 4.4272% |

## Connected damage and absolute front

The table gives the exact native upper-axis crossing at x≈6. Initial contour positions are 5.92185466 m on the coarse mesh and 5.92169979 m on B. Advance is measured from each mesh’s own contour; signed support distance is from x=6. Delta=1e-4 extent is a separate connected-damage diagnostic.

| Case/time | Absolute .05 position (m) | Advance (m) | Beyond support (m; /hx) | Delta=1e-4 extension (m; /hx) |
|---|---:|---:|---:|---:|
| base / 0.0700 | 5.93944124 | 0.01758658 | -0.06055876; -0.9689 | 0.05750727; 0.9201 |
| base / 0.0750 | 5.98673348 | 0.06487882 | -0.01326652; -0.2123 | 0.12126176; 1.9402 |
| base / 0.0800 | 6.10867797 | 0.18682331 | 0.10867797; 1.7388 | 0.24578072; 3.9325 |
| B / 0.0700 | 5.93929842 | 0.01759862 | -0.06070158; -1.9425 | 0.05664851; 1.8128 |
| B / 0.0750 | 5.98848290 | 0.06678310 | -0.01151710; -0.3685 | 0.12024786; 3.8479 |
| B / 0.0800 | 6.10906335 | 0.18736356 | 0.10906335; 3.4900 | 0.24535781; 7.8514 |
| C / 0.0750 | 5.99399187 | 0.07213720 | -0.00600813; -0.0961 | 0.16730154; 2.6768 |
| C / 0.0775 | 6.04692296 | 0.12506829 | 0.04692296; 0.7508 | 0.22646639; 3.6235 |
| C / 0.0800 | 6.11803069 | 0.19617602 | 0.11803069; 1.8885 | 0.30222069; 4.8355 |

Both spatial meshes cross the absolute .05 level beyond the original support in (.075,.080]. C narrows that bracket to (.075,.0775]. All observed .05-positive cells belong to the seed component. The 2D left/right advances agree within 3.6e-14 m; full-curve and axis observations are retained separately. Maximum 16/32 extraction change is 2.093e-05 m, far below the physical advance and time-pair difference. No smoothing or phase clipping is used.

At .080, halving hx changes advance by .540 mm (.29%); halving dt changes it by 9.353 mm (5.01%). The much lower-amplitude delta=1e-4 extent is more sensitive to time: .2458→.3022 m (23%). The contour’s passage is robust; these differences preclude treating its measured length or onset as a converged quantity.

## Regional fracture, work and signed deformation

Regional energies use .5 B_i(F_i+K mass_i), K=2, including SIPG face contributions. They telescope to the native fracture increment. Shared-face allocation can be signed; it is not an independent positive fracture density. Remote and other increments remain zero in every completed case.

| .080 case | Central added Ef | Old taper | Initially intact axial | Max axial delta phi | W/hy |
|---|---:|---:|---:|---:|---:|
| base | 0.032112032 | 0.01791518 | 0.0073300673 | 0.11123438 | 2.093576 |
| B | 0.032091568 | 0.018216248 | 0.0073032936 | 0.1100358 | 2.093708 |
| C | 0.032616558 | 0.019194676 | 0.0089057983 | 0.11358255 | 2.098882 |

Energies are J/m, from each original seed. C’s axial added fracture is 21.5% above the coarse result despite similar peak phase increment; its distribution and extent retain time sensitivity. The normal pulse broadens from W/hy=1.394, rather than collapsing. Fixed hy and two axial meshes do not qualify normal resolution.

Physical-work remainder is Wext−deltaE−Db−Dc−Df, excluding the numerical phase remainder. The near-zero scheme ledger includes that remainder and is not the physical-work accuracy measure.

| Matching window / case | Added fracture (J/m) | Signed remainder | Accumulated absolute | Remainder / added Ef |
|---|---:|---:|---:|---:|
| 0.060–0.070 / base | 0.0096378333 | 2.1870836e-05 | 2.1870836e-05 | 0.2269% |
| 0.060–0.070 / half | 0.010004641 | 5.8391422e-06 | 5.8391422e-06 | 0.0584% |
| 0.060–0.070 / Afinish | 0.01027365 | 1.4631665e-06 | 1.4631665e-06 | 0.0142% |
| 0.000–0.080 / base | 0.057357279 | 7.2633724e-05 | 7.2633724e-05 | 0.1266% |
| 0.000–0.080 / B | 0.057611109 | 7.2691816e-05 | 7.2691816e-05 | 0.1262% |
| 0.070–0.080 / base | 0.04477107 | 3.9028798e-05 | 3.9028798e-05 | 0.0872% |
| 0.070–0.080 / C | 0.047764015 | 1.2548543e-05 | 1.2548543e-05 | 0.0263% |

Signed and accumulated absolute sums essentially coincide: cancellation does not conceal a larger defect. A’s remainders decrease by approximately four with each halving; C’s late remainder decreases 3.11×. C’s complete inherited-history remainder is 3.0121775e-05 J/m, 0.0496% of total added fracture. Pre-.060 history remains coarse and common; it is not retrospectively refined.

At first connected delta=1e-4 detection (upper-axis cut), absolute axial energy and full inherited accumulated global remainder provide scale:

| Case | Detection bracket (s) | Axial added Ef (J/m) | Absolute global remainder through detection (J/m) |
|---|---|---:|---:|
| base | (0.06000, 0.06500] | 2.609241e-06 | 2.015237e-05 |
| A | (0.06125, 0.06250] | 7.145889e-07 | 1.200845e-05 |
| B | (0.06000, 0.06500] | 7.087376e-06 | 2.017831e-05 |

B also retains a .060 precursor: max axial delta=4.44e-10 and axial energy 3.56e-13 J/m. That does not establish meaningful onset. Global work agreement is not a local onset-energy certificate.

No signed fault deformation is clipped. At .080, B/C have 28/18 sections with opposite-sign rates at the retained detector threshold, and no detected multiple active bands. Integrated negative w is −5.08e-8/−2.61e-7 m²/s against positive .04743/.05318; cancellation fractions are 2.14e-6/9.82e-6. Last-interval negative local friction/radiation work is −3.38e-8/−8.76e-8 J/m, while global section dissipation remains positive. These tiny signed contributions follow the work-adjoint common section force; pointwise positivity is not asserted. Raw signed histories and the full bulk cohesive inventory are retained.

## Restart, preservation and reproducibility

A’s initial 4500 s cap stopped an unaccepted .070 trial. Its accepted step19 restarts exactly, and one 1477 s continuation completes the eighth interval. The original failed attempt remains in its log and process cost. C’s first launch exits before stepping because its .070 terminal snapshot saved a pending dt=1.3877787807814457e-17 s; the reduce-only override properly rejects .0025.

The C-only checker hook requires a consistent native accepted parent, a roundoff-sized positive terminal interval and a requested dt no larger than the last physical dt. It changes only simulator pending time/dt to .0725/.0025; accepted material time, fields, old_time_step and checkpoint bytes remain untouched. Restore stress/cohesive/phase/slip/age differences are exactly zero. The first actual endpoint is .0725; all four physical intervals are .0025. Existing core guards and the original checker remain unchanged. This is an explicit benchmark scheduling extension, not production checkpoint conversion.

Native reconstruction, bounds/irreversibility, regional energy sums, section-rate/power identity, single publication, original KKT/mechanics and clock gates pass. Protected binaries, original .060/.070 parents, old step89 and unrelated manual source hashes are unchanged. Incoming checker source is recovered byte-for-byte by removing the opt-in hook. Final checkpoint paths/hashes and exact run commands are in `final-verification.json` and each `command.json`. No new final-snapshot recovery or whole-step rejection test is claimed.

Disk preparation losslessly compressed 294 closed diagnostic files, 5,489,027,152 raw bytes to 2,115,617,059 gzip bytes, checking raw SHA256 before replacement. New exports use the same verified workflow. Checkpoints/runtime libraries remain available. This creates no data loss and does not alter the inherited cleanup-loss record. One observer assertion initially used sequential weight summation; pairwise summation repairs roundoff with the same 1e-11 tolerance. That failure and retry are recorded.

Reading/reproduction: `run_resolution_matched.py` supplies `coarse` or the explicit x-refined `axial` fixture, distinct labels, retained binaries, two ranks and verified compression. `analyze_resolution60_overnight.py --plots` regenerates the tables, cuts and maps. The budget driver/watcher and bounded continuation scripts record exact commands. Existing labels are preserved; reproduction requires fresh labels/paths. The old preset `fine` refines y and is not this experiment.

Final C invocation uses the half-step `output/restart/02` parent, dt=.0025, end=.08, end-step=20, ranks=2, checkpoint-steps=1, `--checker build-particle-free/resolution60-overnight/checker/libbulk_tip_checks.release.so --extend-terminal-restart`. Build only that checker with `cmake --build build-particle-free/resolution60-overnight/checker --target bulk_tip_checks -j1`; its configured Aspect_DIR is the retained bulk-tip/native build. All other configuration and material inputs are frozen above.

Artifacts (PNG with matching PDF, plus CSV/JSON):

- [Common-time comparison table](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_overnight/common-time-comparison.csv) and [every accepted observation](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_overnight/history.csv).
- [Three timesteps at onset](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_overnight/timestep-onset-profiles.png), [two axial meshes](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_overnight/axial-mesh-profiles.png), [late timesteps](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_overnight/late-timestep-profiles.png): phi0, phi and delta together.
- [Both-end axis tracking](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_overnight/front-tracking.png), [full seed-connected absolute contour](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_overnight/global-absolute-front.png), [refined phase map](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_overnight/B-phase-map.png), [late half-step map](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_overnight/C-phase-map.png).
- [Energy, slip and width](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_overnight/energy-slip-width.png), [native cuts](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_overnight/endpoint-profiles.csv), [contour positions](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_overnight/contours.csv), [final verification](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_overnight/final-verification.json).

## Paper decision and next question

Retain the fixed-direction example for the qualitative claims: damage extends through connected DG superlevel support from the taper into initially intact material, reaches appreciable phase, and its seed-connected absolute .05 contour passes both original ends while remote material stays quiet. State the tested displacement range and crossing brackets. Do not claim precise onset, converged length, general convergence order, normal-mesh qualification, a rupture speed, general turning/branching or mature-production readiness. A smaller ledger remainder alone would not establish these claims.

The fixed-direction stage can close for this purpose. The next model question is how to choose/evolve general fault direction and carry RSF state while retaining the actual-measure work adjoint and full bulk cohesive energy. Resolve that closure before selecting one bounded turning experiment. Final examples, free-direction and unstable-RSF runs remain deferred; none is launched here.
