# Final local timestep check of the fourfold-rate propagation pilot

**The separate region of fast slip beyond the right tip survives timestep halving on this mesh.** Both fine midpoints show a peak at about x = 6.08 m. The interval-averaged fine rate exceeds 5 mm/s on exactly the same 18 stations as the coarse rate. Over those stations, signed integrated slip changes by −0.28%. Damage on the upper flank also persists, but the number of initially intact cells reached by the connected 0.001 contour changes from two to three. This closes the local physical check through 6.40 ms. The imposed increase in boundary velocity, single spatial mesh and subcell tip motion leave spontaneous propagation and its speed unresolved.

![Midpoint slip rates, integrated slip and phase increments at the right tip](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/overnight-recovery/rate-final40/last-interval.png)

The dotted lines mark the original tip and 5 mm/s rate threshold. The bottom panels use one color scale for phase added since 6.32 ms. White lines show absolute endpoint phase contours and cyan marks the cell boundary between the initial seed support and initially intact material.

## Slip beyond the original tip

| Timestep | Midpoint (ms) | Interpolated 5 mm/s span (m) | Width (mm) | Peak (mm/s) | Rate at x = 6 m (mm/s) |
|---|---:|---|---:|---:|---:|
| 40 μs, first half | 6.34 | 6.0512–6.1238 | 72.7 | 6.44 | 3.48 |
| 80 μs | 6.36 | 6.0427–6.1328 | 90.1 | 7.34 | 3.90 |
| 40 μs, second half | 6.38 | 6.0306–6.1397 | 109.1 | 8.17 | 4.37 |

The peak strengthens and the threshold expands in both directions during the fine interval. Summing slip over stations above threshold at each stage gives 4.7341e−8 m² on the fine branch versus 4.9077e−8 m² on the coarse branch. Using the same 18 stations throughout removes the effect of changing membership. Across all stations the duration-averaged rate differs by 0.94% in weighted RMS relative to the coarse rate. [Station brackets, quadrature integrals and registered sections](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/overnight-recovery/rate-final40/secondary-group.json).

## Phase, fracture and work at 6.40 ms

| Change since the common 6.32 ms parent | 1 × 80 μs | 2 × 40 μs |
|---|---:|---:|
| Integrated slip, whole fault (m²) | 5.40122e−6 | 5.37161e−6 |
| Integrated slip, same 18 stations (m²) | 4.90773e−8 | 4.89422e−8 |
| Added fracture energy (J/m) | 5.23379e−5 | 5.28526e−5 |
| Signed physical-work error (J/m) | +3.5470e−10 | +9.9906e−11 |
| Accumulated absolute physical-work error (J/m) | 3.5470e−10 | 9.9906e−11 |
| Phase added in initially intact cells, area integral (m²) | 4.07004e−7 | 3.99998e−7 |
| Largest coefficient increment in initially intact cells | 5.53875e−4 | 5.13141e−4 |
| Advance of connected right φ = 0.05 contour (mm) | 0.2102 | 0.2508 |
| Initially intact cells reached at φ = 0.001 / 0.01 | 2 / 0 | 3 / 0 |

Halving the timestep changes total slip by −0.55% and fracture work by +0.98%. The work defects are below 0.001% of the added fracture energy on both branches. The area integral of new phase in initially intact cells changes by −1.72%. The spatial L² difference between the endpoint fields is 6.46% of the coarse phase increment from the parent. All native coefficient increments are nonnegative and the endpoint coefficients range from −2.71e−19 to 0.60. [Endpoint fields, per-cell statistics and work ledger](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/overnight-recovery/rate-final40/endpoint.json).

The additional fine-branch cell lies on the upper flank at (6.109375, 2.140625) m. Its largest phase coefficient reaches 0.001010 versus 0.000915 on the coarse branch. Both sampling densities give the same cell counts. At φ = 0.01, all connected damage still lies in cells with initial seed support. The φ = 0.05 advance differs by 40.7 μm between branches, about 19% of the coarse advance. Both advances are below 0.009 of a 31.25 mm cell.

The same six stations at each end registered on both branches. The fine branch had registered all twelve by 6.36 ms, with maximum sampled phase across any such section of 3.45e−13 at registration. The coarse maximum was 6.11e−12. The six right-hand stations are at x = 6.314–6.343 m, beyond the observed fast-slip group. Registration at these phase levels cannot substantiate extension to those stations. The physical evidence supports accelerated sliding and small connected upper-flank damage under the prescribed fourfold loading. It leaves the location of a developed advancing tip unresolved.

## Acceptance and cost

| Step and endpoint | Native step time | Line trials | Final mechanics | Final phase KKT |
|---|---:|---:|---:|---:|
| Coarse 80, 6.40 ms | 55.6 min | 95 | 1.55e−16 | 7.30e−8 |
| Fine 80, 6.36 ms | 51.5 min | 117 | 9.55e−18 | 8.69e−8 |
| Fine 81, 6.40 ms | 34.7 min | 229 | 2.57e−18 | 7.18e−9 |

Each fine step published once after passing the original mechanics (1e−9) and phase (1e−7) gates. The two fine steps used 1.55 times the coarse native step time for the same interval. The four-rank fine launcher exited with code zero after 86 min 48 s of elapsed run time. Its final snapshot is `build-particle-free/overnight-recovery/rate-final40/output/restart/01`. [Terminal record and checkpoint hashes](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/overnight-recovery/rate-final40/summary.json) · [Matched inputs](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/overnight-recovery/rate-final40/matched-inputs.json) · [Native solver log](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/overnight-recovery/rate-final40/native.log.gz).

## Matched experiment

The common parent is the published fourfold-rate step 79 at 6.32 ms. The coarse branch advances once to 6.40 ms and samples its rate at 6.36 ms. The fine branch advances through 6.36 and 6.40 ms and samples its rates at 6.34 and 6.38 ms. All physical increments in this comparison start at 6.32 ms. The fine branch inherits the fourfold prescribed boundary velocity already active at its parent. [Coarse result](STAGE4_TRUE_RESIDUAL_RESTART.md) · [Six parent hashes](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/overnight-recovery/rate-true-residual/parent79.json).

Both branches use the same executable (SHA-256 `5e6bdb65646c94610e2605675773d7701d87ea9819758e355be8ae8d95fa4014`), checker, four MPI ranks, 256 × 128 mesh, section stations, midpoint law and acceptance criteria. The worktree was on `pf-rsf-particle-free` at `c07eea8e77e02dd464f5e050e773083963c5ccb2` with local changes. The fine run copied the verified intermediate checkpoint into its own restart directory. Input differences are the timestep settings, terminal step number and output path. The native timestep override also changes from 80 to 40 μs. [Accepted parent and binary identities](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/overnight-recovery/rate-true-residual/comparison.json).

## Measurements

The secondary group is a contiguous sequence of stations entirely beyond the original right tip at x = 6 m with signed midpoint rate ≥ 5 mm/s. Its outer limits are bracketed by adjacent stations. Linear interpolation locates a crossing within each bracket. The stage-specific slip integral uses the actual section weights and midpoint duration, summing Δt ds V over qualifying stations. A second integral holds the coarse group's section IDs fixed, including any that fall below threshold in a fine stage. This measures how much the same material slipped over the common time window. The fine interval-mean rate is the duration-weighted average of its two stages.

An initially intact cell has all nine seed Bernstein coefficients zero. The physical area times the mean coefficient gives its exact phase integral on this affine grid. Connectivity is checked with both DG traces on grids of 8 and 16 samples per axis. The endpoint comparison uses the native fracture energy and tracks signed and accumulated absolute errors in external work less stored energy change and the three physical dissipations. [Endpoint reducer](../../../benchmarks/reconstructed_fault/particle_free/stage4_propagation_pilot/compare_flank_interval.py) · [Group and phase reducer](../../../benchmarks/reconstructed_fault/particle_free/stage4_propagation_pilot/compare_final_rate_interval.py).

## Reproduction

From the worktree root, use fresh run and output labels.

```sh
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py final-local-fourfold-repeat 9090 \
  python3 build-particle-free/overnight-recovery/launch-final40.py rate-final40-repeat \
  build-particle-free/overnight-recovery/rate-true-residual/output/restart/02 79 81 .00004 9000 4
```

To reduce the retained outputs and regenerate the figure, run from the worktree root.

```sh
pilot=benchmarks/reconstructed_fault/particle_free/stage4_propagation_pilot
evidence=benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/overnight-recovery/rate-final40
coarse=build-particle-free/overnight-recovery/rate-true-residual/output
fine=build-particle-free/overnight-recovery/rate-final40/output
seed=build-particle-free/stage4-propagation-pilot/analysis_views/through70
python3 "$pilot/compare_flank_interval.py" "$coarse" 79 "$coarse" "$fine" "$evidence/endpoint.json"
python3 "$pilot/compare_final_rate_interval.py" "$coarse" "$fine" "$seed" "$evidence"
```
