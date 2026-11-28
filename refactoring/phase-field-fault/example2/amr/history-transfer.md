# Accepted growing-fault history across refinement

The midpoint path publishes one strip table, resistance-law state and work ledger per accepted physical endpoint. A mesh-only pass leaves these records and the physical clock in place. Refining transfers the last accepted FE vectors using the simulator's solution transfer. A rejected step also needs the older accepted FE vector, so that after remeshing the same extrapolation and old phase are available for the repeated interval.

Set `Material model / Phase field fault / Target strip width` in metres to fix the physical station partition. The implementation divides the original projected in-box range into equal widths no larger than this setting. Zero selects the legacy uniform-grid spacing. After refinement it rebuilds only the local cell intersections, native integration rules and ownership. The material checks the stored width and edges on restart and continues to read uniform v1 (RSF age per section) and v2 (law-owned state) archives.

## Transfer operator

The pre-refinement material signal registers a variable-size p4est attachment for each owned cell. A geometrically unchanged cell transfers its native records directly, including across an ownership change. A refined cell sends one reconstruction per physical strip intersection to the new owner. This requires no all-rank native array or geometric point search. The existing v1/v2 checkpoint reader/writer still uses its original gathered format.

On each old Cartesian cell intersected with strip k, the test functions are 1, parent reference x and parent reference y. All integrals use physical area. The old native rule supplies the source moments. The relaxed bulk stress components use affine trial functions. Cohesive strain, signed accumulated shear and accumulated absolute shear use phase times an affine trial function. The phase is evaluated from the accepted Q2 polynomial on each side of the mesh change. Projecting with the phase factor preserves exact zero at intact points and avoids imposing finite strain where the phase tends to zero and cohesive compliance vanishes. The resulting phase-weighted mass is assembled directly without dividing by small phase. For the scalar histories both affine slopes are reduced by a common factor if necessary to guarantee absolute shear ≥ |signed shear| at all clipped polygon vertices before multiplying by phase. This preserves the two zeroth moments. Do not require a limited slope to preserve its first moment. The source-side centered solve uses long-double accumulators on thin cuts. At least four Gauss points per chord and per station interval integrate the highest-degree phase-weighted test moments exactly on a Cartesian cut. The old uniform-grid quadrature requirement is unchanged.

The solution is not interpolated across strip boundaries. A new material quadrature point is evaluated using its old parent reference coordinate and exactly its native material ID. Applied to a generally nonlinear history, the projection can change energy at zero physical duration. A jump in DG phase can separately change the discrete fracture energy when the penalty length changes. Neither contribution is external work or time-integrated dissipation.

## Reproduce the loaded transfer and continuation

The point exporter checks all four possibly nonzero plane-strain stress and strain components (xx, yy, zz, xy). The production remap applies componentwise to the stored symmetric tensors. The benchmark sets a quadratic shear velocity on the walls to load a measurable in-cell stress gradient. It starts with 8 × 4 cells, uses a 30° physical seed and 18 fixed sections, and refines a portion of the accepted mesh after the 5 ms endpoint. The 16 × 8 control retains the same station resolution and material settings. Native tables contain the accepted phase, previous phase, tension-positive stress, cohesive strain, and signed and absolute shear at the cut-cell quadrature. A pre-solve observation after restart reads the older accepted vectors, before the next physical solve.

Run from the worktree root with a fresh destination. Each ASPECT process uses one numerical thread. The pre-AMR binary is the independent local uniform writer from the granite pilot. The optional final argument exercises the original native v1 archived generation as well as the v2 continuation.

```sh
cmake --build build-granite-amr/plugin -j1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 \
  python3 benchmarks/granite_fault_growth/run_transfer_cases.py \
  build-6b/release/aspect-release \
  build-granite-amr/plugin/libgrowing_fault_checks.release.so \
  build-bp3-fresh/install/bin/aspect-release \
  build-granite-amr/transfer-final \
  --v1-generation build-5a-repair/lifecycle2/clean/output/restart/03
```

The independent check fits the source moments by weighted rectangular least squares and evaluates each reconstructed tensor at each child point. Phase prolongation uses physical coordinates at off-node native points, a subdivided Bernstein enclosure, and an independent irreversible-increment comparison. A separate physical face integral visits each actual interior overlap once to measure the change in penalty energy. Strict continuation uses the existing `1e-12 + 1e-12 |reference|` allowance. The report preserves each failed column as well as its index and difference for a change of MPI layout.

## Measured zero-duration change at 5 ms

| Parent and resulting mesh | Ranks | Cells on a different rank than their parent | Δ bulk / J m⁻¹ | Δ cohesive / J m⁻¹ | Δ discrete fracture / J m⁻¹ |
| --- | ---: | ---: | ---: | ---: | ---: |
| RSF, 32 → 77 | 1 | 0 | −7.35893e−8 | −8.54947e−8 | +1.730443054 |
| RSF, 32 → 77 | 2 | 12 | −7.35893e−8 | −8.54947e−8 | +1.730443054 |
| RSF, 128 → 308 | 2 | 52 | −3.03943e−8 | −1.06225e−8 | +1.323739324 |
| Linear dashpot, 32 → 77 | 1 | 0 | −1.49334e−6 | −2.09018e−6 | +1.730451490 |

The two RSF 8 × 4 runs carried 18 strip records, 18 ages and 18 adiabatic friction stations. The dashpot uses prescribed constant 200 Pa and carries zero constitutive state values or adiabatic stations. In each case the section metadata and last physical work ledger compare exactly across the mesh change. The 8 × 4 RSF transfer carried signed shear of both signs (−1.16625e−5 to +1.21434e−5). At the 1,116 exactly intact destination points all damage-associated history is zero. Constant parent-cell stress would miss the 8 × 4 result by 8.724 Pa. The independently reconstructed point stress is within 2.71e−13 Pa of the actual tensor; the largest integrated stress moment defect is 3.56e−15 Pa m². The measured fracture-energy shift agrees with independent penalty integration to 3.3e−14 J/m on the serial 8 × 4 mesh and declines at the finer resolution.

An early unweighted affine reconstruction of cohesive strain passed the algebraic tensor moment checks but failed the physical energy test. On 128 → 308 cells it raised cohesive energy from 2.15190e−6 to 3.87953e−5 J/m as it extrapolated finite strain into nearly intact material. That scratch trajectory is retained separately in `build-granite-amr/history/out-h2-loaded/`. The phase-weighted result on the same parent is 2.14128e−6 J/m. The first, longer post-transfer trial on the 308-cell mesh was stopped while solving the additional 10 ms endpoint. The focused finer input terminates normally after the required 5 ms transfer (17 seconds on two ranks); the 32 → 77 trajectories run through 15 ms.

For the recorded run the first three complete parents are in `build-granite-amr/transfer-final/`; the termination-limited finer run and the restart cases are in `build-granite-amr/transfer-completion/`. The latter used the same command above with the last positional path changed to `build-granite-amr/transfer-completion` and `--parents build-granite-amr/transfer-final`. The pre-AMR writer SHA-256 is `836f7e773da8487e05dcd26d2b94d86656c5084d5f198a9827e70e7fa6907ce5`; the observer DSO SHA-256 is `0097908c0b3d09185c8a5cfd991fb7ad3e140a525f911d93a98c80f0606e4bc0`. The final reader with the order guard has SHA-256 `afe996fedf60d88c6a45e01bcd508157aaff6dbcd3f1fe3c431ca3b72998f547`.

## Restored state and later physical solves

All four refined restarts restored exactly the accepted native tensors and scalar histories keyed by cell and native quadrature row, endpoint and older FE phase, ordered physical strip records, law state, adiabatic stations when applicable, clock, and serialized work ledger before another nonlinear solve. Changing the reader layout moved 37 of 77 cells for each of 1→2 RSF, 2→1 RSF and 1→2 dashpot. The 1→1 RSF restart also had zero difference in every published field, strip and ledger column at both 10 and 15 ms.

| Same checkpoint, different layout | Strict columns outside budget at 10 / 15 ms | Largest strip-angle difference, rad |
| --- | --- | ---: |
| RSF, 1→2 | 0 / 3 | 4.30373e−9 |
| RSF, 2→1 | 3 / 2 | 3.39502e−9 |
| Dashpot, 1→2 | 2 / 2 | 5.77406e−9 |

These changed-layout solves pass the existing physical angle and arclength budgets at every station. The largest recorded nodal pressure difference was 5.55e−12 Pa. The strict outliers involve pressure, a native zz stress and the prospective angle and arclength in unformed strip 17. The exact first row for each affected column is in `history-transfer.json`, alongside every per-column maximum. A width change from 0.125 to 0.124 m was rejected on checkpoint loading even though it yielded the same number of strips. An attempted remesh using Gauss(3) was rejected before history transfer. The old v2 writer ran from initial conditions and resumed with the final reader to 10 and 15 ms; both endpoints met the strict continuation budget. The original v1 generation 03 from the 5A RSF run resumed with both binaries to 15 ms and also passed.

Starting the loaded fixture independently on 1 and 2 ranks has an earlier difference. Before the first remesh, strip 16 selected −0.485436860397 and −0.485436623503 rad at 5 ms. Its 2.36894e−7 rad separation exceeds the existing 1.48544e−7 rad angular allowance. The induced arclength difference is 1.75210e−8 m against 1.40300e−8 m. Both persist at 10 and 15 ms because the strip formed on both ranks' trajectories. No other strip angle or length exceeded its scientific tolerance. Native xy stress differs by up to 2.23e−7 Pa before transfer and 5.63e−6 Pa at 15 ms. Identical accepted-checkpoint restarts isolate transfer from the independently computed direction choice; the transfer makes no fitted-angle decision. The fresh 30° generally loaded case remains outside the cross-layout direction tolerance at that single strip.

The minimum admitted Gauss(4) rule was exercised in a separate fresh 32 → 77 serial calculation with the subsequent simulator build `b726cce858f38fe884fe70d3b06439b8b54efb19ed7e0cae5711590e50de6706`. Its signed tensor integral defect was at most 3.11e−15 Pa m² and its independently fitted point stress defect was at most 1.57e−13 Pa. Current and previous off-node phase errors were below 1.63e−15. The independently integrated penalty shift differed by 7.8e−15 J/m. The focused run uses the `gauss-four` input generated by the same driver above. As a negative control, running the final checker on `build-granite-amr/history/out-s-loaded` (the discarded unweighted remap) fails its pointwise tensor reconstruction at Axx with 4.65350e−6 strain error while its stress check still passes.
