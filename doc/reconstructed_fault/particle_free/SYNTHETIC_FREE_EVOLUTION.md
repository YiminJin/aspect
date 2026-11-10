# Freely evolving synthetic specimen — 9 November 2026

**Both selected BE histories completed. The response is central-band broadening plus modest, low-amplitude axial extension. Stop this refinement sequence: numerical work loss still dominates the added fracture energy.**

This is a bounded fixed-reference, single-direction synthetic verification control. It does not establish resolved crack propagation, natural-fault calibration, general turning/branching, or mature-fault production readiness.

## Retained setup and initialization

Worktree `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`, entry and exit HEAD `5f1fc6f3c656dfb7e1ea03e2a6507258929dbca2`, with the incoming dirty specimen/checker/documentation work preserved. No core or checker rebuild, solver edit, parameter calibration, tolerance change, checkpoint conversion or commit. The executable identity is independent of HEAD:

- Core: `build-particle-free/bulk-tip/native/aspect-release`, SHA-256 `5d8f9ab1adc94662f7b96a9b63267657a18bc0674dfb51693e763f5877c18e3e`.
- Checker: `build-particle-free/bulk-tip/synthetic-checks/libbulk_tip_checks.release.so`, SHA-256 `1086b40397853e164ca9a93b41101f4621546ccf476018797d59a883f89d70ea`.

The [selected specimen](SYNTHETIC_PROPAGATION_BENCHMARK.md) and `bulk_tip/synthetic_propagation.prm` are unchanged: G=1 MPa, eta=100 MPa s, Gc=2/3 J/m², c=1000 Pa, ell=.125 m, m=4, Dc=.0005 m, confinement 200 Pa and initial age .5 s. The 8×4 m body has a 2 m seed half-length, compatible uniform-shear velocities on all four walls, h=.125 m (64×32 cells), native DGQ2 Bernstein phase and the retained section RSF closure. Direct physical stress/cohesive history, quadrature and BE endpoint conventions are retained.

Both runs start fresh at t=0. The ordinary zero-clock fracture minimization equilibrates the initial obstacle, followed immediately by freely evolving physical steps. The preflight-only `--fixed-phase` flag is absent; neither run imports a frozen .04 s parent. Initial stress, cohesive tensor, slip, age, Gauss fields, native phase coefficients and fracture gradients are identical between runs; exported seed data also exactly match the earlier preflight. Initial fracture energy is 2.157351615561 J/m. No phase or age reset occurs.

## Completed runs and acceptance

| Run | dt (s) | Intervals | End (s) | Wall time | Max mechanics residual | Max phase KKT | Max ledger defect |
|---|---:|---:|---:|---:|---:|---:|---:|
| `synthetic-growth-be-dt0025-local` | 0.0025 | 24 | .06 | 652.91 s | 4.343e-10 | 3.321e-08 | 5.745e-15 |
| `synthetic-growth-be-dt00125-local` | 0.00125 | 48 | .06 | 1235.38 s | 4.403e-10 | 6.680e-08 | 5.817e-15 |

Runs were sequential on two MPI ranks with one thread each. Total numerical wall time is 1888.29 s (31.47 min; about 1.05 rank-hours), within the budget. A single sandbox MPI socket startup failed before simulation; `synthetic-growth-be-dt0025` preserves that log. The two physical histories above both exit zero. No server jobs, mesh refinement, extra material cases, or third time grid were launched.

The existing mechanics tolerance 1e-9 and phase tolerance 1e-7 were retained. Accepted histories pass the existing checker; independent postprocessing verifies phase bounds/irreversibility, the section power adjoint and exact fracture-energy partition. Maximum spatial partition error is 4.885e-15 J/m; maximum section-work reconstruction error is 5.898e-17 J/m. These checks establish accounting and acceptance, not physical accuracy.

## What evolves

Using max phase increment >1e-4 as a declared diagnostic threshold, central-band evolution is first detected at .030 s on both grids: onset brackets are (.0275,.030] and (.02875,.030] s. Axial initially intact material first exceeds that threshold at .055 s: brackets (.0525,.055] and (.05375,.055] s. The axial onset is the same for thresholds 1e-6 and 1e-3. A fine-grid-in-time axial increment of only 2.6e-9 at .05375 s is not counted as meaningful extension. Thresholds define observations; solver tolerances are unchanged.

At .04 s the evolving band has already added about .0004882 J/m of fracture energy. Releasing phase only at that time would suppress earlier evolution. Later changes reach the old tip taper and then the adjacent initially intact axial cells. Transverse thresholds widen **inside** the old normal support; no meaningful phase appears outside that support. Boundary and remote phase increments remain exactly zero in both exported histories.

| Region, disjoint relative to initial support | Max delta phi, dt=.0025 | Max delta phi, dt=.00125 | Added fracture, dt=.0025 (J/m) | Added fracture, dt=.00125 (J/m) | Fine share |
|---|---:|---:|---:|---:|---:|
| Axial intact: x<2 or x>6 | 0.027897702 | 0.027635831 | 0.000529143087 | 0.000521598638 | 5.090% |
| Transverse intact: 2<x<6, abs(y−2)>.25 | 0 | 0 | 0 | 0 | 0.000% |
| Old tip band: 1.5≤abs(x−4)<2, abs(y−2)<.25 | 0.03371767 | 0.033439524 | 0.00279148342 | 0.00277347793 | 27.066% |
| Old central band: abs(x−4)<1.5, abs(y−2)<.25 | 0.036838815 | 0.036835332 | 0.006953643 | 0.00695204188 | 67.844% |

The full SIPG energy, including face terms, is partitioned with the exact quadratic chain rule `delta_E_i = .5*(g_i(final)+g_i(seed))*(B_i(final)-B_i(seed))`. Rows are assigned to their cells and the disjoint regions above. This is a signed regional attribution, not a unique positive pointwise face-energy density. No negative contribution is clipped. The exported volume density alone is insufficient; its differences are retained separately in JSON.

### Axial and transverse threshold motion

The native Q2 polynomial is recovered from all 36 Gauss samples per cell (maximum reconstruction defect below 2e-15), then quadratic roots locate thresholds on y=2 for the axial trace and x=4 for the normal trace. Both one-sided DG traces are included. These measurements avoid quantizing displacement to Gauss points, but do not establish sub-cell physical resolution. Left/right changes agree to roundoff.

| Phase level | Initial right axial position (m) | Coarse final (m) | Fine final (m) | Coarse outward shift / h | Fine outward shift / h |
|---|---:|---:|---:|---:|---:|
| 0.001 | 5.996066647 | 6.102639817 | 6.102538662 | 0.852585 | 0.851776 |
| 0.01 | 5.977292851 | 6.054290893 | 6.053971012 | 0.615984 | 0.613425 |
| 0.1 | 5.909843953 | 5.924338941 | 5.924198007 | 0.115960 | 0.114832 |
| 0.5 | 5.761553699 | 5.761553699 | 5.761553699 | 0.000000 | 0.000000 |

The phi=.01 front moves about .077 m per end relative to its own initial threshold, but ends only about .054 m beyond the initial zero-support endpoint x=6. The phi=.1 front remains inside the original support and phi=.5 has no axial motion. Thus extension is low-amplitude and less than one cell; visible fully developed crack growth is not a pass criterion. The central phi=.5 half-width changes from 0.068630937 m to 0.075425897/0.075425237 m, an increase of about .0544h per side.

## Timestep sensitivity

Weighted volume norms use actual Gauss JxW and section norms use the actual ds weights. The phase-increment relative norm divides by the fine solution’s change from the common initial seed, rather than its much larger pre-existing phase field.

| Difference, coarse minus fine | t=.04 s | t=.06 s |
|---|---:|---:|
| Max absolute phase difference | 8.08612e-07 | 0.000279614 |
| Phase-increment relative L2 | 0.000125726 | 0.00184754 |
| Max absolute cumulative slip difference (m) | 1.31962e-08 | 2.03303e-07 |
| Cumulative slip relative L2 | 2.1783e-05 | 0.000134678 |
| Endpoint section rate relative L2 | 0.00174605 | 0.01092 |
| Age relative L2 | 0.0032612 | 0.00381509 |
| Max absolute shear-stress difference (Pa) | 0.0135551 | 0.514262 |

Final max cumulative slip is 0.00064832591/0.000648291224 m; final max section rate is 0.0151824313/0.0152399699 m/s. Added fracture differs by 0.2650%. The largest phase discrepancy lies in the old tip band; the central-band maximum difference is 8.33e-06. These are measured two-grid sensitivities, not a formal convergence order or spatial qualification.

Both cases retain signed local deformation. At the final endpoint 126 sections have opposite-sign rates under the documented relative/absolute diagnostic threshold; no multiple active bands are detected. Integrated cancellation is 0.8521%/0.8468%. Local negative work is retained and the full signed work agrees with the section adjoint. No clipping or new localization rule was used.

## Physical work and numerical loss

Physical error is `Wext − delta_Etotal − Db − Dc − Df`, before subtracting BE losses or residual terms. All interval errors here are positive, so their signed cumulative sums and accumulated absolute sums coincide; both are explicitly computed. The window contains only intervals starting at or after .04 s and ending by .06 s.

| dt (s) | Interval | Added fracture (J/m) | Signed work error (J/m) | Accumulated absolute error (J/m) | Error / added fracture |
|---|---|---:|---:|---:|---:|
| 0.0025 | 0–.06 | 0.010274270 | 0.213323993 | 0.213323993 | 20.7629 |
| 0.0025 | .04–.06 | 0.009786119 | 0.068307573 | 0.068307573 | 6.9800 |
| 0.00125 | 0–.06 | 0.010247118 | 0.106743927 | 0.106743927 | 10.4170 |
| 0.00125 | .04–.06 | 0.009758893 | 0.034163849 | 0.034163849 | 3.5008 |

| Signed full-history accounting term (J/m) | dt=.0025 | dt=.00125 |
|---|---:|---:|
| BE bulk stress loss | 0.208823818 | 0.104471512 |
| BE cohesive traction loss | 0.00404004844 | 0.00203667102 |
| Nonlinear cohesive/phase remainder | -0.000163122696 | -8.03839774e-05 |
| BE fracture quadratic loss | 0.00062306433 | 0.000316020642 |
| Phase constraint/residual work | -1.58331388e-11 | -3.06385422e-11 |
| Internal minus external mechanical work | -1.85217038e-07 | -1.06657062e-07 |

Bulk stress loss dominates: it is .208824/.104472 J/m over the full histories and .066456/.033231 J/m over the late window. The small negative cohesive remainder is retained, not hidden in a positive dissipation. Reconstructing the ledger with its proper signs agrees below 2e-18 J/m. A near-roundoff ledger therefore coexists with a large physical-work error. The global error includes the bulk elastic loading of the whole specimen; it does not by itself measure a local phase error. The separate phase/slip comparison above is essential.

Halving dt almost halves the error, but the fine-grid error is still 10.42 times the full added fracture and 3.50 times the late-window addition. This meets the user’s stopping criterion. **No further BE refinement, mesh refinement or longer propagation run is justified in this task.**

## Cost, evidence and next issue

| Metric | dt=.0025 | dt=.00125 |
|---|---:|---:|
| Nonlinear iterations | 47 | 86 |
| Outer Krylov iterations | 8117 | 13471 |
| Active-set passes | 177 | 304 |
| Internal search failures, later recovered | 4 | 7 |

The coarse .0525 endpoint costs about 233 s; the fine .0525 and .05375 endpoints cost about 258 and 247 s. The existing shift/search corrections recover and pass the unchanged gates. These are costly internal search attempts, not accepted failed states. Easy preflight intervals would substantially underpredict active-phase cost.

Implemented solver behavior is unchanged. Locally verified: two complete free-phase histories, identical initialization, bound/acceptance and work/energy reconstruction checks. Physically compared: fixed-mesh timestep sensitivity, early broadening and weak late extension. Not qualified: resolved crack-front speed, spatial convergence, energetically accurate fracture growth, general geometry or production use.

**Next justified work is time-integration accuracy and affordability**, using these retained BE histories as controls. Establish improved state accuracy and physical-work accounting through point/small coupled controls before repeating a longer growth history. The shared midpoint option retains its separately documented near-intact search limitation; this experiment neither repairs nor qualifies it. Do not infer adoption from its smaller ledger remainder. Mesh and additional specimen cases remain deferred.

[Commands, measures and evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/synthetic_growth/README.md) · [analysis code](../../../benchmarks/reconstructed_fault/particle_free/bulk_tip/analyze_synthetic_growth.py) · [machine-readable comparison](../../../benchmarks/reconstructed_fault/particle_free/evidence/synthetic_growth/comparison.json). Exact run inputs, raw fields, solver logs, executable/checker identities and checkpoint generations remain under `build-particle-free/bulk-tip/runs/` in the two named fresh directories. Existing outputs, protected binaries and manual work remain preserved. Historical losses remain 39/45 pilot frames, early temporary logs and the unavailable old `bulk_continuation.zip`.

![History and energy comparison](../../../benchmarks/reconstructed_fault/particle_free/evidence/synthetic_growth/history.png)

![Phase and phase increments](../../../benchmarks/reconstructed_fault/particle_free/evidence/synthetic_growth/phase.png)
