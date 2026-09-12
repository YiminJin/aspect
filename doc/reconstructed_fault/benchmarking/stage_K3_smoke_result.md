# K3: accepted-fixture one-rank Release smoke result

The explicitly approved second invocation **passes the bounded smoke** through
initialization and two real steps. It ran once in 43.412 s, with 585 MiB peak
RSS, below the 180-s cap. No automatic retry, production change, parameter scan,
MPI run, refinement or support adjustment followed. The first failed attempt
is preserved separately. This is a feasibility/feedback verification, not a
converged K3 reference or passage of the unresolved Gate K2.

## Execution and genuine convergence

The unchanged selected peak is .00225 m/s. Accepted times are 0, 2 and
**3.712347940221483 s**; real dt values are 2 and **1.712347940221483 s**.
The artificial 2-s initialization interval is not counted as physical time.
All 17 fault nodes are free at each accepted state; none is at V_min.

| Step | Absolute bulk residual | Bulk stopping target | Surface RMS residual (Pa) | Surface/fixed scale |
|---|---:|---:|---:|---:|
| 0 | 5.706e-12 | 1.587e-5 | 6.120e-8 | 6.430e-11 |
| 1 | 1.041e-10 | 3.411e-4 | 1.861e-11 | 1.743e-14 |
| 2 | 8.635e-11 | 3.192e-8 | 6.135e-11 | 9.228e-15 |

The bulk target includes the already accepted precision allowance; it was not
modified. There are **16 returned linear directions**, all with fresh residual
<= requested target. The largest fresh/target ratio is .97457. Initial phase
convergence reaches 1.220e-9 relative residual; the two real phase solves reach
1.738e-9 and 3.621e-12, below the unchanged 1e-8 criterion. Passing is based
on these criteria and per-state gates, not just process exit zero.

## Noncommitting phase probe and lifecycle

The corrected fingerprint handles pre-mechanics absence of initialized V. The
run verifies normal and injected-exception restoration before initial mechanics
and again during real-step probes. The forced path substitutes half-H, executes
production residual assembly into scratch data, then throws before returning.
All substituted H is restored; before/after live state/solver fingerprints
match. Stable particle IDs verify entry H equals the preceding committed H.
No live history field is refreshed merely for visualization.

The paired probes use the same current CPDI domains, before particle advection:

| Probe at step 2 | Production residual norm |
|---|---:|
| R(phi1; H0) | 9.6384631e-5 |
| R(phi1; H1) | 2.0085839e-3 |
| R(phi2; H1) | 7.2722463e-15 |

The H0 residual need not vanish on the later particle domains. Its difference
from the H1 result, the stable-ID handoff, and the subsequent converged solve
demonstrate actual history-driven phase feedback without exporting all CPDI
weights/gradients. The real-step exact Theta updates agree at vertices to
5.55e-17 s; particle H increments are nonnegative. At zero, supplied Theta=200 s,
initial H and particle stress remain retained rather than physically evolved.

The reconstructed fault has exactly unchanged coordinates and node count at
all three accepted states. No topology or geometry evolution was permitted.

## Supported integrals and seam behavior

Both diagnostics remain necessary. Full I_h is unchanged as a definition;
there is no tail renormalization. Containment is independently remeasured from
the exported Q1 profile at surface vertices and actual bulk quadrature columns.
The supported crack-strain integrals use the actual assembler QP locations,
associations and JxW; post-commit history diagnostics use saved old C/I_h.

| Step | Maximum omitted h fraction | Maximum total normalization error |
|---|---:|---:|
| 0 | 5.83027e-5 | 5.23649e-5 |
| 1 | 5.83182e-5 | 5.24831e-5 |
| 2 | 6.09998e-5 | **9.56659e-5** |

The final normalization passes **narrowly**, with only 4.334e-6 margin below
1e-4. This is not a claim of support convergence. At step 2 the ranges across
measured columns, in m/s, are:

- instantaneous integral: [.002257877449, .002257918025];
- signed history integral: [-8.872386e-8, -2.825469e-8];
- total supported integral: [.002257788725, .002257876016].

These are separate columnwise measurements; adding extrema from different
columns is not a substitute for the recorded per-column total. CSV files retain
each column's V and all three integrals. The history term increases the support
deficit even though h omission alone is smaller.

There are **0 and 358 measured periodic crossing events** in the two real
steps. Wrapping does not imply failure. The following whole-fault ranges include
seam/endpoints and therefore also bound seam-localized variations:

| Field | Final along-fault range | Fraction of predeclared feedback scale |
|---|---:|---:|
| H (Pa), stable initial normal rows | .00102194 | .02597% |
| phi, fixed mesh rows | 2.14044e-6 | .22447% |
| I_h (m) | .000414821 | .27317% |
| C (Pa) | .000806622 | .00279% |
| V (m/s) | 4.13546e-8 | .06615% |

No observed along-fault variation is comparable to the intended transverse
feedback. Maximum phi is .59999801, below .8; minimum sampled/nodal phi is
positive. These observations support this bounded 1-D interpretation only.

## Primary independent reference on exact accepted times

The independent continuum reference was rerun using the actual exported
time/dt/U sequence and the fully resolved smoke parameters. Its histories were
initialized once from the documented independent initialization, never reset
from later ASPECT fields. The earlier production-initialized reference remains
diagnostic decomposition only and is not substituted as the primary reference.

| Quantity at final time | Production surface mean | Independent reference | Relative difference |
|---|---:|---:|---:|
| V (m/s) | .002258036359 | .002257753777 | +.01252% |
| C (Pa) | 383.8062933 | 382.8401560 | +.25236% |
| Theta (s) | .4829435247 | .4830068220 | -.01310% |
| I_h (m) | 108.2581849 | 108.2867776 | -.02640% |
| Accumulated slip (m) | .008506844129 | .008506605578 | +.002804% |

Keep the initialization differences visible: production/reference C0 are
318.747656/317.742650 Pa, I0 are 108.098095/108.134923 m, and maximum phi0 are
.599910323/.599702855. The initial maximum transverse mean-phi error is
2.07468e-4; the final total-field error is 2.13652e-4. These errors were not
subtracted from the physical trajectory.

Feedback measured after initialization:

- max H1-H0: **3.85048 Pa** in production versus 3.93555 Pa in the continuum
  reference; the preexisting core H maximum does not grow;
- max transverse mean phi2-phi1: **.000990097** versus .000953568;
- I2-I1: **.159226 m** versus .151855 m.

Production phi1 also adjusts by 1.17541e-6 with retained H0. This small
initial/discrete-domain adjustment is not counted as H1-driven phi2 feedback.
Raw bulk shear-stress errors against the independent evaluated stress have
RMS/max values 1.090/3.188 Pa at zero, 5.146/17.793 Pa at step 1, and
4.239/14.934 Pa at step 2. Evaluation uses the constrained old FE stress from
the actual mechanical working state, not newly committed particle history.
These are coarse-smoke errors, not convergence estimates.

## Artifacts and scope

All artifacts below are under
`benchmarks/reconstructed_fault/uniform_shear/evolving/`:

- `smoke.log`, `smoke.resources.json`: current successful invocation;
- `smoke/guard_0.json` through `guard_2.json`, `crack_integrals_*.csv`,
  `normal_profiles_*.csv`, `H_rows_*.csv`, phase-probe and stable-ID entry CSVs;
- `smoke-reference/`: primary independent exact-time reference and profiles;
- `smoke-comparison.json`, `.log`, and `compare_smoke.py`: all errors, fresh
  linear checks, weak balances and lifecycle comparisons;
- `smoke/comparison_phase_*.csv`: full transverse errors, not initial-error
  corrected curves;
- ParaView bulk collection `smoke/solution.pvd`, fault files
  `smoke/reconstructed_faults/reconstructed_faults-0000{0,1,2}.vtu`, and
  `smoke/particles/particles-0000{0,1,2}.pvtu`;
- `attempt1-fingerprint-failure/`: archived first invocation's directory/log/
  resources, alongside the previously saved failed plugin/header snapshots.

Exact additional commands, both successful and capped at 120 s:

```sh
OPENBLAS_NUM_THREADS=1 timeout 120 python3 benchmarks/reconstructed_fault/uniform_shear/evolving/reference.py --parameters benchmarks/reconstructed_fault/uniform_shear/evolving/smoke/parameters.prm --ramp-peak .00225 --accepted-times benchmarks/reconstructed_fault/uniform_shear/evolving/smoke --output benchmarks/reconstructed_fault/uniform_shear/evolving/smoke-reference
OPENBLAS_NUM_THREADS=1 timeout 120 python3 benchmarks/reconstructed_fault/uniform_shear/evolving/compare_smoke.py
```

The smoke wrapper measures **43.41240324 s**, **598776 KiB** peak RSS, exit 0,
one direct Release process and a 180-s process-group cap. ASPECT's internal
timer reports approximately 40 s. Tested plugin SHA256 is
`85b6ace73782ddf646faa1663c4e5fe406e4a76054078dc9b2db543d9aa6a5cd`;
the unchanged executable/parameter hashes and HEAD are in the resource JSON.
No new ASPECT tests or build were necessary for this approval; the existing
corrected Release target and passing cheap checks were reused.

Stop here for review. No additional simulation or broader K3 campaign is
implied by this successful smoke, and no Gate-K2 convergence claim is made.
