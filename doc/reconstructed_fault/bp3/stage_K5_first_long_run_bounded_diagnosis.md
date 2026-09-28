# First long run: bounded onset and friction-timing diagnosis

Follow-up: the subsequent frozen mechanical probes are reported in
`stage_K5_shallow_mechanical_probe.md`. The statements below about missing
mechanical tests describe this earlier saved-output audit, not the current
verification status; the original observations are retained unchanged.

## Decision and limits

The shallow structure is present in the raw saved nodal fields, not introduced
by plotting. It first develops without any active lower-bound nodes. It is not
a pure two-node checkerboard from its inception: the early structure spans
several nodes, later producing many two/three-element trough separations.
The single mesh cannot distinguish fault-grid, bulk-grid and finite-width
contributions. An under-restored alternating mechanical mode remains a
hypothesis, not a result of this audit.

The deep slowdown is quantitatively consistent with reduced shear/compression
ratio and nearly steady aged state. The state must be timed before the update.
The numerical attribution below uses projected nodal tractions and is explicitly
**not** an independently reconstructed production weak balance.

**The requested frozen mechanical perturbation test has not been performed.**
None of the supplied checkpoints precedes the instability. The actual
quadrature loads/mass matrices and early retained bulk-history state are not
exported. A late checkpoint, a new unrelated fixture, or a nodal reaction-matrix
surrogate would not answer the requested early-state question. No ASPECT run,
history mutation, parameter change or production-code edit was made.

## Data and reproducibility

Input root: `benchmarks/reconstructed_fault/bp3/first_long_run/output/`.
Offline driver:

```sh
MPLCONFIGDIR=/tmp/aspect-bp3-slip-plot python3 \
  benchmarks/reconstructed_fault/bp3/analyze_long_run_fault.py \
  benchmarks/reconstructed_fault/bp3/first_long_run/output
```

Outputs are in `output/bounded_diagnosis/`:

- `onset_metrics.csv`: every available accepted slip-table step, dip metric,
  source of rates, floor contacts and actual solver active count where present.
- `onset_neighbours.csv`, `first_dip_neighbours.png`: raw node IDs/coordinates,
  rates, reconstructed incoming/updated state and slip around the first 10% dip.
- `trough_spacing.csv`, `node_spacing_cells.csv`: measured wavelengths and
  actual containing cells from the initial mesh exports.
- `native_csv_agreement.csv`: independent native/CSV serialization check.
- `state_reconstruction_check.csv`, `deep_nodal_proxy.csv`, `summary.json`.

The copy is not a single complete end-state snapshot. The slip table and
accepted log reach step 5188 (383.222145 yr); available sparse profiles extend
to step 5211, with six later indexed profiles absent. One indexed native VTU
is absent. Only the common complete history prefix is used for state timing
and the deep comparison; its latest available profile is step 5187. An exact
duplicate initialization block is explicitly skipped without editing input.

## 1. Onset, ordering and spatial scale

For adjacent nodes define a dip when
`V_i < 0.9 min(V_(i-1), V_(i+1))`, with both neighbours above 1e-12 m/s,
in 5–18 km. This is an observational 10% detection threshold, not a changed
solver criterion. Requiring a local minimum avoids classifying a monotone
front as ringing. Sensitivity to the detection threshold is shown below.

| First depth below both neighbours | Step | Physical time (yr) | Location (km) |
|---|---:|---:|---:|
| 0.1% | 11 | 36.259648 | 16.9 |
| 1% | 19 | 43.761270 | 16.4 |
| 5% | 24 | 45.464281 | 16.2 |
| 10% | 27 | 46.506281 | 16.2 |
| 20% | 38 | 50.442697 | 16.1 |
| 50% | 185 | 88.255075 | 15.4 |

There is no qualifying initial dip. The earliest resolved structure is within
the 15–18 km transition, not first born in the interior of the shallow VW zone.
The first saved 10% dip shallower than 15 km is step 213, 95.150607 yr, at
14.9 km. Two significant minima exist by step 109 (71.254789 yr); three by
step 187 (88.638050 yr).

Selected adjacent-node values at step 27:

| Stored node | Down dip (km) | V/Vp | Incoming Theta (yr) |
|---:|---:|---:|---:|
| 995 | 16.0 | 0.328366 | 0.70120 |
| 994 | 16.1 | 0.133738 | 1.20653 |
| 993 | 16.2 | 0.112552 | 1.65236 |
| 992 | 16.3 | 0.125127 | 1.63844 |
| 991 | 16.4 | 0.138880 | 1.58605 |

At the same 16.2-km node, V/Vp declines through 0.123369, 0.117564,
0.112552, 0.108134 on steps 25–28 while incoming Theta grows through
1.47828, 1.56479, 1.65236, 1.74154 yr. Thus structured incoming state already
participates in amplifying the dip by the 10% detection time; that observation
does not identify whether the original seed is spatial or temporal.

The solver log reports its **first lower-active node at step 222**, approximately
96.60 yr. The next saved profile, step 223, has an exact V=1e-20 m/s node at
14.9 km. A floor contact inferred by subtracting nearly equal accumulated
slips is not treated as an exact active-set flag: the driver labels inferred
rates separately and records the actual solver counts. All nodes are free
at the earlier onset states. Lower-bound clipping therefore did not initiate
the instability, although it participates in its later saturated form.

Measured geometry:

- Adjacent shallow fault vertices: **100 m**.
- Cell containing the first 10% dip: `7_9:213230100`, level 9, square side
  **97.65625 m**. The 25/40/80-km controls also lie in cells of that size.
- Regularization length: **ell=400 m**, not a statement that the entire
  nonzero profile has width 400 m.
- Trough separations at steps 109/187/223 have medians 600/450/500 m.
  By steps 1060, 2864 and 5187 the median is 300 m, with many 200-m spacings.

Later oscillations are therefore only two to three fault elements apart,
but fault and bulk resolution are almost equal and ell is only about four
cell widths. There is no independent length-scale variation here. Quantized
trough coordinates alone do not prove a fault-Q1 origin, nor do these data
show a fixed wavelength proportional to ell. A claim of one specific spatial
cause would exceed the evidence.

Ordering checks: geometry and node IDs are identical across the available
profile files; stored order is opposite increasing down-dip order. Sorting
applies the same permutation to all fields. Native VTU and CSV V, Theta and
cumulative slip agree **exactly** at checked steps 0, 11, 26, 221 and 5172.
The dip plot uses original adjacent-node V values, without interpolation,
smoothing or differentiation of slip. The visual oscillation is real in the
discrete solution, not an ordering/plot-reconstruction artefact.

## 2. Deep slowdown with correctly timed state

Production work measure (current design/specification):

\[
R_i=\sum_q J_q\chi_q N_i(q)
[q_q-\mu(V_q,\widehat\Theta_{k-1,q})\sigma_{n,q}-\eta^dV_q].
\]

V and incoming Theta are Q1 interpolants at the same bulk quadrature points.
C=0. Normal stress includes the background. The exported `q_weak_Pa` and
`sigma_n_weak_Pa` are **M^-1 times their production weak load vectors**.
They are not raw QP tractions. In particular,
`mu(V_i,Theta_i) * (M^-1 normal_load)_i` is not generally
`(M^-1 friction_load)_i`. The absent mass/weak-friction/QP exports prevent an
exact offline reassembly of R. Close nodal balance is not proof of the actual
weak residual or permission to retune it.

Incoming state is reconstructed **forward once** from the supplied initial
state, using every available accepted dt and V (saved V when available;
otherwise the recorded dt*V slip increment). The exact aging expression is
used with expm1. It is never reset from subsequent saved Theta. Across saved
profiles the maximum relative discrepancy is **1.82e-11**; at the three deep
controls it is **7.38e-13**. This reconstruction is sufficient for the reported
offline traction sensitivities, not an exact restart or substitute for the
production 1e-12 state audit. Cancellation at very small rates is explicitly
handled only for this offline reconstruction, not by changing production V.

Latest common comparison: **step 5187, 383.212470 yr**. The 80-km control is
the actual node at **79.984134 km**, not an invented 80-km interpolated point.

| Location (km) | V/Vp | Projected q (MPa) | Projected sigma_n (MPa) | Incoming Theta (s) | Nodal proxy imbalance (Pa) |
|---:|---:|---:|---:|---:|---:|
| 25 | 0.256069 | 26.034080 | 50.325352 | 31,274,873 | +1.366 |
| 40 | 0.302081 | 26.262806 | 50.607576 | 26,479,859 | -0.025 |
| 79.984 | 0.356543 | 26.673178 | 51.235350 | 22,425,154 | +1.087 |

Initial projected shear was approximately 26.5461 MPa and normal stress
50 MPa. Radiation damping is only 0.0012–0.0016 Pa at these final rates.
The proxy uses the exact regularized asinh friction law. At these states its
logarithmic limit agrees to diagnostic precision. With a=.025 and b=.015:

\[
\Delta\ln V \simeq \Delta(q/\sigma_n)/a
 -(b/a)\ln(\Theta_{k-1}/\Theta_0)-\Delta(\eta^dV/\sigma_n)/a.
\]

The change of q/sigma is split symmetrically into
`Delta q * (1/sigma + 1/sigma0)/2` and
`(q+q0) * Delta(1/sigma)/2`, avoiding an arbitrary order of substitutions.

| Location | Shear contribution to Delta ln V | Normal contribution | State contribution | Observed Delta ln V |
|---:|---:|---:|---:|---:|
| 25 km | -0.408262 | -0.135987 | -0.818024 | -1.362274 |
| 40 km | -0.225256 | -0.253597 | -0.718166 | -1.197019 |
| 79.984 km | +0.100447 | -0.513263 | -0.618445 | -1.031262 |

These proxy sums differ from observed log-rate changes by at most 1.1e-6 at
the three controls. Shear/compression/state account for approximately
30/10/60%, 19/21/60%, and -10/50/60% of the respective logarithmic slowdowns.
The 80-km shear **increases** and offsets some slowdown; enhanced compression
more than cancels it. This is an algebraic attribution, not three independent
causal experiments. In particular V*Theta_in/Dc is 1.00107, 0.99988 and
0.99944: state is nearly steady, so its approximately 60% share follows
largely from b/a=.6. It is not evidence of an independent external state load.

The timing error matters most earlier: at step 5 (1.227268 yr), substituting
newly committed Theta adds **16.71, 3.68 and 1.05 kPa** to the nodal friction
proxy at 25/40/80 km, respectively. At step 27 the corresponding errors are
0.984/1.672/1.735 kPa. These are not mechanical residual failures; they are
the error of combining an accepted traction with the wrong state time level.

## 3. Frozen coupling test: unavailable early state and exact next action

Available complete checkpoints:

| Slot | Accepted step | Physical years |
|---|---:|---:|
| 03 | 4661 | 367.519552 |
| 01 | 4885 | 370.582483 |
| 02 | 5110 | 380.194662 |

They already contain strong oscillatory state/stress history. Graphical VTUs
and nodal histories do not supply the original particle ownership, retained
bulk history and frozen working FE state necessary for the early solve.
Moreover, these server checkpoints were produced with 32 ranks/deal.II 9.6.0
and 64-bit indices; local restart compatibility is not qualified by the earlier
same-binary four-rank tests. Do not silently substitute a local historical
100/200-km run or the old manufactured kink operator.

**Recommended next action:** obtain an accepted **step-10 checkpoint
(31.400709 yr)** from the same server build/configuration, or reproduce only
that short prefix on that build using the saved clock. This precedes the
first small seed. At the identical beginning-of-step-11 history, compare two
small feasible free-rate directions in a tapered patch around 16–17 km:
a smooth mode and a neighbouring-node alternating mode, normalized in the
same production work mass norm. Retain incoming Theta and both particle and
working FE old stress. Keep all production tolerances and constraints.

For each mode d, measure both frozen-bulk and bulk-relaxed restoring actions:

\[
A\delta x=Bd,\qquad
\delta R_\Gamma=G\delta x-K_Vd,\qquad
T(d)=(K_V-GA^{-1}B)d.
\]

Report d^T T(d)/(d^T M d), actual work-weighted traction components, and
response at adjacent nodes, with fresh bulk residual and directional checks
against the same frozen residual. Do not assume the operator or surface
Jacobian is SPD; sign and cancellation are part of the diagnostic. No
accepted histories may be published. A K-only comparison is insufficient:
it omits precisely the bulk relaxation whose spatial response is in question.

Choose the next bounded comparison **after** that measurement:

- If the alternating mode has anomalously weak/non-restoring bulk-relaxed
  traction relative to the smooth mode, prioritize one local spatial/coupling
  resolution comparison with unchanged incoming history and timestep.
- If it has adequate positive restoring traction, prioritize one identical-
  starting-state full step versus two half-steps to test split-state feedback.

The current files do not justify choosing one branch as demonstrated. No
further convergence campaign, solver modification or long continuation is
required to obtain this discriminating result.

## Checks performed

Four offline algebraic controls passed: monotone-profile rejection, inserted
dip detection, exact two-factor q/sigma decomposition, and the logarithmic
friction limit. Forward state reconstruction and native/CSV equality provide
additional checks against the actual output. This audit does **not** claim
an exact quadrature friction residual, an early-state operator probe, or a
qualified restart. The unresolved evidence is explicitly preserved rather
than replaced by a manufactured surrogate.
