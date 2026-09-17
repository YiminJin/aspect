# Seven-step committing independent-trace comparison

## Scope and reproducibility

**Decision:** separating the traces reduces the within-free notch, but does
not arrest accumulated-slip mismatch over the tested interval or remove the
stress concentration. At 4.190129 yr the slip jump is -1.76093 mm and still
growing; the free endpoint is 2.03287% below Vp. The normal-stress range in
the fixed junction window is 9.64% larger, although its work-weighted RMS
variation is 2.29% smaller. This is not evidence for a qualified junction
repair. No continuation beyond step 7 was run.

This is the user-authorized fresh-start comparison in
`stage_K5_trace_replay_addendum.md`, not a production change to the default
continuous-Q1 representation. Completed results are in
`benchmarks/reconstructed_fault/bp3/trace-replay-seven-local4/`.
The control is the saved `work-replay-50-local4` trajectory. Both use the same
initial physical fields and accepted timestep sequence through step 7,
132230424.76671731 s (about 4.19 yr). The artificial initialization interval
remains 4e6 s and accumulates no slip or aging.

Run and analysis commands:

```
cmake --build build-pf-cpdi --target aspect.exe.release -j4
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
build-pf-cpdi/aspect-release --test 'Stage-I*'
python3 benchmarks/reconstructed_fault/bp3/run_trace_replay.py
python3 benchmarks/reconstructed_fault/bp3/analyze_trace_replay.py
```

The launch script starts exactly one four-rank process with a 2400 s hard cap,
refuses an existing output directory, records binary/input hashes and the
tracked source patch, and retains strict timestep-controller checks. There is
no automatic retry. No continuation past the seven accepted real steps is
authorized here.

## What the experiment changes

The free-side trace at 40 km is stored at node 795. The deep-side trace is
constant on segment 794 and eliminated through its prescribed node 794 at
40.04998 km. The latter storage coordinate does **not** move the physical
junction: it represents the constant deep trace at 40+ as well. Every deep
segment remains prescribed at Vp.

The source/test/trial weights on segment 794 are (1,0). Its state uses the
deep-side history too. All shallower segments retain Q1 interpolation. Each
node's state advances once after acceptance using its own accepted rate and
the ordinary exact frozen-rate aging law. Each side's accumulated slip obeys

\[
 d_k^\pm=d_{k-1}^\pm+\Delta t_k V_k^\pm,\qquad
 J_k=d_k^- -d_k^+=J_{k-1}+\Delta t_k(V_k^- -V_p).
\]

There is no added penalty or constitutive spring for J. Any control of its
growth must come from the existing bulk elastic/Maxwell response and surface
balance. Therefore a converged free-endpoint row alone does not establish
bounded accumulated mismatch.

Geometry, material and background interpolation, full Ih, fixed phase,
support, boundary corrections, work measure, mesh, tolerances and history
publication remain unchanged. Separate state interpolation is lagged during
mechanics: it adds no candidate-state derivative. The benchmark raw-stress
observer now calls the same manager rate interpolation as mechanics and uses
the cached mechanical test weights for native weak loads.

## Definitions and verification

The within-free chord notch is
`[V(39.90)+V(40-)]/(2 Vp) - V(39.95)/Vp`. Also report the deficit below **both**
free neighbours, separately from `V(40-)-Vp`. Slip-gradient reporting uses
the last 50 m of the free interval; J is an actual discontinuity and is not
divided by an arbitrarily chosen length to make a gradient.

Raw stress is the accepted constitutive tensor computed from accepted u,p,V
and the frozen working FE old stress. It is not the newly committed particle
Maxwell array exported at that same accepted time. Normal traction is
50 MPa + p - tau:N. Comparisons match the physical bulk QP coordinates and
weights, and use common 39.8–40.2 km and 37–43 km windows. Native one-sided
weak rows are separately labelled; node 795 now receives only free-side work.

Offline checks reproduce correctly timed weak friction/residual loads, verify
deep Vp/state/slip, the one-sided Theta at mechanical QPs, the fixed-state
tangent, and exactly one state/slip update per accepted step. They keep the
initialization-induced difference in the mechanical solution visible.

## Results

### 1. Kinematics and state

| At step 7, 4.190129 yr | Shared trace | Independent traces |
|---|---:|---:|
| V(39.90 km)/Vp | 0.982584152 | 0.981553768 |
| V(39.95 km)/Vp | 0.966302401 | 0.973694913 |
| V(40-)/Vp | 1 | 0.979671331 |
| V(40+)/Vp, and every deep rate | 1 | 1 |
| Within-free chord notch / Vp | 0.024989675 | 0.006917637 |
| Deficit below both free neighbours / Vp | 0.016281751 | 0.005976418 |
| Slip(40-) [m] | 0.132230425 | 0.130469493 |
| Slip(40+) [m] | 0.132230425 | 0.132230425 |
| Slip jump [mm] | 0 | -1.760932 |
| Last-free-element slip gradient [m/m] | 5.67251e-5 | 9.09693e-6 |
| Committed Theta(39.95 km) [s] | 8278909.972 | 8216073.688 |
| Committed Theta(40-) [s] | 8000000 | 8165969.226 |
| Committed Theta(40+) [s] | 8000000 | 8000000 |

The chord notch falls **72.32%**, the below-both-neighbours deficit falls
**63.29%**, and the last free-element slip gradient falls **83.96%**. A
remaining free-side local minimum is visible; it has not been replaced by a
perfectly monotone curve. The new rate jump is distinct from that minimum.

| Step | Time [yr] | Independent V(40-)/Vp - 1 | Slip jump [mm] |
|---:|---:|---:|---:|
| 0 | 0 | +0.000016299 | 0 |
| 1 | 0.078815 | +0.000010437 | +0.000025960 |
| 2 | 0.157494 | -0.000324250 | -0.000779130 |
| 3 | 0.307772 | -0.000997192 | -0.005508207 |
| 4 | 0.594802 | -0.002410141 | -0.027339200 |
| 5 | 1.143029 | -0.005313053 | -0.119258888 |
| 6 | 2.190142 | -0.010853073 | -0.477892105 |
| 7 | 4.190129 | -0.020328669 | -1.760931824 |

The mismatch is **1.3317% of the deep accumulated slip** at the end, and the
last interval alone adds -1.28304 mm. Both the instantaneous rate deficit
and the mismatch fraction are increasing. Thus this trajectory supplies no
evidence of saturation. It does not prove unbounded long-time growth either.

Initial histories were identical, not imported from a notched checkpoint.
Changing the endpoint equation changes the initial mechanical solution:
the initial chord notch is 1.84435e-5 Vp versus 5.23662e-6 Vp in the control,
and the independent endpoint starts 1.62985e-5 Vp above Vp. The initial slip
is exactly zero on both sides; supplied Theta and zero particle stress are
retained. These small initial-solution differences are included throughout,
not subtracted from the result.

Mechanics 7 used **Theta6**, including 8086964.467 s at the free endpoint and
8000000 s on the deep side. The table reports **Theta7 after commitment**.
The mechanical QP exports verify that the prescribed-side segment reads its
own preceding state and not a mixture with the free endpoint's state.

### 2. Current stress, not published stress history

Same physical QPs, all admitted positive-chi samples in **39.8–40.2 km**:

| Quantity | Shared trace | Independent traces |
|---|---:|---:|
| Pressure min/max [kPa] | -81.247 / 90.362 | -79.525 / 90.834 |
| Pressure range [kPa] | 171.610 | 170.358 |
| tau_xx min/max [kPa] | -14.138 / 89.814 | -13.466 / 89.797 |
| tau_yy min/max [kPa] | -99.190 / 35.561 | -98.519 / 31.215 |
| tau_xy min/max [kPa] | -37.114 / 56.731 | -34.085 / 56.484 |
| -tau:N range [kPa] | 74.410 | 69.099 |
| Total sigma_n min/max [MPa] | 49.910618 / 50.081402 | 49.904323 / 50.091563 |
| Total sigma_n range [kPa] | 170.784 | 187.240 |
| sigma_n work-weighted RMS about its mean [kPa] | 27.440 | 26.812 |
| sigma_n work-weighted mean [MPa] | 49.998012499 | 49.998012522 |

The independent-trace normal minimum moves from 39.99030 km to **40.00921 km**,
and its maximum moves from 39.97598 km to **40.00039 km**. Both new extreme
samples are just on the deep side. At identical physical QPs the maximum
normal-traction change is 30.565 kPa; the pressure change is 15.087 kPa.
The pressure range alone would miss the increased combined normal-stress
extrema. No tensile normal traction develops in this window.

Normal-stress ranges for shared/split traces grow from 2.959/3.140 kPa at
step 4, through 11.262/12.388 kPa at step 5 and 45.102/50.050 kPa at step 6,
to the values above. The new jump has **not** removed the growing
concentration. Its existence in both runs means this comparison does not
attribute all concentration to the newly permitted jump.

The 13–20 km control window is effectively unchanged: its range is
1130.616263 versus 1130.616078 kPa. The 59–61 km prescribed control range is
65.067455 versus 65.067459 kPa. The response to this trace change is localized
around the junction rather than a global stress-reference shift.

### 3. Weak equilibrium and the remaining notch

At the unchanged 39.95-km test function, native weighted normal traction is
49.998068 MPa in the control and 49.997544 MPa in the split run; weighted shear
traction is 26.522374 versus 26.523488 MPa. This small change in averages
coexists with the raw normal extrema above.

The two incident-element residual contributions at node 796 are
**-179883.352 / +179883.352 Pa*m** in the control and
**-78044.167 / +78044.167 Pa*m** in the split run. Thus the opposing weak
contributions shrink, but are not eliminated.

The new free endpoint receives no work from segment 794. Its native physical
row is 2.02e-8 Pa*m, with work mass 25.0953 m, at the final accepted solution.
Its one-sided weak mean is 49.994434 MPa normal and 26.523555 MPa shear.
These are **not** directly comparable to the old two-sided prescribed row's
means. The old prescribed row carried -1.681086e6 Pa*m of physical residual,
balanced by the prescription reaction rather than an RSF free equation.
Independent offline evaluation reproduces all selected physical rows,
including prescribed rows, within 8.64e-6 Pa*m against individual traction
loads of order 1e9 Pa*m.

There is therefore no failed endpoint force balance hiding the jump. The
new free equation genuinely balances, but does not enforce accumulated-slip
continuity. Elastic-history feedback in this resolved calculation does not
arrest the mismatch by the requested comparison time.

### 4. Verification and cost

- Release executable and plugin built with `-j4`; only existing unrelated
  range-loop-copy warnings in the diagnostic header were printed.
- `Stage-I*`: **90 assertions in 11 test cases passed**. An initial sandboxed
  invocation also passed with MPI socket warnings; the unrestricted focused
  invocation passed without those warnings. No full test suite was run.
- Exactly initialization plus **seven real steps**, no step 8; strict saved
  time/dt checks passed. Final stop: `BP3 replay complete`.
- **64/64 fresh linear checks passed**, 1230 total Krylov iterations.
  Final normalized bulk/surface residuals: **2.01948e-13 / 4.25841e-13**.
  Final dimensional bulk residual: 1.13722e-4; surface RMS: 1.43378e-6 Pa.
- All 441 free nodes remained free; every deep rate remained exactly Vp.
- Independent exact-aging relative error <=2.22e-16 at every accepted state;
  each slip increment equals accepted V*dt. Deep Theta remains 8e6 s.
- Initial nodal Theta, particle IDs, inert H and all particle Maxwell tensor
  components match the original baseline bitwise on all four ranks.
- First Maxwell update: error 5.37e-8 Pa on a 5280.04 Pa scale, with zero old
  FE stress. All 385920 stable particle IDs preserve inert H. Geometry and
  completed Ih remain unchanged; mature C remains zero.
- Saved mechanical QPs verify side-consistent preceding Theta, friction,
  fixed-state tangent and zero candidate-state tangent at all eight states.
  Existing independent-trace derivative/work qualification is reused; no
  new finite-difference solve was needed for lagged state interpolation.
- Analysis matches physical QP positions, phase, Ih and weights exactly.
  Re-evaluated chi is compared at near-roundoff allowance 1e-12 relative
  (tiny material-mixture arithmetic differences after different particle
  motion); no production tolerance changed.
- **741.96 s** elapsed, four ranks. Reported peak child RSS **1525440 KiB
  (1.455 GiB)** is a per-process high-water mark, not measured aggregate memory.
- No restart qualification or long-time/spatial convergence claim is made
  for the experimental representation. It remains fresh-start-only.

## Evidence and next decision

`execution.json`, `provenance.json`, `source.patch`, `clock.csv`, `run.prm`
and `run.log` preserve the run. The `analysis/` directory contains
`histories.csv`, explicit `endpoint_traces.csv`, `nodes.csv`,
`element_budgets.csv`, `raw_stress.csv`, `matching_differences.csv`, matched
raw-QP files for steps 0/1/2/7, `summary.json`, and `comparison.png`.
The `state_qp_<step>_rank*.csv` files are last-Jacobian mechanical inputs used
to audit timing/tangents; `work_qp_<step>_rank*.csv` are the accepted current
constitutive stress used in the stress comparison.
The executable SHA256 is
`15f8d5242eaf3824723f60d35cdd4e4d551111fd25523a4492a8e6a032e1a597`;
the plugin SHA256 is
`0ade17a513f53ea2b7f1864dc02a8d88e2a3a993fce05f4e718db804b9b3f077`.
The complete preserved run occupies approximately 833 MiB.

**Recommendation:** do not promote the independent trace as a BP3 repair.
It is useful evidence that the shared prescribed endpoint amplifies the
neighbouring notch, but the fresh committing test trades much of that
oscillation for a growing slip discontinuity without materially reducing the
stress concentration. A physical/numerical junction condition must be reviewed
before another committing method is chosen; a longer split-trace replay alone
would not establish its admissibility. The present seven-step data cannot
decide long-time saturation or spatial convergence of the jump-induced stress.
