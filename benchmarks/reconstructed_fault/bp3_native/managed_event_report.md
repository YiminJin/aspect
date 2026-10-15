# Fixed-state population/near-face audit and short timestep test

The selected diagnostic task is complete. Keep **4×4, minimum 12, maximum 24**
as the working baseline; the completed 5×5 result remains the density control.
No density sweep or second long management trajectory was run. Resampling and
near-face trace selection both have measurable operator effects. Neither the
previous recurrence passes nor this bounded test establishes spatial accuracy.
No sampling rule, physical parameter, solver tolerance or production default changed.

Baseline: `fault-propagation`, `c33edaaeb145e5d3dce155bb815552ef80efc5f9`.
All incoming P2/stationary/moving evidence and original binaries are preserved.
Exact identities, snapshots, analysis and builds are under
`build-bp3-native/managed-event-audit/`; commands, exits, environment and wall
measurements are `refactoring_boundary/evidence/bp3-event-*.{json,log}`.
The unchanged core SHA256 starts `bf6f605431cd`; maintained BP3 plugin starts
`5256f361d3e8`. `run-artifacts.json` pins the actual diagnostic observer/source.

The same translated 60-degree strengthening fixture retains 4136 graded cells,
149538 DoFs, 59 surface nodes, native 3×3 mechanical quadrature, the discrete
phase/I_h, physical Vp=1e-9 m/s, homogeneous G/eta, full bottom velocity and the
20 m filtered normal input. The moving particle-owned D3 Maxwell law remains
non-objective, with no competing FE d_R history. Surface theta/slip ownership,
prescribed geometry/phase, native RK2, LLS history reconstruction and CPDI stay.

## One lower-end population event, with mechanical fields frozen

Earlier profiles were saved, but no complete later particle/FE state or usable
checkpoint existed. One baseline-only prefix through step 44 was therefore
needed to capture the event. This acquisition replay is not the timestep test.
Its 45 accepted surface profiles (steps 0–44, all nine reported fields) are
**bit-identical** to retained `managed-v3`, including after both scoped probes.
The 5×5 trajectory was only read, never rerun or overwritten.

At the terminal native RK2 stage, after sorting/outflow and immediately before
count management, rebuild the actual moved domains and reconstruct stress through
native particle-to-composition interpolation. Assemble a separate native surface
system at frozen u, p, phi, V, theta and I_h. Repeat immediately after native
addition/removal, before mechanics. Only stress blocks are temporarily replaced;
restore the complete FE solution afterward. Bulk/old fields, pressure loads,
surface mass, state and rate equality are asserted. No stress/state/slip update
or nonlinear correction belongs to either probe.

With Q denoting the existing particle→FE reconstruction, the fixed-strain test
isolates delta tau_trial = beta delta(Q tau_particle), beta=exp(-G dt/eta).
Its maximum defect is 1.71e-8 Pa. Pressure does not change; the raw-normal change
comes from the tensor contribution. Filtered normal and friction loads respond
through the unchanged filter and fixed-V/theta friction law.

Selected clock: step 44, t=6.871021411e9 s, dt=2e8 s. There are 7277 cell
crossings and 232 outflows before the first snapshot. Count management alone
changes **65681→65761** particles: 316 births and 236 deliberate removals;
65445 survivors retain position, cell and all eight properties exactly.
In the projected lower-200-m window, including endpoint completion, there are
159 births and 70 removals. Births inherit native interpolated history rather
than zero memory. Both complete populations, FE coefficients and native RNG
states are saved once as `population-44-{before,after}-*` for later frozen work.
These diagnostic snapshots are not a supported ASPECT restart format.

| Fixed-state quantity | Maximum change | Weighted RMS / other local measure |
|---|---:|---:|
| Lower-window reconstructed FE history, in-plane tensor norm | 19.868 MPa | .33267 MPa RMS |
| Lower-window direct particle LLS history | 28.984 MPa | .41417 MPa RMS |
| Reconstructed stress FE coefficient | 13.629 MPa | — |
| Consistent nodal shear traction | 159.320 kPa | lower-window RMS 27.079 kPa |
| Consistent nodal raw normal / tensor contribution | 92.368 kPa | lower-window RMS 15.699 kPa |
| Consistent nodal filtered normal | 29.571 kPa | lower-window RMS 6.851 kPa |
| Pressure contribution | 0 | exactly unchanged |
| Assembled residual row | 1.16125e6 Pa m | vector L2 1.17127e6 Pa m |
| Assembled shear / raw-normal / filtered-normal row | 1.03180e6 / 5.98199e5 / 2.43713e5 Pa m | maxima at lower endpoint |

These are actual weak loads before mass inversion as well as traction projections.
The absolute sum of residual-row changes is 1.43147e6 Pa m (signed sum 1.27050e6);
the table does not infer accuracy from a small signed global mean. The largest
point-history change lies in the lower completion flank at (89.71875,144.18202)m,
phi=.022864 and chi=.00024143/m, so its peak is not a representative load average.

The FE reconstruction and direct LLS evaluation already differ before management:
lower-window max 257.890 MPa, weighted RMS 66.0335 MPa; after, 257.890/66.0340 MPa.
This is a distinct representation discrepancy, not the resampling jump and not
an independent analytic error estimate. Large local history changes must not be
hidden by the smaller traction projections. One event does not explain the
previous long-trajectory density sensitivity in full.

## Near-face rule and assembled effect

The unchanged production path uses deal.II RemotePointEvaluation with default
reference tolerance 1e-6 and averages all eligible traces. At step 34, actual
particle 40014 lies 1.257 micrometers from a face of a 7.8125-m cell. Native
averaged-trace versus host-cell update differs by **3.23229 MPa** in the stored
in-plane tensor norm. An independent FEValues evaluation of each eligible trace
reproduces the native average within 4.21e-26 /s. Independent box-cell inclusion
also reproduces the two eligible cell IDs and every sweep count.

Holding FE fields and incoming particle history fixed, reference offsets
[-4,-1.1,-.9,0,.9,1.1,4]×1e-6 give trace counts **[1,1,2,2,2,1,1]**.
The shear stress changes from -79.73195 MPa on one unique trace to -77.85942 MPa
in the averaging band to -75.98684 MPa on the other trace. Thus changes in the
eligible set create finite jumps at the edges of the tolerance band. This is
consistent with discontinuous cellwise FE gradients; recurrence correctness
alone does not control the size of these jumps.

For a separate frozen operator probe, temporarily substitute only that particle's
host-based predicted tensor for its actual native committed tensor, reconstruct
and assemble, then restore it exactly. This is a sensitivity of the committed
history operator, not another physical update or a new accepted solution:
maximum weak residual change **1.02698e5 Pa m**; nodal shear/raw/filtered changes
**8.093/2.354/.710 kPa**, with pressure exactly unchanged. The largest row is
at s=676.893 m, in the fault interior. The lower-end effect of this particular
particle is only a filter tail. It does not reproduce the worst lower-end
5.35/10.38-MPa host/average differences in the retained 4×4/5×5 runs; their full
later states were not saved. Those observations remain unresolved evidence.

At these same frozen fields and incoming history, the exact Maxwell coefficient
alone reduces the 3.23229-MPa sampling difference to .64646/.064646/.032323 MPa
for dt=4e7/4e6/2e6 s. This is an instantaneous sensitivity calculation, not a
trajectory refinement result or convergence-order claim. No change to averaging,
its tolerance or particle density is justified solely to make this diagnostic small.

## Short physical timestep comparison

Restore native `convection time step, reconstructed fault time step` controllers
from `bp3-fresh-fe.prm` and `server/parameters.prm`: maximum first/global dt
4e6/4e7 s, maximum relative increase 91%, logarithmic state-change limit .2.
Compare against 2e6-s first/global caps, with all other settings fixed, to
T=2e7 s (.634 yr). No function timestep model is active.

Both retain the **200000-s step-zero constitutive interval**, so their initial
profiles, mesh, phase, quadrature and I_h are identical. The fresh full-BP3
input's 4e6-s initialization interval is deliberately not imported: changing it
would change the initial discrete model. Native law-specific timestep restriction
is inactive for this uniform strengthening branch (a>b). Predicted maximum
log-theta change even at the caps is .00698/.00242, below .2: caps, growth and
the endpoint set these schedules, not the state-change bound.

| Actual result | Production controller settings | Tighter caps |
|---|---:|---:|
| Positive accepted dt [s] | 4e6, 7.64e6, 8.36e6 | ten ×2e6 |
| Runtime [s] | 135.97 | 248.50 |
| Maximum normalized nonlinear residual, target <1e-8 | 3.976e-12 | 9.478e-9 |
| Maximum surface RMS [Pa] | 2.186e-5 | .03198 |
| Newton updates / Krylov iterations, including step zero | 8 / 375 | 19 / 806 |
| Maximum particle Maxwell defect [Pa] | 3.279e-8 | 9.096e-9 |
| Maximum theta relative defect | 2.221e-16 | 2.221e-16 |
| Maximum step slip / Dc | 1.0666 | .25557 |
| Crossings / births / removals | 0 / 0 / 0 | 0 / 0 / 0 |

All stress/state/slip once-only, survivor, domain/cache, tangent and frozen-field
checks pass. There is exactly one publication at step zero and each positive
accepted step, with no second increment from diagnostics. This short interval
measures early mechanical temporal sensitivity; it does not establish temporal
accuracy at late population events. In particular, enabled controllers do not
force an active state bound or a small slip/Dc in uniform strengthening.

Common-time profiles are compared at t=0,4e6,2e7 s without temporal interpolation.
Initial discrepancies are zero. Final differences below are tighter minus primary;
RMS uses the existing Q1 surface metric. Scales are Vp, theta0=8e6 s, 1, Vp*T=.02 m,
and 50 MPa, respectively.

| Field | Final maximum absolute difference | Maximum / scale | RMS / scale |
|---|---:|---:|---:|
| V | 5.22469e-12 m/s | .52247% | .43123% |
| theta | 35507.5 s | .44384% | .35248% |
| Omega | .000927758 | .09278% | .08263% |
| Slip | 8.14058e-5 m | .40703% | .32728% |
| Shear | 11.156 kPa | .02231% | .00650% |
| Raw normal | 47.134 kPa | .09427% | .00735% |
| Filtered normal | 15.864 kPa | .03173% | .00444% |
| Pressure contribution | 51.796 kPa | .10359% | .00946% |
| Tensor normal contribution | 4.661 kPa | .00932% | .00364% |

One tighter setting measures sensitivity, not a convergence order. Pressure and
tensor terms remain separate; no difference is assigned wholesale to projection.
`assessment.json` retains all common-time metrics, signed/absolute weak sums,
trace sweeps, final roughness and lifecycle checks.

## Changes, verification and next step

Changed only `include/aspect/particle/manager.h` (test friendship, no data/layout
or algorithm change), the existing `coupled_checks.cc` observer (default-false
scoped event probes), three `managed-{event,production-time,production-time-tight}.prm`
fixtures, `analyze_managed_event.py`, this report and the two recovery statuses.
No new diagnostic framework or production physical option. The observer copies
the initialized native RK2 and observes only its terminal-stage decision; it does
not duplicate integration or alter the native integrator name/property layout.
The pre-management hook is required because the existing signal is post-management.
Extra rebuilds invalidate derived caches but preserve physical state; the complete
replay's exact profile match checks their noninterference. Full snapshots are
restricted to the selected event; no continuous whole-particle output is added.

Initial and final observer compilation/link pass. The original and final
one-step gates pass with bit-identical retained profiles; final source differs
from tested trajectory source only in whitespace, checked explicitly. ASPECT
astyle 2.04 formatting is restricted to new code, and Python syntax/analysis and
whitespace checks pass. No complete core build, unit-suite repeat, MPI, restart
or cutback experiment was needed or claimed for this diagnostic-only change.
The event acquisition costs 574.80 s; the short pair costs 384.47 s total.
All five launches, including the two gates, total 990.53 s (16.51 min), excluding
compilation; every launch exits successfully within its explicit step/wall bound.
At most three serial ASPECT runs overlap, within the four-process preference.

Keep the density baseline and sampling rule. The next specific refinement should
address lower-end history reconstruction/trace-gradient resolution at fixed saved
states, assessing local history and weak loads separately, before another density
increase. A future late-time timestep test needs a matched restartable state; this
early pair cannot supply that qualification. No first-event or later campaign
is launched, and the previous controlled-error exit remains open.
