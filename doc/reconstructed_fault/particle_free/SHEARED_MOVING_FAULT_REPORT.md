# Bounded moving-fault shear and rotation comparison

**All four runs completed, with local near-fault convergence but failed global
stress/energy refinement.** The near-fault tensor, shear traction and cumulative
slip differences decrease approximately linearly in physical timestep. Pressure-inclusive
normal traction does not show a clear temporal order. Whole-domain tensor differences
instead grow from **6.70% to 22.36%** on timestep halving; the 32²–64² difference
is **37.78%**. The growing discrepancy is concentrated at the **upper-left domain
corner**, away from the mature fault ends. No global or production qualification
is justified by successful nonlinear convergence.

[Protocol](SHEARED_MOVING_FAULT_PROTOCOL.md) ·
[Evidence and reproduction](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/sheared_moving_fault/README.md) ·
[Field profiles](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/sheared_moving_fault/sheared-fields.png) ·
[Geometry/spin/energy/cost](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/sheared_moving_fault/sheared-diagnostics.png) ·
[Bulk discrepancy maps](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/sheared_moving_fault/sheared-boundary-differences.png)

## Tested state and controls

Root `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`, revision
`cf06525471ce89789f1ac10905dc79f3aac7562a`. Entry had seven modified documentation
files from the preceding frozen split-Q2 work, plus retained untracked research
and manual artifacts. They remain preserved. Native source/CMake/tests and all
six current binaries are unchanged; all 63 protected artifacts match entry hashes.
The binary's older configure-time Git banner is not used as its source identity.
No rebuild, general refactor, isolated rotation recheck, production/server run or
checkpoint conversion was performed. Earlier unit/lifecycle evidence is reused
for unchanged code; this task adds exactly four new native trajectories.

The existing `run_material_native.py` fixture uses **ordinary uncut DGQ2 and
SSPRK3**, opt-in direct physical stress, bounded MacCormack Q1 profile transport,
accepted-trace tensor inflow and full-normal support. G=10 Pa, eta=10 Pa s,
growth disabled, native q8, accepted-gradient total-material Jaumann Lie ordering,
endpoint reaction and solver criteria are unchanged. Separated history remains
the default. No frozen split-Q2 approximation enters these calculations.

The boundary velocity is `(0.4+0.5*y,0)` m/s on left/right/bottom; top retains the
fixture's natural mechanical boundary. Duration is 0.4 s. All cases start from
the same analytic compact profile, curve, loading and initial constitutive
interval of .1 s. The initial .1 m label-spacing parameter produces **18 labels**;
their initial coordinates match bitwise across all four cases. There is no
surface refinement or resampling. All initial exported polynomial fields and
surface fields match **bitwise across the three temporal controls**. The finer
mesh uses the same physical initialization, with its own spatial discretization:
initial normalization differs by 2.837% and initial slip rate by .02550% in the
whole-curve L2 norm. This spatial difference is not hidden as a timestep effect.

The nominal affine accumulated spin is -0.1 rad. The natural top boundary makes
that affine field a loading estimate, not an exact solution. Actual spin is
`sum(dt*(d_x v_y^- - d_y v_x^-)/2)` sampled at accepted material labels;
curve orientation is measured separately.

| Mesh | dt (s) | Accepted steps | Mean label spin (rad) | Spin range (rad) | Wall time (s) | Linear solves / iterations |
|---|---:|---:|---:|---:|---:|---:|
| 32² | 0.1 | 4 | -0.093257 | -0.097076 to -0.091645 | 81.81 | 46 / 1717 |
| 32² | 0.05 | 8 | -0.093897 | -0.097242 to -0.092550 | 130.07 | 74 / 2711 |
| 32² | 0.025 | 16 | -0.094347 | -0.097352 to -0.093067 | 226.51 | 119 / 4335 |
| 64² | 0.025 | 16 | -0.094502 | -0.097397 to -0.093222 | 649.63 | 113 / 4201 |

Four MPI ranks ran sequentially, one thread each. The four simulations took
1088.02 s in total including initialization and output. Native CFL substeps per
physical step were 12–13, 6–7, 3–6 and 6–9 respectively. These are unchanged
SSPRK3 subcycles, not an assertion that the full lagged Lie/endpoint method is
third-order. Direct tensor storage is 27,648 scalar coefficients at 32² and
110,592 at 64²; label count stays fixed.

The measured spin is about **5.25–5.58 degrees clockwise**. The final chord turns
only .230–.243 degrees clockwise, as expected to differ from material spin in a
flow with strain. The curve stays at least **.9845 m** from the domain boundary
at every accepted snapshot. Finer-run ends finish at (-.680117,-.248193) and
(1.015248,.252589) m. Its peak final label slip rate is approximately .0621 m/s;
its maximum cumulative slip over the .4 s interval is .007212 m.

## Near-fault temporal comparison

All quantities below are pairwise sensitivities, not errors against an exact
continuum solution. Bulk stress uses the full 3-D plane-strain tensor norm,
including tau_zz=-tau_xx-tau_yy. A common Eulerian tube of radius .2 m follows
the finest-temporal reference curve; its central material coordinate is
.2 <= a <= .8, and its ends are the outer fifths, with nearest-point round caps.
These diagnostic caps do not extend the physical localization. Whole-domain
norms are retained separately. Surface quantities are sampled on each run's own
curve at identical material coordinates, with a common reference arclength
measure. Thus geometry differences are included.

Surface quadrature is split at labels, regional boundaries and every Cartesian
cell crossing in both compared runs; Gauss6 integrates the squared polynomial
traction differences. Bulk q8 is used. Self-convergence order is
`log2(||u_dt-u_dt/2|| / ||u_dt/2-u_dt/4||)` using absolute norms and common
regions/measures. It is one observed order, not an asymptotic guarantee.

The normal convention is tensile-positive `tau_nn-p`; effective frictional
compression is `5+p-tau_nn`. Bulk shear is `t.tau.n`; total driving shear adds
the fixture's initial material-label background `2.7+.15*cos(3*x_initial)`.
Cumulative slip is the endpoint integral `sum(dt*V)` over accepted physical
steps; the .1 s initialization interval is excluded.

Each table entry is **central / combined ends**, in percent except order.

| Quantity | .1 versus .05 s | .05 versus .025 s | Observed order |
|---|---:|---:|---:|
| Tensor stress, .2 m tube | 1.0599 / 1.2210 | 0.5616 / 0.6243 | 0.912 / 0.966 |
| Bulk shear traction | 0.3168 / 0.7645 | 0.1813 / 0.3973 | 0.805 / 0.948 |
| Deviatoric normal traction | 1.7315 / 1.6133 | 0.9249 / 0.8377 | 0.891 / 0.933 |
| Normal traction including pressure | 0.5480 / 0.9636 | 0.6465 / 0.7991 | -0.248 / 0.260 |
| Effective compression | 0.0831 / 0.1462 | 0.0986 / 0.1219 | -0.248 / 0.260 |
| Projected work shear traction | 0.3101 / 0.6928 | 0.1509 / 0.3560 | 1.040 / 0.964 |
| Slip rate | 3.4746 / 2.2952 | 1.6367 / 1.4570 | 1.070 / 0.663 |
| Cumulative slip | 21.5819 / 29.4129 | 12.2784 / 16.3859 | 0.980 / 1.062 |

Cumulative slip still differs by **12.28% centrally and 16.39% at the ends**
between the two smallest timesteps despite its near-first-order absolute
self-convergence. A small final slip-rate difference is not a small integrated
slip error. The left endpoint is more sensitive: its corresponding final slip-rate
difference is 5.57%, versus .980% at the right. Total normal traction's central
order is negative even though its deviatoric part approaches first order; pressure
and the boundary-coupled mechanics cannot be omitted from the conclusion.

The whole-curve geometry vertex Linf difference decreases .2177 → .1067 mm
on timestep halving. Central/endpoint geometry errors are retained explicitly
below, together with the spatial comparison. Values are Linf in **mm**.

| Comparison | Central | Ends | Left | Right |
|---|---:|---:|---:|---:|
| dt_to_half | 0.2131 | 0.2177 | 0.2177 | 0.1266 |
| half_to_quarter | 0.0996 | 0.1067 | 0.1067 | 0.0482 |
| space | 0.2026 | 0.2030 | 0.1522 | 0.2030 |

## One finer mesh, same physical timestep and labels

32² versus 64² at dt=.025 s, percentages in the same regional norms. The one
mesh pair measures sensitivity of the complete spatial discretization (including
Q1 profile, quadrature and native CFL subcycling); it does not establish a spatial
order or a converged whole-domain reference.

| Quantity | Central | Combined ends | Left end | Right end |
|---|---:|---:|---:|---:|
| Tensor stress, .2 m tube | 0.2430 | 0.1978 | 0.1415 | 0.2449 |
| Bulk shear traction | 0.2694 | 0.1720 | 0.0789 | 0.2304 |
| Deviatoric normal traction | 0.1460 | 0.1824 | 0.1495 | 0.2175 |
| Normal traction including pressure | 0.7837 | 0.7667 | 0.7790 | 0.7283 |
| Effective compression | 0.1187 | 0.1162 | 0.1391 | 0.0814 |
| Projected work shear traction | 0.2452 | 0.1491 | 0.0886 | 0.1922 |
| Slip rate | 0.8729 | 0.7970 | 2.6877 | 0.5989 |
| Cumulative slip | 1.1543 | 1.1102 | 2.6106 | 0.9436 |

Near-fault tensor differences are .243%/.198% central/ends and slip differences
are about 1.15%/1.11%. This is encouraging local sensitivity, conditional on the
retained domain boundary model. The full-domain difference of 37.78% prevents
using the finer run as a globally converged reference. Finer-grid initialization
and profile normalization errors remain part of this coupled mesh comparison.

## Global failure is localized at the domain corner

The saved-field audit partitions the squared full-tensor difference into a
.25 m boundary rim and .5 x .5 m corner boxes. Boundaries align with both meshes,
so Gauss4 exactly integrates these Q2 polynomial squared differences. No new
native run or model change was needed.

| Comparison | Whole-domain relative stress difference | Squared difference in boundary rim | Squared difference in upper-left corner |
|---|---:|---:|---:|
| dt_to_half | 6.698% | 97.1290% | 94.7184% |
| half_to_quarter | 22.364% | 99.9069% | 99.8576% |
| space | 37.783% | 99.9931% | 99.9868% |

Sampled peak tensor norms grow 18.9 → 27.9 → 58.8 Pa across the 32² temporal
runs; the 64² run gives 45.7 Pa. All these sampled maxima are at the upper-left
corner cell; they are not certified Linf values. The final incoming flux through
the upper .5 m of the left boundary grows in magnitude .562 → 1.093 → 6.604 W/m;
the fine-mesh value is 3.815 W/m. This is far from the finite localization ends.
The finest 32² run even has negative net top energy flux, unlike the other three.

This identifies a **boundary/corner-dominated coupled accuracy failure**, not
its unique cause. A mixed velocity/natural-traction corner, frozen accepted-interior
inflow and transport/reaction/projection feedback there have not been separated.
Neither a fault-end-only explanation nor a claim that SSPRK3 or Jaumann rotation
is independently at fault follows from these four runs. Small local normal-traction
errors also inherit globally coupled pressure. No loading, inflow, solver tolerance
or dissipation was changed to conceal this evidence.

## Work and energy

The mature physical bulk storage is E=integral(tau:tau/(4G)). Auxiliary Z has
no physical energy. The endpoint ledger uses
`Delta E - dt*(P_mech - P_slip - D_Maxwell + P_dilatation - F_out)`.
The coupled ledger replaces `-P_slip` by background work minus friction and
radiation. Mechanical power is evaluated from the actual shear solution; it is
**not set to zero** using the earlier pure-translation assumption. The same
endpoint constitutive scheme is retained. A trapezoidal integration of accepted
powers is only a diagnostic, not a different executed scheme.

Normalized values divide accumulated absolute remainder by the trapezoidal sum
of magnitudes of mechanical/slip/Maxwell/dilatation/net-flux work. Signed and
absolute errors are both given in J/m. Endpoint and coupled remainders agree
within the small native surface-work residual.

| Mesh / dt | Endpoint signed | Endpoint absolute | Endpoint / active | Trapezoid signed | Trapezoid absolute | Trapezoid / active |
|---|---:|---:|---:|---:|---:|---:|
| 32² / 0.1 | -0.411021 | 0.411021 | 13.3631% | -0.016414 | 0.036410 | 1.1838% |
| 32² / 0.05 | -0.236620 | 0.236620 | 7.5987% | -0.033874 | 0.037001 | 1.1882% |
| 32² / 0.025 | -0.389971 | 0.389971 | 11.0005% | -0.204740 | 0.205193 | 5.7882% |
| 64² / 0.025 | -0.243464 | 0.243464 | 7.3648% | -0.109186 | 0.109611 | 3.3157% |

Endpoint remainder decreases then increases; trapezoidal signed cancellation
particularly understates the coarse absolute remainder. Thus neither diagnostic
supports monotonically improving global energy accuracy. The native rotation
projection losses are 3.00e-5, 2.13e-4, 8.62e-3 and 2.82e-3 J/m; source-projection
losses are 1.03e-6, 2.74e-7, 1.07e-7 and 7.50e-8 J/m. Reaction projection alone
is far too small to explain the global balance defect. Net transport-energy
changes (.01286, .04065, .29572, .16264 J/m) include boundary exchange and
numerical error and must not be labeled dissipation.

| Mesh / dt | Final outward energy flux (W/m) | Raw face mechanical power | Weak mechanical power | Raw minus weak (W/m) |
|---|---:|---:|---:|---:|
| 32² / 0.1 | -0.276399 | 12.093213 | 11.505878 | 0.587334 |
| 32² / 0.05 | -0.586214 | 11.916039 | 11.500097 | 0.415942 |
| 32² / 0.025 | -8.017413 | 12.139418 | 11.504613 | 0.634805 |
| 64² / 0.025 | -3.512181 | 9.399009 | 11.487817 | -2.088808 |

Weak mechanical power stays near 11.5 W/m while raw face power does not converge.
The independent q8 slip-work adjoint defect is at roundoff, and final q8 surface
work residuals are at most 1.36e-11 W/m. Raising saved-state quadrature to q12
changes bulk storage/mechanical/net-flux powers only at roundoff, but surface
work residuals become 4.33e-7–2.11e-6 W/m: native q8 cancellation alone is not
an independently converged surface balance. These small integration effects
cannot account for the global energy/flux changes above.

Final normalization-measure errors are -.04166%, +.01702%, +.02054%, -.05632%.
Profile mass changes are +.05072%, +.02581%, +.01936%, -.003747%; bounded profile
transport is not a conservative phase-density scheme. Saved-state q8→q12 changes
regional spatial stress-difference norms by at most .176% (relative change of
the norm), and the whole-domain norm by 1.3e-14 relative. This does not reverse
the reported local/global distinction.

## Verification and remaining work

All **44 physical steps** publish history and geometry exactly once, with zero
timestep retries. Independent replay of geometry agrees within 1.03e-16 m;
Q1 profile replay within 4.45e-16, donor-bound violation below 2.78e-16, age
source within 1.78e-15 s and diagnostic memory source within 1.39e-17 Pa m.
Native stage-energy storage agrees with the independently reconstructed fields.
Region partitions sum to the full reported tube/curve norms. Initial temporal
states are bitwise identical; fixed label geometry across meshes is verified.
No rejection/restart change was introduced, and no extra lifecycle trajectory
was run beyond the requested four cases.

**Implemented:** benchmark analysis/figures and compact provenance; native method
unchanged. **Locally verified:** accepted transactions, finite measurable spin,
bounded in-domain curve/profile, and near-fault field sensitivities.
**Physical comparison:** conditional numerical refinement only, with unresolved
cumulative-slip accuracy, pressure-inclusive normal-traction order and a failed
global boundary/energy comparison. **Production readiness:** not established.
Mature physical histories remain distinct from cohesive-interface histories and
production BP3. The earlier frozen split-Q2 result is preserved, but does not
resolve this boundary-corner failure. Inherited losses of 39/45 native-pilot
frames and earlier temporary logs remain unrepaired.

**Next justified experiment:** one bounded prescribed-velocity affine-shear
boundary control with an explicit smooth nonzero initial stress and analytic
time-dependent exterior history, compared with the retained accepted-trace
inflow. Isolate the upper-left mixed-boundary/inflow feedback before another
refinement matrix or attributing the global failure to tensor approximation or
stress rotation. If it separates transport from coupled mechanics, then repeat
a short coupled control with compatible mechanical corner data. These are
proposals, not additional runs performed or automatically selected here.
