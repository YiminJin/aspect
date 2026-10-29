# Event-resolved oblique admission

**The prescribed append passes tied insertion, physical damage admission and
history/checkpoint checks. Temporal response differences are small; quantitative
physical-energy qualification remains incomplete at the sliding-dissipation
scale.** No free path or fully decohered extension is demonstrated.

[Selected scope and gates](COHESIVE_INTERFACE_ADMISSION_SELECTION.md) ·
[figures, data and commands](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_admission/INDEX.md).
Baseline: `pf-rsf-particle-free`, `f1f9f29cf40c32d1ddcea000b724375b64853532`,
with prior local work preserved. Production/BP3 and physical calibration are unchanged.

## Experiment and admission

Reuse the fitted native 16×8 Q2/P1disc kernel, horizontal old segment and
26.565° append. Prepare the block at axial strain/memory .0015, physical time
.2 s, zero slip/damage and dimensionless age 1.001. The initial bulk energy is
4.5 J/m and is subtracted from subsequent storage changes. Apply
u_boundary=(gamma*x,−gamma*y), gamma_dot=.001/s through .65 s, then hold
boundary displacement to .66 s. The append is installed at .25 s while its
required weak multiplier is **2478.78 Pa**, below 3000 Pa. Insertion changes no
physical history, stress or energy and advances no physical clock.

The whole physical trajectory agrees **bitwise** with a preinstalled candidate
at the same timestep. The material clock is retained in prospective material.
Bisection from an immutable incoming state locates loss of the discrete tied
certificate. The lower tied endpoint is accepted; the remaining timestep uses
the unchanged irreversible cohesive–RSF return and full current pressure.

| Measurement | Δt=.005 s | Δt=.0025 s |
|---|---:|---:|
| Discrete tied-loss bracket (s) | [.5814467144, .5814467239] | [.5814419651, .5814419746] |
| First accepted positive-damage time (s) | .585 | .5825 |
| Final peak surface damage | .02687483 | .02694529 |
| Final integrated damage (m) | .00558906 | .00560302 |
| Final peak absolute jump (μm) | 17.9166 | 17.9635 |
| Fully decohered length (m) | 0 | 0 |

The analytic homogeneous tied Maxwell control loses admissibility at
**.5814372075 s**. Thus the <1e−8 s bisection bracket resolves a *discrete* event;
it does not eliminate timestep error or qualify physical first activation.
The timestep shift is 4.75 μs. At common times .585/.6/.625/.65/.66 s, integrated
response differences stay below .25%; the final peak difference is .2615%.
The declared 2% temporal-response target passes on this mesh.

All new-segment Gauss points develop small damage after admission; the old
horizontal segment remains intact. The damaged append length is .2795085 m,
with a≥.01 over .240688 m at the endpoint, but **a=1 is never reached**.
This is broad activation of the prescribed segment, not tip-localized propagation.
During the hold, integrated damage increases from .00544826 to .00560302 m.
Stopping boundary motion does not remove stress or freeze the histories.

## Weak solve, pressure and lifecycle

Q2 trace functions change sign. The new reference uses an unrestricted signed
primal/dual predictor, general Gauss-tie nullspaces and complete bounded weak
certification. It does not inherit the straight Q1 fixture's positivity rule or
clamp entire cells when only one Gauss constraint is active. A separate partial-
Gauss tie reference retains six free modes and gives a reduced derivative error
of **2.34e−9**. Exact tie equalities, rather than a damage floor, set intact jumps.

The largest coupled tangent FD error on the fine trajectory is **1.78e−8**
(tolerance 1e−6); maximum weak residual is **1.77e−11 Pa** (tolerance 1e−8 Pa).
The minimum sided contact pressure over accepted positive steps is **11802.95 Pa**
(the prepared initial value is 11800 Pa). The restricted,
mass-scaled symmetric part of the complete incremental tangent remains positive,
minimum **7.33e6 Pa/m**. This is a local admissible-branch diagnostic of the
nonsymmetric pressure/RSF problem, not a global potential-Hessian certificate.

Repeated trials, deliberately invalid publication, retry, duplicate publication
and reference restart across activation pass. The native adapter additionally
checks three activated states with the **compiled unchanged C++ point return**,
full pressure response, Maxwell recurrence, complete weak force and BE ledger
before publishing the actual native Gauss histories. It advances physical
histories once and separately preserves the single geometry publication.
Actual ASPECT checkpoint/restart is bitwise identical to uninterrupted replay.
Maximum native scaled source error is **8.68e−19**, weak residual **1.15e−12 Pa**
and ledger error **6.44e−15 J/m**. The native host scheduler's millisecond ticks
are solely checkpoint callbacks; physical timesteps come from the saved sequence.

This native adapter verifies and publishes an independently solved mixed-kernel
sequence; it is **not a production outer Newton/topology solver**. Default-off
replay leaves the preceding native kink-owner control bitwise unchanged.

## Energy interpretation and retained failure

At the fine endpoint, new fracture energy is **.00560302 J/m**, sliding loss
**6.35315e−5 J/m**, Maxwell loss **.0551991 J/m**, and trapezoidal external work
**3.0981230 J/m**. The cumulative absolute physical-work remainder decreases
from **7.98671e−5** to **4.00062e−5 J/m** on halving; the fine signed remainder
is −3.98829e−5 J/m, so cancellation is not hiding the error.

The fine remainder is **.7140% of new fracture energy** (passes the declared
1% screen), but **62.97% of sliding loss** (fails). It is only .001291% of total
work; that small work-normalized value does not qualify the sliding balance.
The original endpoint-Maxwell accounting is retained. Using trapezoidal Maxwell
loss as a diagnostic alternative still leaves 3.56997e−5 J/m absolute remainder;
it does not repair the sliding-scale failure.

An analytic homogeneous Maxwell block without a fracture interface, using the
same BE loading/time sequence, gives **3.79919e−5 J/m** absolute remainder on
the fine schedule—about 95% of the interface result. The absolute difference
between the two stepwise remainders is 2.01430e−6 J/m. This identifies a dominant
bulk time-discretization contribution; it is **not subtracted from the acceptance
error** and does not prove the small sliding work accurate. BE ledger closure
(maximum 4.55e−15 J/m in the reference) is reported separately.

## Scope and next action

Gc is counted once through surface damage; phi remains provisional geometry.
The corner keeps zero vector jump under the retained continuous-trace,
zero-opening rule. No old direction/history is rotated or reset. Pressure space
and bulk mesh remain fixed. The axial loading is a component test on prescribed
planes: other, unrepresented maximum-shear planes would reach intact strength
earlier. It is **not** an unrestricted nucleation or direction-selection specimen.
The older 65.4% early straight-growth spatial failure and pointwise tip sensitivity
remain unchanged. No new spatial campaign, production/BP3 change, MPI, transport,
AMR or free-path evolution was undertaken.

The admission/lifecycle milestone is supported. Before longer propagation or a
quantitative sliding-energy claim, the smallest justified follow-up is a bounded
temporal work/Maxwell-dissipation consistency check on this same activated
append, using the analytic bulk control and preserving the failed raw screen.
Do not tune fracture/friction, add damping or reinterpret ledger closure as a
physical fix. Freely sliding through the corner still requires an explicit
normal-contact/kinematic choice; no such choice is made here.

Reference trajectories cost .46 s/.84 s plus .44 s for the preinstalled control;
the three native owner/checkpoint invocations cost 3.73/1.67/1.57 s, sequentially.
The existing mixed kernels were reused. No server or long specimen run was launched.
