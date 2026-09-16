# K5: frozen-cohesion mechanism diagnostic — stopped with sufficient evidence

## Decision summary

The data establish a concrete contributing mechanism: the frozen-profile
cohesive law retains an almost elastic slip-hardening resistance. Its growth
is not canceled by the shear driving traction. Free-fault slowdown then
increases the accumulated-slip mismatch against the prescribed deep rate,
and the junction pressure dipole follows that mismatch. In the clean,
same-timestep comparison at 29.24 yr, freezing only the initial evaluated
cohesive force reduces the last-element slip gradient by 73% and the raw
junction pressure range by 75%. An exact weak-row decomposition also shows
that growth inherited from the prescribed vertex alone exceeds the reaction
at the original first lower-bound contact.

This is not the sole mechanism: the modified run still slows, and its later
timestep sequence differs. It does not establish that contact is permanently
prevented or that frozen cohesion is a valid production model. Per the user's
conditional stop instruction, the run was stopped after accepted step 52
(60.1876 yr); unfinished step 53 is excluded. No additional solve is needed
to support this conclusion. Recommend a physical review of the frozen-profile
cohesive spring's role in BP3, not another replay or an automatic code change.

The user authorized removal of the benchmark timestep-matching guard after
the guarded run in `stage_K5_cohesion_replay.md`. This means allowing the
unchanged production controllers to select smaller steps, not disabling their
stability restrictions. The saved comparison times remain upper caps, with
additional accepted substeps as necessary. End at 2232176379.2516127 s
(70.73340112 yr), not at a fixed timestep number. The existing fresh-only
diagnostic is rerun from initialization; the interrupted evidence is preserved.

Only benchmark `replay_time_step.h` gains an opt-in adaptive comparison clock.
The frozen evaluated initial cohesive force, consistent K tangent, original
nodal Theta, physical problem, mesh, support, I_h, pressure and solver settings
remain those of the preceding experiment. Default production constitutive
equations remain unchanged. Run:

`python3 benchmarks/reconstructed_fault/bp3/run_frozen_cohesion.py --adaptive`

Output: `benchmarks/reconstructed_fault/bp3/frozen-cohesion-adaptive-50-local4/`.
Initial cost estimate was 25--35 minutes on four local MPI ranks. This
underestimated the additional production-controlled substeps: after about
58 yr the controller selected steps of order 0.1 yr. No production timestep
restriction was disabled to meet the initial runtime estimate. The
discriminating question is whether the slower early accumulation of the
junction slip deficit persists through the original late comparison time,
and whether pressure concentration follows that deficit rather than the
cohesive force directly. No second physical counterfactual is introduced.

## Mechanism to test

The specification's common cohesive-state equation, with C denoting
T_coh, is

\[
 C_k^{\rm eval}(s)=\frac{\kappa_{\Gamma,k}V_k(s)
       +\beta_{\Gamma,k}I_{k-1}(s)\widehat C_{k-1}(s)}{I_k(s)},
 \quad \kappa_\Gamma=\eta_\Gamma(1-\beta_\Gamma).
\]

The evaluated response is subsequently projected to committed Q1 C. It is
not a recurrence evaluated independently at the vertices. For frozen I,

\[
 C_k^{\rm eval}-\widehat C_{k-1}
 =\kappa_\Gamma\left(V_k/I-\widehat C_{k-1}/\eta_\Gamma\right).
\]

Here G_Gamma=32.03812032 GPa, eta_Gamma=1e26 Pa s, and I near the last free
node is 12960.25 m. Thus G_Gamma/I is about 2.47 MPa/m. For MPa-scale C,
the growth threshold I C/eta is of order 1e-16 m/s, far below Vp=1e-9 m/s.
The zero frozen-profile crack-strain **history correction** does not imply
a zero cohesive-state update.

The exact residual convention is R=q-C-mu sigma_n-damping V. At fixed bulk
unknowns, q's direct derivative is -2 kappa_bulk chi S:S while the cohesive
resistance derivative is +kappa_Gamma/I: both resist increasing V in R.
There is no equal-and-opposite algebraic cancellation. The fixed initial
background includes initial C only; it does not grow with later C.

1. Frozen I_h leaves an approximately elastic cohesive stiffness G_Gamma/I_h.
   C grows with slip; fixed background shear does not cancel its growth.
2. In the velocity-strengthening region, this reduces q-C available for
   friction. The steady-state logarithmic slope is sigma_n(a-b), about
   0.5 MPa here. Test this scale against the actual weak force changes,
   keeping preceding-state versus steady-state friction explicitly separate.
3. The imposed Vp below 40 km cannot participate in this slowdown. Accumulated
   slip deficit divided by the unchanged 50-m last-element spacing creates
   a growing junction gradient. Compare pressure-dipole amplitude against
   this gradient for the original and frozen-force trajectories.
4. Separately retain any remaining slowdown, nodal-state timing differences,
   and deep-tip stress concentration. Similarity is not proof of a universally
   linear pressure law or a physically justified replacement cohesive model.

The physical distinction is important: at fixed I_h the cohesive law is an
interface Maxwell spring, with elastic stiffness G_Gamma/I_h and relaxation
time eta_Gamma/G_Gamma. The 70-year experiment is effectively elastic compared
with its approximately 99-million-year relaxation time. This spring survives
even though the phase field is no longer evolved. The ordinary BP3 friction
condition does not contain that independent slip-hardening spring.

In the continuous fixed-profile limit (before spatial history projection),
`dot C+C/tau=k_Gamma V`, with `k_Gamma=G_Gamma/I_h` and
`tau=eta_Gamma/G_Gamma`. Its elastic energy per area is
`W=I_h C^2/(2G_Gamma)`, giving `dot W=C V-I_h C^2/eta_Gamma`.
Thus growth represents additional elastic storage, not a roundoff error or
a missing algebraic cancellation. Frozen-force diagnostics remove that
restoring stiffness; promoting them requires a model/energy decision.

The relevant sensitivity scale is not the approximately 26.5 MPa total
frictional traction. It is sigma_n(a-b), approximately 0.5 MPa in the VS
region. At 15.31 yr the added baseline cohesive force relative to the
frozen-force run is 0.699663 MPa; only 0.222709 MPa is compensated by extra
q. The resulting 0.476954 MPa reduction in available frictional traction is
almost one logarithmic rate interval on that quasi-steady scale. Exact weak
friction uses preceding nodal Theta and nonuniform quadrature-point rates;
this scale estimate is not substituted for that discrete law.

For a permanently frozen profile, replacing C by C_star makes the residual
`(tau_bg-C_star)+delta_q-mu sigma_n-damping V`; the constant resistance can
be absorbed algebraically into an effective fixed shear background. This
explains what the counterfactual removes, but does not make its normally
evolved C/H shadow histories an energy-consistent constitutive replacement.
The specification currently includes the spring. Changing that physical
choice is separate from correcting a software defect.

The already saved early comparison also excludes tension as the initiating
mechanism: substantial slowdown and a junction pressure dipole are present
while every sampled normal traction remains compressive. The chronology is
compatible with hardening -> slip deficit -> stress concentration, rather
than assuming that tensile normal stress first locks the node.

The earlier 35-km frozen-history intervention must not be misreported as
moving the dominant old dipole. As recorded in
`stage_K5_normal_stress_consolidated_report.md` section 4, it produced a new
35-km feature and weakened, but did not relocate, the dominant 40-km feature.
That is compatible with nearly elastic retained stress recording the prior
accumulated slip mismatch. The present replay changes that accumulated
mismatch from initialization; it does not merely move the instantaneous
junction in an already stressed state. The older probe used a different
comparison configuration and is qualitative supporting evidence, not another
point in the present 50-m quantitative comparison.

## Result

### Exact retained-cohesion contribution at the original junction

The frozen-profile retained term is exactly `beta M C_(k-1)` in the production
surface residual. Using its saved mass matrix, no quadrature reconstruction
or nodal-recurrence assumption is needed. For row 796 (39.95 km), the fractions
from vertices 795/796/797 are approximately 1/6, 2/3, 1/6. Vertex 795 is the
prescribed Vp vertex at 40 km. Its cohesive history continues growing even
though its own mechanical residual row is replaced by the velocity constraint.
That history enters the last free row through genuine Q1 weak coupling.
Here "prescribed-side" means the **prescribed vertex trace**: the product
N_796 N_795 is supported on the last RSF element, 39.95--40 km. This is not a
claim that the free test function extends arbitrarily into the deep prescribed
region. The measured contact effect belongs to this tested 50-m discrete
junction; no continuum or fault-grid convergence claim is inferred from it.

Values below are actual baseline row-weight-normalized contributions in MPa.
"Growth" subtracts initial retained C0 using the **current** mass matrix.

| Baseline step | Prescribed memory | Self memory | Other free neighbour | Current instantaneous term | Prescribed memory growth | Physical R | R with only that growth removed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 0.457916 | 0.840003 | 0.279438 | 0.524835 | 0.376304 | ~0 | +0.376304 |
| 12 | 0.798397 | 0.928703 | 0.393458 | 0.210397 | 0.716786 | -0.375776 | +0.341009 |
| 13 | 0.957362 | 0.961840 | 0.434808 | 0.032428 | 0.875750 | -1.098618 | -0.222867 |

At first contact (step 12), prescribed-side memory growth alone exceeds the
bound reaction and reverses the frozen-state residual sign. At step 13 it
does not suffice alone: self and neighbouring retained growth plus the
coupled response also matter. This is a force-budget sufficiency statement
at fixed bulk/V/Theta, not another solve or a prediction that all other fields
would remain fixed after re-equilibration. No new counterfactual was run.

This localizes one concrete mechanism beyond broad hardening: ongoing
prescribed-side slip accumulates cohesive resistance which is sampled by a
neighbouring **free** weak row. There is no corresponding guarantee that q
from that side increases by the same amount. It is a consequence of the
specified cohesive state and Q1 coupling, not evidence of an assembly error.

The lower-bound decision is a **weak nodal** decision. Setting V_796 to its
bound does not set quadrature-point V to that value: on the last element it
still contains N_795 Vp. Likewise, interpolated retained C and Theta still
contain neighbouring states. Consequently C and friction contributions to
row 796 remain finite when that node contacts the bound. This is distinct
from interpreting a local pointwise RSF law at V_min, and does not require
tension to initiate the slowdown. The separately established nodal-Theta
interpolation effect can amplify this weak-row effect; it is not removed or
retested in the present replay.

### Adaptive replay

**Stopped deliberately, not failed:** 53 accepted states (0--52), through
1899376326.5656066 s = 60.18760383 yr. Four MPI ranks used 3725.79 s wall
time (62.10 minutes). The unaccepted step 53 was interrupted with SIGTERM
to the identified MPI launcher; all ranks exited. Its partial iterations
are not trajectory data. The ordinary checkpoint in restart/02 is the
last complete step-52 snapshot; this opt-in frozen-force diagnostic remains
fresh-start-only for mechanics. No diagnostic restart continuation occurred.
The nonzero exit and runner assertion are consequences of that intentional
stop, documented separately in stop_record.json, not a numerical failure.

The replay reproduces the original comparison clock through step 10,
29.24190894 yr; its first additional substep follows that state. Consequently
the following early comparison isolates the force intervention without an
intervening timestep-history difference:

| At 29.24190894 yr | Original C | Frozen initial force |
|---|---:|---:|
| Last free V/Vp | 0.152487 | 0.729546 |
| Last-element accumulated-slip gradient | 0.012465718 | 0.003365138 |
| Junction raw bulk pressure minimum (MPa) | -19.882718 | -4.883373 |
| Junction raw bulk pressure maximum (MPa) | 18.816164 | 4.803925 |
| Pressure range / slip gradient (GPa) | 3.104425 | 2.878722 |

The gradient falls by 73.0%, and the pressure range by 75.0%. Pressure here
is the actual Q1 bulk field sampled at identical exported vertices in
39--40.5 km, |normal distance| <=1500 m, not pressure at selected surface
normal-stress extrema. The similar pressure/gradient ratios support the
accumulated-slip-gradient mechanism; they are not an exact scalar law.

At the next common time, 55.85900364 yr (original step 11, diagnostic step
26), last-free V/Vp is 0.0448697 versus 0.200419, and the gradients are
0.028511364 versus 0.012015237. The diagnostic still slows: freezing C does
not remove all causes of slowdown. This later comparison includes a changed
time-discrete history and particle-transfer sequence and is not a same-clock
measurement of the cohesive contribution alone.

At 55.859 yr the raw normal-traction range on the last mixed free/prescribed
element is 3.61259--95.33494 MPa originally, versus 32.92670--67.02874 MPa
with frozen force. The corresponding minimum's decomposition is:

| Case | Surface coordinate (m) | Delta p (MPa) | -Delta tau:N (MPa) | sigma_n (MPa) |
|---|---:|---:|---:|---:|
| Original | 39997.5077 | -40.89227 | -5.49514 | 3.61259 |
| Frozen force | 39975.3542 | -13.07036 | -4.00294 | 32.92670 |

These are raw production constitutive samples, not the much smoother
row-mass-normalized weak traction plotted in comparison.png. The absolute
global minimum still comes from the separate bottom/deep fault tip at about
115.47 km: -9.10755 versus -8.48891 MPa at this common time. Freezing C does
not remove that separate boundary feature.

At the deliberately stopped diagnostic state (60.1876 yr), last-free
V/Vp=0.1165005 and the last-element slip gradient is 0.01432275. All 440
free nodes remain free. This time has no saved baseline accepted state;
do not compare it directly to the baseline's 70.7334-yr endpoint or claim
that its original 68.6196-yr contact has been prevented. The remaining
slowdown is material, and its origin is not fully separated here.

### Recovering pressure without an additional mechanical solve

Adaptive inserted states are not all heavy-output states. Where a final
accepted state lacks VTU pressure, the existing read-only checkpoint export
can recover its Q1 polynomial on the same junction vertices. This is not a
restart continuation: it stops immediately after export and commits no state.
The parser was checked against the already saved original step-13 export
and VTU: all 1730 unique junction vertices agree **exactly after the VTU's
float32 conversion**. Maximum double-versus-exported difference is 1.99064 Pa,
the expected output rounding at a roughly 56-MPa pressure magnitude.
The actual history/pressure export remains separate from ordinary graphical
output. The export's advanced checkpoint clock must be step k+1 with
time-dt equal to accepted-state k; it must not be mistaken for a new solve.
Only the existing **baseline** export was used for this offline validation.
No new checkpoint-export process was needed after the stop. In particular,
there is no heavy bulk-pressure output at diagnostic step 26; the 55.859-yr
comparison uses its actual saved raw surface samples, not a nearby bulk
output silently substituted for that time.

## Verification and artifacts

- Release BP3 plugin rebuilt with `cmake --build benchmarks/reconstructed_fault/bp3/build -j4`;
  successful build log: `/tmp/bp3-cohesion-adaptive-build.log`. Core executable
  and frozen-force constitutive branch were unchanged in this continuation.
- `python3 benchmarks/reconstructed_fault/bp3/run_frozen_cohesion.py --adaptive`:
  accepted states 0--52 verified, then intentionally stopped. The previous
  guarded run and the baseline are untouched.
- All 53 accepted states have genuine bulk/surface convergence. Maximum
  reported final relative residuals are 5.6471e-13 bulk and 9.9302e-9 surface,
  both below the unchanged 1e-8 target. All 249 recorded fresh-linear checks
  passed; one belongs to the interrupted step 53, leaving 248 for accepted
  states. Maximum fresh/target ratio is 0.9975724.
- The existing four-rank frozen-force derivative checks passed again:
  K error 3.60677e-7 -> 3.50878e-8 when the perturbation shrank from
  1e-15 to 1e-16 m/s; G pressure error 9.30134e-13. The existing snapshot
  and trial-invariance tests were reused, not broadened.
- `python3 benchmarks/reconstructed_fault/bp3/analyze_frozen_cohesion.py --adaptive`:
  **passed**. Checks each accepted nodal Theta update against its own preceding
  state at 1e-12 relative tolerance, slip accumulation, exact deep Vp, unchanged
  geometry/backgrounds, I_h agreement at 1e-12, and the full weak force identity.
  The independent preceding-Theta friction check differs by at most 2.11e-15
  in mu. Initial Theta/C/I_h/backgrounds are identical; initial V differs by
  at most 8.27e-25 m/s. The old guarded prefix agrees to roundoff.
- The saved pressure-polynomial parser validation passed on all 1730 vertices
  against existing step-13 graphical output, as described above. No new bulk
  integration, smoothing, or altered graphical fields were introduced.
- No broad regression suite, further counterfactual, or production model
  change was run for this continuation. The runner and analyzer changes are
  benchmark-local. `run_theta_exact.py` gained an export-only alternate-case
  selector; that new selector was not exercised with a new ASPECT process and
  must not be described as independently runtime-qualified.

The result directory is
`benchmarks/reconstructed_fault/bp3/frozen-cohesion-adaptive-50-local4/`.
It retains `run.log`, `execution.json`, `stop_record.json`, `provenance.json`,
the initial force snapshot, accepted fields, and rotating checkpoints.
The `comparison/` directory contains:

- `summary.json`, `common_times.csv`, `series.csv`;
- `force_budgets.csv`, `cohesive_memory.csv`, `steady_rate_scale.csv`;
- `junction_profiles.csv`, `raw_support_extrema.csv`;
- `pressure_slip_mechanism.csv`, `pressure_pattern.csv`, `comparison.png`.

At 29.24 yr the pressure-pattern correlation over 1730 common vertices is
0.99172. Scaling the original field by the independently measured slip-gradient
ratio leaves 20.6% relative RMS error: good evidence of the same dominant
dipole, but not an exact pressure-amplitude formula. At 15.31 yr the analogous
correlation is 0.99612 and error 12.8%.

## Next decision

Review whether the pre-existing, permanently frozen BP3 fault should carry
this extra finite-width cohesive spring, or whether a separately justified
frictional constitutive specialization is needed. The diagnostic supports
that model question; it does **not** authorize simply freezing production C.
Retain the known nodal-state interpolation and bulk-history transfer issues
as separate mechanisms. No claim of a unique cause, continuum junction
convergence, permanent elimination of bound contact, or repaired deep-tip
boundary behavior follows from this bounded result.
