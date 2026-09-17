# K5 slip-rate notch: frozen-history mechanism experiments

## Decision summary

The gradient-kink hypothesis has experimental support, with an important
distinction: a change in slope provides the disturbance; the present coupled
spatial operator can turn it into an oscillatory rate response. With histories
frozen, a +1% Vp disturbance at 40 km makes the 39.95-km free node **slower**
by 0.592% Vp. A half-amplitude test agrees. An interior 25-km control creates
the same alternating response without a friction transition or an existing
prescribed-slip junction. Direct operator measurement identifies a positive
nearest-neighbour coupling, dominated by the Maxwell/source term remaining
after bulk relaxation, not by normal-stress feedback.

These tests isolate a spatial mechanism; state evolution subsequently
amplifies it. They do not establish that a particular mass-lumping, mesh,
or finite-width correction is justified. The latest 15-km profile is a broad
aging-driven slowdown, not a local minimum. The small early 18.1-km dip is
absent at the final revised-work state. No production equations, tolerances,
histories or accepted trajectory were changed. Four disposable solves are
complete; no more runs are needed for this mechanism review.

## 1. Which features are present?

The authoritative comparison here is the **mature, revised-work** 50-m junction
fixture, not the older cohesive or volume-weighted trajectories. Its accepted
step 10 is at **29.24190894 yr**. Surface spacing is 100 m at 15/18/25 km and
50 m at the 40-km junction; the fault-crossed bulk spacing is approximately
97.7 m. All rates below are manager/Q1 nodal values, not interpolated graphics.

| Location | Saved observation | Interpretation supported by these data |
|---|---|---|
| 15 km | V/Vp = 3.24664e-5 at step 10; monotone through 14–16 km | Strong broad slowdown and slope change, not a narrow local minimum |
| 18 km | At step 5, V18/Vp = 0.770936 and V18.1/Vp = 0.768947 | A small transient downstream dip, 0.001989 Vp below the preceding node |
| 18 km, late | Step-10 V18/Vp = 0.292663, V18.1/Vp = 0.295460; monotone over 17–19 km | The local dip does not persist in the latest formulation at this time |
| 40 km | V39.90/Vp = 0.957303, V39.95/Vp = 0.839369, V40/Vp = 1 | Strong genuine undershoot at the last free node |

The historical mature **volume-rule** step 10 had V18/Vp = 0.381265 followed
by V18.2/Vp = 0.321816. That trajectory is not interchangeable with the
revised-work baseline: both mechanical sampling/weighting and evolved history
differ. Its larger dip must not silently be described as the current result.

The user's common-feature observation is visible in the actual one-sided
rate gradients. At step 5, at 18 km they change from **+1.00064e-12** to
**-1.98919e-14 s^-1**. At step 10, at 40 km they change from
**+3.21263e-12** to **zero s^-1**. At 15 km they remain positive, changing
from **1.05933e-16** to **3.82846e-16 s^-1**. Thus a gradient change is common,
but a sign-reversing rate slope/local notch is not common at every time.
The accumulated-slip gradient jump at 40 km is **-0.00213375224** at step 10;
at 18 km it is **-0.00021462804**, even though the instantaneous late rate
profile is monotone. Stress can therefore retain a feature after the rate dip
has weakened. Correlation with a gradient is not, on its own, a causal proof.

See [profiles](../../../benchmarks/reconstructed_fault/bp3/notch-mechanism/notches_by_time.png),
[extrema inventory](../../../benchmarks/reconstructed_fault/bp3/notch-mechanism/extrema_inventory.csv),
and [one-sided gradients](../../../benchmarks/reconstructed_fault/bp3/notch-mechanism/gradient_changes.csv).

### Initial aging explains the broad shallow slowdown

Official supplied Theta0 is **8000 s** in the VW region, versus steady
Dc/Vp = **8e6 s**. The initial mechanical root does not imply steady state.
After the first real step, Theta near 14/15 km is approximately **2.14360e6 s**.
At fixed Vp and 50 MPa this is a **4.19310-MPa** increase in frictional
resistance. The independent homogeneous, fixed-stress estimate

\[
 V_2/V_p\simeq(\Theta_0/\Theta_1)^{b/a}
\]

gives **0.000227992** at 14 km; the actual coupled step-2 value is
**0.000232294** (1.89% higher). This simple control already explains most of
the early broad VW slowdown without a spatial notch or tensile stress.
It is a local diagnostic, not the reference for the whole nonuniform fault.
At 18 km and deeper, the supplied state starts at steady Dc/Vp; the analogous
first-step state-only resistance change is only a few Pa. There the subsequent
slowdown also involves elastic loading transferred from the shallow region.

One additional representation detail is retained, not corrected away:
the actual projected surface composition gives a15 = **0.01056236** and
a18 = **0.02442745** at the audited step, rather than the sharp-fault endpoint
values 0.010/0.025. This follows from the current particle/surface composition
projection. The frozen background already belongs to this initialized
representation. It is a confounder for quantitative 15/18-km attribution,
not a newly demonstrated production bug or permission to replace that mapping.

## 2. The exact equation being tested

The current mature work rule uses each owned bulk Stokes quadrature point once:

\[
 R_i=\sum_q J_q\chi_q N_i
   [q_q-\mu(V_q,\Theta_q)\sigma_{n,q}-\eta_d V_q],
\quad V_q=\sum_jN_jV_j,
\quad\Theta_q=\sum_jN_j\Theta_j^{old}.
\]

Here q and sigma are current constitutive tractions computed from the frozen
**working FE old stress**, current velocity/pressure and crack-strain source.
They are not the newly committed particle stress or a post-solve history field.
C=0 in this mature model. All new experiments use the lagged state, exactly
as the saved A control; no aging update occurs in any new solve.

For the frozen linearization,

\[
 A\delta x=B\delta V,\qquad
 \delta R=G\delta x-K_V\delta V
          =-S_\Gamma\delta V,\qquad
 S_\Gamma=K_V-GA^{-1}B.
\]

Consequently an imposed-rate perturbation obeys
S_FF delta V_F = -S_FP delta V_P on free nodes. A well-converged solve does
not guarantee a monotone response: neither the consistent Q1 matrix nor this
bulk-relaxed operator is required to satisfy a discrete maximum principle.
No assumption that K_V or S_Gamma is SPD is made by these tests.

In straight fault coordinates, the source has shear component chi(s,n)V(s)/2.
Its two-dimensional strain-compatibility expression contains
**-partial_s partial_n[chi V]**. Along-fault variation therefore creates
elastic incompatibility; a sudden slope change can localize the response.
This identifies why gradients matter, but does not assert that every pressure
peak or slope change is a discretization error.

## 3. Frozen-history causal experiments

All cases copy the same accepted step-9 checkpoint. The physical interval is
15.30625726–29.24190894 yr, dt = **439775721.3758315 s**. Ordinary advection
and transfer occur identically before mechanics; per-rank fingerprints of
the incoming represented fields agree with the previously verified lagged A
control. Bulk boundary conditions, friction, pressure treatment, phase, Ih,
support, source, mesh and tolerances are unchanged. The alterations below are
explicit **diagnostic constraints**, not an alternative accepted BP3 loading.

| Experiment | Normalized rate response | Result |
|---|---|---|
| +0.01 Vp at the prescribed 40-km node only | delta V39.95/delta V40 = **-0.591699**; delta V39.90/delta V40 = **+0.0435688** | Boundary disturbance generates undershoot followed by overshoot |
| +0.005 Vp, otherwise identical | **-0.591885**, **+0.0436466** | Adjacent response differs only 0.0313%; not a large-perturbation effect |
| Pin the 25-km node at its saved converged value +0.01 Vp | Immediate shallow/deep neighbours: **-0.459824/-0.450361**; next neighbours: **+0.123276/+0.118351** | Same alternating pattern away from friction/BC transitions |

The 25-km unperturbed rate is locally smooth and the material is wholly VS.
Its old Theta is retained, not made uniform or reset. The unperturbed saved
solution satisfies the extra point constraint exactly. This control establishes
that neither a special friction coefficient at 40 km nor a new state update
is needed to generate a notch response. It does not claim that an imposed
interior point force is part of BP3.

[Measured response table](../../../benchmarks/reconstructed_fault/bp3/notch-mechanism/coupled_boundary_response.csv),
[junction plot](../../../benchmarks/reconstructed_fault/bp3/notch-mechanism/boundary_impulse.png),
[interior plot](../../../benchmarks/reconstructed_fault/bp3/notch-mechanism/interior_impulse.png).

### Direct versus bulk-mediated coupling

The fourth experiment holds **all other rates** at the saved converged values
and adds the same junction impulse. Only bulk u,p re-equilibrate. Unreplaced
surface residuals then measure a column of S_Gamma. They are intentionally
not zero on these diagnostically prescribed rows. This is not claimed as
free-surface residual convergence.

At row 796 (39.95 km), column 795 (40 km), multiply the derivative by Vp and
divide by row measure **49.99709446 m**. The actual decomposition is:

| Contribution to S_Gamma[796,795] Vp / row measure | MPa |
|---|---:|
| Direct Maxwell/source Q1 term | **+4.111046** |
| Bulk shear relaxation | **-1.691972** |
| Direct friction-rate derivative | **+0.227426** |
| Normal-stress feedback through friction | **+0.009737** |
| Radiation damping | +7.7252e-10 |
| Total tangent at the baseline | **+2.656238** |

Thus bulk relaxation removes only part of the positive nearest-neighbour
source coupling; the remaining shear contribution is **+2.419074 MPa**.
It dominates the sign. The finite 1% residual difference is +2.655642 MPa;
the 0.0224% difference is accounted for by evaluating the nonlinear friction
over a finite increment. The tabulated tangent instead uses the analytic
baseline mu_V and the affine bulk pressure/stress response.

The Q1 origin of a non-monotone local response is independently visible
without re-solving bulk. A uniform consistent reaction matrix gives the
recurrence delta V_(i-1)+4 delta V_i+delta V_(i+1)=0, with decaying ratio
**-2+sqrt(3) = -0.267949**. Using the **actual** saved work mass gives
**-0.269052**, and the full fixed-bulk K_V gives **-0.269564** at the adjacent
node. The fully coupled value is -0.5917. The simple mass result explains
the alternating tendency, but must not be substituted for the full operator.

In the freely responding 1% junction test, the actual last-row incremental
balance is also revealing:

| Change in row-mean contribution | Pa |
|---|---:|
| Shear from changed bulk strain | **-56763.529** |
| Shear from changed crack-strain source | **+53655.725** |
| Friction change from V | **-3275.802** |
| Friction change from sigma_n | **+168.033** |
| Friction cross term | -0.034984 |

Driving changes sum to -3107.804 Pa, balancing the friction change. Normal
feedback is secondary here. This is not tension-induced locking: all free
nodes remain off the lower bound, and the baseline last-two-element normal
tractions are **47.208–52.673 MPa**, compressive throughout.

See [operator column](../../../benchmarks/reconstructed_fault/bp3/notch-mechanism/bulk_relaxed_operator_column.csv),
[tangent decomposition](../../../benchmarks/reconstructed_fault/bp3/notch-mechanism/operator_column_decomposition.json),
and [incremental force budgets](../../../benchmarks/reconstructed_fault/bp3/notch-mechanism/impulse_force_budget_Pa.csv).

## 4. Mechanism and limits of the conclusion

The evidence-supported sequence is:

1. The initial shallow state ages strongly, producing the broad VW slowdown
   and changes of rate/slip gradient through the 15–18-km transition.
2. Deeper elastic response and the hard Vp constraint at 40 km produce another
   sharp gradient change. These are localized spatial disturbances.
3. The represented work-measure surface/bulk operator has positive
   nearest-neighbour coupling and a non-monotone inverse response. A localized
   increase can therefore demand a neighbouring decrease to satisfy the weak
   equation. The 25-km experiment shows this is not unique to the 40-km law.
4. Once a node slows, aging increases its state and frictional resistance.
   The already completed candidate-state and timestep tests show amplification,
   not removal: same-input candidate coupling deepened the notch, and smaller
   substeps deepened it further with only modest contraction of changes.

At the last free node, the original accepted weak residual is the cancellation
of **+2.981115e6 Pa m** on its free/free element and **-2.981115e6 Pa m** on
its free/prescribed element. Pointwise balance on each element is not enforced.
These values and the new positive operator column explain how a converged
undershoot can occur without an active-set or Newton failure.

**What is not established:** these tests do not uniquely separate an intrinsic
finite-width response from bulk/fault space mismatch or all Q1 quadrature/
representation effects. In particular, the fault grid is finer than the
fault-crossed bulk grid near 40 km. Positive off-diagonal terms alone are not
proof of a coding bug, and mass lumping is not automatically a justified fix.
The transient 18.1-km dip is consistent with the same oscillatory mechanism,
but its entire history has not been causally apportioned by this test. At
15 km the present data support a broad state-driven slowdown, not a local
notch that still needs to be removed.

**Recommended next decision:** qualify the tangential bulk/fault response with
one frozen-history manufactured gradient-kink problem, comparing compatible
bulk/fault resolutions at fixed physical width and load. Measure the
bulk-relaxed operator and rate overshoot, not only a smoother BP3 plot. This
distinguishes a space-resolution correction from a formulation decision before
changing RSF transitions, imposing smoothing, or adopting a lumped rule.
No such correction or additional run is made in this task.

## 5. Verification, cost and reproducibility

| New four-rank case | Wall seconds | Fresh-linear checks | Krylov iterations | Final relative bulk/surface residual |
|---|---:|---:|---:|---|
| 1% junction impulse | 130.952 | 7 | 267 | 2.26243e-13 / 1.87760e-12 |
| 0.5% junction impulse | 136.629 | 7 | 267 | 2.27621e-13 / 1.87767e-12 |
| 25-km interior control | 133.357 | 7 | 268 | 2.26151e-13 / 1.87926e-12 |
| Bulk-relaxed column, all V prescribed | 50.120 | 2 | 39 | 2.51366e-10 / 0 by prescribed-row exclusion |

Total simulation time **451.057 s (7.52 min)**. Largest measured child peak
RSS **1,502,784 KiB**, approximately 1.43 GiB; this is not an aggregate MPI
memory measurement. No long trajectory, timestep/refinement campaign, new
history update, or broad test suite was run.

All **23** fresh-linear checks pass their requested targets. Existing complete
K/G derivative checks and pivoted-inverse/UMFPACK comparisons remain enabled.
All four solves satisfy the unchanged nonlinear criteria and intentionally
throw **after** diagnostics to exercise ordinary rollback. Complete owned
bulk values, manager V, particle IDs/positions/properties, surface properties
and geometry are unchanged afterward. Incoming per-rank fingerprints agree
with the saved A control; original/copied checkpoint hashes are unchanged;
no accepted-state output is written. Exit 1 alone is not treated as success.

Offline independent reconstruction of the actual step-10 work loads agrees
with the production rows to maximum errors **7.63e-8 Pa** (shear),
**1.62e-7 Pa** (normal), **2.36e-6 Pa** (friction) and **2.37e-6 Pa** (residual),
after division by row measure. The composition is interpolated **before**
conversion/clipping to material fractions, exactly as production requires.
The uniform reaction-stencil analytic check and zero-forcing control pass.

Base source revision: `3335d3d26c298ff5aaeba77062b0a77c8d20f0b5`.
Only guarded benchmark constraint/export additions were made in `bp3.cc`
and `within_step_diagnostic.h`; no `source/` or production-header edits.
New scripts, this report/addendum, and separate evidence files are uncommitted.
Release plugin builds used **-j4**, with the pre-existing range-loop-copy
warning in the derivative audit. Python compilation and `git diff --check`
pass. One initial MPI launch was blocked by sandbox socket permissions before
ASPECT started; its directory/log is preserved separately as `impulse-0.01/`.
This was not a failed mechanical solve or a parameter-retuned retry.

Commands (runners refuse to overwrite evidence):

```sh
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
python3 benchmarks/reconstructed_fault/bp3/run_notch_probe.py 0.01 --label impulse-0.01-local4
python3 benchmarks/reconstructed_fault/bp3/run_notch_probe.py 0.005 --label impulse-0.005-local4
python3 benchmarks/reconstructed_fault/bp3/run_notch_probe.py 0.01 --location 25000 --label interior-25000-local4
python3 benchmarks/reconstructed_fault/bp3/run_notch_probe.py 0.01 --clamp-others --label operator-column-local4
python3 benchmarks/reconstructed_fault/bp3/analyze_notch_mechanism.py
```

Each case under `benchmarks/reconstructed_fault/bp3/notch-mechanism/` has
`run.log`, `execution.json`, `provenance.json`, noncommitting surface/history
CSV, incoming fingerprints and derivative checks. Plugin/executable hashes
distinguish the incremental diagnostic builds. The final
[analysis manifest](../../../benchmarks/reconstructed_fault/bp3/notch-mechanism/analysis_provenance.json)
records the analysis source and reused input hashes. Large disposable
checkpoint copies remain local evidence; they need not be committed.
